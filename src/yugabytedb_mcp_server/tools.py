"""MCP tool implementations for yugabytedb-mcp-server.

All three tools acquire a connection from the lifespan-owned ConnectionPool,
run their query, and return JSON. Read tools wrap the query in
`BEGIN READ ONLY ... ROLLBACK` for transaction-level enforcement. Write tool
runs through the guardrail blocklist before executing.

When OIDC auth is active, each tool call wraps its SQL execution in
SET ROLE / RESET ROLE so the query runs under the database role corresponding
to the authenticated user's identity.
"""

import json
import logging
from contextlib import contextmanager
from typing import Any, Dict, List

from fastmcp import Context
from fastmcp.server.dependencies import get_access_token
from psycopg.sql import SQL, Identifier

from .guardrails import GuardrailConfig, QueryBlockedError, validate_write_query

logger = logging.getLogger("yugabytedb-mcp.tools")


def _execute(cur, query, params: tuple | None = None) -> None:
    """Thin execute wrapper that logs at DEBUG before running.

    `query` may be a plain string or a psycopg.sql.Composed/SQL object.
    """
    if params is not None:
        logger.debug("SQL: %s | params=%r", query, params)
        cur.execute(query, params)
    else:
        logger.debug("SQL: %s", query)
        cur.execute(query)


def _apply_transform(value: str, transform: str) -> str:
    """Apply a named transform to a claim value to derive a DB role name."""
    if transform == "strip_domain":
        return value.split("@", 1)[0]
    return value


class IdentityError(Exception):
    """Raised when an authenticated token lacks the required identity claim."""


def _get_db_role(ctx: Context) -> str | None:
    """Extract the database role from the authenticated user's OIDC token.

    Returns None when auth is disabled or no token is present (the pool's
    default credentials will be used).

    Raises IdentityError when a token IS present but the required claim is
    missing — falling back to pool credentials in this case would be a
    privilege escalation.
    """
    try:
        token = get_access_token()
    except RuntimeError:
        return None

    if token is None:
        return None

    lifespan = ctx.request_context.lifespan_context
    claim_name = lifespan.get("identity_claim", "email")
    transform = lifespan.get("identity_transform", "none")

    claim_value = token.claims.get(claim_name)
    if not claim_value:
        raise IdentityError(
            f"Token present but required claim {claim_name!r} is missing or empty. "
            f"Cannot determine database role for authenticated user."
        )

    role = _apply_transform(str(claim_value), transform)
    logger.debug("Resolved DB role %r from claim %r=%r", role, claim_name, claim_value)
    return role


@contextmanager
def _conn_as_role(pool, role: str | None):
    """Acquire a connection and optionally SET ROLE for the duration.

    If `role` is not None, executes SET ROLE before yielding and RESET ROLE
    in the finally block so the connection is returned to the pool clean.
    """
    with pool.connection() as conn:
        if role is not None:
            with conn.cursor() as cur:
                cur.execute(SQL("SET ROLE {}").format(Identifier(role)))
            logger.debug("SET ROLE %s", role)
        try:
            yield conn
        finally:
            if role is not None:
                with conn.cursor() as cur:
                    cur.execute("RESET ROLE")
                logger.debug("RESET ROLE")


def summarize_database(ctx: Context, schema: str = "public") -> List[Dict[str, Any]]:
    """
    Summarize a database schema: list every table with its column schema and
    row count.

    Use this to explore the database structure before writing queries —
    `run_read_only_query` against `information_schema.tables` would also work
    but this is a more compact summary.

    Args:
        ctx: MCP context (injected automatically).
        schema: Schema name to inspect (default: ``public``).
    """
    logger.info("summarize_database called (schema=%s)", schema)
    summary: List[Dict[str, Any]] = []
    pool = ctx.request_context.lifespan_context["pool"]

    try:
        role = _get_db_role(ctx)
    except IdentityError as e:
        logger.error("Identity resolution failed: %s", e)
        return [{"error": str(e)}]

    with _conn_as_role(pool, role) as conn:
        logger.debug("Acquired connection from pool for summarize_database")
        with conn.cursor() as cur:
            try:
                _execute(cur, "BEGIN READ ONLY")
                _execute(
                    cur,
                    """
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = %s
                    ORDER BY table_name
                    """,
                    (schema,),
                )
                tables = [row[0] for row in cur.fetchall()]
                logger.debug("Schema %s has %d tables: %s", schema, len(tables), tables)

                for table in tables:
                    _execute(
                        cur,
                        """
                        SELECT column_name, data_type
                        FROM information_schema.columns
                        WHERE table_schema = %s AND table_name = %s
                        ORDER BY ordinal_position
                        """,
                        (schema, table),
                    )
                    schema_info = [
                        {"column_name": col, "data_type": dtype}
                        for col, dtype in cur.fetchall()
                    ]

                    _execute(cur, f'SELECT COUNT(*) FROM {schema}."{table}"')
                    row_count = cur.fetchone()[0]
                    logger.debug(
                        "Table %s.%s: %d columns, %d rows",
                        schema, table, len(schema_info), row_count,
                    )

                    summary.append({
                        "table": table,
                        "row_count": row_count,
                        "schema": schema_info,
                    })

            except Exception as e:
                logger.error("Error summarizing schema %s: %s", schema, e, exc_info=True)
                summary.append({"error": str(e)})
            finally:
                try:
                    _execute(cur, "ROLLBACK")
                except Exception as e:
                    logger.error("Failed to ROLLBACK in summarize_database: %s", e)

    logger.info("summarize_database returning %d entries for schema=%s", len(summary), schema)
    return summary


def run_read_only_query(ctx: Context, query: str) -> str:
    """
    Run a read-only SQL query under BEGIN READ ONLY and return the rows as
    JSON.

    Any data-mutating statement is rejected by the database itself because
    of the read-only transaction.

    Args:
        ctx: MCP context (injected automatically).
        query: SQL statement (typically SELECT) to execute.
    """
    logger.info("run_read_only_query called")
    logger.debug("Query: %s", query)
    pool = ctx.request_context.lifespan_context["pool"]

    try:
        role = _get_db_role(ctx)
    except IdentityError as e:
        logger.error("Identity resolution failed: %s", e)
        return json.dumps({"error": str(e)})

    with _conn_as_role(pool, role) as conn:
        logger.debug("Acquired connection from pool for run_read_only_query")
        with conn.cursor() as cur:
            try:
                _execute(cur, "BEGIN READ ONLY")
                _execute(cur, query)
                rows = cur.fetchall()
                column_names = [desc[0] for desc in cur.description]
                result = [dict(zip(column_names, row)) for row in rows]
                logger.info(
                    "run_read_only_query returned %d rows × %d columns",
                    len(rows), len(column_names),
                )
                return json.dumps(result, indent=2, default=str)
            except Exception as e:
                logger.error("Error executing read-only query: %s", e, exc_info=True)
                return f"Error executing query: {e}"
            finally:
                try:
                    _execute(cur, "ROLLBACK")
                except Exception as e:
                    logger.error("Failed to ROLLBACK read-only transaction: %s", e)


def run_write_query(ctx: Context, query: str) -> str:
    """
    Execute a write SQL statement (INSERT/UPDATE/DELETE/MERGE/TRUNCATE/DDL)
    after guardrail validation. Returns a JSON object with `rows_affected`
    on success or `error` on failure.

    Guardrails reject the highest-risk statement classes (DROP DATABASE,
    ALTER SYSTEM, role/privilege ops, COPY, filesystem functions, dblink,
    multi-statement queries, INSERTs over the configured row limit, and
    optionally UPDATE / DELETE without WHERE). This list is best-effort;
    Claude Desktop also surfaces a confirmation prompt because of the
    tool's destructiveHint.

    Args:
        ctx: MCP context (injected automatically).
        query: SQL statement to execute.
    """
    logger.info("run_write_query called")
    logger.debug("Query: %s", query)
    lifespan = ctx.request_context.lifespan_context
    pool = lifespan["pool"]
    guardrail_config: GuardrailConfig = lifespan["guardrail_config"]

    try:
        validate_write_query(query, guardrail_config)
    except QueryBlockedError as e:
        logger.warning("Query blocked by guardrail: %s", e)
        return json.dumps({"error": str(e), "blocked_by_guardrail": True})

    try:
        role = _get_db_role(ctx)
    except IdentityError as e:
        logger.error("Identity resolution failed: %s", e)
        return json.dumps({"error": str(e)})

    with _conn_as_role(pool, role) as conn:
        logger.debug("Acquired connection from pool for run_write_query")
        with conn.cursor() as cur:
            try:
                _execute(cur, query)
                conn.commit()
                logger.info("run_write_query committed: %d rows affected", cur.rowcount)
                return json.dumps({"rows_affected": cur.rowcount})
            except Exception as e:
                conn.rollback()
                logger.error("Write query failed, rolled back: %s", e, exc_info=True)
                return json.dumps({"error": str(e)})
