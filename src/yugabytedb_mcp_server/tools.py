"""MCP tool implementations for yugabytedb-mcp-server.

All three tools acquire a connection from the lifespan-owned ConnectionPool,
run their query, and return JSON. Read tools wrap the query in
`BEGIN READ ONLY ... ROLLBACK` for transaction-level enforcement. Write tool
runs through the guardrail blocklist before executing.
"""

import json
import logging
from typing import Any, Dict, List

from fastmcp import Context

from .guardrails import GuardrailConfig, QueryBlockedError, validate_write_query

logger = logging.getLogger("yugabytedb-mcp.tools")


def _execute(cur, query: str, params: tuple | None = None) -> None:
    """Thin execute wrapper that logs at DEBUG before running."""
    if params is not None:
        logger.debug("SQL: %s | params=%r", query.strip(), params)
        cur.execute(query, params)
    else:
        logger.debug("SQL: %s", query.strip())
        cur.execute(query)


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

    with pool.connection() as conn:
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

    with pool.connection() as conn:
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

    with pool.connection() as conn:
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
