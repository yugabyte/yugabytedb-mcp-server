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
import re
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

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
    """Apply a named transform to a claim value to derive a DB role name.

    Backward-compat helper used by the no-map path (when
    ``YB_MCP_IDENTITY_MAP`` is unset). The map-file path replaces this with
    an explicit map lookup — see ``_apply_map`` and ``_load_identity_map``.
    """
    if transform == "strip_domain":
        return value.split("@", 1)[0]
    return value


class IdentityError(Exception):
    """Raised when an authenticated token lacks the required identity claim."""


# ---------------------------------------------------------------------------
# v2 identity mapping — YSQL native OIDC parity.
#
# Design mirrors YSQL's `matching_claim_key` + `ysql_ident_conf_csv` (see
# https://docs.yugabyte.com/stable/yugabyte-platform/security/authentication/oidc-authentication-aad/):
#
# - `YB_MCP_IDENTITY_CLAIM` accepts dotted paths (`realm_access.roles`,
#   `cognito:groups`) — dot walks nested dicts, colon is a literal key char.
# - Claim value may be a list (e.g. Cognito's `cognito:groups`, Keycloak's
#   `realm_access.roles`, Azure's `groups`). The caller (agent) selects one
#   via the tool's `requested_role` parameter; server clamps against the
#   JWT's list.
# - `YB_MCP_IDENTITY_MAP` points at a `pg_ident.conf`-style file with
#   literal or regex entries. When configured, replaces the legacy
#   `_apply_transform` path.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MapEntry:
    """One line from the identity map file.

    ``pattern`` is the raw system-value field (kept for error messages);
    ``compiled`` is the compiled regex when ``is_regex`` is True.
    """
    name: str
    pattern: str
    role: str
    is_regex: bool
    compiled: Optional[re.Pattern] = field(default=None, compare=False)


def _extract_claim(claims: Dict[str, Any], path: str) -> Any:
    """Walk a dotted claim path through the JWT claims dict.

    A dot (``.``) separates path segments; colons (``:``) are part of the
    key name — Cognito's ``cognito:groups`` is one top-level key, not two
    nested keys. Missing intermediate segments raise ``IdentityError``.

    Fast path: if ``path`` has no dot, this is a single dict lookup.
    """
    if "." not in path:
        if path not in claims:
            raise IdentityError(f"Claim {path!r} not found in token")
        return claims[path]

    parts = path.split(".")
    cursor: Any = claims
    for i, segment in enumerate(parts):
        if not isinstance(cursor, dict):
            raise IdentityError(
                f"Claim path {path!r}: cannot descend into non-dict at "
                f"{'.'.join(parts[:i]) or '<root>'}"
            )
        if segment not in cursor:
            raise IdentityError(
                f"Claim path {path!r}: key {segment!r} missing at "
                f"{'.'.join(parts[:i]) or '<root>'}"
            )
        cursor = cursor[segment]
    return cursor


def _load_identity_map(path: str) -> List[MapEntry]:
    """Parse a ``pg_ident.conf``-style identity map file.

    Format (matches PostgreSQL's user-name-map file format, used by YSQL's
    ``ysql_ident_conf_csv``):

    - Each non-empty, non-comment line has three space-separated fields:
      ``<map_name> <system_value> <db_role>``.
    - Lines starting with ``#`` (or having ``#`` mid-line for a trailing
      comment) are treated as comments.
    - ``system_value`` starting with ``/`` is a regex pattern — the rest of
      the field (no closing ``/``) is compiled with ``re.compile`` and
      matched with ``fullmatch``. The ``db_role`` field may reference
      capture groups via ``\\1``, ``\\2``, ....
    - Malformed lines raise ``ValueError`` — caller (server startup) should
      let this propagate to fail-closed instead of silently accepting a
      typo'd map.
    """
    entries: List[MapEntry] = []
    with open(path) as f:
        for lineno, raw_line in enumerate(f, start=1):
            # Strip inline comments and surrounding whitespace.
            line = raw_line.rstrip("\n")
            hash_idx = line.find("#")
            if hash_idx >= 0:
                line = line[:hash_idx]
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 2)
            if len(parts) != 3:
                raise ValueError(
                    f"{path}:{lineno}: expected 3 space-separated fields "
                    f"(<map_name> <system_value> <db_role>); got {len(parts)}: {raw_line!r}"
                )
            name, system_value, role = parts
            is_regex = system_value.startswith("/")
            compiled: Optional[re.Pattern] = None
            if is_regex:
                pattern_body = system_value[1:]
                try:
                    compiled = re.compile(pattern_body)
                except re.error as e:
                    raise ValueError(
                        f"{path}:{lineno}: invalid regex {system_value!r}: {e}"
                    ) from e
            entries.append(MapEntry(
                name=name,
                pattern=system_value,
                role=role,
                is_regex=is_regex,
                compiled=compiled,
            ))
    return entries


def _apply_map(
    value: str,
    entries: List[MapEntry],
    map_name: str,
) -> Optional[str]:
    """Try to resolve ``value`` to a DB role via the map entries.

    Iterates entries in order; returns the first match's role (with
    ``\\1``-style substitutions applied for regex entries). Returns ``None``
    when no entry under ``map_name`` matches the value — caller decides
    whether "unmapped" is an error (typical) or a fall-through case (never,
    in this codebase).
    """
    for entry in entries:
        if entry.name != map_name:
            continue
        if entry.is_regex:
            # PostgreSQL pg_ident.conf matches anchored — use fullmatch.
            m = entry.compiled.fullmatch(value)
            if m is not None:
                return m.expand(entry.role)
        else:
            if value == entry.pattern:
                return entry.role
    return None


def _pick_role(candidates: List[str], requested_role: Optional[str]) -> str:
    """Choose one role from a list of candidates.

    The user picks any PG role that appears in their
    roles/groups claim. Here the "user" is the agent, and ``requested_role``
    is its choice; server clamps against the candidate list so the agent
    cannot pick a role that isn't in the JWT.

    Behavior:
    - Empty candidates → ``IdentityError`` (nothing to pick from).
    - Single candidate → auto-pick.
    - Multiple candidates + ``requested_role`` in list → return it.
    - Multiple candidates + ``requested_role`` NOT in list → ``IdentityError``.
    - Multiple candidates + ``requested_role is None`` → default to first
      with a WARNING (documented behavior; agent should pass
      ``requested_role`` for determinism).
    """
    if not candidates:
        raise IdentityError(
            "None of the identity-claim values resolved to a permitted DB role. "
            "Check YB_MCP_IDENTITY_MAP configuration."
        )
    if len(candidates) == 1:
        return candidates[0]
    if requested_role is None:
        logger.warning(
            "Identity claim resolved to multiple roles %s but no requested_role "
            "was passed. Defaulting to first entry %r. Pass requested_role=<name> "
            "to disambiguate.",
            candidates, candidates[0],
        )
        return candidates[0]
    if requested_role in candidates:
        return requested_role
    raise IdentityError(
        f"requested_role={requested_role!r} is not in the caller's identity-claim "
        f"candidates {candidates}. The agent must pick a role that appears in the "
        f"JWT's mapped list."
    )


def _get_db_role(ctx: Context, requested_role: Optional[str] = None) -> Optional[str]:
    """Extract the database role from the authenticated user's OIDC token.

    Returns None when auth is disabled or no token is present (the pool's
    default credentials will be used).

    v2 behavior (see ``_extract_claim`` / ``_apply_map`` / ``_pick_role``):
    - ``identity_claim`` may be a dotted path (``realm_access.roles``,
      ``cognito:groups``).
    - Claim may be a list — ``requested_role`` selects one; server clamps
      against the JWT's list.
    - When ``YB_MCP_IDENTITY_MAP`` is configured, the resolved claim value(s)
      go through ``_apply_map`` (pg_ident.conf-style lookup). Otherwise the
      legacy ``_apply_transform`` path runs (v1 backward-compat).

    Raises ``IdentityError`` when a token IS present but the claim resolves
    to nothing — falling back to pool credentials would be a privilege
    escalation.
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
    identity_map: Optional[List[MapEntry]] = lifespan.get("identity_map")
    identity_map_name: str = lifespan.get("identity_map_name", "default")

    _missing_msg = (
        f"Token present but required claim {claim_name!r} is missing or empty. "
        f"Cannot determine database role for authenticated user."
    )

    # 1. Extract claim (supports dotted paths).
    try:
        claim_value = _extract_claim(token.claims, claim_name)
    except IdentityError:
        # Normalize to the v1 error message shape so existing callers /
        # tests get an identical string on the "missing claim" path.
        raise IdentityError(_missing_msg)

    # 2. Normalize the claim to a list of non-empty string values. This is
    #    where scalar and list-valued claims converge into one code path.
    if isinstance(claim_value, list):
        raw_values = [str(v) for v in claim_value if v]
        if not raw_values:
            raise IdentityError(_missing_msg)
    else:
        if not claim_value:
            raise IdentityError(_missing_msg)
        raw_values = [str(claim_value)]

    # 3. Resolve each raw value to a candidate role.
    #    - With a map: apply pg_ident-style lookup; drop unmapped values.
    #    - Without a map (v1 backward-compat): apply transform to each.
    if identity_map is not None:
        candidates: List[str] = []
        for v in raw_values:
            mapped = _apply_map(v, identity_map, identity_map_name)
            if mapped is not None:
                candidates.append(mapped)
    else:
        candidates = [_apply_transform(v, transform) for v in raw_values]

    # 4. Pick one role.
    role = _pick_role(candidates, requested_role)

    logger.debug(
        "Resolved DB role %r from claim %r (raw=%s, candidates=%s, requested=%r, mapped=%s)",
        role, claim_name, raw_values, candidates, requested_role,
        identity_map is not None,
    )
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


def summarize_database(
    ctx: Context,
    schema: str = "public",
    requested_role: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Summarize a database schema: list every table with its column schema and
    row count.

    Use this to explore the database structure before writing queries —
    `run_read_only_query` against `information_schema.tables` would also work
    but this is a more compact summary.

    Args:
        ctx: MCP context (injected automatically).
        schema: Schema name to inspect (default: ``public``).
        requested_role: When the identity-claim JWT value is a list (e.g.
            ``cognito:groups=["writer","reader"]``), pick which role to
            SET ROLE to. Must be a value that appears in the mapped
            candidate list — the server clamps against the JWT. Ignored
            when the claim is a scalar (single email/sub/etc.).
    """
    logger.info("summarize_database called (schema=%s)", schema)
    summary: List[Dict[str, Any]] = []
    pool = ctx.request_context.lifespan_context["pool"]

    try:
        role = _get_db_role(ctx, requested_role=requested_role)
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


def run_read_only_query(
    ctx: Context,
    query: str,
    requested_role: Optional[str] = None,
) -> str:
    """
    Run a read-only SQL query under BEGIN READ ONLY and return the rows as
    JSON.

    Any data-mutating statement is rejected by the database itself because
    of the read-only transaction.

    Args:
        ctx: MCP context (injected automatically).
        query: SQL statement (typically SELECT) to execute.
        requested_role: When the identity-claim JWT value is a list (e.g.
            ``cognito:groups=["writer","reader"]``), pick which role to
            SET ROLE to. Must be a value that appears in the mapped
            candidate list — the server clamps against the JWT.
    """
    logger.info("run_read_only_query called")
    logger.debug("Query: %s", query)
    pool = ctx.request_context.lifespan_context["pool"]

    try:
        role = _get_db_role(ctx, requested_role=requested_role)
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


def run_write_query(
    ctx: Context,
    query: str,
    requested_role: Optional[str] = None,
) -> str:
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
        requested_role: When the identity-claim JWT value is a list, pick
            which role to SET ROLE to for this write. Must be a value that
            appears in the mapped candidate list — server clamps against
            the JWT.
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
        role = _get_db_role(ctx, requested_role=requested_role)
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
