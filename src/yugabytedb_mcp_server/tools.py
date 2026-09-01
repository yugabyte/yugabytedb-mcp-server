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
import math
import re
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from fastmcp import Context
from fastmcp.server.dependencies import get_access_token
from psycopg.sql import SQL, Identifier

from .guardrails import GuardrailConfig, QueryBlockedError, validate_query

logger = logging.getLogger("yugabytedb-mcp.tools")


def _sanitize_for_json(value: Any) -> Any:
    """Convert values into JSON-safe forms before serialization.

    Handles two classes of bugs in the previous `json.dumps(..., default=str)`
    path:

    - Float `NaN`, `Infinity`, `-Infinity` — Python's `json.dumps` emits them
      as bare tokens, which are invalid JSON per RFC 8259 (JS `JSON.parse`
      rejects them). Map non-finite floats to `None`.
    - `bytes` / `memoryview` — the previous `default=str` produced the lossy
      Python bytes-repr (`"b'\\xde\\xad\\xbe\\xef'"`). Encode as
      `{"$hex": "deadbeef"}` for a lossless, well-defined round-trip.

    Recursive so nested `list` / `dict` / array-column values are covered.
    Any type not matched here falls through to `json.dumps`'s `default=str`
    for datetime / Decimal / UUID / IPv4Address etc. (existing coverage).
    """
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"$hex": bytes(value).hex()}
    if isinstance(value, list):
        return [_sanitize_for_json(v) for v in value]
    if isinstance(value, tuple):
        return [_sanitize_for_json(v) for v in value]
    if isinstance(value, dict):
        return {k: _sanitize_for_json(v) for k, v in value.items()}
    return value


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
#   literal or regex entries. When configured, each token claim value is
#   looked up in the map to derive the DB role. Without a map, the raw
#   claim value is used as the role name directly.
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


def _strip_inline_comment(line: str) -> str:
    """Return ``line`` with any pg_ident-style comment stripped.

    Only treats ``#`` as a comment marker when it starts the line or is
    preceded by whitespace. This preserves regexes / role names that
    contain a literal ``#`` mid-token (e.g. ``/user#\\d+/``).
    """
    i = 0
    while True:
        j = line.find("#", i)
        if j < 0:
            return line
        if j == 0 or line[j - 1] in " \t":
            return line[:j]
        i = j + 1


def _validate_db_role_template(
    role: str, compiled: Optional[re.Pattern], path: str, lineno: int,
) -> None:
    """Fail-closed at load time on a db_role template with a broken
    backreference (``\\N`` referencing a group that doesn't exist).

    Without this, ``m.expand(role)`` would raise ``re.error`` at request
    time for every match, which surfaces as an unhandled 500 rather than
    a clean startup refusal.
    """
    if compiled is None:
        # Literal mapping — no expansion, so any bytes are fine.
        return
    # sre_parse.parse_template validates group references against the
    # pattern; anything malformed raises re.error here at LOAD time.
    try:
        re.compile(role)  # noqa: F841 -- just to reject obviously invalid
    except re.error:
        pass  # role isn't itself a regex; we only care about backref shape
    # The real check: try an actual expand against a synthetic match. Use a
    # string that satisfies the pattern (or the empty string if it doesn't
    # anchor), catch the specific "invalid group reference" family.
    try:
        # Compile a throwaway template checker: re.Match.expand triggers
        # sre_parse.parse_template internally on the template string. We
        # only need to know if expand would raise for THIS pattern's group
        # count, which we can get with .groups.
        n_groups = compiled.groups
        for ref in re.finditer(r"\\(\d+)", role):
            g = int(ref.group(1))
            if g > n_groups:
                raise ValueError(
                    f"{path}:{lineno}: db_role template {role!r} references "
                    f"capture group \\{g}, but the pattern only has "
                    f"{n_groups} group{'s' if n_groups != 1 else ''}."
                )
    except ValueError:
        raise
    except re.error as e:
        raise ValueError(
            f"{path}:{lineno}: db_role template {role!r} is invalid: {e}"
        ) from e


def _load_identity_map(path: str) -> Dict[str, List[MapEntry]]:
    """Parse a ``pg_ident.conf``-style identity map file, indexed by map_name.

    Returns a dict of ``map_name → ordered list of that map's entries``. The
    per-name grouping is done once at startup so ``_apply_map`` doesn't
    re-filter every entry per value.

    Format (matches PostgreSQL's user-name-map file format, used by YSQL's
    ``ysql_ident_conf_csv``):

    - Each non-empty, non-comment line has three space-separated fields:
      ``<map_name> <system_value> <db_role>``.
    - ``#`` starts a comment only at line-start or after whitespace, so a
      literal ``#`` inside a regex or role name is preserved.
    - ``system_value`` starting with ``/`` is a regex pattern — the rest of
      the field (no closing ``/``) is compiled with ``re.compile`` and
      matched with ``fullmatch``. The ``db_role`` field may reference
      capture groups via ``\\1``, ``\\2``, ....
    - Malformed lines raise ``ValueError`` — caller (server startup) should
      let this propagate to fail-closed instead of silently accepting a
      typo'd map. Group-reference validity (``\\N`` in db_role) is also
      checked here.
    """
    by_name: Dict[str, List[MapEntry]] = {}
    with open(path) as f:
        for lineno, raw_line in enumerate(f, start=1):
            line = _strip_inline_comment(raw_line.rstrip("\n")).strip()
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
            _validate_db_role_template(role, compiled, path, lineno)
            by_name.setdefault(name, []).append(MapEntry(
                name=name,
                pattern=system_value,
                role=role,
                is_regex=is_regex,
                compiled=compiled,
            ))
    return by_name


def _apply_map(
    value: str,
    identity_map: Dict[str, List[MapEntry]],
    map_name: str,
) -> Optional[str]:
    """Try to resolve ``value`` to a DB role via the ``map_name`` entries.

    Iterates the entries under ``map_name`` in order; returns the first
    match's role (with ``\\1``-style substitutions applied for regex
    entries). Returns ``None`` when no entry matches the value — caller
    decides whether "unmapped" is an error (typical) or a fall-through
    case (never, in this codebase). Also returns ``None`` if a regex
    match expands to an empty string (a capture group matching ""); an
    empty role name is never a valid SET ROLE target and would fail with
    a confusing DB error.
    """
    for entry in identity_map.get(map_name, ()):
        if entry.is_regex:
            # PostgreSQL pg_ident.conf matches anchored — use fullmatch.
            m = entry.compiled.fullmatch(value)
            if m is not None:
                expanded = m.expand(entry.role)
                if expanded == "":
                    # Empty expansion → treat as unmapped so the caller
                    # gets a clean IdentityError instead of a downstream
                    # `SET ROLE ""` failure.
                    continue
                return expanded
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
    - Empty candidates → ``IdentityError``.
    - Any ``requested_role`` not present in ``candidates`` → ``IdentityError``
      (regardless of candidate count — a mismatch is never silently ignored).
    - Single candidate + ``requested_role is None`` → auto-pick.
    - Multiple candidates + ``requested_role is None`` → ``IdentityError``
      (fail-closed on ambiguity; the agent must disambiguate explicitly).
    - Any candidate count + ``requested_role`` in list → return it.
    """
    if not candidates:
        raise IdentityError(
            "None of the identity-claim values resolved to a permitted DB role. "
            "Check YB_MCP_IDENTITY_MAP configuration."
        )
    if requested_role is not None:
        if requested_role in candidates:
            return requested_role
        raise IdentityError(
            f"requested_role={requested_role!r} is not in the caller's "
            f"identity-claim candidates {candidates}. The agent must pick a "
            f"role that appears in the JWT's mapped list."
        )
    # requested_role is None from here on.
    if len(candidates) == 1:
        return candidates[0]
    raise IdentityError(
        f"Identity claim resolved to multiple roles {candidates} but no "
        f"requested_role was passed. Pass requested_role=<name> to pick one — "
        f"the server refuses to default arbitrarily to avoid granting the "
        f"more-privileged role by accident."
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
      raw claim value is used verbatim as the DB role name.

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
    claim_name = lifespan.get("identity_claim", "sub")
    identity_map: Optional[Dict[str, List[MapEntry]]] = lifespan.get("identity_map")
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
    #
    # DB-22135 fail-closed at request time: any *composite* claim value
    # (list, tuple, dict) without a map means the caller could hand us
    # arbitrary group names / structures which we'd feed verbatim to
    # `SET ROLE` — no allowlist. Vishal's re-open comment: "fail closed
    # whenever there is no allowlist/map and the claim is not a fixed
    # enumerated value (regardless of claim name)". Dicts and tuples are
    # rare but not fixed either. The startup guard can only see the claim
    # NAME, so it misses composite claims served under unrecognized names
    # (`roles`, `entitlements`, custom scopes) — catch them here where
    # the actual value type is known.
    if isinstance(claim_value, (list, tuple)):
        raw_values = [str(v) for v in claim_value if v]
        if not raw_values:
            raise IdentityError(_missing_msg)
        if identity_map is None:
            raise IdentityError(
                f"Claim {claim_name!r} resolved to a "
                f"{type(claim_value).__name__} of {len(raw_values)} "
                f"value(s), but YB_MCP_IDENTITY_MAP is unset. A composite "
                f"claim without a map has no allowlist — every value "
                f"would be a candidate SET ROLE target. Configure "
                f"YB_MCP_IDENTITY_MAP to translate IdP values to a fixed "
                f"PG role set, or switch YB_MCP_IDENTITY_CLAIM to a "
                f"scalar claim like `sub` or `email`."
            )
    elif isinstance(claim_value, dict):
        if identity_map is None:
            raise IdentityError(
                f"Claim {claim_name!r} resolved to a dict, but "
                f"YB_MCP_IDENTITY_MAP is unset. A composite claim "
                f"without a map has no allowlist and no defined mapping "
                f"to a scalar role name. Configure YB_MCP_IDENTITY_MAP, "
                f"or switch YB_MCP_IDENTITY_CLAIM to a scalar claim like "
                f"`sub` or `email`."
            )
        # With a map: fall through with a synthesized "raw value" that
        # the map lookup can key off. Since the map is admin-controlled,
        # a stringified dict is fine as a lookup key — the operator
        # decides whether to accept it.
        if not claim_value:
            raise IdentityError(_missing_msg)
        raw_values = [str(claim_value)]
    else:
        if not claim_value:
            raise IdentityError(_missing_msg)
        raw_values = [str(claim_value)]

    # 3. Resolve each raw value to a candidate role.
    #    - With a map: apply pg_ident-style lookup; drop unmapped values.
    #    - Without a map: use the claim value verbatim as the DB role name.
    if identity_map is not None:
        candidates: List[str] = []
        for v in raw_values:
            mapped = _apply_map(v, identity_map, identity_map_name)
            if mapped is not None:
                candidates.append(mapped)
    else:
        candidates = list(raw_values)

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
    lifespan = ctx.request_context.lifespan_context
    pool = lifespan["pool"]
    # DB-22159 round-2: previously only run_read_only_query /
    # run_write_query enforced the statement_timeout; a `COUNT(*)` over a
    # huge table via summarize_database could hold a pool connection
    # indefinitely. Apply the same cap here.
    statement_timeout_ms = lifespan["statement_timeout_ms"]

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
                    f"SET LOCAL statement_timeout = '{statement_timeout_ms}ms'",
                )

                # Initial tables lookup — if this fails there's nothing to
                # iterate; report the error and unwind.
                try:
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
                except Exception as e:
                    logger.error(
                        "Error listing tables in schema %s: %s",
                        schema, e, exc_info=True,
                    )
                    summary.append({"error": str(e)})
                    tables = []
                logger.debug("Schema %s has %d tables: %s", schema, len(tables), tables)

                # DB-22138: per-table error handling. A single problematic
                # object (view over a since-dropped table, permission-denied
                # table, division-by-zero in a view, foreign-table connection
                # error…) previously aborted the whole loop and discarded
                # every other table's row count. Wrap each iteration in a
                # SAVEPOINT so the failed COUNT rolls back and iteration
                # continues; record the error against the offending table.
                for i, table in enumerate(tables):
                    sp = f"tbl_{i}"
                    try:
                        _execute(cur, SQL("SAVEPOINT {}").format(Identifier(sp)))
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

                        # DB-22158: `schema` is interpolated unquoted here
                        # — pre-existing gap, deferred. The per-table
                        # SAVEPOINT above does not narrow or widen that
                        # surface; it only prevents one bad object from
                        # aborting the loop.
                        _execute(cur, f'SELECT COUNT(*) FROM {schema}."{table}"')
                        row_count = cur.fetchone()[0]
                        _execute(cur, SQL("RELEASE SAVEPOINT {}").format(Identifier(sp)))
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
                        logger.warning(
                            "Error summarizing %s.%s (recording per-table error "
                            "and continuing): %s",
                            schema, table, e,
                        )
                        # Roll back to the pre-COUNT state and release the
                        # savepoint so the next iteration's SAVEPOINT is
                        # clean. Cleanup failures are logged but don't stop
                        # iteration — the outer ROLLBACK will unwind the
                        # whole transaction if the connection is truly
                        # unrecoverable.
                        try:
                            _execute(cur, SQL("ROLLBACK TO SAVEPOINT {}").format(Identifier(sp)))
                            _execute(cur, SQL("RELEASE SAVEPOINT {}").format(Identifier(sp)))
                        except Exception as cleanup_err:
                            logger.error(
                                "Failed to roll back savepoint %s for %s.%s: %s",
                                sp, schema, table, cleanup_err,
                            )
                        summary.append({"table": table, "error": str(e)})
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
    Run a read-only SQL query under BEGIN READ ONLY and return the result
    as JSON: `{"columns": [<name>, ...], "rows": [[<val>, ...], ...]}`.

    Columns and rows are returned as parallel arrays rather than a
    list-of-dicts so duplicate output-column names (e.g. `SELECT 1 AS id,
    2 AS id` or `SELECT *` over a join) are preserved losslessly. In the
    previous list-of-dicts shape, dict-key collision silently dropped all
    but the last duplicate.

    The query is validated against the same dangerous-function blocklist
    used by run_write_query before it reaches the database: side-effecting
    built-ins (COPY … TO PROGRAM, pg_read_file, dblink, pg_sleep, …) are
    rejected even though BEGIN READ ONLY would let some of them run.

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
    lifespan = ctx.request_context.lifespan_context
    pool = lifespan["pool"]
    guardrail_config: GuardrailConfig = lifespan["guardrail_config"]
    # DB-22159 resource caps
    max_query_len = lifespan["max_query_len"]
    statement_timeout_ms = lifespan["statement_timeout_ms"]
    max_result_rows = lifespan["max_result_rows"]
    max_result_bytes = lifespan["max_result_bytes"]

    # Reject oversized queries BEFORE parsing or opening a connection.
    # Vishal observed a ~1MB write query burning ~6s of CPU in the
    # guardrail parser alone; keep the pre-parse surface flat. Byte
    # length (UTF-8) matches the DoS rationale and the env-var
    # documentation.
    query_bytes = len(query.encode("utf-8"))
    if query_bytes > max_query_len:
        logger.warning(
            "run_read_only_query rejected: query %d bytes exceeds "
            "max_query_len=%d",
            query_bytes, max_query_len,
        )
        return json.dumps({
            "error": (
                f"Query length {query_bytes} bytes exceeds "
                f"YB_MCP_MAX_QUERY_LEN ({max_query_len} bytes)."
            ),
            "blocked_by_guardrail": True,
        })

    # Read-side guardrail (PR #9): blocks pg_read_file, COPY … TO
    # PROGRAM, pg_sleep, and other dangerous SELECT-shaped surfaces.
    try:
        validate_query(query, guardrail_config, read_only=True)
    except QueryBlockedError as e:
        logger.warning("Read query blocked by guardrail: %s", e)
        return json.dumps({"error": str(e), "blocked_by_guardrail": True})

    try:
        role = _get_db_role(ctx, requested_role=requested_role)
    except IdentityError as e:
        logger.error("Identity resolution failed: %s", e)
        return json.dumps({"error": str(e)})

    with _conn_as_role(pool, role) as conn:
        logger.debug("Acquired connection from pool for run_read_only_query")
        # Control cursor drives BEGIN / SET LOCAL / ROLLBACK. The named
        # cursor below runs the user's SELECT as a server-side cursor
        # (`DECLARE … CURSOR FOR SELECT`) so rows are fetched on demand
        # via `FETCH FORWARD N` instead of being materialized in a
        # client-side buffer at execute() time. This is what actually
        # makes the byte cap enforceable — a wide-row query no longer
        # gets buffered wholesale before we count bytes.
        with conn.cursor() as ctrl_cur:
            try:
                _execute(ctrl_cur, "BEGIN READ ONLY")
                # DB-22159: SET LOCAL statement_timeout scopes the cap to
                # this transaction only — the timeout dies with the ROLLBACK
                # below, so it doesn't leak across pool checkouts.
                _execute(
                    ctrl_cur,
                    f"SET LOCAL statement_timeout = '{statement_timeout_ms}ms'",
                )
                # DB-22159 round-2: server-side cursor + small itersize.
                # psycopg3's client-side cursor buffers ALL rows at
                # ``execute()`` time, defeating any downstream byte cap;
                # a named cursor issues DECLARE/FETCH so the DB streams
                # to us. itersize=10 keeps per-FETCH memory bounded to
                # ~10 rows at whatever their intrinsic width happens to
                # be, so a `SELECT repeat('x', N) FROM …` pathological
                # case still buffers at most 10*N bytes per round-trip
                # instead of every row × N.
                with conn.cursor(name="mcp_read_only") as cur:
                    cur.itersize = 10
                    _execute(cur, query)
                    rows = []
                    approx_bytes = 0
                    truncated_by_rows = False
                    truncated_by_bytes = False
                    for row in cur:
                        if len(rows) >= max_result_rows:
                            truncated_by_rows = True
                            break
                        # Approximate serialized cost: string length of
                        # each value plus a couple of bytes for JSON
                        # delimiters (`, ` between values, `[]` for the
                        # row). Cheap and bounded — psycopg has already
                        # materialized this row in memory to hand it to
                        # us, so the string coercion doesn't add
                        # asymptotic overhead.
                        row_bytes = sum(len(str(v)) for v in row) + 2 * len(row)
                        if approx_bytes + row_bytes > max_result_bytes:
                            truncated_by_bytes = True
                            break
                        rows.append(row)
                        approx_bytes += row_bytes
                    truncated = truncated_by_rows or truncated_by_bytes
                    column_names = [desc[0] for desc in cur.description]
                # Use the parallel-arrays shape (PR #9 / DB-22203) so
                # duplicate column names don't collapse; still robust to a
                # `SELECT a.id, b.id FROM ...` after a join.
                result: dict = {
                    "columns": column_names,
                    "rows": [list(row) for row in rows],
                }
                if truncated:
                    result["truncated"] = True
                    result["returned_rows"] = len(rows)
                    reason = (
                        "YB_MCP_MAX_RESULT_ROWS "
                        f"({max_result_rows})"
                        if truncated_by_rows
                        else f"YB_MCP_MAX_RESULT_BYTES ({max_result_bytes} bytes)"
                    )
                    result["note"] = (
                        f"Result set exceeded {reason}. Add a LIMIT clause "
                        f"or narrow the query to see the full result."
                    )
                logger.info(
                    "run_read_only_query returned %d rows × %d columns "
                    "(truncated=%s)",
                    len(rows), len(column_names), truncated,
                )
                sanitized = _sanitize_for_json(result)
                return json.dumps(sanitized, indent=2, default=str, allow_nan=False)
            except Exception as e:
                logger.error("Error executing read-only query: %s", e, exc_info=True)
                return f"Error executing query: {e}"
            finally:
                try:
                    _execute(ctrl_cur, "ROLLBACK")
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
    # DB-22159 resource caps
    max_query_len = lifespan["max_query_len"]
    statement_timeout_ms = lifespan["statement_timeout_ms"]

    # Reject oversized queries BEFORE parsing — the guardrail's sqlparse
    # walker spikes CPU on very large inputs (Vishal measured ~6.4s on a
    # ~1MB query). Cheap first-line defense. Measured in bytes to match
    # the env var's documented unit; `len(str)` alone counts code points,
    # which diverges for multibyte UTF-8.
    query_bytes = len(query.encode("utf-8"))
    if query_bytes > max_query_len:
        logger.warning(
            "run_write_query rejected: query %d bytes exceeds "
            "max_query_len=%d",
            query_bytes, max_query_len,
        )
        return json.dumps({
            "error": (
                f"Query length {query_bytes} bytes exceeds "
                f"YB_MCP_MAX_QUERY_LEN ({max_query_len} bytes)."
            ),
            "blocked_by_guardrail": True,
        })

    try:
        validate_query(query, guardrail_config, read_only=False)
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
                # DB-22159 + DB-22131 round 2: SET LOCAL statement_timeout
                # runs before EVERY write, unconditionally. It's the sole
                # bound on runtime for INSERT (VALUES, SELECT, DEFAULT
                # VALUES), UPDATE, DELETE, and DDL — the static row cap
                # `YB_MCP_MAX_INSERT_ROWS` was retired in DB-22131 round 2.
                # SET LOCAL opens the implicit transaction that psycopg's
                # default (autocommit=False) uses, and the timeout dies
                # on commit so it doesn't leak across pool checkouts.
                _execute(
                    cur,
                    f"SET LOCAL statement_timeout = '{statement_timeout_ms}ms'",
                )
                _execute(cur, query)
                conn.commit()
                logger.info("run_write_query committed: %d rows affected", cur.rowcount)
                return json.dumps({"rows_affected": cur.rowcount})
            except Exception as e:
                conn.rollback()
                logger.error("Write query failed, rolled back: %s", e, exc_info=True)
                return json.dumps({"error": str(e)})
