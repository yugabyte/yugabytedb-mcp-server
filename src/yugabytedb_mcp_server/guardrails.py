import logging
import re
from dataclasses import dataclass

import sqlparse

logger = logging.getLogger("yugabytedb-mcp.guardrails")


class QueryBlockedError(Exception):
    """Raised when a query is rejected by a guardrail check."""
    pass


@dataclass
class GuardrailConfig:
    max_insert_rows: int = 1000
    require_where_on_update: bool = False
    require_where_on_delete: bool = False


_BLOCKED_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Database / schema destruction
    (re.compile(r"\bDROP\s+DATABASE\b", re.I), "DROP DATABASE is not allowed"),
    (re.compile(r"\bDROP\s+SCHEMA\b", re.I), "DROP SCHEMA is not allowed"),
    (re.compile(r"\bALTER\s+DATABASE\b", re.I), "ALTER DATABASE is not allowed"),
    (re.compile(r"\bCREATE\s+DATABASE\b", re.I), "CREATE DATABASE is not allowed"),

    # Role / privilege manipulation
    (re.compile(r"\bGRANT\b", re.I), "GRANT is not allowed"),
    (re.compile(r"\bREVOKE\b", re.I), "REVOKE is not allowed"),
    (re.compile(r"\bCREATE\s+ROLE\b", re.I), "CREATE ROLE is not allowed"),
    (re.compile(r"\bALTER\s+ROLE\b", re.I), "ALTER ROLE is not allowed"),
    (re.compile(r"\bDROP\s+ROLE\b", re.I), "DROP ROLE is not allowed"),
    (re.compile(r"\bCREATE\s+USER\b", re.I), "CREATE USER is not allowed"),
    (re.compile(r"\bALTER\s+USER\b", re.I), "ALTER USER is not allowed"),
    (re.compile(r"\bDROP\s+USER\b", re.I), "DROP USER is not allowed"),

    # Filesystem access / arbitrary code execution
    (re.compile(r"\bCOPY\b.+\b(TO|FROM)\b", re.I | re.S), "COPY TO/FROM is not allowed"),
    (re.compile(r"\bLOAD\s+", re.I), "LOAD is not allowed"),
    # Match anonymous code blocks: bare `DO $$…$$` AND `DO LANGUAGE plpgsql $$…$$`.
    # The pre-match _strip_strings pass replaces dollar-quoted bodies with `''`,
    # so the bare `DO $$…$$` case becomes `DO ''` here — include that shape
    # explicitly. The LANGUAGE form still parses as `DO LANGUAGE plpgsql ''`.
    (re.compile(r"\bDO\s+(?:\$|LANGUAGE\b|'')", re.I),
     "Anonymous code blocks (DO) are not allowed"),
    (re.compile(r"\bCREATE\s+EXTENSION\b", re.I), "CREATE EXTENSION is not allowed"),

    # Server configuration
    (re.compile(r"\bALTER\s+SYSTEM\b", re.I), "ALTER SYSTEM is not allowed"),
    (re.compile(r"\bRESET\s+ALL\b", re.I), "RESET ALL is not allowed"),

    # Dangerous built-in functions
    (re.compile(r"\bpg_sleep\b", re.I), "pg_sleep is not allowed"),
    (re.compile(r"\bpg_read_file\b", re.I), "pg_read_file is not allowed"),
    (re.compile(r"\bpg_read_binary_file\b", re.I), "pg_read_binary_file is not allowed"),
    (re.compile(r"\bpg_write_file\b", re.I), "pg_write_file is not allowed"),
    (re.compile(r"\bpg_ls_dir\b", re.I), "pg_ls_dir is not allowed"),
    (re.compile(r"\blo_import\b", re.I), "lo_import is not allowed"),
    (re.compile(r"\blo_export\b", re.I), "lo_export is not allowed"),
    (re.compile(r"\bdblink\b", re.I), "dblink is not allowed"),

    # Schema isolation. `SET search_path` was blocked; the semantically
    # equivalent `set_config('search_path', …)` and `pg_catalog.set_config(…)`
    # bypassed it, so block `set_config` unconditionally (a GUC setter has no
    # legitimate use case via this tool — search_path is the security-relevant
    # one, but any GUC change is out of scope).
    (re.compile(r"\bSET\s+search_path\b", re.I), "SET search_path is not allowed"),
    (re.compile(r"\bset_config\b", re.I), "set_config is not allowed"),
    (re.compile(r"\bCREATE\s+SCHEMA\b", re.I), "CREATE SCHEMA is not allowed"),
]


def _strip_comments(sql: str) -> str:
    """Remove SQL comments so they cannot hide malicious patterns."""
    return sqlparse.format(sql, strip_comments=True).strip()


def _strip_strings(sql: str) -> str:
    """Replace every string literal in `sql` with `''`.

    This prevents pattern-based checks from firing on keywords or text that
    appear inside string values — e.g. `INSERT ... VALUES ('grant access')`
    must not trip the GRANT block, and `UPDATE t SET memo='reset where needed'`
    must not satisfy `require_where_on_update`.

    Handles three sqlparse tokenizations:

    - `Token.Literal.String.Single` — `'…'`, `E'…'`, `N'…'`. Strip.
    - `Token.Literal` (bare) — dollar-quoted bodies `$$…$$` and `$tag$…$tag$`.
      Strip so `DO $$ GRANT … $$` can't hide the GRANT.
    - `Token.Literal.String.Symbol` — double-quoted identifiers `"my_table"`.
      Preserve — these are names, not string data, and stripping them would
      erase identifier information the caller may legitimately be using.
    """
    try:
        parsed = sqlparse.parse(sql)
    except Exception:
        return sql
    if not parsed:
        return sql

    parts: list[str] = []
    for stmt in parsed:
        for tok in stmt.flatten():
            ttype = tok.ttype
            if ttype is None:
                parts.append(str(tok))
                continue
            ttype_str = str(ttype)
            if ttype_str == "Token.Literal.String.Single" or ttype_str == "Token.Literal":
                parts.append("''")
            else:
                parts.append(str(tok))
    return "".join(parts)


def _count_values_rows(sql: str) -> int | None:
    """Count top-level row tuples in a VALUES clause.

    Returns None when no VALUES keyword is found (e.g. INSERT ... SELECT),
    in which case no row-count limit applies.

    Caller is expected to pass string-stripped SQL (see `_strip_strings`) so
    that `)(` inside a string literal does not falsely inflate the count.
    """
    match = re.search(r"\bVALUES\b", sql, re.I)
    if not match:
        return None

    rest = sql[match.end():]
    depth = 0
    row_count = 0
    for ch in rest:
        if ch == "(":
            if depth == 0:
                row_count += 1
            depth += 1
        elif ch == ")":
            depth -= 1
    return row_count


def _has_top_level_where(sql: str) -> bool:
    """Check if a WHERE keyword exists outside of any parenthesized subexpression.

    Caller is expected to pass string-stripped SQL (see `_strip_strings`) so
    that a WHERE appearing inside a string literal does not falsely satisfy
    `require_where_on_update` / `require_where_on_delete`.
    """
    depth = 0
    upper = sql.upper()
    i = 0
    while i < len(upper):
        ch = upper[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(depth - 1, 0)
        elif depth == 0 and upper[i:i+5] == "WHERE" and (i == 0 or not upper[i-1].isalnum() and upper[i-1] != "_"):
            end = i + 5
            if end >= len(upper) or (not upper[end].isalnum() and upper[end] != "_"):
                return True
        i += 1
    return False


def validate_query(sql: str, config: GuardrailConfig, read_only: bool) -> None:
    """Validate a SQL query against the guardrail blocklist.

    Called by both `run_read_only_query` (with `read_only=True`) and
    `run_write_query` (with `read_only=False`). The dangerous-function
    blocklist runs in both modes; the write-shape checks (INSERT row limit,
    require-WHERE) only run when `read_only=False`.

    Raises `QueryBlockedError` with a human-readable reason if the query is
    rejected. Returns None when the query passes.
    """
    logger.debug("Validating query (%d chars, read_only=%s)", len(sql), read_only)
    stripped = sql.strip()
    if not stripped:
        raise QueryBlockedError("Empty query")

    if stripped.startswith("\\"):
        raise QueryBlockedError(
            "psql meta-commands (e.g. \\c, \\d, \\!) are not supported. "
            "Please use standard SQL statements."
        )

    cleaned = _strip_comments(stripped)
    if not cleaned:
        raise QueryBlockedError("Query is empty after removing comments")

    statements = [s for s in sqlparse.split(cleaned) if s.strip()]
    if len(statements) > 1:
        raise QueryBlockedError(
            "Multi-statement queries are not allowed. "
            "Please submit one statement at a time."
        )

    # Strip string literals BEFORE running the pattern-based checks so that
    # keywords/text inside string values don't cause false positives.
    for_matching = _strip_strings(cleaned)

    for pattern, reason in _BLOCKED_PATTERNS:
        if pattern.search(for_matching):
            logger.warning("Query blocked: %s", reason)
            raise QueryBlockedError(reason)

    if read_only:
        # Read path: the DB-level BEGIN READ ONLY transaction already blocks
        # writes at the DB layer; the blocklist above handles read-tool-
        # specific abuses (pg_read_file, COPY ... TO PROGRAM, dblink, etc.).
        # No further checks needed.
        logger.debug("Read-only query passed guardrail checks")
        return

    upper_cleaned = for_matching.upper().lstrip()
    if upper_cleaned.startswith("INSERT"):
        row_count = _count_values_rows(for_matching)
        if row_count is not None and row_count > config.max_insert_rows:
            raise QueryBlockedError(
                f"INSERT contains {row_count} rows, which exceeds the "
                f"maximum of {config.max_insert_rows} rows per statement. "
                f"Please split into smaller batches."
            )

    if config.require_where_on_update and upper_cleaned.startswith("UPDATE"):
        if not _has_top_level_where(for_matching):
            raise QueryBlockedError(
                "UPDATE without a WHERE clause is not allowed "
                "(YB_MCP_REQUIRE_WHERE_ON_UPDATE is enabled)."
            )

    if config.require_where_on_delete and upper_cleaned.startswith("DELETE"):
        if not _has_top_level_where(for_matching):
            raise QueryBlockedError(
                "DELETE without a WHERE clause is not allowed "
                "(YB_MCP_REQUIRE_WHERE_ON_DELETE is enabled)."
            )

    logger.debug("Write query passed guardrail checks")
