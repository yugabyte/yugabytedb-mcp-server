import logging
import re
from dataclasses import dataclass

import sqlparse

logger = logging.getLogger("yugabytedb-mcp.guardrails")


class QueryBlockedError(Exception):
    """Raised when a write query is rejected by a guardrail check."""
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
    (re.compile(r"\bDO\s+\$", re.I), "Anonymous code blocks (DO $$) are not allowed"),
    (re.compile(r"\bCREATE\s+EXTENSION\b", re.I), "CREATE EXTENSION is not allowed"),

    # Server configuration
    (re.compile(r"\bALTER\s+SYSTEM\b", re.I), "ALTER SYSTEM is not allowed"),
    (re.compile(r"\bRESET\s+ALL\b", re.I), "RESET ALL is not allowed"),

    # Dangerous built-in functions
    (re.compile(r"\bpg_sleep\b", re.I), "pg_sleep is not allowed"),
    (re.compile(r"\bpg_read_file\b", re.I), "pg_read_file is not allowed"),
    (re.compile(r"\bpg_write_file\b", re.I), "pg_write_file is not allowed"),
    (re.compile(r"\blo_import\b", re.I), "lo_import is not allowed"),
    (re.compile(r"\blo_export\b", re.I), "lo_export is not allowed"),
    (re.compile(r"\bdblink\b", re.I), "dblink is not allowed"),

    # Schema isolation
    (re.compile(r"\bSET\s+search_path\b", re.I), "SET search_path is not allowed"),
    (re.compile(r"\bCREATE\s+SCHEMA\b", re.I), "CREATE SCHEMA is not allowed"),
]


def _strip_comments(sql: str) -> str:
    """Remove SQL comments so they cannot hide malicious patterns."""
    return sqlparse.format(sql, strip_comments=True).strip()


def _count_values_rows(sql: str) -> int | None:
    """Count top-level row tuples in a VALUES clause.

    Returns None when no VALUES keyword is found (e.g. INSERT ... SELECT),
    in which case no row-count limit applies.
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
    """Check if a WHERE keyword exists outside of any parenthesized subexpression."""
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


def validate_write_query(sql: str, config: GuardrailConfig) -> None:
    """Validate a write query against all guardrail checks.

    Raises QueryBlockedError with a human-readable reason if the query is
    rejected. Does nothing (returns None) when the query passes all checks.
    """
    logger.debug("Validating write query (%d chars)", len(sql))
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

    for pattern, reason in _BLOCKED_PATTERNS:
        if pattern.search(cleaned):
            logger.warning("Write query blocked: %s", reason)
            raise QueryBlockedError(reason)

    upper_cleaned = cleaned.upper().lstrip()
    if upper_cleaned.startswith("INSERT"):
        row_count = _count_values_rows(cleaned)
        if row_count is not None and row_count > config.max_insert_rows:
            raise QueryBlockedError(
                f"INSERT contains {row_count} rows, which exceeds the "
                f"maximum of {config.max_insert_rows} rows per statement. "
                f"Please split into smaller batches."
            )

    if config.require_where_on_update and upper_cleaned.startswith("UPDATE"):
        if not _has_top_level_where(cleaned):
            raise QueryBlockedError(
                "UPDATE without a WHERE clause is not allowed "
                "(YB_MCP_REQUIRE_WHERE_ON_UPDATE is enabled)."
            )

    if config.require_where_on_delete and upper_cleaned.startswith("DELETE"):
        if not _has_top_level_where(cleaned):
            raise QueryBlockedError(
                "DELETE without a WHERE clause is not allowed "
                "(YB_MCP_REQUIRE_WHERE_ON_DELETE is enabled)."
            )

    logger.debug("Write query passed guardrail checks")
