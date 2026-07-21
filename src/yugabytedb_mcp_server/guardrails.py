"""SQL guardrail for the YugabyteDB MCP server.

Every query submitted through `run_read_only_query` or `run_write_query` goes
through `validate_query()`. The check runs against the sqlparse AST — not
against raw text — so keywords inside string literals, dollar-quoted bodies,
and double-quoted identifiers cannot trip or bypass the blocklist.

Blocklists:

- ``_BLOCKED_SINGLE_KEYWORDS`` — a bare keyword whose presence anywhere in
  the statement is rejected (GRANT, REVOKE, LOAD, COPY, DO).
- ``_BLOCKED_KEYWORD_PAIRS`` — two-word phrases like ``DROP DATABASE`` /
  ``ALTER SYSTEM``. Matched only when the two keywords appear adjacent
  (whitespace / comment tokens skipped), so a DROP DATABASE embedded in a
  string literal does not match (the literal tokenizes as one String token,
  not two Keyword tokens).
- ``_BLOCKED_FUNCTIONS`` — dangerous built-in function names. Detected as a
  Name token that is immediately followed by ``(``. Handles dotted forms
  like ``pg_catalog.pg_read_file(…)`` because sqlparse flattens the dotted
  name into separate Name tokens.
- ``SET search_path`` — SET keyword followed by an identifier named
  ``search_path``. Blocks all SET-search-path variants (SET, SET LOCAL,
  SET SESSION) without blocking legitimate SET commands like
  ``SET LOCAL statement_timeout``.

Write-only shape checks (skipped when `read_only=True`):

- INSERT row-count limit: counts ``Parenthesis`` children of the top-level
  ``Values`` group, not raw ``(`` occurrences, so a ``)(`` inside a string
  literal cannot inflate the count.
- ``require_where_on_update`` / ``require_where_on_delete``: checks for a
  ``Where`` group at the top level of the parsed statement, so a WHERE
  inside a subquery or inside a string literal does not satisfy the check.
  Uses ``stmt.get_type()`` so ``WITH x AS (…) UPDATE …`` is still
  recognized as an UPDATE.
"""

import logging
from dataclasses import dataclass

import sqlparse
from sqlparse import tokens as T
from sqlparse.sql import Parenthesis, Statement, Values, Where

logger = logging.getLogger("yugabytedb-mcp.guardrails")


class QueryBlockedError(Exception):
    """Raised when a query is rejected by a guardrail check."""
    pass


@dataclass
class GuardrailConfig:
    max_insert_rows: int = 1000
    require_where_on_update: bool = False
    require_where_on_delete: bool = False


# Two-word keyword phrases, matched only when both tokens are Keyword tokens
# adjacent to each other (whitespace / comments skipped). Because sqlparse
# tokenizes a keyword-shaped substring inside a string literal as
# `Token.Literal.String.Single` — not as `Token.Keyword` — no phrase inside
# a string can match.
_BLOCKED_KEYWORD_PAIRS: dict[tuple[str, str], str] = {
    # Database / schema destruction
    ("DROP", "DATABASE"): "DROP DATABASE is not allowed",
    ("DROP", "SCHEMA"): "DROP SCHEMA is not allowed",
    ("ALTER", "DATABASE"): "ALTER DATABASE is not allowed",
    ("CREATE", "DATABASE"): "CREATE DATABASE is not allowed",
    # Role / user manipulation
    ("CREATE", "ROLE"): "CREATE ROLE is not allowed",
    ("ALTER", "ROLE"): "ALTER ROLE is not allowed",
    ("DROP", "ROLE"): "DROP ROLE is not allowed",
    ("CREATE", "USER"): "CREATE USER is not allowed",
    ("ALTER", "USER"): "ALTER USER is not allowed",
    ("DROP", "USER"): "DROP USER is not allowed",
    # Server / extension configuration
    ("CREATE", "EXTENSION"): "CREATE EXTENSION is not allowed",
    ("ALTER", "SYSTEM"): "ALTER SYSTEM is not allowed",
    ("RESET", "ALL"): "RESET ALL is not allowed",
    ("CREATE", "SCHEMA"): "CREATE SCHEMA is not allowed",
}


# Single-keyword bans. A single Keyword token whose normalized value is in
# this set is rejected on sight.
_BLOCKED_SINGLE_KEYWORDS: dict[str, str] = {
    # DCL — role / privilege manipulation
    "GRANT": "GRANT is not allowed",
    "REVOKE": "REVOKE is not allowed",
    # File / library loading
    "LOAD": "LOAD is not allowed",
    # Bulk data ops always require TO/FROM/PROGRAM; no legitimate use case
    # via this tool.
    "COPY": "COPY TO/FROM is not allowed",
    # Anonymous code blocks — the code inside is a dollar-quoted body which
    # sqlparse tokenizes as a single Literal token; there is no way to
    # inspect its contents, and the block runs arbitrary procedural code.
    # Block DO in any form (bare `DO $$…$$` and `DO LANGUAGE plpgsql $$…$$`).
    "DO": "Anonymous code blocks (DO) are not allowed",
}


# Function names blocked in any context. Matched against Name tokens that are
# immediately followed by `(`. Handles schema-qualified calls like
# `pg_catalog.pg_read_file(…)` because sqlparse's `flatten()` emits the
# unqualified name as its own Name token.
_BLOCKED_FUNCTIONS: frozenset[str] = frozenset({
    "pg_sleep",
    "pg_read_file",
    "pg_read_binary_file",
    "pg_write_file",
    "pg_ls_dir",
    "pg_ls_logdir",
    "pg_ls_waldir",
    "pg_stat_file",
    "lo_import",
    "lo_export",
    "lo_from_bytea",
    "dblink",
    "dblink_exec",
    "dblink_connect",
    "dblink_send_query",
    # set_config('search_path', …) is equivalent to `SET search_path`; block
    # the function form unconditionally (it can change any GUC, none of which
    # is a legitimate MCP-tool operation).
    "set_config",
})


# Privileged catalog tables/views blocked in any context. Reading these
# discloses credentials (pg_authid.rolpassword, pg_shadow.passwd,
# pg_user_mappings.umoptions) or raw statistics that can leak data
# distributions (pg_statistic). Matched against bare Name tokens; string
# literals and double-quoted identifiers are not Name tokens, so
# `SELECT 'pg_authid' AS x` and `SELECT "pg_authid" AS c` do NOT match.
# Schema-qualified references (`pg_catalog.pg_authid`) match because sqlparse
# emits `pg_authid` as its own Name token.
_BLOCKED_TABLES: frozenset[str] = frozenset({
    "pg_authid",
    "pg_shadow",
    "pg_user_mappings",
    "pg_statistic",
})


def _is_keyword_token(tok) -> bool:
    """True if the token's ttype is Token.Keyword or any subtype
    (Keyword.DDL, Keyword.DML, Keyword.DCL, Keyword.CTE)."""
    return tok.ttype is not None and tok.ttype in T.Keyword


def _is_name_token(tok) -> bool:
    """True if the token's ttype is Token.Name (unquoted identifier). Does
    NOT match Token.Literal.String.Symbol (double-quoted identifiers) —
    those are preserved as identifier names and never treated as callable
    functions."""
    return tok.ttype is not None and tok.ttype in T.Name


def _is_punctuation(tok) -> bool:
    return tok.ttype is not None and tok.ttype in T.Punctuation


def _significant_tokens(iterable) -> list:
    """Return a list of tokens with whitespace and comment tokens stripped."""
    result = []
    for t in iterable:
        if t.is_whitespace:
            continue
        if t.ttype is not None and str(t.ttype).startswith("Token.Comment"):
            continue
        result.append(t)
    return result


def _check_blocked_ast(stmt: Statement) -> None:
    """Walk the parsed statement, rejecting blocked keywords, keyword pairs,
    and function calls. Recurses into subqueries because `Statement.flatten()`
    yields leaf tokens across the whole tree.

    A keyword-shaped substring inside a string literal is tokenized by
    sqlparse as a single Token.Literal.String.Single (or Token.Literal for a
    dollar-quoted body), so no string-literal contents are ever inspected
    against the blocklists — the class-A false-positive from the audit ticket
    cannot occur by construction.
    """
    tokens = _significant_tokens(stmt.flatten())

    for i, tok in enumerate(tokens):
        if _is_keyword_token(tok):
            kw = tok.value.upper()

            # Single-keyword bans.
            if kw in _BLOCKED_SINGLE_KEYWORDS:
                raise QueryBlockedError(_BLOCKED_SINGLE_KEYWORDS[kw])

            # Two-keyword-pair bans.
            if i + 1 < len(tokens):
                nxt = tokens[i + 1]
                if _is_keyword_token(nxt):
                    pair_key = (kw, nxt.value.upper())
                    if pair_key in _BLOCKED_KEYWORD_PAIRS:
                        raise QueryBlockedError(_BLOCKED_KEYWORD_PAIRS[pair_key])

            # SET search_path (handles SET, SET LOCAL, SET SESSION variants).
            # `search_path` is not a keyword; sqlparse tokenizes it as Name.
            # Scan forward from SET until we hit the value side (`TO` or `=`)
            # or run out of tokens; if any intermediate Name is `search_path`,
            # block. This still allows `SET LOCAL statement_timeout = '5s'`.
            if kw == "SET":
                for j in range(i + 1, len(tokens)):
                    nxt = tokens[j]
                    if _is_name_token(nxt) and nxt.value.lower() == "search_path":
                        raise QueryBlockedError("SET search_path is not allowed")
                    if _is_keyword_token(nxt) and nxt.value.upper() in ("TO", "="):
                        break
                    if _is_punctuation(nxt) and nxt.value == "=":
                        break

        elif _is_name_token(tok):
            # Function-call detection: Name followed (immediately, whitespace
            # already stripped) by `(`. Blocks both plain `pg_read_file(…)`
            # and dotted `pg_catalog.pg_read_file(…)` — the latter is
            # flattened by sqlparse into `pg_catalog` Name, `.` Punctuation,
            # `pg_read_file` Name, `(` Punctuation.
            name = tok.value.lower()
            if name in _BLOCKED_FUNCTIONS:
                if i + 1 < len(tokens):
                    nxt = tokens[i + 1]
                    if _is_punctuation(nxt) and nxt.value == "(":
                        raise QueryBlockedError(f"{name} is not allowed")
            # Privileged-catalog detection: bare Name match against
            # _BLOCKED_TABLES. No `(` gate — the reference isn't a function
            # call. False-positive risk (a user column named `pg_authid` etc.)
            # is theoretical and acceptable for a security guard against
            # credential/statistics disclosure.
            elif name in _BLOCKED_TABLES:
                raise QueryBlockedError(f"Access to {name} is not allowed")


def _count_values_rows(sql: str) -> int | None:
    """Count top-level row tuples in a VALUES clause.

    Uses the parsed AST: sqlparse groups a `VALUES (…), (…), (…)` clause as
    a single `Values` node whose direct children are `Parenthesis` groups
    (one per row) separated by punctuation commas. A `(`, `)`, or `,`
    appearing inside a string literal cannot affect the count because those
    characters are tokenized as part of the enclosing String token, not as
    separate tokens.

    Returns None when the statement has no top-level Values group
    (e.g. `INSERT … SELECT`), in which case no row-count limit applies.
    """
    parsed = sqlparse.parse(sql)
    if not parsed:
        return None
    stmt = parsed[0]
    for tok in stmt.tokens:
        if isinstance(tok, Values):
            return sum(1 for child in tok.tokens if isinstance(child, Parenthesis))
    return None


def _has_top_level_where(sql: str) -> bool:
    """Return True if a `Where` group exists at the top level of the statement.

    sqlparse groups the WHERE clause of the outermost UPDATE/DELETE/SELECT as
    a top-level `Where` node. A WHERE inside a nested subquery lives inside a
    `Parenthesis` group and is not visible at the top level. A WHERE inside a
    string literal is not a token at all (the whole literal is one String
    token). So this correctly distinguishes real top-level WHEREs from
    lookalikes inside subqueries or strings.
    """
    parsed = sqlparse.parse(sql)
    if not parsed:
        return False
    stmt = parsed[0]
    return any(isinstance(t, Where) for t in stmt.tokens)


def _strip_comments(sql: str) -> str:
    """Remove SQL comments so they cannot hide malicious patterns from
    multi-statement / psql-meta detection. The AST walker itself already
    ignores comment tokens, but comments-inside-comments and other edge
    cases are simplest to handle by stripping up front."""
    return sqlparse.format(sql, strip_comments=True).strip()


def validate_query(sql: str, config: GuardrailConfig, read_only: bool) -> None:
    """Validate a SQL query.

    Both `run_read_only_query` (with `read_only=True`) and `run_write_query`
    (with `read_only=False`) call this function. The dangerous-keyword and
    dangerous-function blocklists run in both modes; the write-shape checks
    (INSERT row limit, require-WHERE) only run when `read_only=False`.

    Raises `QueryBlockedError` with a human-readable reason on rejection.
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

    # Multi-statement rejection via sqlparse.split (respects strings correctly).
    statements = [s for s in sqlparse.split(cleaned) if s.strip()]
    if len(statements) > 1:
        raise QueryBlockedError(
            "Multi-statement queries are not allowed. "
            "Please submit one statement at a time."
        )

    parsed = sqlparse.parse(cleaned)
    if not parsed or not parsed[0].tokens:
        raise QueryBlockedError("Query is empty after parsing")
    stmt = parsed[0]

    # AST-based blocklist. Runs in both read and write modes.
    _check_blocked_ast(stmt)

    if read_only:
        # Read path: the DB-level BEGIN READ ONLY transaction blocks writes
        # at the DB layer; the AST blocklist above blocks the read-tool-
        # specific abuses (pg_read_file, COPY, dblink, pg_sleep, DO, GRANT,
        # …). No further checks needed.
        logger.debug("Read-only query passed guardrail checks")
        return

    # Write path: additional shape checks. `stmt.get_type()` recognizes
    # `WITH … UPDATE` as UPDATE, so the require-WHERE check still fires on
    # a CTE-prefixed UPDATE/DELETE.
    stmt_type = (stmt.get_type() or "").upper()

    if stmt_type == "INSERT":
        row_count = _count_values_rows(cleaned)
        if row_count is not None and row_count > config.max_insert_rows:
            raise QueryBlockedError(
                f"INSERT contains {row_count} rows, which exceeds the "
                f"maximum of {config.max_insert_rows} rows per statement. "
                f"Please split into smaller batches."
            )

    if config.require_where_on_update and stmt_type == "UPDATE":
        if not _has_top_level_where(cleaned):
            raise QueryBlockedError(
                "UPDATE without a WHERE clause is not allowed "
                "(YB_MCP_REQUIRE_WHERE_ON_UPDATE is enabled)."
            )

    if config.require_where_on_delete and stmt_type == "DELETE":
        if not _has_top_level_where(cleaned):
            raise QueryBlockedError(
                "DELETE without a WHERE clause is not allowed "
                "(YB_MCP_REQUIRE_WHERE_ON_DELETE is enabled)."
            )

    logger.debug("Write query passed guardrail checks")
