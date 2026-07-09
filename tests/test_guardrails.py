"""Exhaustive unit tests for yugabytedb_mcp_server.guardrails.

No DB, no network. Run with: uv run pytest tests/test_guardrails.py
"""
import pytest

from yugabytedb_mcp_server.guardrails import (
    GuardrailConfig,
    QueryBlockedError,
    validate_query,
    _count_values_rows,
    _has_top_level_where,
    _strip_comments,
)


@pytest.fixture
def cfg():
    return GuardrailConfig(
        max_insert_rows=10,
        require_where_on_update=False,
        require_where_on_delete=False,
    )


@pytest.fixture
def strict_cfg():
    return GuardrailConfig(
        max_insert_rows=10,
        require_where_on_update=True,
        require_where_on_delete=True,
    )


# ---------------------------------------------------------------------------
# Allowed queries
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sql", [
    "INSERT INTO t (id) VALUES (1)",
    "INSERT INTO t (id) VALUES (1), (2), (3)",
    "UPDATE t SET c = 1",                # WHERE not required by default cfg
    "UPDATE t SET c = 1 WHERE id = 1",
    "DELETE FROM t",                     # WHERE not required by default cfg
    "DELETE FROM t WHERE id = 1",
    "MERGE INTO t USING s ON t.id = s.id WHEN MATCHED THEN UPDATE SET c = s.c",
    "TRUNCATE TABLE t",
    "CREATE TABLE t (id INT)",
    "ALTER TABLE t ADD COLUMN c TEXT",
    "DROP TABLE t",                       # Only DROP DATABASE/SCHEMA blocked
    "INSERT INTO t SELECT * FROM s",      # INSERT ... SELECT has no row-count enforcement
])
def test_allows(sql, cfg):
    validate_query(sql, cfg, read_only=False)  # raises if blocked


# ---------------------------------------------------------------------------
# Database / schema destruction
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sql,fragment", [
    ("DROP DATABASE postgres", "DROP DATABASE"),
    ("drop database postgres", "DROP DATABASE"),
    ("DROP SCHEMA public CASCADE", "DROP SCHEMA"),
    ("ALTER DATABASE postgres SET search_path = bad", "ALTER DATABASE"),
    ("CREATE DATABASE evil", "CREATE DATABASE"),
])
def test_blocks_db_destruction(sql, fragment, cfg):
    with pytest.raises(QueryBlockedError) as exc:
        validate_query(sql, cfg, read_only=False)
    assert fragment in str(exc.value)


# ---------------------------------------------------------------------------
# Role / privilege manipulation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sql", [
    "GRANT ALL ON t TO public",
    "REVOKE SELECT ON t FROM public",
    "CREATE ROLE attacker",
    "ALTER ROLE attacker LOGIN",
    "DROP ROLE attacker",
    "CREATE USER u WITH PASSWORD 'p'",
    "ALTER USER u",
    "DROP USER u",
])
def test_blocks_role_manipulation(sql, cfg):
    with pytest.raises(QueryBlockedError):
        validate_query(sql, cfg, read_only=False)


# ---------------------------------------------------------------------------
# Filesystem access / code execution
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sql", [
    "COPY t TO '/tmp/exfil.csv'",
    "COPY t FROM '/etc/passwd'",
    "COPY (SELECT * FROM secrets) TO '/tmp/x'",
    "LOAD 'libsomething.so'",
    "DO $$ BEGIN PERFORM 1; END $$",
    "CREATE EXTENSION dblink",
])
def test_blocks_filesystem_and_code(sql, cfg):
    with pytest.raises(QueryBlockedError):
        validate_query(sql, cfg, read_only=False)


# ---------------------------------------------------------------------------
# Server configuration
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sql", [
    "ALTER SYSTEM SET max_connections = 1",
    "RESET ALL",
])
def test_blocks_server_config(sql, cfg):
    with pytest.raises(QueryBlockedError):
        validate_query(sql, cfg, read_only=False)


# ---------------------------------------------------------------------------
# Dangerous built-in functions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sql", [
    "SELECT pg_sleep(60)",
    "SELECT pg_read_file('/etc/passwd')",
    "SELECT pg_write_file('/tmp/x', 'data')",
    "SELECT lo_import('/etc/passwd')",
    "SELECT lo_export(1, '/tmp/x')",
    "SELECT * FROM dblink('host=evil', 'SELECT secret')",
])
def test_blocks_dangerous_functions(sql, cfg):
    with pytest.raises(QueryBlockedError):
        validate_query(sql, cfg, read_only=False)


# ---------------------------------------------------------------------------
# Schema isolation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sql", [
    "SET search_path = secret_schema",
    "CREATE SCHEMA evil",
])
def test_blocks_schema_isolation(sql, cfg):
    with pytest.raises(QueryBlockedError):
        validate_query(sql, cfg, read_only=False)


# ---------------------------------------------------------------------------
# Multi-statement
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sql", [
    "INSERT INTO t VALUES (1); DROP TABLE t",
    "SELECT 1; SELECT 2",
    "UPDATE t SET c = 1 WHERE id = 1; DELETE FROM t",
])
def test_blocks_multistatement(sql, cfg):
    with pytest.raises(QueryBlockedError) as exc:
        validate_query(sql, cfg, read_only=False)
    assert "Multi-statement" in str(exc.value)


# ---------------------------------------------------------------------------
# Comment obfuscation cannot hide blocked patterns
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sql", [
    "/* harmless */ DROP DATABASE x",
    "-- harmless\nDROP DATABASE x",
    "DROP /* trick */ DATABASE x",
    "-- comment\nGRANT ALL ON t TO public",
])
def test_strips_comments_before_check(sql, cfg):
    with pytest.raises(QueryBlockedError):
        validate_query(sql, cfg, read_only=False)


# ---------------------------------------------------------------------------
# psql meta-commands
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sql", [
    "\\c postgres",
    "\\d",
    "\\!",
])
def test_blocks_psql_meta(sql, cfg):
    with pytest.raises(QueryBlockedError) as exc:
        validate_query(sql, cfg, read_only=False)
    assert "meta-command" in str(exc.value)


# ---------------------------------------------------------------------------
# Empty queries
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sql", [
    "",
    "   ",
    "-- only a comment",
    "/* only a block comment */",
])
def test_blocks_empty(sql, cfg):
    with pytest.raises(QueryBlockedError):
        validate_query(sql, cfg, read_only=False)


# ---------------------------------------------------------------------------
# Bulk INSERT row limit
# ---------------------------------------------------------------------------

def test_bulk_insert_under_limit(cfg):
    rows = ", ".join(["(1)"] * cfg.max_insert_rows)
    validate_query(f"INSERT INTO t VALUES {rows}", cfg, read_only=False)


def test_bulk_insert_over_limit(cfg):
    rows = ", ".join(["(1)"] * (cfg.max_insert_rows + 1))
    with pytest.raises(QueryBlockedError) as exc:
        validate_query(f"INSERT INTO t VALUES {rows}", cfg, read_only=False)
    assert "exceeds the maximum" in str(exc.value)


def test_bulk_insert_select_no_limit(cfg):
    # INSERT ... SELECT has no VALUES, so row-count limit does not apply
    validate_query("INSERT INTO t SELECT * FROM huge_table", cfg, read_only=False)


# ---------------------------------------------------------------------------
# Optional WHERE enforcement
# ---------------------------------------------------------------------------

def test_update_without_where_blocked_when_strict(strict_cfg):
    with pytest.raises(QueryBlockedError) as exc:
        validate_query("UPDATE t SET c = 1", strict_cfg, read_only=False)
    assert "UPDATE without a WHERE" in str(exc.value)


def test_update_with_where_allowed_when_strict(strict_cfg):
    validate_query("UPDATE t SET c = 1 WHERE id = 1", strict_cfg, read_only=False)


def test_delete_without_where_blocked_when_strict(strict_cfg):
    with pytest.raises(QueryBlockedError) as exc:
        validate_query("DELETE FROM t", strict_cfg, read_only=False)
    assert "DELETE without a WHERE" in str(exc.value)


def test_delete_with_where_allowed_when_strict(strict_cfg):
    validate_query("DELETE FROM t WHERE id = 1", strict_cfg, read_only=False)


# ---------------------------------------------------------------------------
# Helper-function unit tests
# ---------------------------------------------------------------------------

def test_count_values_rows_simple():
    assert _count_values_rows("INSERT INTO t VALUES (1), (2), (3)") == 3


def test_count_values_rows_nested():
    # Inner parens (e.g. composite types) shouldn't be counted as rows
    assert _count_values_rows("INSERT INTO t VALUES ((1, 'a')), ((2, 'b'))") == 2


def test_count_values_rows_no_values():
    assert _count_values_rows("INSERT INTO t SELECT * FROM s") is None


def test_has_top_level_where_simple():
    assert _has_top_level_where("UPDATE t SET c = 1 WHERE id = 1")
    assert not _has_top_level_where("UPDATE t SET c = 1")


def test_has_top_level_where_in_subquery_doesnt_count():
    # WHERE inside a subquery shouldn't satisfy the top-level requirement
    assert not _has_top_level_where(
        "UPDATE t SET c = (SELECT MAX(x) FROM s WHERE s.id = 1)"
    )


def test_strip_comments_removes_both_styles():
    assert "DROP" in _strip_comments("/* hide */ DROP TABLE t -- end")
    assert "hide" not in _strip_comments("/* hide */ DROP TABLE t")


# ---------------------------------------------------------------------------
# DB-22131: keyword-in-string-literal must not trip the guardrail
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sql", [
    # (A) legitimate writes whose data happens to contain a blocked keyword
    "INSERT INTO audit(action) VALUES ('grant access to vault')",
    "UPDATE tickets SET note = 'please revoke the old token'",
    "INSERT INTO runbook(step) VALUES ('never run DROP DATABASE x')",
    "INSERT INTO logs(msg) VALUES ('user called pg_read_file to debug')",
    "INSERT INTO logs(msg) VALUES ('COPY the file TO the vault')",
    "INSERT INTO logs(msg) VALUES ('DO NOT touch this')",
    "UPDATE t SET note = 'ALTER SYSTEM was rejected' WHERE id = 1",
])
def test_string_literal_keywords_do_not_false_positive(sql, cfg):
    validate_query(sql, cfg, read_only=False)  # must not raise


# ---------------------------------------------------------------------------
# DB-22131: advertised protections must not be bypassable by equivalent syntax
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sql", [
    # set_config('search_path', …) is equivalent to SET search_path — block both
    "SELECT set_config('search_path', 'evil', false)",
    "SELECT set_config( 'search_path' , 'evil' , false )",
    'SELECT set_config("search_path", \'evil\', false)',
])
def test_blocks_set_config_search_path_bypass(sql, cfg):
    with pytest.raises(QueryBlockedError):
        validate_query(sql, cfg, read_only=False)


@pytest.mark.parametrize("sql", [
    "DO $$ BEGIN PERFORM 1; END $$",
    "DO LANGUAGE plpgsql $$ BEGIN PERFORM 1; END $$",
    "do language plpgsql $$ begin perform 1; end $$",
])
def test_blocks_do_block_regardless_of_language_clause(sql, cfg):
    with pytest.raises(QueryBlockedError):
        validate_query(sql, cfg, read_only=False)


def test_where_in_string_literal_does_not_satisfy_strict_where(strict_cfg):
    # DB-22131 class C: 'reset where needed' inside a string must not
    # count as a real WHERE clause.
    with pytest.raises(QueryBlockedError) as exc:
        validate_query(
            "UPDATE accounts SET memo='reset where needed'",
            strict_cfg, read_only=False,
        )
    assert "WHERE" in str(exc.value)


def test_parens_in_string_literal_do_not_inflate_insert_row_count(cfg):
    # DB-22131 class C: ')(' inside a string was previously counted as a
    # new row tuple. Verify a 1-row INSERT with such a value passes.
    validate_query(
        "INSERT INTO t (msg) VALUES ('has )( parens')",
        cfg, read_only=False,
    )


# ---------------------------------------------------------------------------
# DB-22129: read-only path still blocks side-effecting SQL
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sql", [
    "COPY (SELECT 1) TO PROGRAM 'id > /tmp/poc'",
    "SELECT pg_read_file('/etc/hostname')",
    "SELECT pg_read_binary_file('/etc/hostname')",
    "SELECT * FROM dblink('host=evil', 'SELECT secret')",
    "SELECT pg_sleep(60)",
    "SELECT lo_import('/etc/passwd')",
    "SELECT set_config('search_path', 'evil', false)",
])
def test_read_only_blocks_dangerous_reads(sql, cfg):
    with pytest.raises(QueryBlockedError):
        validate_query(sql, cfg, read_only=True)


@pytest.mark.parametrize("sql", [
    "SELECT 1",
    "SELECT * FROM t WHERE id = 1",
    "SELECT count(*) FROM information_schema.tables",
    "WITH x AS (SELECT 1) SELECT * FROM x",
])
def test_read_only_allows_normal_selects(sql, cfg):
    validate_query(sql, cfg, read_only=True)  # must not raise


def test_read_only_skips_insert_row_count(cfg):
    # A SELECT is never an INSERT, so the row-count check should no-op even
    # if the query mentions VALUES.
    validate_query("SELECT * FROM (VALUES (1), (2), (3)) AS v(x)", cfg, read_only=True)


def test_read_only_skips_require_where(strict_cfg):
    # SELECTs don't start with UPDATE/DELETE, so the WHERE-required check
    # should not fire on the read path.
    validate_query("SELECT * FROM t", strict_cfg, read_only=True)


# ---------------------------------------------------------------------------
# AST-based detection edge cases — new behaviors from the parser rewrite
# ---------------------------------------------------------------------------

def test_blocks_schema_qualified_dangerous_function(cfg):
    # pg_catalog.pg_read_file(…) — sqlparse flattens the dotted name into
    # separate Name tokens, so the unqualified `pg_read_file` still matches
    # the function blocklist.
    with pytest.raises(QueryBlockedError) as exc:
        validate_query("SELECT pg_catalog.pg_read_file('/etc')", cfg, read_only=True)
    assert "pg_read_file" in str(exc.value)


def test_allows_pg_read_file_as_string_literal(cfg):
    # The literal 'pg_read_file' inside a WHERE clause must not match the
    # function blocklist — it's not a Name token, it's inside a String.Single.
    validate_query(
        "SELECT * FROM my_table WHERE name = 'pg_read_file'",
        cfg, read_only=True,
    )


def test_allows_pg_read_file_as_quoted_identifier(cfg):
    # A double-quoted identifier "pg_read_file" is tokenized as
    # String.Symbol (a Name, not a String literal). It's an identifier ref,
    # not a call — no following `(` — so it must not be blocked.
    validate_query(
        'SELECT "pg_read_file" FROM my_table',
        cfg, read_only=True,
    )


def test_allows_set_local_statement_timeout(cfg):
    # `SET LOCAL statement_timeout = '5s'` is a legitimate performance knob
    # that PR #5 will use internally — must not be blocked by the
    # SET-search_path rule.
    validate_query(
        "SET LOCAL statement_timeout = '5s'",
        cfg, read_only=False,
    )


def test_blocks_set_search_path_all_variants(cfg):
    for sql in [
        "SET search_path TO evil",
        "SET search_path = evil",
        "SET LOCAL search_path TO evil",
        "SET SESSION search_path TO evil",
    ]:
        with pytest.raises(QueryBlockedError) as exc:
            validate_query(sql, cfg, read_only=False)
        assert "search_path" in str(exc.value)


def test_blocks_grant_inside_dollar_quoted_do_block(cfg):
    # DO $$ … $$ is blocked outright (any DO block is rejected), so even
    # if it contained an attempted GRANT the outer DO block catches it first.
    with pytest.raises(QueryBlockedError) as exc:
        validate_query("DO $$ GRANT ALL ON t TO evil $$", cfg, read_only=False)
    assert "DO" in str(exc.value) or "Anonymous" in str(exc.value)


def test_top_level_where_detected_after_cte(strict_cfg):
    # WITH x AS (…) UPDATE t SET c = 1 — stmt.get_type() returns 'UPDATE'
    # and the top-level Where check should fire. Without a top-level WHERE,
    # this should be rejected under strict config.
    with pytest.raises(QueryBlockedError):
        validate_query(
            "WITH x AS (SELECT 1) UPDATE t SET c = 1",
            strict_cfg, read_only=False,
        )
    # But with a real top-level WHERE, it should pass.
    validate_query(
        "WITH x AS (SELECT 1) UPDATE t SET c = 1 WHERE id = 1",
        strict_cfg, read_only=False,
    )


# ---------------------------------------------------------------------------
# DB-22172: read-tool multi-statement escape via COMMIT/END prefix
# ---------------------------------------------------------------------------
# The read tool wraps queries in `BEGIN READ ONLY ... ROLLBACK`, but a caller
# whose input starts with COMMIT/END closes the read-only transaction before
# subsequent statements run — those then auto-commit outside the wrapper. The
# multi-statement rejection in validate_query() catches this on both read and
# write paths.

@pytest.mark.parametrize("sql", [
    "COMMIT; INSERT INTO t VALUES (1)",
    "END; DROP TABLE t",
    "COMMIT; UPDATE t SET c = 1",
    "END; CREATE TABLE t (id int)",
    "COMMIT; DELETE FROM t",
    "COMMIT ; INSERT INTO t VALUES (1)",       # extra whitespace before ;
    "commit; insert into t values (1)",         # lowercase
    "COMMIT;\nINSERT INTO t VALUES (1)",       # newline separator
    "COMMIT; SELECT 1",                          # even a second SELECT is blocked
])
def test_multi_statement_escape_blocked_read_only(sql, cfg):
    """DB-22172: read tool must reject multi-statement input so a COMMIT/END
    prefix cannot end the BEGIN READ ONLY transaction and let downstream
    statements auto-commit outside the read-only wrapper."""
    with pytest.raises(QueryBlockedError) as exc:
        validate_query(sql, cfg, read_only=True)
    assert "Multi-statement" in str(exc.value)


@pytest.mark.parametrize("sql", [
    "COMMIT; INSERT INTO t VALUES (1)",
    "END; DROP TABLE t",
])
def test_multi_statement_escape_blocked_write(sql, cfg):
    """DB-22172: same multi-statement rejection on the write path (regression
    check — was already covered by DB-22131 multi-statement tests, but explicit
    for the DB-22172 scenario)."""
    with pytest.raises(QueryBlockedError) as exc:
        validate_query(sql, cfg, read_only=False)
    assert "Multi-statement" in str(exc.value)


def test_bare_commit_alone_not_multi_statement(cfg):
    """A lone COMMIT is one statement — not blocked by the multi-statement
    check. (It's still harmless via the read tool: ends an empty read-only
    txn.) Documents current behavior; if we want to also block bare
    transaction control statements, that's a follow-up ticket."""
    # No raise expected: single statement, not on the blocklist.
    validate_query("COMMIT", cfg, read_only=True)
    validate_query("END", cfg, read_only=True)
    validate_query("ROLLBACK", cfg, read_only=True)
