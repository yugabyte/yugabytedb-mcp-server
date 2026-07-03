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
# _strip_strings helper
# ---------------------------------------------------------------------------

def test_strip_strings_replaces_single_quoted():
    from yugabytedb_mcp_server.guardrails import _strip_strings
    assert _strip_strings("SELECT 'GRANT'") == "SELECT ''"


def test_strip_strings_replaces_dollar_quoted():
    from yugabytedb_mcp_server.guardrails import _strip_strings
    # Dollar-quoted strings should also be replaced so DO $$ ... GRANT ... $$
    # can't hide GRANT inside a code block.
    stripped = _strip_strings("DO $$ GRANT ALL ON t TO evil $$")
    assert "GRANT" not in stripped


def test_strip_strings_preserves_identifiers():
    from yugabytedb_mcp_server.guardrails import _strip_strings
    # Double-quoted identifiers must NOT be treated as strings — otherwise
    # `UPDATE "GRANT"` would look like `UPDATE ''` and no longer be a valid
    # keyword to match. sqlparse tokenizes them as Token.Name, not String.
    assert '"my_table"' in _strip_strings('SELECT * FROM "my_table"')
