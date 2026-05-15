"""Exhaustive unit tests for yugabytedb_mcp_server.guardrails.

No DB, no network. Run with: uv run pytest tests/test_guardrails.py
"""
import pytest

from yugabytedb_mcp_server.guardrails import (
    GuardrailConfig,
    QueryBlockedError,
    validate_write_query,
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
    validate_write_query(sql, cfg)  # raises if blocked


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
        validate_write_query(sql, cfg)
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
        validate_write_query(sql, cfg)


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
        validate_write_query(sql, cfg)


# ---------------------------------------------------------------------------
# Server configuration
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sql", [
    "ALTER SYSTEM SET max_connections = 1",
    "RESET ALL",
])
def test_blocks_server_config(sql, cfg):
    with pytest.raises(QueryBlockedError):
        validate_write_query(sql, cfg)


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
        validate_write_query(sql, cfg)


# ---------------------------------------------------------------------------
# Schema isolation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sql", [
    "SET search_path = secret_schema",
    "CREATE SCHEMA evil",
])
def test_blocks_schema_isolation(sql, cfg):
    with pytest.raises(QueryBlockedError):
        validate_write_query(sql, cfg)


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
        validate_write_query(sql, cfg)
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
        validate_write_query(sql, cfg)


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
        validate_write_query(sql, cfg)
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
        validate_write_query(sql, cfg)


# ---------------------------------------------------------------------------
# Bulk INSERT row limit
# ---------------------------------------------------------------------------

def test_bulk_insert_under_limit(cfg):
    rows = ", ".join(["(1)"] * cfg.max_insert_rows)
    validate_write_query(f"INSERT INTO t VALUES {rows}", cfg)


def test_bulk_insert_over_limit(cfg):
    rows = ", ".join(["(1)"] * (cfg.max_insert_rows + 1))
    with pytest.raises(QueryBlockedError) as exc:
        validate_write_query(f"INSERT INTO t VALUES {rows}", cfg)
    assert "exceeds the maximum" in str(exc.value)


def test_bulk_insert_select_no_limit(cfg):
    # INSERT ... SELECT has no VALUES, so row-count limit does not apply
    validate_write_query("INSERT INTO t SELECT * FROM huge_table", cfg)


# ---------------------------------------------------------------------------
# Optional WHERE enforcement
# ---------------------------------------------------------------------------

def test_update_without_where_blocked_when_strict(strict_cfg):
    with pytest.raises(QueryBlockedError) as exc:
        validate_write_query("UPDATE t SET c = 1", strict_cfg)
    assert "UPDATE without a WHERE" in str(exc.value)


def test_update_with_where_allowed_when_strict(strict_cfg):
    validate_write_query("UPDATE t SET c = 1 WHERE id = 1", strict_cfg)


def test_delete_without_where_blocked_when_strict(strict_cfg):
    with pytest.raises(QueryBlockedError) as exc:
        validate_write_query("DELETE FROM t", strict_cfg)
    assert "DELETE without a WHERE" in str(exc.value)


def test_delete_with_where_allowed_when_strict(strict_cfg):
    validate_write_query("DELETE FROM t WHERE id = 1", strict_cfg)


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
