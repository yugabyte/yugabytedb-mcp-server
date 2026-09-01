"""Exhaustive unit tests for yugabytedb_mcp_server.guardrails.

No DB, no network. Run with: uv run pytest tests/test_guardrails.py
"""
import pytest

from yugabytedb_mcp_server.guardrails import (
    GuardrailConfig,
    QueryBlockedError,
    validate_query,
    _has_top_level_where,
    _strip_comments,
)


@pytest.fixture
def cfg():
    return GuardrailConfig(
        require_where_on_update=False,
        require_where_on_delete=False,
    )


@pytest.fixture
def strict_cfg():
    return GuardrailConfig(
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
    "INSERT INTO t DEFAULT VALUES",       # Single-row default insert, no VALUES tuple
    "INSERT INTO t SELECT * FROM s",      # INSERT … SELECT bounded by statement_timeout
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
# INSERT shapes — all allowed and bounded by SET LOCAL statement_timeout
# ---------------------------------------------------------------------------
# the static ``YB_MCP_MAX_INSERT_ROWS`` cap was retired.
# Every write goes through statement_timeout in ``run_write_query`` before
# executing, so a runaway INSERT — VALUES, SELECT, DEFAULT, whatever
# shape — is bounded by the timeout. These tests pin that every INSERT
# shape passes the guardrail cleanly.

def test_bulk_insert_values_allowed(cfg):
    rows = ", ".join(["(1)"] * 100)
    validate_query(f"INSERT INTO t VALUES {rows}", cfg, read_only=False)


def test_insert_select_allowed(cfg):
    """``INSERT … SELECT`` bounded by SET LOCAL statement_timeout."""
    validate_query(
        "INSERT INTO t SELECT * FROM huge_table", cfg, read_only=False,
    )


def test_insert_select_with_cte_allowed(cfg):
    """CTE-prefixed INSERT … SELECT is the same shape, still allowed."""
    validate_query(
        "WITH src AS (SELECT id FROM other) INSERT INTO t SELECT * FROM src",
        cfg, read_only=False,
    )


def test_insert_default_values_allowed(cfg):
    """Regression: ``INSERT … DEFAULT VALUES`` inserts one row with all
    column defaults."""
    validate_query("INSERT INTO t DEFAULT VALUES", cfg, read_only=False)


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
# CREATE FUNCTION / CREATE PROCEDURE — arbitrary code + privilege escalation
# ---------------------------------------------------------------------------

def test_create_function_rejected(cfg):
    """CREATE FUNCTION runs arbitrary PL code and can grant privileges via
    SECURITY DEFINER — no legitimate MCP-tool use case."""
    with pytest.raises(QueryBlockedError) as exc:
        validate_query(
            "CREATE FUNCTION f() RETURNS int LANGUAGE SQL AS $$ SELECT 1 $$",
            cfg, read_only=False,
        )
    assert "CREATE FUNCTION" in str(exc.value)


def test_create_function_security_definer_rejected(cfg):
    """The SECURITY DEFINER variant is the most dangerous form —
    privilege escalation to function owner. Same blocklist entry catches
    it."""
    with pytest.raises(QueryBlockedError):
        validate_query(
            "CREATE FUNCTION f() RETURNS int "
            "SECURITY DEFINER LANGUAGE SQL AS $$ SELECT 1 $$",
            cfg, read_only=False,
        )


def test_create_procedure_rejected(cfg):
    """CREATE PROCEDURE is CREATE FUNCTION's sibling — same class of
    concern."""
    with pytest.raises(QueryBlockedError):
        validate_query(
            "CREATE PROCEDURE p() LANGUAGE plpgsql AS $$ BEGIN END $$",
            cfg, read_only=False,
        )


def test_alter_function_rejected(cfg):
    """ALTER FUNCTION can flip SECURITY DEFINER on an existing function —
    same escalation risk as CREATE."""
    with pytest.raises(QueryBlockedError):
        validate_query(
            "ALTER FUNCTION f() SECURITY DEFINER", cfg, read_only=False,
        )


def test_create_or_replace_function_rejected(cfg):
    """ sqlparse tokenizes ``CREATE OR REPLACE`` as a single
    Keyword.DDL, so the two-keyword pair matcher on ``('CREATE','FUNCTION')``
    never fires. The dedicated ``_check_create_or_alter_routine`` scanner
    catches this form."""
    with pytest.raises(QueryBlockedError) as exc:
        validate_query(
            "CREATE OR REPLACE FUNCTION f() RETURNS int "
            "LANGUAGE SQL AS $$ SELECT 1 $$",
            cfg, read_only=False,
        )
    assert "CREATE FUNCTION" in str(exc.value)


def test_create_or_replace_function_security_definer_rejected(cfg):
    """The escalation form of the CREATE-OR-REPLACE variant — SECURITY
    DEFINER on a redefined function. Same block."""
    with pytest.raises(QueryBlockedError):
        validate_query(
            "CREATE OR REPLACE FUNCTION f() RETURNS int "
            "SECURITY DEFINER LANGUAGE SQL AS $$ SELECT 1 $$",
            cfg, read_only=False,
        )


def test_create_or_replace_procedure_rejected(cfg):
    with pytest.raises(QueryBlockedError) as exc:
        validate_query(
            "CREATE OR REPLACE PROCEDURE p() "
            "LANGUAGE plpgsql AS $$ BEGIN END $$",
            cfg, read_only=False,
        )
    assert "CREATE PROCEDURE" in str(exc.value)


# ---------------------------------------------------------------------------
# Bulk-copy shapes intentionally left ALLOWED
# ---------------------------------------------------------------------------
# `CREATE TABLE … AS SELECT` and `SELECT … INTO` are unbounded row copies
# in the same family as `INSERT … SELECT`, but the team chose to allow them
# — they're commonly used to materialize a snapshot from an existing query
# and blocking them broke a valid workflow. These tests pin that choice so
# a future guardrail expansion doesn't silently regress it.

def test_create_table_as_select_allowed(cfg):
    validate_query(
        "CREATE TABLE snap AS SELECT * FROM src", cfg, read_only=False,
    )


def test_create_unlogged_table_as_select_allowed(cfg):
    validate_query(
        "CREATE UNLOGGED TABLE snap AS SELECT * FROM src",
        cfg, read_only=False,
    )


def test_select_into_allowed(cfg):
    validate_query("SELECT * INTO snap FROM src", cfg, read_only=False)


def test_create_table_plain_still_allowed(cfg):
    """Regression: ``CREATE TABLE t (col int)`` (no AS SELECT) is a
    bounded DDL statement — no rows copied, must not be blocked."""
    validate_query("CREATE TABLE t (id int, name text)", cfg, read_only=False)


# ---------------------------------------------------------------------------
# Helper-function unit tests
# ---------------------------------------------------------------------------

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
# keyword-in-string-literal must not trip the guardrail
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
# advertised protections must not be bypassable by equivalent syntax
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
    # class C: 'reset where needed' inside a string must not
    # count as a real WHERE clause.
    with pytest.raises(QueryBlockedError) as exc:
        validate_query(
            "UPDATE accounts SET memo='reset where needed'",
            strict_cfg, read_only=False,
        )
    assert "WHERE" in str(exc.value)


def test_parens_in_string_literal_do_not_inflate_insert_row_count(cfg):
    # class C: ')(' inside a string was previously counted as a
    # new row tuple. Verify a 1-row INSERT with such a value passes.
    validate_query(
        "INSERT INTO t (msg) VALUES ('has )( parens')",
        cfg, read_only=False,
    )


# ---------------------------------------------------------------------------
# read-only path still blocks side-effecting SQL
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
# read-tool multi-statement escape via COMMIT/END prefix
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
    """ read tool must reject multi-statement input so a COMMIT/END
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
    """ same multi-statement rejection on the write path (regression
    check — was already covered by multi-statement tests, but explicit
    for the scenario)."""
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


# ---------------------------------------------------------------------------
# privileged catalog tables (credential / statistics disclosure)
# ---------------------------------------------------------------------------
# `SELECT rolname, rolpassword FROM pg_authid` was called out in the# repro list. The function blocklist doesn't catch it (pg_authid is a table,
# not a function), so we cover it via a table-name blocklist walked in the
# same AST pass. Applies to both read and write paths.


@pytest.mark.parametrize("sql", [
    "SELECT rolname, rolpassword FROM pg_authid",
    "SELECT * FROM pg_authid",
    "SELECT rolname FROM pg_catalog.pg_authid",
    "SELECT * FROM pg_shadow",
    "SELECT * FROM pg_catalog.pg_shadow",
    "SELECT * FROM pg_user_mappings",
    "SELECT * FROM pg_statistic",
    "SELECT * FROM pg_catalog.pg_statistic",
    # JOIN target
    "SELECT a.oid FROM pg_class a JOIN pg_authid b ON a.relowner = b.oid",
    # Subquery reference
    "SELECT (SELECT rolpassword FROM pg_authid LIMIT 1) AS x",
    # CTE reference
    "WITH c AS (SELECT * FROM pg_authid) SELECT * FROM c",
])
def test_blocks_privileged_catalog_reads_read_path(sql, cfg):
    """ reads of credential/statistics catalogs are guardrail-blocked
    on the read path regardless of whether Postgres RBAC would also reject."""
    with pytest.raises(QueryBlockedError, match="not allowed"):
        validate_query(sql, cfg, read_only=True)


@pytest.mark.parametrize("sql", [
    "UPDATE pg_authid SET rolpassword = 'x'",
    "DELETE FROM pg_shadow",
    "INSERT INTO pg_user_mappings VALUES (1, 2, ARRAY['x'])",
])
def test_blocks_privileged_catalog_writes(sql, cfg):
    """ same tables are also blocked on the write path — an operator
    running the server as a role with catalog-modification privileges must not
    be able to mutate them via run_write_query."""
    with pytest.raises(QueryBlockedError, match="not allowed"):
        validate_query(sql, cfg, read_only=False)


@pytest.mark.parametrize("sql", [
    # String literal — sqlparse tokenizes as String.Single, not Name.
    "SELECT 'pg_authid' AS x",
    # Double-quoted identifier — tokenized as Symbol, not Name.
    'SELECT 1 AS "pg_authid"',
    # Different, unprivileged tables must not be flagged.
    "SELECT * FROM pg_roles",
    "SELECT * FROM pg_stat_activity",
    "SELECT * FROM pg_catalog.pg_class",
    "SELECT * FROM information_schema.tables",
])
def test_privileged_catalog_blocklist_no_false_positives(sql, cfg):
    """String literals, double-quoted identifiers, and unprivileged catalogs
    that only share a prefix with blocked names must pass through."""
    # No raise expected.
    validate_query(sql, cfg, read_only=True)
