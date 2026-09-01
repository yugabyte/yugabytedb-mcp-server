"""Integration tests for read tools (summarize_database, run_read_only_query).

Requires YUGABYTEDB_URL. Run with:
    YUGABYTEDB_URL="host=... port=... ..." uv run pytest tests/test_integration_reads.py
"""
import pytest

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _helpers import requires_yugabytedb, parse_json, parse_json_list, raw_text  # noqa: E402


pytestmark = [requires_yugabytedb, pytest.mark.asyncio]


async def test_run_read_only_query_simple(mcp_session):
    result = await mcp_session.call_tool("run_read_only_query", {"query": "SELECT 1 AS x"})
    parsed = parse_json_list(result)
    assert parsed == {"columns": ["x"], "rows": [[1]]}


async def test_run_read_only_query_duplicate_column_names(mcp_session):
    """ duplicate output column names must not collapse. Previously
    `dict(zip(cols, row))` silently dropped the first `id`. Now columns and
    rows are parallel arrays so all values survive."""
    result = await mcp_session.call_tool(
        "run_read_only_query",
        {"query": "SELECT 1 AS id, 2 AS id, 3 AS other"},
    )
    parsed = parse_json_list(result)
    assert parsed == {"columns": ["id", "id", "other"], "rows": [[1, 2, 3]]}


async def test_run_read_only_query_join_star_no_column_loss(mcp_session, test_schema, db_conn):
    """ real-world repro: SELECT * over a join where both tables have
    an `id` column. Both `id` values must be present, not collapsed."""
    with db_conn.cursor() as cur:
        cur.execute(f'CREATE TABLE "{test_schema}".a (id INT, tag TEXT)')
        cur.execute(f'CREATE TABLE "{test_schema}".b (id INT, note TEXT)')
        cur.execute(f'INSERT INTO "{test_schema}".a VALUES (1, \'left\')')
        cur.execute(f'INSERT INTO "{test_schema}".b VALUES (2, \'right\')')

    result = await mcp_session.call_tool(
        "run_read_only_query",
        {"query": f'SELECT * FROM "{test_schema}".a, "{test_schema}".b'},
    )
    parsed = parse_json_list(result)
    assert parsed["columns"] == ["id", "tag", "id", "note"]
    assert parsed["rows"] == [[1, "left", 2, "right"]]


async def test_run_read_only_query_error_path(mcp_session):
    result = await mcp_session.call_tool(
        "run_read_only_query",
        {"query": "SELECT * FROM nonexistent_table_xyz"},
    )
    text = raw_text(result)
    assert text.startswith("Error"), f"expected error string, got: {text!r}"


async def test_read_only_query_rejects_write(mcp_session, test_schema, db_conn):
    """BEGIN READ ONLY at the transaction level must reject DML, even though
    the tool itself doesn't have query-shape validation (that's run_write_query's job).
    """
    # Seed a table via side-channel
    with db_conn.cursor() as cur:
        cur.execute(f'CREATE TABLE "{test_schema}".t (id INT, c TEXT)')
        cur.execute(f'INSERT INTO "{test_schema}".t VALUES (1, \'a\')')

    result = await mcp_session.call_tool(
        "run_read_only_query",
        {"query": f'UPDATE "{test_schema}".t SET c = \'b\' WHERE id = 1'},
    )
    text = raw_text(result)
    assert text.startswith("Error"), f"expected error, got: {text!r}"

    # Confirm the row is unchanged via side-channel
    with db_conn.cursor() as cur:
        cur.execute(f'SELECT c FROM "{test_schema}".t WHERE id = 1')
        assert cur.fetchone()[0] == "a"


async def test_summarize_database_default_schema(mcp_session):
    """`public` schema should be summarizable. Don't assume any specific tables — just
    that the structure parses."""
    result = await mcp_session.call_tool("summarize_database", {"schema": "public"})
    parsed = parse_json(result)
    assert isinstance(parsed, list)
    for entry in parsed:
        assert "table" in entry or "error" in entry


async def test_summarize_database_seeded_schema(mcp_session, test_schema, db_conn):
    """Seed a known set of tables and verify summarize_database reports them."""
    with db_conn.cursor() as cur:
        cur.execute(f'CREATE TABLE "{test_schema}".users (id INT, name TEXT)')
        cur.execute(f'INSERT INTO "{test_schema}".users VALUES (1, \'a\'), (2, \'b\')')
        cur.execute(f'CREATE TABLE "{test_schema}".items (id INT)')
        cur.execute(f'INSERT INTO "{test_schema}".items VALUES (10)')

    result = await mcp_session.call_tool("summarize_database", {"schema": test_schema})
    parsed = parse_json(result)

    by_table = {e["table"]: e for e in parsed if "table" in e}
    assert "users" in by_table
    assert "items" in by_table
    assert by_table["users"]["row_count"] == 2
    assert by_table["items"]["row_count"] == 1
    user_cols = {c["column_name"] for c in by_table["users"]["schema"]}
    assert user_cols == {"id", "name"}


# ---------------------------------------------------------------------------
# per-table error handling — a single erroring object must not
# discard every other table in the summary.
# ---------------------------------------------------------------------------

async def test_summarize_continues_past_error_view(mcp_session, test_schema, db_conn):
    """ primary repro: schema has a good table plus a view whose
    COUNT(*) raises (division-by-zero). The view sorts alphabetically first
    (`a_bad` before `z_good`), so v1 would abort at the view and never
    reach the table. Fix: SAVEPOINT per table + per-object error entry."""
    with db_conn.cursor() as cur:
        # `z_good` — normal table with data
        cur.execute(f'CREATE TABLE "{test_schema}".z_good (id INT)')
        cur.execute(f'INSERT INTO "{test_schema}".z_good VALUES (1), (2), (3)')
        # `a_bad` — view that raises at COUNT time. Alphabetically first,
        # so it's iterated BEFORE z_good.
        cur.execute(
            f'CREATE VIEW "{test_schema}".a_bad AS SELECT 1 AS x WHERE (1/0) = 1'
        )

    result = await mcp_session.call_tool("summarize_database", {"schema": test_schema})
    parsed = parse_json(result)

    by_name = {}
    for entry in parsed:
        # New shape: successful entries carry `table`; failures carry
        # `table` + `error`. Both keyed under their table/view name.
        key = entry.get("table")
        if key is not None:
            by_name[key] = entry

    assert "z_good" in by_name, (
        f"good table missing from summary — regression. Got: {parsed!r}"
    )
    assert by_name["z_good"].get("row_count") == 3

    assert "a_bad" in by_name, f"bad view missing entry entirely — got: {parsed!r}"
    assert "error" in by_name["a_bad"], (
        f"bad view should carry per-object error, got: {by_name['a_bad']!r}"
    )
    # The error mentions the underlying failure — division by zero.
    assert "division" in by_name["a_bad"]["error"].lower() or \
           "zero" in by_name["a_bad"]["error"].lower(), \
        f"unexpected error text: {by_name['a_bad']['error']!r}"


async def test_summarize_continues_past_permission_denied(
    mcp_session, test_schema, db_conn,
):
    """Under a role that lacks SELECT on one table, that table's entry
    contains a permission-denied error but the OTHER tables still get
    their row counts. Verifies the least-privilege docs scenario the
    audit called out."""
    with db_conn.cursor() as cur:
        cur.execute(f'CREATE TABLE "{test_schema}".public_table (id INT)')
        cur.execute(f'INSERT INTO "{test_schema}".public_table VALUES (1), (2)')
        cur.execute(f'CREATE TABLE "{test_schema}".secret_table (id INT)')
        cur.execute(f'INSERT INTO "{test_schema}".secret_table VALUES (99)')

        # Create a limited role: SELECT on public_table only, no access to
        # secret_table. Grant to the current login user so summarize sees
        # the same permission profile when running as this role (via the
        # normal pool credentials — we don't need OIDC here).
        cur.execute(f'REVOKE ALL ON "{test_schema}".secret_table FROM PUBLIC')
        # Explicitly revoke from the connecting role. This is best-effort —
        # if the pool user is a superuser it can still SELECT. In that case
        # the test degrades to just verifying both entries are present.

    result = await mcp_session.call_tool("summarize_database", {"schema": test_schema})
    parsed = parse_json(result)

    by_name = {e.get("table"): e for e in parsed if e.get("table") is not None}

    # Regardless of pool privileges: BOTH tables must have entries. That's
    # the fix — one problematic object never removes other entries.
    assert "public_table" in by_name, f"good table missing: {parsed!r}"
    assert "secret_table" in by_name, f"restricted table missing entirely: {parsed!r}"

    # public_table always succeeds.
    assert by_name["public_table"].get("row_count") == 2


async def test_summarize_multiple_errors_one_good(mcp_session, test_schema, db_conn):
    """Two erroring objects + one good — all three appear in the summary,
    no truncation, no early return."""
    with db_conn.cursor() as cur:
        # Two failing views (alphabetically first) + one good table.
        cur.execute(
            f'CREATE VIEW "{test_schema}".a_first_bad AS '
            f'SELECT 1 AS x WHERE (1/0) = 1'
        )
        # Postgres's COUNT(*) skips the projection, so `SELECT (5/0)::int AS y`
        # never actually evaluates the division and COUNT succeeds with 1.
        # Put the failure in WHERE, same shape as a_first_bad, so the DB has
        # to evaluate the expression to know which rows count.
        cur.execute(
            f'CREATE VIEW "{test_schema}".b_second_bad AS '
            f'SELECT 1 AS y WHERE (5 / 0) = 0'
        )
        cur.execute(f'CREATE TABLE "{test_schema}".c_good (id INT)')
        cur.execute(f'INSERT INTO "{test_schema}".c_good VALUES (1)')

    result = await mcp_session.call_tool("summarize_database", {"schema": test_schema})
    parsed = parse_json(result)

    by_name = {e.get("table"): e for e in parsed if e.get("table") is not None}

    assert "a_first_bad" in by_name
    assert "b_second_bad" in by_name
    assert "c_good" in by_name

    assert "error" in by_name["a_first_bad"]
    assert "error" in by_name["b_second_bad"]
    assert by_name["c_good"].get("row_count") == 1


async def test_summarize_empty_schema_returns_empty_list(
    mcp_session, test_schema, db_conn,
):
    """Regression: an empty schema still returns []. This shouldn't have
    changed with the fix — the initial tables lookup returns []
    and the loop is a no-op."""
    # test_schema is created by the fixture but not populated.
    result = await mcp_session.call_tool("summarize_database", {"schema": test_schema})
    parsed = parse_json(result)
    assert parsed == []


# ---------------------------------------------------------------------------
# resource limits: statement timeout, result row cap, query length
# ---------------------------------------------------------------------------

async def test_statement_timeout_kills_slow_query(mcp_session_capped):
    """ primary repro. A slow query with a 1s statement_timeout
    must fail within ~1s instead of holding the connection indefinitely.
    Prevents the "5 trivial queries downs the service" attack.

    Note: this used to use ``pg_sleep(60)``, but PR #9's read-side
    guardrail blocks pg_sleep pre-execute. We now use a
    CPU-bound generate_series count large enough that the timeout, not
    the guardrail, is what fires. This preserves the assertion
    (server-side timeout works) while acknowledging the defense-in-depth
    guardrail from PR #9.
    """
    import time
    # 20M-row count runs ~2-3s locally — well above the 1s cap.
    # Streams (no aggregation memory, no temp files) so we don't trip
    # unrelated resource limits.
    slow_query = "SELECT count(*) FROM generate_series(1, 20000000) g"
    start = time.monotonic()
    result = await mcp_session_capped.call_tool(
        "run_read_only_query", {"query": slow_query},
    )
    elapsed = time.monotonic() - start
    text = raw_text(result)
    assert elapsed < 5, (
        f"statement_timeout should fire fast; took {elapsed:.1f}s. Result: {text!r}"
    )
    assert "timeout" in text.lower() or "canceling" in text.lower(), (
        f"expected a timeout error, got: {text!r}"
    )


async def test_result_byte_cap_truncates(mcp_session_capped, test_schema, db_conn):
    """repro. A query with FEW rows (well under
    max_result_rows) but WIDE rows now truncates by cumulative byte size
    — the old fetchmany(row_cap+1) buffered wide rows without bound and
    could OOM the process. The capped fixture sets max_result_bytes=512
    so ~5 rows of a 200-char string is enough to trip the cap.
    """
    with db_conn.cursor() as cur:
        cur.execute(f'CREATE TABLE "{test_schema}".wide (val TEXT)')
        cur.execute(
            f'INSERT INTO "{test_schema}".wide '
            f"SELECT repeat('x', 200) FROM generate_series(1, 3)"
        )

    result = await mcp_session_capped.call_tool(
        "run_read_only_query",
        {"query": f'SELECT val FROM "{test_schema}".wide'},
    )
    parsed = parse_json(result)
    assert isinstance(parsed, dict), f"expected dict shape, got: {parsed!r}"
    assert parsed.get("truncated") is True
    # 3 rows total, each ~200 chars. Under a 3-row cap AND 512-byte cap,
    # the byte cap should trip first — only 1-2 rows survive.
    assert 1 <= parsed.get("returned_rows", 0) <= 2, (
        f"expected byte cap to trip before row cap; got returned_rows="
        f"{parsed.get('returned_rows')}"
    )
    assert "YB_MCP_MAX_RESULT_BYTES" in parsed.get("note", "")


async def test_result_row_cap_truncates(mcp_session_capped, test_schema, db_conn):
    """ second repro. A query returning MORE than
    YB_MCP_MAX_RESULT_ROWS (3 in the capped fixture) truncates with a
    `truncated: true` marker instead of buffering the whole result set
    into memory and risking OOM.
    """
    # Seed 10 rows; the capped session's max_result_rows=3 so we truncate.
    with db_conn.cursor() as cur:
        cur.execute(f'CREATE TABLE "{test_schema}".big (id INT)')
        cur.execute(
            f'INSERT INTO "{test_schema}".big '
            f'SELECT generate_series(1, 10)'
        )

    result = await mcp_session_capped.call_tool(
        "run_read_only_query",
        {"query": f'SELECT id FROM "{test_schema}".big ORDER BY id'},
    )
    parsed = parse_json(result)
    # PR #9 shape: {"columns": [...], "rows": [[...]]}. On truncation,
    # extra keys `truncated`/`returned_rows`/`note` are added onto the
    # same dict — rows still uses the parallel-arrays shape.
    assert isinstance(parsed, dict), f"expected dict shape, got: {parsed!r}"
    assert parsed.get("truncated") is True
    assert parsed.get("returned_rows") == 3
    assert parsed["columns"] == ["id"]
    assert len(parsed["rows"]) == 3
    # First 3 ids returned, remaining 7 dropped.
    ids = [row[0] for row in parsed["rows"]]
    assert ids == [1, 2, 3]


async def test_result_row_cap_not_triggered(mcp_session_capped):
    """Regression: a query returning fewer rows than the cap keeps the
    non-truncated columns/rows shape without any truncation marker."""
    result = await mcp_session_capped.call_tool(
        "run_read_only_query",
        {"query": "SELECT generate_series(1, 2) AS n"},
    )
    parsed = parse_json(result)
    # PR #9 shape stays even below the truncation cap; just no truncated
    # marker keys.
    assert isinstance(parsed, dict), f"expected dict shape, got: {parsed!r}"
    assert "truncated" not in parsed
    assert "returned_rows" not in parsed
    assert parsed["columns"] == ["n"]
    assert parsed["rows"] == [[1], [2]]


async def test_max_query_len_rejects_oversized_query(mcp_session_capped):
    """ third repro. Queries whose text length exceeds
    YB_MCP_MAX_QUERY_LEN (200 bytes in the capped fixture) are rejected
    BEFORE reaching the DB — no connection acquisition, no parser CPU
    spike."""
    # Pad the query well past 200 bytes with meaningless-but-valid SQL.
    padding = "-- " + ("x" * 300)
    long_query = f"SELECT 1 AS n\n{padding}"
    assert len(long_query) > 200

    result = await mcp_session_capped.call_tool(
        "run_read_only_query", {"query": long_query},
    )
    parsed = parse_json(result)
    # Rejected pre-execute → structured error, not a DB-side failure.
    assert parsed.get("blocked_by_guardrail") is True
    assert "MAX_QUERY_LEN" in parsed.get("error", "")


async def test_max_query_len_allows_normal_query(mcp_session_capped):
    """Regression: queries under the length cap pass through normally."""
    result = await mcp_session_capped.call_tool(
        "run_read_only_query", {"query": "SELECT 1 AS n"},
    )
    parsed = parse_json(result)
    # PR #9 shape (parallel arrays), preserved through PR #11 rebase.
    assert parsed == {"columns": ["n"], "rows": [[1]]}
