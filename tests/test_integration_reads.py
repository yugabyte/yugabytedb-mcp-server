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
    rows = parse_json_list(result)
    assert rows == [{"x": 1}]


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
