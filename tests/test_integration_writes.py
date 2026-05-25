"""Integration tests for run_write_query — full round-trips with read-back verification.

Every write tests sends a query via the MCP tool, then re-queries the DB
via a direct side-channel connection to confirm the change actually landed
(not just that the tool returned success).

Requires YUGABYTEDB_URL.
"""
import pytest

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _helpers import requires_yugabytedb, parse_json, raw_text  # noqa: E402


pytestmark = [requires_yugabytedb, pytest.mark.asyncio]


async def test_insert_round_trip(mcp_session, test_schema, db_conn):
    # Seed empty table via side-channel
    with db_conn.cursor() as cur:
        cur.execute(f'CREATE TABLE "{test_schema}".t (id INT, name TEXT)')

    # Insert via MCP tool
    result = await mcp_session.call_tool(
        "run_write_query",
        {"query": f'INSERT INTO "{test_schema}".t (id, name) VALUES (1, \'alice\')'},
    )
    payload = parse_json(result)
    assert payload == {"rows_affected": 1}, payload

    # Read back via side-channel — confirm the row is present
    with db_conn.cursor() as cur:
        cur.execute(f'SELECT id, name FROM "{test_schema}".t')
        rows = cur.fetchall()
    assert rows == [(1, "alice")]


async def test_update_round_trip(mcp_session, test_schema, db_conn):
    with db_conn.cursor() as cur:
        cur.execute(f'CREATE TABLE "{test_schema}".t (id INT, c TEXT)')
        cur.execute(f'INSERT INTO "{test_schema}".t VALUES (1, \'old\'), (2, \'keep\')')

    result = await mcp_session.call_tool(
        "run_write_query",
        {"query": f'UPDATE "{test_schema}".t SET c = \'new\' WHERE id = 1'},
    )
    payload = parse_json(result)
    assert payload == {"rows_affected": 1}, payload

    with db_conn.cursor() as cur:
        cur.execute(f'SELECT id, c FROM "{test_schema}".t ORDER BY id')
        rows = cur.fetchall()
    assert rows == [(1, "new"), (2, "keep")]


async def test_delete_round_trip(mcp_session, test_schema, db_conn):
    with db_conn.cursor() as cur:
        cur.execute(f'CREATE TABLE "{test_schema}".t (id INT)')
        cur.execute(f'INSERT INTO "{test_schema}".t VALUES (1), (2), (3)')

    result = await mcp_session.call_tool(
        "run_write_query",
        {"query": f'DELETE FROM "{test_schema}".t WHERE id IN (1, 3)'},
    )
    payload = parse_json(result)
    assert payload == {"rows_affected": 2}, payload

    with db_conn.cursor() as cur:
        cur.execute(f'SELECT id FROM "{test_schema}".t ORDER BY id')
        rows = cur.fetchall()
    assert rows == [(2,)]


async def test_bulk_insert_under_limit_succeeds(mcp_session, test_schema, db_conn):
    with db_conn.cursor() as cur:
        cur.execute(f'CREATE TABLE "{test_schema}".t (id INT)')

    rows = ", ".join(f"({i})" for i in range(10))
    result = await mcp_session.call_tool(
        "run_write_query",
        {"query": f'INSERT INTO "{test_schema}".t VALUES {rows}'},
    )
    payload = parse_json(result)
    assert payload == {"rows_affected": 10}

    with db_conn.cursor() as cur:
        cur.execute(f'SELECT COUNT(*) FROM "{test_schema}".t')
        assert cur.fetchone()[0] == 10


async def test_bulk_insert_over_limit_blocked(mcp_session_strict, test_schema, db_conn):
    """With YB_MCP_MAX_INSERT_ROWS=5, an INSERT of 10 rows must be blocked
    by the guardrail AND have no DB side effect."""
    with db_conn.cursor() as cur:
        cur.execute(f'CREATE TABLE "{test_schema}".t (id INT)')

    rows = ", ".join(f"({i})" for i in range(10))
    result = await mcp_session_strict.call_tool(
        "run_write_query",
        {"query": f'INSERT INTO "{test_schema}".t VALUES {rows}'},
    )
    payload = parse_json(result)
    assert payload.get("blocked_by_guardrail") is True
    assert "exceeds the maximum" in payload.get("error", "")

    # Confirm DB unchanged
    with db_conn.cursor() as cur:
        cur.execute(f'SELECT COUNT(*) FROM "{test_schema}".t')
        assert cur.fetchone()[0] == 0


async def test_strict_update_without_where_blocked(mcp_session_strict, test_schema, db_conn):
    with db_conn.cursor() as cur:
        cur.execute(f'CREATE TABLE "{test_schema}".t (id INT, c TEXT)')
        cur.execute(f'INSERT INTO "{test_schema}".t VALUES (1, \'a\'), (2, \'b\')')

    result = await mcp_session_strict.call_tool(
        "run_write_query",
        {"query": f'UPDATE "{test_schema}".t SET c = \'BAD\''},
    )
    payload = parse_json(result)
    assert payload.get("blocked_by_guardrail") is True
    assert "WHERE" in payload.get("error", "")

    # Confirm rows unchanged
    with db_conn.cursor() as cur:
        cur.execute(f'SELECT c FROM "{test_schema}".t ORDER BY id')
        assert [r[0] for r in cur.fetchall()] == ["a", "b"]


async def test_strict_delete_without_where_blocked(mcp_session_strict, test_schema, db_conn):
    with db_conn.cursor() as cur:
        cur.execute(f'CREATE TABLE "{test_schema}".t (id INT)')
        cur.execute(f'INSERT INTO "{test_schema}".t VALUES (1), (2)')

    result = await mcp_session_strict.call_tool(
        "run_write_query",
        {"query": f'DELETE FROM "{test_schema}".t'},
    )
    payload = parse_json(result)
    assert payload.get("blocked_by_guardrail") is True

    with db_conn.cursor() as cur:
        cur.execute(f'SELECT COUNT(*) FROM "{test_schema}".t')
        assert cur.fetchone()[0] == 2


async def test_write_query_postgres_error_returns_json(mcp_session, test_schema, db_conn):
    """Errors from the DB (not from guardrails) should also come back as a
    well-formed JSON response, not crash the tool."""
    result = await mcp_session.call_tool(
        "run_write_query",
        {"query": f'INSERT INTO "{test_schema}".does_not_exist VALUES (1)'},
    )
    payload = parse_json(result)
    assert "error" in payload
    assert "blocked_by_guardrail" not in payload  # different code path than guardrails
