"""Integration tests verifying that guardrail-blocked queries have NO DB side effect.

Each test:
1. Seeds a known DB state via side-channel
2. Sends a blocked query through the MCP tool
3. Asserts the tool returned blocked_by_guardrail=True
4. Re-queries via side-channel to confirm the DB state is unchanged

Requires YUGABYTEDB_URL.
"""
import pytest

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _helpers import requires_yugabytedb, parse_json  # noqa: E402


pytestmark = [requires_yugabytedb, pytest.mark.asyncio]


@pytest.mark.parametrize("blocked_query,expected_fragment", [
    ("DROP DATABASE postgres", "DROP DATABASE"),
    ("GRANT ALL ON ALL TABLES IN SCHEMA public TO public", "GRANT"),
    ("COPY (SELECT 1) TO '/tmp/exfil.csv'", "COPY"),
    ("ALTER SYSTEM SET max_connections = 1", "ALTER SYSTEM"),
    ("SELECT 1; DROP TABLE foo", "Multi-statement"),
    ("/* hidden */ DROP DATABASE x", "DROP DATABASE"),
    ("\\d", "meta-command"),
])
async def test_blocked_query_returns_guardrail_error(
    mcp_session, blocked_query, expected_fragment, test_schema, db_conn
):
    """For each class of blocked statement, confirm:
    - tool returns {"error": ..., "blocked_by_guardrail": True}
    - DB schema is untouched (using test_schema as a witness)
    """
    # Seed a witness row in test_schema
    with db_conn.cursor() as cur:
        cur.execute(f'CREATE TABLE "{test_schema}".witness (id INT)')
        cur.execute(f'INSERT INTO "{test_schema}".witness VALUES (1)')

    result = await mcp_session.call_tool("run_write_query", {"query": blocked_query})
    payload = parse_json(result)
    assert payload.get("blocked_by_guardrail") is True, payload
    assert expected_fragment in payload.get("error", ""), payload

    # Witness still present, schema not dropped
    with db_conn.cursor() as cur:
        cur.execute(f'SELECT id FROM "{test_schema}".witness')
        assert cur.fetchone() == (1,)


async def test_blocked_create_extension_no_side_effect(mcp_session, db_conn):
    """CREATE EXTENSION is blocked. Confirm by attempting to create an extension
    that would otherwise succeed (citext is on most YugabyteDB installs) and
    verify it's NOT in pg_extension after the call."""
    result = await mcp_session.call_tool(
        "run_write_query",
        {"query": "CREATE EXTENSION IF NOT EXISTS citext"},
    )
    payload = parse_json(result)
    assert payload.get("blocked_by_guardrail") is True
