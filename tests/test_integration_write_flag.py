"""Integration tests for the --enable-write-query / YB_MCP_ENABLE_WRITE_QUERY flag.

Verifies that the run_write_query tool is only available when explicitly enabled.

Requires YUGABYTEDB_URL.
"""
import pytest

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _helpers import requires_yugabytedb  # noqa: E402


pytestmark = [requires_yugabytedb, pytest.mark.asyncio]


async def test_write_tool_absent_by_default(mcp_session):
    """With default config, run_write_query must NOT be listed."""
    tools = await mcp_session.list_tools()
    tool_names = {t.name for t in tools.tools}
    assert "run_write_query" not in tool_names
    assert "summarize_database" in tool_names
    assert "run_read_only_query" in tool_names


async def test_write_tool_present_when_enabled(mcp_session_write_enabled):
    """With YB_MCP_ENABLE_WRITE_QUERY=true, run_write_query must be listed."""
    tools = await mcp_session_write_enabled.list_tools()
    tool_names = {t.name for t in tools.tools}
    assert "run_write_query" in tool_names
    assert "summarize_database" in tool_names
    assert "run_read_only_query" in tool_names
