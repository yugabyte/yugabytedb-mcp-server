"""Unit tests for MCP tool registration.

Verifies:
- `summarize_database` and `run_read_only_query` are both `readOnlyHint=True,
  destructiveHint=False`. `run_read_only_query`'s guardrail (DB-22129) strips
  the dangerous-function surface before execution, so the tool's advertised
  read-only semantic holds.
- `run_write_query` is gated behind `enable_write_query` and carries the
  destructive annotation when enabled.
"""
import asyncio
from dataclasses import replace

import pytest

from yugabytedb_mcp_server import server as server_module
from yugabytedb_mcp_server.server import ServerConfig, YugabyteDBMCPServer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_BASE_CONFIG = ServerConfig(
    yugabytedb_url="host=localhost port=5433 user=yugabyte dbname=yugabyte",
    transport="stdio",
    host="127.0.0.1",
    port=8000,
    stateless_http=False,
    ssl_root_cert_secret_arn=None,
    ssl_root_cert_key=None,
    ssl_root_cert_path="/tmp/yb-root.crt",
    ssl_root_cert_secret_region=None,
    require_where_on_update=False,
    require_where_on_delete=False,
    auth_provider=None,
    enable_write_query=False,
    identity_claim="email",
    identity_map_path=None,
    identity_map_name="mcp",
    allow_superuser_role=False,
    pool_min_size=1,
    pool_max_size=3,
    statement_timeout_ms=30_000,
    max_result_rows=10_000,
    max_result_bytes=50 * 1024 * 1024,
    max_query_len=100_000,
)


@pytest.fixture
def with_config():
    """Set the module-level CONFIG so `YugabyteDBMCPServer()` can register
    tools without going through `main()` / `parse_config()`."""
    original = getattr(server_module, "CONFIG", None)
    server_module.CONFIG = _BASE_CONFIG
    yield
    if original is None:
        try:
            del server_module.CONFIG
        except AttributeError:
            pass
    else:
        server_module.CONFIG = original


@pytest.fixture
def with_write_enabled():
    """Same as with_config but with enable_write_query=True."""
    original = getattr(server_module, "CONFIG", None)
    server_module.CONFIG = replace(_BASE_CONFIG, enable_write_query=True)
    yield
    if original is None:
        try:
            del server_module.CONFIG
        except AttributeError:
            pass
    else:
        server_module.CONFIG = original


def _get_tool(mcp, name):
    """Look up a registered tool by name via `list_tools()`."""
    tools = asyncio.run(mcp.list_tools())
    for t in tools:
        if t.name == name:
            return t
    raise AssertionError(
        f"tool {name!r} not registered; got {[t.name for t in tools]}"
    )


# ---------------------------------------------------------------------------
# run_read_only_query: read-only annotation (guardrail-backed)
# ---------------------------------------------------------------------------

class TestReadToolAnnotation:
    """`run_read_only_query`'s dangerous-function blocklist (DB-22129) strips
    the RCE / file-read / dblink / set_config / privileged-catalog surface
    before execution, so the tool truly is read-only end-to-end. Advertise
    accordingly."""

    def test_read_only_query_is_read_only_hint(self, with_config):
        server = YugabyteDBMCPServer()
        tool = _get_tool(server.mcp, "run_read_only_query")
        assert tool.annotations.readOnlyHint is True

    def test_read_only_query_is_not_destructive_hint(self, with_config):
        server = YugabyteDBMCPServer()
        tool = _get_tool(server.mcp, "run_read_only_query")
        assert tool.annotations.destructiveHint is False


# ---------------------------------------------------------------------------
# summarize_database: regression — must stay read-only
# ---------------------------------------------------------------------------

class TestSummarizeAnnotation:
    """Regression: `summarize_database` is genuinely read-only (only queries
    information_schema + COUNT(*)) — keep readOnlyHint=True."""

    def test_summarize_database_read_only_hint(self, with_config):
        server = YugabyteDBMCPServer()
        tool = _get_tool(server.mcp, "summarize_database")
        assert tool.annotations.readOnlyHint is True

    def test_summarize_database_not_destructive(self, with_config):
        server = YugabyteDBMCPServer()
        tool = _get_tool(server.mcp, "summarize_database")
        assert tool.annotations.destructiveHint is False


# ---------------------------------------------------------------------------
# run_write_query: gated behind enable_write_query
# ---------------------------------------------------------------------------

class TestWriteToolRegistration:

    def test_write_query_not_registered_by_default(self, with_config):
        """Default config has enable_write_query=False → tool is not exposed."""
        server = YugabyteDBMCPServer()
        tools = asyncio.run(server.mcp.list_tools())
        names = {t.name for t in tools}
        assert "run_write_query" not in names

    def test_write_query_registered_when_enabled(self, with_write_enabled):
        server = YugabyteDBMCPServer()
        tool = _get_tool(server.mcp, "run_write_query")
        assert tool.annotations.readOnlyHint is False
        assert tool.annotations.destructiveHint is True
        assert tool.annotations.idempotentHint is False
