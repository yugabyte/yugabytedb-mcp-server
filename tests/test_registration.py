"""Unit tests for MCP tool registration — DB-22130.

Verifies that `run_read_only_query` is advertised with `readOnlyHint=False,
destructiveHint=True` so MCP clients (Claude Desktop, Cursor, ...) present
the destructive-action confirmation prompt. Prior to the fix the tool was
annotated `readOnlyHint=True` even though it can modify its environment
(COPY, dblink, side-effecting functions — see DB-22129 blocklist), so
clients skipped the confirmation.

Also verifies:
- `summarize_database` keeps `readOnlyHint=True` (it truly is read-only).
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
    stateless_http=False,
    ssl_root_cert_secret_arn=None,
    ssl_root_cert_key=None,
    ssl_root_cert_path="/tmp/yb-root.crt",
    ssl_root_cert_secret_region=None,
    max_insert_rows=1000,
    require_where_on_update=False,
    require_where_on_delete=False,
    auth_provider=None,
    enable_write_query=False,
    identity_claim="email",
    identity_transform="none",
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
# DB-22130: run_read_only_query annotation must be destructive
# ---------------------------------------------------------------------------

class TestReadToolAnnotation:
    """DB-22130 — run_read_only_query was `readOnlyHint=True` but it can
    modify its environment (COPY, dblink, pg_read_file, etc. — see DB-22129).
    MCP clients trusting readOnlyHint skipped the confirmation prompt. Fix:
    flip to `readOnlyHint=False, destructiveHint=True, idempotentHint=True`."""

    def test_read_only_query_is_not_read_only_hint(self, with_config):
        server = YugabyteDBMCPServer()
        tool = _get_tool(server.mcp, "run_read_only_query")
        assert tool.annotations.readOnlyHint is False, (
            "DB-22130: run_read_only_query must not advertise readOnlyHint=True — "
            "MCP clients skip the confirmation prompt for readOnly tools."
        )

    def test_read_only_query_is_destructive_hint(self, with_config):
        server = YugabyteDBMCPServer()
        tool = _get_tool(server.mcp, "run_read_only_query")
        assert tool.annotations.destructiveHint is True, (
            "DB-22130: run_read_only_query must advertise destructiveHint=True "
            "so clients present the destructive-action confirmation."
        )

    def test_read_only_query_is_idempotent_hint(self, with_config):
        server = YugabyteDBMCPServer()
        tool = _get_tool(server.mcp, "run_read_only_query")
        # Idempotent because BEGIN READ ONLY + rollback means re-running the
        # same allowed query has no side effect. The AST blocklist rejects
        # side-effecting statements before execution.
        assert tool.annotations.idempotentHint is True


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
