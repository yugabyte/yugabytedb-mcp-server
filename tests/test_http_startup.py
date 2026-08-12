"""Unit tests for the HTTP-mode startup guard (DB-22139).

Covers `_check_http_startup` — the fail-closed check that runs before
uvicorn opens the socket. Pre-fix the server bound `0.0.0.0:8000` and
accepted anonymous /mcp requests; the new check refuses to start when
HTTP mode is combined with a non-loopback host and no auth provider.

Also covers `_is_loopback` and `_env_bool` since they're small.
"""
import os
import sys
from dataclasses import replace
from unittest.mock import patch

import pytest

from yugabytedb_mcp_server import server as server_module
from yugabytedb_mcp_server.server import (
    _check_http_startup,
    _is_loopback,
    _env_bool,
    ServerConfig,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_BASE_CONFIG = ServerConfig(
    yugabytedb_url="host=localhost port=5433 user=yugabyte dbname=yugabyte",
    transport="http",
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
    identity_transform="none",
    identity_map_path=None,
    identity_map_name="mcp",
    pool_min_size=1,
    pool_max_size=5,
    statement_timeout_ms=30_000,
    max_result_rows=10_000,
    max_result_bytes=50 * 1024 * 1024,
    max_query_len=100_000,
)


@pytest.fixture
def with_config():
    """Install a config on the server module for the duration of one test."""
    def _apply(**overrides):
        cfg = replace(_BASE_CONFIG, **overrides)
        server_module.CONFIG = cfg
        return cfg

    original = getattr(server_module, "CONFIG", None)
    yield _apply
    if original is None:
        try:
            del server_module.CONFIG
        except AttributeError:
            pass
    else:
        server_module.CONFIG = original


@pytest.fixture(autouse=True)
def clean_env():
    """Prevent test env pollution — clear the DB-22139 escape hatch env
    and the Origin allowlist before every test."""
    with patch.dict(os.environ, {}, clear=False):
        for k in (
            "MCP_ALLOW_UNAUTHENTICATED",
            "MCP_ALLOWED_ORIGINS",
            "MCP_BASE_URL",
        ):
            os.environ.pop(k, None)
        yield


# ---------------------------------------------------------------------------
# _is_loopback
# ---------------------------------------------------------------------------

class TestIsLoopback:
    @pytest.mark.parametrize("host", [
        "127.0.0.1",
        "::1",
        "localhost",
        "LOCALHOST",       # case-insensitive
        " 127.0.0.1 ",     # whitespace tolerated
    ])
    def test_loopback_hosts(self, host):
        assert _is_loopback(host) is True

    @pytest.mark.parametrize("host", [
        "0.0.0.0",
        "192.168.1.1",
        "10.0.0.1",
        "mcp.example.com",
        "",
    ])
    def test_non_loopback_hosts(self, host):
        assert _is_loopback(host) is False


# ---------------------------------------------------------------------------
# _env_bool
# ---------------------------------------------------------------------------

class TestEnvBool:
    def test_true(self):
        with patch.dict(os.environ, {"X": "true"}):
            assert _env_bool("X") is True

    def test_true_case_insensitive(self):
        with patch.dict(os.environ, {"X": "TRUE"}):
            assert _env_bool("X") is True

    def test_false_default_unset(self):
        os.environ.pop("X", None)
        assert _env_bool("X") is False

    @pytest.mark.parametrize("val", ["1", "yes", "on", "y", "false", "0", ""])
    def test_only_true_is_true(self, val):
        """Matches the parse_config idiom — only the literal `true` (case-
        insensitive) is True. `1`, `yes`, etc. are False. Documents current
        behavior; DB-22186 tracks broadening this in the follow-up release."""
        with patch.dict(os.environ, {"X": val}):
            assert _env_bool("X") is False


# ---------------------------------------------------------------------------
# _check_http_startup — DB-22139 fail-closed guard
# ---------------------------------------------------------------------------

class TestCheckHttpStartup:
    """Fail-closed guard that runs before uvicorn opens the socket."""

    def test_stdio_transport_skips_the_check(self, with_config, caplog):
        """stdio has no network — the guard should no-op regardless of
        host/auth config."""
        with_config(transport="stdio", host="0.0.0.0", auth_provider=None)
        # Would exit(1) under http; must not under stdio.
        _check_http_startup("0.0.0.0")  # doesn't raise
        # No CRITICAL log for the auth gap either.
        assert not any(r.levelname == "CRITICAL" for r in caplog.records)

    def test_loopback_no_auth_with_allowlist_starts(self, with_config, caplog):
        """Loopback bind without auth is fine WHEN an Origin allowlist is
        configured — DNS-rebinding defense catches browser attacks that
        would otherwise reach 127.0.0.1."""
        with_config(host="127.0.0.1", auth_provider=None)
        with patch.dict(os.environ, {"MCP_ALLOWED_ORIGINS": "https://mcp.example.com"}), \
             caplog.at_level("WARNING"):
            _check_http_startup("127.0.0.1")  # doesn't exit
        assert not any(r.levelname == "CRITICAL" for r in caplog.records)

    def test_loopback_no_auth_no_allowlist_refuses_to_start(self, with_config, caplog):
        """DB-22139 round-2: loopback + no auth + no Origin allowlist =
        a browser DNS-rebinding attack can reach the loopback bind. Fail
        closed rather than warn-and-continue."""
        with_config(host="127.0.0.1", auth_provider=None)
        with caplog.at_level("CRITICAL"):
            with pytest.raises(SystemExit) as exc:
                _check_http_startup("127.0.0.1")
        assert exc.value.code == 1
        combined_log = " ".join(r.message for r in caplog.records)
        assert "DNS-rebinding" in combined_log
        assert "MCP_ALLOWED_ORIGINS" in combined_log

    def test_loopback_no_auth_no_allowlist_with_escape_starts(
        self, with_config, caplog,
    ):
        """The same escape hatch (MCP_ALLOW_UNAUTHENTICATED=true) that
        lets the public + no-auth case run also covers the no-allowlist
        case, since both live in the same defense-in-depth layer."""
        with_config(host="127.0.0.1", auth_provider=None)
        with patch.dict(os.environ, {"MCP_ALLOW_UNAUTHENTICATED": "true"}), \
             caplog.at_level("WARNING"):
            _check_http_startup("127.0.0.1")  # doesn't exit

    def test_loopback_with_auth_starts(self, with_config):
        """Loopback with auth is definitely fine."""
        with_config(host="127.0.0.1", auth_provider="cognito")
        _check_http_startup("127.0.0.1")  # doesn't exit

    def test_public_with_auth_starts(self, with_config, caplog):
        """Non-loopback bind with auth is the intended production shape."""
        with_config(host="0.0.0.0", auth_provider="cognito")
        _check_http_startup("0.0.0.0")  # doesn't exit
        # No CRITICAL log.
        assert not any(r.levelname == "CRITICAL" for r in caplog.records)

    def test_public_no_auth_no_escape_refuses_to_start(self, with_config, caplog):
        """DB-22139 primary repro: HTTP mode + non-loopback host +
        no auth + no escape hatch → sys.exit(1) with a CRITICAL log
        explaining what went wrong."""
        with_config(host="0.0.0.0", auth_provider=None)
        with caplog.at_level("CRITICAL"):
            with pytest.raises(SystemExit) as exc:
                _check_http_startup("0.0.0.0")
        assert exc.value.code == 1
        combined_log = " ".join(r.message for r in caplog.records)
        assert "MCP_AUTH_PROVIDER" in combined_log
        assert "0.0.0.0" in combined_log

    def test_public_no_auth_with_escape_starts_with_warning(
        self, with_config, caplog,
    ):
        """MCP_ALLOW_UNAUTHENTICATED=true lets HTTP mode run without auth
        on a public host — but with a very loud WARNING so operators see
        it in prod logs."""
        with_config(host="0.0.0.0", auth_provider=None)
        with patch.dict(os.environ, {"MCP_ALLOW_UNAUTHENTICATED": "true"}), \
             caplog.at_level("WARNING"):
            _check_http_startup("0.0.0.0")  # doesn't exit

        combined_log = " ".join(r.message for r in caplog.records)
        assert "UNAUTHENTICATED" in combined_log

    def test_escape_hatch_ignored_when_auth_is_configured(
        self, with_config, caplog,
    ):
        """MCP_ALLOW_UNAUTHENTICATED with auth also configured → no
        warning about unauth mode. The escape hatch is a no-op when
        auth is present."""
        with_config(host="0.0.0.0", auth_provider="cognito")
        with patch.dict(os.environ, {"MCP_ALLOW_UNAUTHENTICATED": "true"}), \
             caplog.at_level("WARNING"):
            _check_http_startup("0.0.0.0")

        assert not any(
            "UNAUTHENTICATED" in r.message for r in caplog.records
        )

    def test_warns_when_origin_allowlist_empty(self, with_config, caplog):
        """Independent of auth: an empty MCP_ALLOWED_ORIGINS means the
        DNS-rebinding defense is off. WARNING but not fatal."""
        with_config(host="127.0.0.1", auth_provider="cognito")
        # No MCP_ALLOWED_ORIGINS, no MCP_BASE_URL in env.
        with caplog.at_level("WARNING"):
            _check_http_startup("127.0.0.1")
        assert any(
            "Origin allowlist" in r.message for r in caplog.records
        )

    def test_no_origin_warning_when_allowlist_present(self, with_config, caplog):
        """Regression: with an allowlist configured, no Origin warning."""
        with_config(host="127.0.0.1", auth_provider="cognito")
        with patch.dict(os.environ, {"MCP_ALLOWED_ORIGINS": "https://mcp.example.com"}), \
             caplog.at_level("WARNING"):
            _check_http_startup("127.0.0.1")
        assert not any(
            "DNS-rebinding" in r.message or "Origin allowlist" in r.message
            for r in caplog.records
        )
