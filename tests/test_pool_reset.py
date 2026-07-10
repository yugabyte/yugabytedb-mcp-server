"""Unit tests for `_pool_reset` (DB-22133 + DB-22202).

`_pool_reset` is the `reset=` callback wired into the psycopg
ConnectionPool. It runs `DISCARD ALL` on every connection return so
session state (role, advisory locks, prepared statements, GUCs, temp
tables) doesn't bleed across pool checkouts.

Unit tests here mock the connection to verify the callback issues
`DISCARD ALL` and handles failures gracefully. End-to-end integration
tests against a real Postgres live in `test_integration_pool_reset.py`.
"""
from unittest.mock import MagicMock

import pytest

from yugabytedb_mcp_server.server import _pool_reset


class TestPoolReset:

    def test_issues_discard_all(self):
        """Callback runs `DISCARD ALL` on the connection."""
        conn = MagicMock()
        _pool_reset(conn)
        conn.execute.assert_called_once_with("DISCARD ALL")

    def test_swallows_execute_failure(self, caplog):
        """If the connection is bad, DISCARD ALL may raise. The callback
        must log-and-continue so the pool's health check on the next
        checkout can replace the connection — raising here would leak
        the exception into psycopg-pool's internal machinery."""
        conn = MagicMock()
        conn.execute.side_effect = RuntimeError("connection closed")

        with caplog.at_level("WARNING"):
            _pool_reset(conn)  # must not raise

        # WARNING was logged.
        combined = " ".join(r.message for r in caplog.records)
        assert "DISCARD ALL" in combined
        assert "connection closed" in combined

    def test_no_side_effects_beyond_execute(self):
        """Sanity: only DISCARD ALL is issued — no cursor management,
        no commit, no rollback. Psycopg's autocommit semantics for the
        pool's post-return path handle transaction state."""
        conn = MagicMock()
        _pool_reset(conn)
        # Only one method call on the conn — .execute("DISCARD ALL").
        # Verifies we're not accidentally doing extra work that could
        # itself fail or leak state.
        assert len(conn.method_calls) == 1
        method_name = conn.method_calls[0][0]
        assert method_name == "execute"
