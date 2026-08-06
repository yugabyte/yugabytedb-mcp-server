"""Unit tests for `_pool_reset` (DB-22133 + DB-22202).

`_pool_reset` is the `reset=` callback wired into the psycopg
ConnectionPool. It runs two statements on every connection return so
session state (role, advisory locks, prepared statements, GUCs, temp
tables) doesn't bleed across pool checkouts:

1. ``DISCARD ALL``
2. ``SELECT pg_advisory_unlock_all()`` — YugabyteDB's DISCARD ALL does
   not release advisory locks (deviation from Postgres 15), so we
   explicitly release them.

Unit tests here mock the connection to verify the callback issues both
statements in the right order and handles failures gracefully.
End-to-end integration tests against a real database live in
`test_integration_pool_reset.py`.
"""
from unittest.mock import MagicMock

import pytest

from yugabytedb_mcp_server.server import _pool_reset


class TestPoolReset:

    def test_issues_discard_and_advisory_unlock(self):
        """Callback runs `DISCARD ALL` then `SELECT pg_advisory_unlock_all()`
        on the connection. The unlock is required because YugabyteDB's
        DISCARD ALL does not release advisory locks (deviation from PG 15)."""
        conn = MagicMock()
        _pool_reset(conn)
        assert conn.execute.call_count == 2
        assert conn.execute.call_args_list[0].args == ("DISCARD ALL",)
        assert conn.execute.call_args_list[1].args == ("SELECT pg_advisory_unlock_all()",)

    def test_flips_autocommit_around_reset(self):
        """DISCARD ALL and pg_advisory_unlock_all() cannot run inside a
        transaction block. The callback must set autocommit True before the
        reset statements and restore False after, so subsequent pool
        checkouts still open implicit transactions the way psycopg's default
        expects."""
        conn = MagicMock()
        _pool_reset(conn)

        # The order of method calls on the connection must be:
        # set_autocommit(True), execute("DISCARD ALL"),
        # execute("SELECT pg_advisory_unlock_all()"), set_autocommit(False).
        calls = [(c[0], c[1]) for c in conn.method_calls]
        assert calls == [
            ("set_autocommit", (True,)),
            ("execute", ("DISCARD ALL",)),
            ("execute", ("SELECT pg_advisory_unlock_all()",)),
            ("set_autocommit", (False,)),
        ], f"unexpected call order: {calls!r}"

    def test_restores_autocommit_even_when_discard_fails(self):
        """If DISCARD raises, the callback must still put the connection
        back into autocommit=False before returning. Otherwise the
        connection sits in autocommit mode until the next checkout,
        breaking psycopg's implicit-transaction contract."""
        conn = MagicMock()
        conn.execute.side_effect = RuntimeError("simulated DISCARD failure")

        _pool_reset(conn)  # must not raise

        calls = [(c[0], c[1]) for c in conn.method_calls]
        # set_autocommit(False) must still fire even after the execute raised.
        assert ("set_autocommit", (False,)) in calls, (
            f"autocommit not restored after DISCARD failure; calls={calls!r}"
        )

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

    def test_swallows_set_autocommit_failure(self, caplog):
        """If `set_autocommit(True)` raises (e.g. connection already
        dropped), the callback must still log-and-continue rather than
        letting the exception leak into psycopg-pool."""
        conn = MagicMock()
        conn.set_autocommit.side_effect = RuntimeError("cannot flip autocommit")

        with caplog.at_level("WARNING"):
            _pool_reset(conn)  # must not raise

        # WARNING logged, describes the failure.
        assert any("cannot flip autocommit" in r.message for r in caplog.records)
