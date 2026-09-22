"""Unit tests for DB-22173: TLS root-cert file-safety hardening.

Covers two independent fixes:

- ``normalize_pem`` now handles any whitespace between armor lines, not
  just the hard-coded 1-space and 2-space patterns it used to check.
- ``_write_cert_atomic`` uses ``O_CREAT|O_EXCL|O_WRONLY|O_NOFOLLOW`` with
  mode ``0o600`` and a same-directory temp file + ``os.replace``, so a
  pre-existing symlink at the destination can't redirect the write to
  an attacker-chosen target and there's no umask race.
"""
import os
import stat
import tempfile

import pytest

from yugabytedb_mcp_server.server import (
    _append_conninfo_param,
    _write_cert_atomic,
    normalize_pem,
)


# ---------------------------------------------------------------------------
# normalize_pem — regex-based whitespace collapse
# ---------------------------------------------------------------------------

_BODY = "MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAxxxx"  # arbitrary base64
_CANONICAL_ONE = (
    "-----BEGIN CERTIFICATE-----\n"
    f"{_BODY}\n"
    "-----END CERTIFICATE-----\n"
)


class TestNormalizePemVariableWhitespace:
    """Pre-fix, the helper only recognized exact 1-space (between BEGIN
    and body / body and END) and exact 2-space (between END and BEGIN
    of the next block) separators. Anything else — 3 spaces, tabs,
    mixed whitespace — collapsed to a single unparseable line and
    libpq rejected the cert with an opaque error."""

    @pytest.mark.parametrize("sep", [" ", "  ", "   ", "\t", "\t ", " \t\t "])
    def test_variable_whitespace_between_begin_and_body(self, sep):
        mangled = f"-----BEGIN CERTIFICATE-----{sep}{_BODY}{sep}-----END CERTIFICATE-----"
        assert normalize_pem(mangled) == _CANONICAL_ONE

    def test_already_canonical_is_idempotent(self):
        assert normalize_pem(_CANONICAL_ONE) == _CANONICAL_ONE

    def test_trailing_and_leading_whitespace_stripped(self):
        mangled = f"  \n\n{_CANONICAL_ONE.strip()}\n  "
        assert normalize_pem(mangled) == _CANONICAL_ONE


class TestNormalizePemChains:
    """Multi-cert bundles from Secrets Manager arrive with anywhere
    from zero to several whitespace chars between one block's END and
    the next block's BEGIN. The zero-separator case is real: a JSON
    round-trip that stripped every internal newline yields a glued
    ``-----END CERTIFICATE----------BEGIN CERTIFICATE-----`` sequence
    that libpq refuses. ``_END_TO_BEGIN`` in ``server.py`` uses ``\\s*``
    (not ``\\s+``) so the empty-separator input still splits cleanly."""

    @pytest.mark.parametrize("sep", ["", "  ", "   ", "\n\n", "\t", " \n \t "])
    def test_end_to_begin_variable_whitespace(self, sep):
        body_a = _BODY
        body_b = _BODY[::-1]  # different content
        mangled = (
            "-----BEGIN CERTIFICATE----- "
            f"{body_a}"
            " -----END CERTIFICATE-----"
            f"{sep}"
            "-----BEGIN CERTIFICATE----- "
            f"{body_b}"
            " -----END CERTIFICATE-----"
        )
        expected = (
            "-----BEGIN CERTIFICATE-----\n"
            f"{body_a}\n"
            "-----END CERTIFICATE-----\n\n"
            "-----BEGIN CERTIFICATE-----\n"
            f"{body_b}\n"
            "-----END CERTIFICATE-----\n"
        )
        assert normalize_pem(mangled) == expected


# ---------------------------------------------------------------------------
# _write_cert_atomic — file-safety
# ---------------------------------------------------------------------------

class TestWriteCertAtomic:
    def test_written_file_is_mode_0600(self, tmp_path):
        """The cert can contain a chain that clients trust to decide
        TLS validity — mode-0600 prevents any local user from reading
        or modifying it."""
        dest = tmp_path / "yb-root.crt"
        _write_cert_atomic(str(dest), _CANONICAL_ONE)
        got = stat.S_IMODE(os.stat(dest).st_mode)
        assert got == 0o600, f"expected 0600, got {oct(got)}"

    def test_write_refuses_to_follow_symlink_at_destination(self, tmp_path):
        """A pre-existing symlink at the destination path used to
        redirect the write via ``open(path, 'w')`` — attacker plants
        ``/tmp/yb-root.crt -> /some/victim`` before startup and the
        server clobbers the victim. Post-fix: the temp file is opened
        with ``O_NOFOLLOW`` and ``os.replace`` swaps the temp in place,
        replacing the symlink itself (not what it points to)."""
        victim = tmp_path / "victim.txt"
        victim.write_text("DO NOT OVERWRITE")
        dest = tmp_path / "yb-root.crt"
        os.symlink(str(victim), str(dest))

        _write_cert_atomic(str(dest), _CANONICAL_ONE)

        # Victim untouched.
        assert victim.read_text() == "DO NOT OVERWRITE"
        # The destination is now a regular file (not a symlink) with the
        # cert content.
        assert not os.path.islink(dest)
        assert dest.read_text() == _CANONICAL_ONE

    def test_no_stray_temp_files_left_on_success(self, tmp_path):
        """The atomic-write helper writes to a `.tmp` sibling then
        renames it in. On success the temp must not linger."""
        dest = tmp_path / "yb-root.crt"
        _write_cert_atomic(str(dest), _CANONICAL_ONE)
        stray = [p for p in tmp_path.iterdir() if p.name.startswith(".yb-root-cert.")]
        assert stray == [], f"stray temp files: {stray}"

    def test_overwrite_replaces_previous_content_atomically(self, tmp_path):
        """A re-run of the server must be able to update an existing
        cert — the atomic-write flow uses ``os.replace``, which
        overwrites even when the destination already exists."""
        dest = tmp_path / "yb-root.crt"
        _write_cert_atomic(str(dest), "OLD\n")
        _write_cert_atomic(str(dest), _CANONICAL_ONE)
        assert dest.read_text() == _CANONICAL_ONE
        # Mode is preserved on re-write.
        assert stat.S_IMODE(os.stat(dest).st_mode) == 0o600


# ---------------------------------------------------------------------------
# _cert_destination + atexit cleanup — no tmpdir leaks per process start
# ---------------------------------------------------------------------------

class TestCertDirCleanup:
    """Every process start under stdio transport spawns a fresh MCP
    server; without cleanup, each start left an owner-only
    ``yb-mcp-cert-*/`` dir in the operator's tmpdir with nothing ever
    reaping it. ``_cert_destination`` now records the dirs it creates
    into a module-level list, and an ``atexit`` hook removes them at
    interpreter shutdown."""

    def test_configured_path_does_not_track_for_cleanup(self, tmp_path):
        """When the operator supplies ``YB_SSL_ROOT_CERT_PATH``, the dir
        belongs to them — we must not add it to the cleanup list."""
        from yugabytedb_mcp_server.server import _cert_destination, _owned_cert_dirs

        # Snapshot the list length; call must not append.
        before = list(_owned_cert_dirs)
        configured = str(tmp_path / "operator-owned-dir" / "yb-root.crt")
        result = _cert_destination(configured)
        assert result == configured
        assert list(_owned_cert_dirs) == before

    def test_default_path_is_tracked_and_cleanup_removes_it(self):
        """A default (no ``YB_SSL_ROOT_CERT_PATH``) invocation makes a
        private ``mkdtemp`` dir, appends it to the tracking list, and
        ``_cleanup_owned_cert_dirs`` wipes it."""
        from yugabytedb_mcp_server.server import (
            _cert_destination,
            _owned_cert_dirs,
            _cleanup_owned_cert_dirs,
        )

        before = len(_owned_cert_dirs)
        path = _cert_destination(None)
        assert len(_owned_cert_dirs) == before + 1
        cert_dir = os.path.dirname(path)
        assert os.path.isdir(cert_dir)
        assert cert_dir.rsplit("/", 1)[-1].startswith("yb-mcp-cert-")

        _cleanup_owned_cert_dirs()
        assert not os.path.exists(cert_dir), (
            f"atexit cleanup should have removed {cert_dir}"
        )

    def test_cleanup_is_best_effort_on_missing_dir(self):
        """Cleanup swallows errors — a dir that already vanished (or
        was never created) must not raise on shutdown."""
        from yugabytedb_mcp_server.server import (
            _owned_cert_dirs,
            _cleanup_owned_cert_dirs,
        )

        _owned_cert_dirs.append("/nonexistent/path/that/never/existed")
        try:
            _cleanup_owned_cert_dirs()  # must not raise
        finally:
            try:
                _owned_cert_dirs.remove(
                    "/nonexistent/path/that/never/existed"
                )
            except ValueError:
                pass


class TestAppendConninfoParam:
    """DB-22185 follow-up: appending ``sslrootcert=<path>`` (and, from
    DB-22159, ``connect_timeout=10``) must use the right separator for
    the conninfo's form. libpq keyword form is space-separated; URI form
    is query-string style. Space-appending to a URI (
    ``postgresql://…?sslmode=verify-full sslrootcert=/tmp/x``) is
    rejected by psycopg. This shared helper backs both append sites, so
    the two paths can't drift out of sync again."""

    def test_keyword_form_uses_space_separator(self):
        url = "host=x port=5433 user=y password=z"
        result = _append_conninfo_param(url, "sslrootcert", "/etc/x.crt")
        assert result == "host=x port=5433 user=y password=z sslrootcert=/etc/x.crt"

    def test_uri_form_without_existing_query_uses_question_mark(self):
        url = "postgresql://u@h:5433/db"
        result = _append_conninfo_param(url, "sslrootcert", "/etc/x.crt")
        assert result == "postgresql://u@h:5433/db?sslrootcert=/etc/x.crt"

    def test_uri_form_with_existing_query_uses_ampersand(self):
        url = "postgresql://u@h:5433/db?sslmode=verify-full"
        result = _append_conninfo_param(url, "sslrootcert", "/etc/x.crt")
        # Space-appending to this input was the specific bug: it would
        # produce "?sslmode=verify-full sslrootcert=/etc/x.crt", which
        # psycopg refuses because the space becomes part of the last
        # query-param VALUE.
        assert result == (
            "postgresql://u@h:5433/db?sslmode=verify-full&sslrootcert=/etc/x.crt"
        )

    def test_postgres_scheme_variant_also_uri_form(self):
        # libpq accepts both ``postgres://`` and ``postgresql://`` — the
        # helper must detect both.
        url = "postgres://u@h:5433/db?sslmode=require"
        result = _append_conninfo_param(url, "connect_timeout", "10")
        assert result == (
            "postgres://u@h:5433/db?sslmode=require&connect_timeout=10"
        )

    def test_connect_timeout_keyword_form_reuses_helper(self):
        # Regression pin for the DB-22159 keyword-form path — the helper
        # replaced an inline ``f"{url} connect_timeout=10"`` so keyword
        # form must still produce the same string.
        url = "host=x port=5433 user=y"
        result = _append_conninfo_param(url, "connect_timeout", "10")
        assert result == "host=x port=5433 user=y connect_timeout=10"
        """Cleanup swallows errors — a dir that already vanished (or
        was never created) must not raise on shutdown."""
        from yugabytedb_mcp_server.server import (
            _owned_cert_dirs,
            _cleanup_owned_cert_dirs,
        )

        # Push a bogus path and confirm cleanup doesn't raise.
        _owned_cert_dirs.append("/nonexistent/path/that/never/existed")
        try:
            _cleanup_owned_cert_dirs()  # must not raise
        finally:
            # Best-effort pop — leave the list clean for other tests.
            try:
                _owned_cert_dirs.remove(
                    "/nonexistent/path/that/never/existed"
                )
            except ValueError:
                pass
