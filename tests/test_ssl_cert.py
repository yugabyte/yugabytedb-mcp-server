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
    the next block's BEGIN."""

    @pytest.mark.parametrize("sep", ["  ", "   ", "\n\n", "\t", " \n \t "])
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
