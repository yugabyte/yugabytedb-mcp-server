"""Unit tests for _sanitize_for_json in tools.py.

Covers DB-22157 (NaN/Infinity produced invalid JSON tokens; bytea was
rendered as lossy Python repr). Also acts as a regression harness for the
existing default=str fallthrough (datetime/Decimal/UUID/IPv4Address).

The tool's runtime path is:

    json.dumps(_sanitize_for_json(rows), allow_nan=False, default=str)

`_sanitize_for_json` recursively converts non-JSON-safe values before
`json.dumps` sees them; `allow_nan=False` then guarantees the emitted
string is RFC-8259-valid JSON that strict parsers (JS `JSON.parse`) accept.
"""
import json
from datetime import date, datetime, timezone
from decimal import Decimal
from ipaddress import IPv4Address
from uuid import UUID

import pytest

from yugabytedb_mcp_server.tools import _sanitize_for_json


# ---------------------------------------------------------------------------
# DB-22157: non-finite floats produce invalid JSON via bare NaN/Infinity tokens
# ---------------------------------------------------------------------------

class TestNonFiniteFloats:
    """Non-finite float values (`nan`, `inf`, `-inf`) must serialize as `null`
    — Python's `json.dumps` emits them as bare tokens which are invalid JSON
    per RFC 8259 and rejected by strict parsers including JavaScript
    `JSON.parse`."""

    @pytest.mark.parametrize("value", [
        float("nan"),
        float("inf"),
        float("-inf"),
    ])
    def test_non_finite_becomes_null(self, value):
        assert _sanitize_for_json([{"v": value}]) == [{"v": None}]

    def test_finite_floats_pass_through(self):
        assert _sanitize_for_json([
            {"v": 1.5},
            {"v": -3.14},
            {"v": 0.0},
        ]) == [
            {"v": 1.5},
            {"v": -3.14},
            {"v": 0.0},
        ]

    def test_nested_list_with_nan(self):
        # array-typed columns (Postgres text[]/int[]/float[]) come back as
        # Python lists — must recurse.
        result = _sanitize_for_json([{"arr": [1.0, float("nan"), 3.0]}])
        assert result == [{"arr": [1.0, None, 3.0]}]

    def test_nested_dict_with_inf(self):
        # jsonb columns come back as Python dicts — must recurse.
        result = _sanitize_for_json([{"j": {"score": float("inf"), "ok": True}}])
        assert result == [{"j": {"score": None, "ok": True}}]

    def test_tuple_treated_as_list(self):
        # Composite/row values come back as tuples — same recursion.
        result = _sanitize_for_json({"row": (1, float("nan"), "x")})
        assert result == {"row": [1, None, "x"]}


# ---------------------------------------------------------------------------
# DB-22157: bytea encoded as lossy Python repr via default=str
# ---------------------------------------------------------------------------

class TestBytesEncoding:
    """`bytea` values were rendered as `"b'\\xde\\xad\\xbe\\xef'"` (Python
    bytes-repr) — lossy and not a defined encoding. Now encoded as
    `{"$hex": "deadbeef"}` for a lossless, well-defined round-trip."""

    def test_bytes_to_hex(self):
        assert _sanitize_for_json([{"b": b"\xde\xad\xbe\xef"}]) == [
            {"b": {"$hex": "deadbeef"}}
        ]

    def test_bytearray_to_hex(self):
        assert _sanitize_for_json([{"b": bytearray(b"\x00\xff")}]) == [
            {"b": {"$hex": "00ff"}}
        ]

    def test_memoryview_to_hex(self):
        assert _sanitize_for_json([{"b": memoryview(b"abc")}]) == [
            {"b": {"$hex": "616263"}}
        ]

    def test_empty_bytes(self):
        assert _sanitize_for_json([{"b": b""}]) == [{"b": {"$hex": ""}}]

    def test_bytes_inside_nested_structure(self):
        result = _sanitize_for_json([{"outer": [b"\x01\x02"]}])
        assert result == [{"outer": [{"$hex": "0102"}]}]


# ---------------------------------------------------------------------------
# End-to-end: sanitize + json.dumps(allow_nan=False) round-trips through
# strict parsers.
# ---------------------------------------------------------------------------

class TestStrictJSONRoundtrip:
    """The runtime path uses `allow_nan=False` — an unsanitized NaN raises
    `ValueError` even from a `default=` hook. These tests verify the
    combined pipeline emits valid JSON for realistic query result shapes."""

    def test_nan_and_bytea_together(self):
        fixture = [
            {"score": float("nan"), "blob": b"\xde\xad", "ok": True},
            {"score": 42.5, "blob": b"", "ok": False},
        ]
        rendered = json.dumps(
            _sanitize_for_json(fixture), allow_nan=False,
        )
        assert json.loads(rendered) == [
            {"score": None, "blob": {"$hex": "dead"}, "ok": True},
            {"score": 42.5, "blob": {"$hex": ""}, "ok": False},
        ]

    def test_allow_nan_false_never_raises_after_sanitize(self):
        """Regression: previously json.dumps(NaN, allow_nan=False) raised
        ValueError. After sanitize, allow_nan=False is safe on any input
        with NaN/Inf mixed in."""
        fixture = [{"a": float("nan"), "b": [float("inf"), float("-inf")]}]
        # If sanitize missed a value, this would raise:
        json.dumps(_sanitize_for_json(fixture), allow_nan=False)


# ---------------------------------------------------------------------------
# Regression: existing default=str coverage must not regress
# ---------------------------------------------------------------------------

class TestFallthroughToDefaultStr:
    """The pre-fix code used `json.dumps(result, default=str)` which handled
    datetime/Decimal/UUID/IPv4Address by stringifying. The new sanitizer must
    not consume those types — they fall through to `default=str` unchanged."""

    def test_datetime_passes_through_untouched(self):
        dt = datetime(2026, 7, 10, 12, 0, tzinfo=timezone.utc)
        sanitized = _sanitize_for_json([{"t": dt}])
        # sanitizer leaves the datetime as a datetime object
        assert isinstance(sanitized[0]["t"], datetime)
        # json.dumps default=str stringifies it
        rendered = json.dumps(sanitized, default=str)
        assert "2026-07-10" in rendered

    def test_date_passes_through(self):
        rendered = json.dumps(_sanitize_for_json([{"d": date(2026, 7, 10)}]), default=str)
        assert "2026-07-10" in rendered

    def test_decimal_passes_through(self):
        rendered = json.dumps(_sanitize_for_json([{"n": Decimal("3.14")}]), default=str)
        assert "3.14" in rendered

    def test_uuid_passes_through(self):
        u = UUID("12345678-1234-5678-1234-567812345678")
        rendered = json.dumps(_sanitize_for_json([{"u": u}]), default=str)
        assert "12345678-1234-5678-1234-567812345678" in rendered

    def test_ipv4_passes_through(self):
        rendered = json.dumps(
            _sanitize_for_json([{"ip": IPv4Address("192.168.1.1")}]),
            default=str,
        )
        assert "192.168.1.1" in rendered


# ---------------------------------------------------------------------------
# Recursion depth / empty structures / primitives
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_empty_list(self):
        assert _sanitize_for_json([]) == []

    def test_empty_dict(self):
        assert _sanitize_for_json({}) == {}

    def test_primitive_string(self):
        assert _sanitize_for_json("hello") == "hello"

    def test_primitive_int(self):
        assert _sanitize_for_json(42) == 42

    def test_primitive_bool(self):
        assert _sanitize_for_json(True) is True
        assert _sanitize_for_json(False) is False

    def test_none(self):
        assert _sanitize_for_json(None) is None

    def test_deeply_nested(self):
        # 4-level nested structure with a NaN at the bottom
        result = _sanitize_for_json(
            {"a": {"b": [{"c": [float("nan")]}]}}
        )
        assert result == {"a": {"b": [{"c": [None]}]}}


# ---------------------------------------------------------------------------
# DB-22203: run_read_only_query response shape must not use dict(zip(cols, row))
# ---------------------------------------------------------------------------

class TestReadResultShape:
    """DB-22203: previously `run_read_only_query` returned
    `[dict(zip(cols, row)) for row in rows]`, which silently dropped
    duplicate column names (e.g. `SELECT * FROM a, b` where both tables
    have `id`, or `SELECT 1 AS id, 2 AS id`). The new shape is
    `{"columns": [...], "rows": [[...], ...]}` — parallel arrays that
    cannot collide.

    These unit tests build the shape directly (no DB required) and verify:
    - it round-trips through the sanitizer + json.dumps pipeline,
    - duplicate column names survive,
    - non-JSON-safe values in rows still get sanitized (NaN → None,
      bytes → $hex).
    """

    def test_duplicate_column_names_preserved(self):
        result = {"columns": ["id", "id", "other"], "rows": [[1, 2, 3]]}
        sanitized = _sanitize_for_json(result)
        assert sanitized == {"columns": ["id", "id", "other"], "rows": [[1, 2, 3]]}
        # And valid JSON — three distinct positions, no dict-key collapse.
        rendered = json.dumps(sanitized, allow_nan=False)
        parsed = json.loads(rendered)
        assert len(parsed["columns"]) == 3
        assert len(parsed["rows"][0]) == 3

    def test_empty_result_set(self):
        result = {"columns": ["a", "b"], "rows": []}
        assert _sanitize_for_json(result) == {"columns": ["a", "b"], "rows": []}

    def test_rows_sanitized_recursively(self):
        # NaN in one column, bytea in another — both must be normalized
        # inside the row arrays.
        result = {
            "columns": ["score", "blob"],
            "rows": [[float("nan"), b"\xde\xad"]],
        }
        sanitized = _sanitize_for_json(result)
        assert sanitized == {
            "columns": ["score", "blob"],
            "rows": [[None, {"$hex": "dead"}]],
        }
        # Strict JSON survives.
        rendered = json.dumps(sanitized, allow_nan=False)
        assert json.loads(rendered) == sanitized
