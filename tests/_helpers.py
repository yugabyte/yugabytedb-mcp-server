"""Test helpers — pure functions and skip markers. Importable from test modules.

Pytest fixtures live in conftest.py; this module is for things that need to be
imported by name (which pytest doesn't expose for conftest.py contents).
"""
import json
import os

import pytest


def yugabytedb_url() -> str | None:
    return os.environ.get("YUGABYTEDB_URL")


requires_yugabytedb = pytest.mark.skipif(
    not yugabytedb_url(),
    reason="YUGABYTEDB_URL not set",
)


def _result_text(result) -> str:
    """Extract the text content from an MCP CallToolResult."""
    if not result.content:
        return ""
    return result.content[0].text


def parse_json(result):
    """Parse a tool result as JSON.

    String-returning tools (run_write_query, run_read_only_query): the JSON
    text is in content[0].text — parse it.
    Non-string-returning tools (summarize_database returns list[dict]): the
    Python value is in structuredContent['result'].
    """
    text = _result_text(result)
    if text:
        return json.loads(text)
    sc = getattr(result, "structuredContent", None)
    if sc is not None and "result" in sc:
        return sc["result"]
    raise RuntimeError("Tool returned no content")


def parse_json_list(result):
    """Same as parse_json but checks text content for the 'Error: ...' prefix
    that run_read_only_query uses for runtime errors."""
    text = _result_text(result)
    if text.startswith("Error"):
        raise RuntimeError(f"Tool returned error: {text}")
    return parse_json(result)


def raw_text(result) -> str:
    """Raw text content of a tool result.

    For non-string-returning tools whose result is delivered via
    structuredContent, returns the JSON serialization for backwards
    compatibility with tests that .startswith('Error') etc.
    """
    text = _result_text(result)
    if text:
        return text
    sc = getattr(result, "structuredContent", None)
    if sc is not None and "result" in sc:
        return json.dumps(sc["result"])
    return ""
