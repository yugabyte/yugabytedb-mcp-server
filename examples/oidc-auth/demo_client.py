#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "httpx",
#   "mcp>=1.2",
# ]
# ///
"""One-command MCP client for the OIDC tutorial.

Drives `yugabytedb-mcp-server` end-to-end using the same browser-based
authorization-code OAuth flow that real MCP clients (Claude Desktop,
Cursor, MCP Inspector) use. No GUI, no Inspector — just one Python
script that opens the browser for sign-in and then exercises the tools.

Run:

    uv run demo_client.py

The script opens your browser to Keycloak's sign-in page. Enter EITHER
user:

    reader@yugabyte.com / Reader123    (reads succeed, writes denied)
    writer@yugabyte.com / Writer123    (reads and writes both succeed)

After sign-in, the script opens an MCP session against the running
server and runs three demo calls.

To try the other user, simply re-run the script. The OAuth request
includes `prompt=login`, which tells Keycloak to ignore any cached
session and always show the login form.

Prerequisites: Keycloak (Step 1), the seed script (Step 2), and the
MCP server (Step 3) all up and running. The shipped realm includes
`http://localhost:9876/callback` as a valid redirect URI for the
script's local OAuth callback listener.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import http.server
import json
import secrets
import sys
import threading
import urllib.parse
import webbrowser

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

KEYCLOAK_AUTHORIZE_URL = (
    "http://localhost:18080/realms/yb-mcp/protocol/openid-connect/auth"
)
KEYCLOAK_TOKEN_URL = (
    "http://localhost:18080/realms/yb-mcp/protocol/openid-connect/token"
)
CLIENT_ID = "yb-mcp-server"
CLIENT_SECRET = "tutorial-secret-not-for-prod"
MCP_URL = "http://localhost:8000/mcp"
CALLBACK_HOST = "127.0.0.1"
CALLBACK_PORT = 9876
REDIRECT_URI = f"http://localhost:{CALLBACK_PORT}/callback"

_received: dict[str, str | None] = {"code": None, "state": None, "error": None}


class _CallbackHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        params = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        _received["code"] = (params.get("code") or [None])[0]
        _received["state"] = (params.get("state") or [None])[0]
        _received["error"] = (params.get("error") or [None])[0]
        body = (
            "<html><body style='font-family: sans-serif; max-width: 480px;"
            " margin: 4em auto; color: #202124;'>"
            "<h2>Signed in.</h2>"
            "<p>You can close this tab and return to the terminal.</p>"
            "</body></html>"
        )
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(body.encode("utf-8"))

    def log_message(self, *args, **kwargs):  # silence default access log
        pass


def authorization_code_flow() -> str:
    """Drive the auth-code-with-PKCE flow and return the access token."""
    code_verifier = secrets.token_urlsafe(64)
    code_challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(code_verifier.encode()).digest())
        .rstrip(b"=")
        .decode("ascii")
    )
    state = secrets.token_urlsafe(16)

    server = http.server.HTTPServer(
        (CALLBACK_HOST, CALLBACK_PORT), _CallbackHandler
    )
    listener = threading.Thread(target=server.handle_request, daemon=True)
    listener.start()

    params = {
        "client_id": CLIENT_ID,
        "response_type": "code",
        "redirect_uri": REDIRECT_URI,
        "scope": "openid email profile",
        "state": state,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "prompt": "login",
    }
    authorize_url = f"{KEYCLOAK_AUTHORIZE_URL}?{urllib.parse.urlencode(params)}"

    print("==> Opening browser for Keycloak sign-in...")
    print(f"    (if your browser does not open, visit:\n     {authorize_url}\n    )")
    webbrowser.open(authorize_url)

    listener.join(timeout=300)
    server.server_close()

    if _received["error"]:
        sys.exit(f"Keycloak returned an error: {_received['error']}")
    if not _received["code"]:
        sys.exit("Timed out waiting for the OAuth callback.")
    if _received["state"] != state:
        sys.exit("State mismatch — possible CSRF.")

    with httpx.Client(timeout=10) as http_:
        r = http_.post(
            KEYCLOAK_TOKEN_URL,
            data={
                "grant_type": "authorization_code",
                "code": _received["code"],
                "redirect_uri": REDIRECT_URI,
                "client_id": CLIENT_ID,
                "client_secret": CLIENT_SECRET,
                "code_verifier": code_verifier,
            },
        )
    if r.status_code != 200:
        sys.exit(f"Token exchange failed ({r.status_code}): {r.text}")
    return r.json()["access_token"]


def email_from_token(access_token: str) -> str:
    payload_b64 = access_token.split(".")[1]
    payload_b64 += "=" * (-len(payload_b64) % 4)
    claims = json.loads(base64.urlsafe_b64decode(payload_b64))
    return claims.get("email") or claims.get("preferred_username") or "<unknown>"


def render(result) -> None:
    for block in result.content:
        text = getattr(block, "text", None)
        if text is None:
            continue
        try:
            print(json.dumps(json.loads(text), indent=2))
        except (json.JSONDecodeError, TypeError):
            print(text)


async def run_mcp(token: str, signed_in_as: str) -> None:
    user_key = signed_in_as.split("@", 1)[0]
    headers = {"Authorization": f"Bearer {token}"}
    async with streamablehttp_client(MCP_URL, headers=headers) as (
        read_stream,
        write_stream,
        _,
    ):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            tools = await session.list_tools()
            tool_names = sorted(t.name for t in tools.tools)
            print(f"==> Connected to MCP server. Tools: {', '.join(tool_names)}\n")

            print("--- 1. Confirm effective database role ---")
            render(
                await session.call_tool(
                    "run_read_only_query",
                    {
                        "query": (
                            "SELECT current_user, session_user, "
                            "current_setting('role') AS effective_role"
                        )
                    },
                )
            )

            print("\n--- 2. SELECT from notes ---")
            render(
                await session.call_tool(
                    "run_read_only_query",
                    {"query": "SELECT id, body FROM notes ORDER BY id"},
                )
            )

            print(f"\n--- 3. INSERT into notes as {user_key} ---")
            render(
                await session.call_tool(
                    "run_write_query",
                    {
                        "query": (
                            "INSERT INTO notes (body) VALUES "
                            f"('hello from {user_key} (demo_client)')"
                        )
                    },
                )
            )


def main() -> None:
    token = authorization_code_flow()
    signed_in_as = email_from_token(token)
    print(f"==> Signed in as {signed_in_as}\n")
    asyncio.run(run_mcp(token, signed_in_as))


if __name__ == "__main__":
    main()
