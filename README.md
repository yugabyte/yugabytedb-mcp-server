# YugabyteDB MCP Server

An [MCP](https://modelcontextprotocol.io/) server for YugabyteDB and PostgreSQL — lets LLMs (Claude Desktop, Cursor, Windsurf, etc.) summarize schemas, run read-only queries, and execute write statements behind a configurable guardrail layer.

## Features

- **`summarize_database`** — list tables with columns and row counts for a schema (read-only)
- **`run_read_only_query`** — execute a SELECT under `BEGIN READ ONLY`; results returned as JSON (read-only)
- **`run_write_query`** — INSERT/UPDATE/DELETE/MERGE/TRUNCATE/DDL gated by a guardrail blocklist (destructive, **disabled by default** — enable with `--enable-write-query` or `YB_MCP_ENABLE_WRITE_QUERY=true`)

Defense in depth: the write tool is annotated `destructiveHint: true`, so Claude Desktop surfaces a confirmation prompt before every call even when the guardrails would let the statement through.

Optional OAuth (AWS Cognito) and Origin-header validation for self-hosted remote deployments.

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- A reachable YugabyteDB or PostgreSQL database
- An MCP client (Claude Desktop, Cursor, Windsurf, etc.)

## Installation

Three install options, in roughly the order of how end users will reach for them:

```bash
# uvx — no install at all; fetches and runs on demand. Handy for one-off use
# and also the form the MCPB Desktop extension uses internally.
uvx yugabytedb-mcp-server --help

# pipx — installs to an isolated venv, puts the script on $PATH.
pipx install yugabytedb-mcp-server

# uv tool — same idea, uv-managed.
uv tool install yugabytedb-mcp-server

# pip — system-level or current-venv install.
pip install yugabytedb-mcp-server
```

After any of the persistent installs (pipx / uv tool / pip), verify with:

```bash
yugabytedb-mcp --help
# or, equivalently:
yugabytedb-mcp-server --help
```

Both console scripts are registered and point at the same entry point —
`yugabytedb-mcp` is the short form, `yugabytedb-mcp-server` matches the
package name and is what `uvx` resolves to by default.

> **Pre-release note**: while v2 is in release-candidate (e.g. `2.0.0rc2`),
> default installs won't pick it up. For now, install with an explicit
> version (`pipx install yugabytedb-mcp-server==2.0.0rc2`) or with
> `--pip-args='--pre'`. This goes away once `2.0.0` stable is published.

For development from source, see [Development](#development) below.

## Configuration

| Environment Variable | CLI flag | Required | Description |
|---|---|---|---|
| `YUGABYTEDB_URL` | `--yugabytedb-url` | Yes | libpq connection string (e.g. `host=… port=5433 dbname=… user=… password=…`). |
| `YB_MCP_TRANSPORT` | `--transport` | No | `stdio` (default) or `http`. |
| `YB_MCP_STATELESS_HTTP` | `--stateless-http` | No | `true` enables stateless Streamable-HTTP — required for multi-replica self-hosted deployments. |
| `YB_MCP_REQUIRE_WHERE_ON_UPDATE` | `--require-where-on-update` | No | Reject UPDATE without a WHERE clause. Default `false`. |
| `YB_MCP_REQUIRE_WHERE_ON_DELETE` | `--require-where-on-delete` | No | Reject DELETE without a WHERE clause. Default `false`. |
| `YB_MCP_ENABLE_WRITE_QUERY` | `--enable-write-query` | No | Enable the `run_write_query` tool. Default `false` (write tool disabled). |
| `MCP_AUTH_PROVIDER` | `--mcp-auth-provider` | No | `cognito` or `oidc`. Leave unset to disable auth. Full OIDC/Cognito setup + per-user identity mapping is documented in [`OIDC.md`](OIDC.md). |
| `MCP_HOST` | `--host` | No | Bind host for HTTP transport. Default `127.0.0.1` (loopback). Set to `0.0.0.0` to expose on all interfaces — auth becomes mandatory in that case (see `MCP_AUTH_PROVIDER`). |
| `MCP_BASE_URL` | — | When auth enabled | Public base URL the server is reachable at (e.g. `https://mcp.example.com`). |
| `MCP_ALLOWED_ORIGINS` | — | No | Comma-separated allowlist of Origin values for DNS-rebinding defense. Case-insensitive (RFC 6454). Defaults to `MCP_BASE_URL`. |
| `MCP_ALLOW_UNAUTHENTICATED` | — | No | Escape hatch to run HTTP mode on a non-loopback host without auth. Dev-only; startup logs a prominent WARNING. |
| `YB_LOG_LEVEL` | — | No | Log level for the `yugabytedb-mcp` logger family (default `INFO`). |
| `YB_AWS_SSL_ROOT_CERT_SECRET_ARN` | `--yb-aws-ssl-root-cert-secret-arn` | No | ARN of an AWS Secrets Manager secret holding the YugabyteDB TLS root certificate. |
| `YB_AWS_SSL_ROOT_CERT_KEY` | `--yb-aws-ssl-root-cert-key` | No | JSON key inside the secret when it stores multiple certs. |
| `YB_AWS_SSL_ROOT_CERT_SECRET_REGION` | `--yb-aws-ssl-root-cert-secret-region` | No | AWS region of the secret. |
| `YB_SSL_ROOT_CERT_PATH` | `--yb-ssl-root-cert-path` | No | Where to write the fetched cert. Default `/tmp/yb-root.crt`. |

For OIDC/Cognito authentication, per-user `SET ROLE` mapping, the identity map
file format, and the `/auth/login` shortcut — see [`OIDC.md`](OIDC.md).

A starter template is in `.env.example`.

## Quickstart — Claude Desktop

Two ways to wire it up. The first uses `uvx` and requires no install at all
— `uv` only. The second assumes you've already run `pipx install` (or
equivalent) and have the `yugabytedb-mcp` script on `$PATH`.

**Option 1 — via `uvx` (no install):**

```json
{
  "mcpServers": {
    "yugabytedb": {
      "command": "uvx",
      "args": ["yugabytedb-mcp-server"],
      "env": {
        "YUGABYTEDB_URL": "host=… port=5433 dbname=… user=… password=…"
      }
    }
  }
}
```

**Option 2 — via an installed script:**

After `pipx install yugabytedb-mcp-server` (or `uv tool install …`):

```json
{
  "mcpServers": {
    "yugabytedb": {
      "command": "yugabytedb-mcp",
      "env": {
        "YUGABYTEDB_URL": "host=… port=5433 dbname=… user=… password=…"
      }
    }
  }
}
```

Locations of `claude_desktop_config.json`:
- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`

Restart Claude Desktop. The three tools will appear with titles and hint badges (read-only icons on the read tools, a confirmation prompt before each `run_write_query` call).

> While `2.0.0rc2` is the only published version, the `uvx` snippet needs
> `["yugabytedb-mcp-server@2.0.0rc2"]` in the args (or `["--pre",
> "yugabytedb-mcp-server"]`). Drop the explicit version once `2.0.0` stable
> is out.

## Other MCP Clients

The same approach works with **Cursor** (Settings → MCP → Add a new global MCP server) and **Windsurf** (Settings → Cascade → MCP Servers → Add custom server) — use either the `uvx` form or the installed-script form from above.

For MCP Inspector against an HTTP-mode server:

```bash
YUGABYTEDB_URL="…" yugabytedb-mcp --transport http
# in another shell:
npx @modelcontextprotocol/inspector
# In the GUI: URL http://localhost:8000/mcp, transport Streamable-HTTP
```

## Tools

| Tool | Title | Hints | What it does |
|---|---|---|---|
| `summarize_database(schema='public')` | "Summarize database schema and row counts" | `readOnlyHint: true` | Lists tables in `schema` with columns and row counts |
| `run_read_only_query(query)` | "Run a read-only SQL query" | `readOnlyHint: true` | Wraps the query in `BEGIN READ ONLY` and returns rows as JSON |
| `run_write_query(query)` | "Run a write SQL query (with guardrails)" | `destructiveHint: true` | Validates the query against the guardrail blocklist, then executes. **Disabled by default** — requires `--enable-write-query`. |

### Guardrails for `run_write_query`

The following statement classes are rejected before execution:

- `DROP DATABASE/SCHEMA`, `ALTER DATABASE`, `CREATE DATABASE`
- Role/privilege ops: `GRANT`, `REVOKE`, `CREATE/ALTER/DROP ROLE`, `CREATE/ALTER/DROP USER`
- Stored code that can run under the owner (`SECURITY DEFINER`): `CREATE FUNCTION`, `CREATE PROCEDURE`, `ALTER FUNCTION`, `ALTER PROCEDURE`
- Filesystem / code execution: `COPY TO/FROM`, `LOAD`, anonymous `DO $$ … $$`, `CREATE EXTENSION`
- Server config: `ALTER SYSTEM`, `RESET ALL`
- Dangerous built-ins: `pg_sleep`, `pg_read_file`, `pg_write_file`, `lo_import`, `lo_export`, `dblink`
- Schema isolation: `SET search_path`, `CREATE SCHEMA`
- Multi-statement queries (anything with a separator semicolon)
- `psql` meta-commands (`\c`, `\d`, `\!`)
- Optionally UPDATE / DELETE without a WHERE clause

Runtime of INSERT / UPDATE / DELETE / DDL is bounded by
`YB_MCP_STATEMENT_TIMEOUT_MS` (SET LOCAL statement_timeout applied to every
write) — a runaway `INSERT … SELECT` or wide `INSERT … VALUES` is killed
by the DB, not by a static row cap.

`CREATE TABLE … AS SELECT` and `SELECT … INTO` are structurally similar unbounded row copies but are intentionally **allowed** — they're the common way to materialize a snapshot from a query.

This list is best-effort, not exhaustive. `destructiveHint: true` is the second line of defense.

## Self-hosted remote mode

For multi-user or shared deployments, run the server as Streamable HTTP behind a reverse proxy with TLS, with Cognito OAuth (or generic OIDC) gating access. The full setup — provider config, per-user `SET ROLE` mapping, the identity map file format, the `/auth/login` shortcut, and security guidance — is in [`OIDC.md`](OIDC.md).

**Secure-by-default:** since the fix, HTTP mode binds `127.0.0.1` by default and **refuses to start** when both of these are true:
- The bind host is non-loopback (`MCP_HOST` set to `0.0.0.0` or a specific address)
- No auth provider is configured (`MCP_AUTH_PROVIDER` unset)

For a shared / networked deployment, set both `MCP_HOST=0.0.0.0` and `MCP_AUTH_PROVIDER`:

```bash
export MCP_HOST=0.0.0.0                # expose beyond loopback (default: 127.0.0.1)
export MCP_AUTH_PROVIDER=cognito
export MCP_BASE_URL=https://mcp.example.com
export COGNITO_USER_POOL_ID=us-west-2_XXXXXXXX
export COGNITO_AWS_REGION=us-west-2
export COGNITO_CLIENT_ID=…
export COGNITO_CLIENT_SECRET=…
export YUGABYTEDB_URL=…
export MCP_ALLOWED_ORIGINS=https://mcp.example.com,https://claude.ai

yugabytedb-mcp --transport http --stateless-http
```

For **dev-only unauthenticated** use on `0.0.0.0`, set `MCP_ALLOW_UNAUTHENTICATED=true` — the server starts with a prominent WARNING. Do not use this in production.

Behavior:

- Requests to `/mcp` without a valid Bearer token return 401.
- Requests with a disallowed `Origin` header return 403 (DNS-rebinding defense).
- `/ping` is unauthenticated and is suitable for liveness probes.
- `/auth/login` exposes a Cognito email+password → token shortcut (details in [`OIDC.md`](OIDC.md)).
- `--stateless-http` is required for multi-replica deployments — without it, MCP session state lives in process memory and round-robin load balancing breaks sessions.

## AWS Secrets Manager for TLS certificates

If your database TLS root certificate is stored in AWS Secrets Manager, the server can fetch and use it automatically. Plaintext PEM is supported; JSON-keyed bundles too (set `YB_AWS_SSL_ROOT_CERT_KEY` to pick one).

```bash
yugabytedb-mcp \
  --yugabytedb-url "host=… port=5433 dbname=… user=… password=… sslmode=verify-full" \
  --yb-aws-ssl-root-cert-secret-arn arn:aws:secretsmanager:us-east-1:…:secret:my-cert \
  --yb-aws-ssl-root-cert-secret-region us-east-1
```

## Docker

```bash
docker build -t mcp/yugabytedb .
docker run -p 8000:8000 -e YUGABYTEDB_URL="…" mcp/yugabytedb yugabytedb-mcp --transport http
```

## Security

- All SQL is run through parameterized queries; user input is never interpolated into statement strings.
- The write tool is **disabled by default** — must be explicitly enabled with `--enable-write-query`.
- The write tool's guardrail list (above) blocks the highest-risk statement classes.
- `destructiveHint: true` ensures Claude Desktop surfaces a per-call confirmation for write operations.
- When OIDC auth is active, per-user `SET ROLE` enforces database-level privilege boundaries per caller. Role names are safely quoted with `psycopg.sql.Identifier`.
- HTTP transport requires a valid Bearer token when `MCP_AUTH_PROVIDER` is configured.
- HTTP transport validates the `Origin` header against `MCP_ALLOWED_ORIGINS` (defaults to `MCP_BASE_URL`).
- HTTPS is the operator's responsibility — terminate TLS at a reverse proxy (nginx, ALB, etc.) in front of the server.
- Run with a least-privilege database role (read-only role for `run_read_only_query`-only deployments; otherwise a role scoped to the target schemas, no superuser).

Report security issues privately to support@yugabyte.com — please do not open public GitHub issues for vulnerabilities.

## Privacy Policy

Yugabyte's privacy policy applies: https://www.yugabyte.com/privacy-policy/

This MCP server does not transmit telemetry. All database access stays between Claude (your MCP client) and your YugabyteDB instance via the connection string you provide. The server logs locally to stderr (controlled by `YB_LOG_LEVEL`) — no remote log aggregation is built in.

## Development

```bash
git clone git@github.com:yugabyte/yugabytedb-mcp-server.git
cd yugabytedb-mcp-server
uv sync
uv run yugabytedb-mcp --help
```

Note: there is **no longer a `src/server.py` you can run directly**. The package layout was reorganized for PyPI distribution (entry point + namespace) so the modules now live under `src/yugabytedb_mcp_server/`. Always invoke via the `yugabytedb-mcp` console script (registered by `uv sync` / `pip install`) — running the module file with `python` would skip the package import machinery and break the relative imports.

Equivalent commands:

```bash
uv run yugabytedb-mcp                 # uses the console script
uv run python -m yugabytedb_mcp_server # uses the __main__.py shim
```

### Testing the connector locally in Claude Desktop

Two paths, depending on how close to the production install experience you want to get:

**Fastest — no MCPB build, just point Claude Desktop at the local entry point**. After `uv sync`, the `yugabytedb-mcp` script is on your `$PATH` (via the active venv). Add this to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "yugabytedb-dev": {
      "command": "/absolute/path/to/repo/.venv/bin/yugabytedb-mcp",
      "env": {
        "YUGABYTEDB_URL": "host=localhost port=5433 dbname=yugabyte user=yugabyte password=yugabyte",
        "YB_LOG_LEVEL": "DEBUG"
      }
    }
  }
}
```

Restart Claude Desktop. Use `~/Library/Logs/Claude/mcp-server-yugabytedb-dev.log` (macOS) to inspect debug output. This skips the MCPB bundling entirely and is the right loop for iterating on tool code.

**Closer to production — build a `.mcpb` and drag it into Claude Desktop**. Requires the [MCPB CLI](https://github.com/modelcontextprotocol/mcpb):

```bash
npm install -g @modelcontextprotocol/mcpb-cli   # one-time
mcpb validate manifest.json                      # static check
mcpb pack .                                      # produces yugabytedb-mcp-server-<version>.mcpb
```

Drag the resulting `.mcpb` into Claude Desktop — the connector installer UI takes it from there, prompting for the `user_config` values defined in `manifest.json`. The `.mcpb` route is closest to what reviewers will exercise. **Note**: the manifest's `mcp_config` runs `uvx yugabytedb-mcp-server`, which fetches the package from PyPI on first launch. Make sure the version referenced by your `.mcpb` is published before sharing the bundle.

## Testing

```bash
# unit tests (no DB, no network)
uv run pytest tests/test_guardrails.py tests/test_auth.py tests/test_identity_mapping.py

# integration tests (require a reachable Postgres-compatible DB)
YUGABYTEDB_URL="host=… port=… …" uv run pytest tests/
```

See [`tests/README.md`](tests/README.md) for the coverage table and the manual Cognito smoke recipe.

## Troubleshooting

- `spawn yugabytedb-mcp ENOENT` from Claude Desktop → ensure the install directory is on the PATH Claude Desktop sees; `pipx ensurepath` or symlink the entry point into `/usr/local/bin`.
- Tools list is empty in the MCP client → restart the client; check `YB_LOG_LEVEL=DEBUG` output for connection errors during lifespan.
- "Invalid or expired transaction" / "Client Not Registered" in HTTP+OAuth mode with multiple replicas → see the self-hosted remote section; `--stateless-http` is mandatory for multi-replica.

## License

[Apache License 2.0](LICENSE).
