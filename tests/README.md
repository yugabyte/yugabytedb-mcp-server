# Tests

Three tiers:

| Tier | What it covers | Requires | Where it runs |
|---|---|---|---|
| Unit | guardrails (exhaustive), auth provider construction, JWT signature verification round-trip | nothing (no DB, no network) | CI |
| Integration | read/write round-trips with DB-side read-back verification; guardrail-blocked queries have no side effects | `YUGABYTEDB_URL` to a reachable Postgres/YugabyteDB | CI (when a DB service is available) and local |
| Manual Cognito smoke | full OAuth flow through a real Cognito user pool | AWS Cognito creds, internet | pre-release |

## Running the tests

### Unit tier only (fast, no DB)

```bash
uv run pytest tests/test_guardrails.py tests/test_auth.py
```

About 1 second; safe to run in any CI environment.

### Integration tier (live DB)

```bash
YUGABYTEDB_URL="host=localhost port=5432 dbname=postgres user=postgres password=postgres" \
  uv run pytest tests/test_integration_reads.py \
                tests/test_integration_writes.py \
                tests/test_integration_guardrails.py
```

Integration tests are auto-skipped when `YUGABYTEDB_URL` is unset.

To stand up a quick YugabyteDB for local testing:

```bash
# Download YugabyteDB locally (one-time) — see https://docs.yugabyte.com/preview/quick-start/
# Then start a single-node cluster:
./bin/yugabyted start --base_dir /tmp/ybd-test

# Default port for the YSQL API is 5433
YUGABYTEDB_URL="host=localhost port=5433 dbname=yugabyte user=yugabyte password=yugabyte" \
  uv run pytest tests/

# Cleanup
./bin/yugabyted stop --base_dir /tmp/ybd-test
rm -rf /tmp/ybd-test
```

CI uses a containerized Postgres (Postgres is wire-compatible enough for the
tools we exercise). For local development, prefer `yugabyted` so you're
running against the same database family the server targets.

### Full suite

```bash
YUGABYTEDB_URL="..." uv run pytest tests/
```

## Coverage table

| Tool / area | Unit | Integration | Notes |
|---|---|---|---|
| `validate_write_query` — allowed shapes (INSERT/UPDATE/DELETE/MERGE/TRUNCATE/CREATE/ALTER/DROP-table) | ✓ | indirect via writes | |
| `validate_write_query` — DROP DATABASE/SCHEMA, ALTER DATABASE, CREATE DATABASE | ✓ | ✓ (no side effect) | |
| `validate_write_query` — role/privilege ops (GRANT/REVOKE/CREATE ROLE/USER/etc.) | ✓ | ✓ (GRANT no side effect) | |
| `validate_write_query` — filesystem ops (COPY TO/FROM, LOAD, DO $$, CREATE EXTENSION) | ✓ | ✓ (COPY, CREATE EXTENSION no side effect) | |
| `validate_write_query` — server config (ALTER SYSTEM, RESET ALL) | ✓ | ✓ (no side effect) | |
| `validate_write_query` — dangerous funcs (pg_sleep, pg_read_file, lo_import, dblink, etc.) | ✓ | — | |
| `validate_write_query` — `SET search_path`, `CREATE SCHEMA` | ✓ | — | |
| `validate_write_query` — multi-statement | ✓ | ✓ | |
| `validate_write_query` — comment obfuscation | ✓ | ✓ | |
| `validate_write_query` — psql meta-commands | ✓ | ✓ | |
| `validate_write_query` — empty query / comment-only | ✓ | — | |
| `validate_write_query` — bulk INSERT row limit | ✓ | ✓ (no DB side effect when blocked) | |
| `validate_write_query` — require WHERE on UPDATE | ✓ | ✓ (no DB side effect when blocked) | |
| `validate_write_query` — require WHERE on DELETE | ✓ | ✓ (no DB side effect when blocked) | |
| Helper `_count_values_rows` | ✓ | — | |
| Helper `_has_top_level_where` | ✓ | — | |
| Helper `_strip_comments` | ✓ | — | |
| `create_auth_provider` — None disables auth | ✓ | — | |
| `create_auth_provider` — unknown provider raises | ✓ | — | |
| `_create_cognito` — missing env raises | ✓ | — | |
| `_create_cognito` — returns `MultiAuth` wrapping `OIDCProxy` with default scopes | ✓ | — | OIDC discovery mocked |
| JWT verify — valid token accepted | ✓ | — | self-signed; no real Cognito |
| JWT verify — expired token rejected | ✓ | — | |
| JWT verify — wrong issuer rejected | ✓ | — | |
| JWT verify — tampered signature rejected | ✓ | — | |
| `run_read_only_query` — simple SELECT round-trip | — | ✓ | |
| `run_read_only_query` — error path (no such table) | — | ✓ | |
| `run_read_only_query` — UPDATE rejected by BEGIN READ ONLY | — | ✓ (no DB side effect) | |
| `summarize_database` — default `public` schema parses | — | ✓ | |
| `summarize_database` — seeded schema reports correct row counts and columns | — | ✓ | |
| `run_write_query` — INSERT round-trip with read-back | — | ✓ | |
| `run_write_query` — UPDATE round-trip with read-back | — | ✓ | |
| `run_write_query` — DELETE round-trip with read-back | — | ✓ | |
| `run_write_query` — bulk INSERT under limit succeeds | — | ✓ | |
| `run_write_query` — bulk INSERT over limit blocked, no DB side effect | — | ✓ | |
| `run_write_query` — strict UPDATE without WHERE blocked, no DB side effect | — | ✓ | |
| `run_write_query` — strict DELETE without WHERE blocked, no DB side effect | — | ✓ | |
| `run_write_query` — Postgres-side error returns JSON (not crash) | — | ✓ | |

## How to test Cognito auth

Three tiers, same shape as above:

### 1. Unit (no AWS, no network)

Already covered by `tests/test_auth.py`:
- Provider construction with mocked OIDC discovery
- Self-signed JWT round-trip through `JWTVerifier` (valid / expired / wrong-issuer / tampered)

These prove the wiring is correct without ever touching a real user pool.

### 2. Local end-to-end with a mock OIDC issuer (optional)

For a full DCR + OAuth flow with no AWS dependency:

```bash
docker run -d --name mock-oidc -p 8080:8080 \
  ghcr.io/navikt/mock-oauth2-server:2

MCP_AUTH_PROVIDER=oidc \
OIDC_CONFIG_URL=http://localhost:8080/default/.well-known/openid-configuration \
OIDC_CLIENT_ID=test \
OIDC_CLIENT_SECRET=test \
MCP_BASE_URL=http://localhost:8000 \
YUGABYTEDB_URL="..." \
  uv run yugabytedb-mcp --transport http
```

In another shell:

```bash
npx mcp-remote http://localhost:8000/mcp
```

Follow the browser flow. Inspect `~/.mcp-auth/` for cached tokens. Useful for
exercising the path Claude Desktop will take without burning sandbox Cognito
state.

### 3. Manual Cognito smoke (pre-release)

Use a sandbox Cognito user pool with a test user. **Do not run against
production.**

```bash
export MCP_AUTH_PROVIDER=cognito
export MCP_BASE_URL=http://localhost:8000
export COGNITO_USER_POOL_ID=eu-north-1_xxxxxxx
export COGNITO_AWS_REGION=eu-north-1
export COGNITO_CLIENT_ID=...
export COGNITO_CLIENT_SECRET=...
export YUGABYTEDB_URL="..."

uv run yugabytedb-mcp --transport http
```

In another shell:

```bash
# 1. Confirm unauthenticated access is rejected
curl -i http://localhost:8000/mcp                  # expect 401

# 2. Health check is unauthenticated
curl -i http://localhost:8000/ping                 # expect 200 {"status":"ok"}

# 3. End-to-end via mcp-remote (browser opens, log in, tokens cached)
rm -rf ~/.mcp-auth/
npx mcp-remote http://localhost:8000/mcp
# After completing the flow, the same command on subsequent runs should
# skip the browser step because tokens are cached.

# 4. Inspect a cached token
ls ~/.mcp-auth/
```

If you have an app client configured for `USER_PASSWORD_AUTH`, you can also
fetch tokens directly with the AWS CLI for scripted smoke tests:

```bash
aws cognito-idp initiate-auth \
  --client-id "$COGNITO_CLIENT_ID" \
  --auth-flow USER_PASSWORD_AUTH \
  --auth-parameters "USERNAME=test@example.com,PASSWORD=..."

# Take the AccessToken from the response and:
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/mcp
```

## Test isolation

Integration tests use a per-test schema (`test_schema` fixture) to avoid
collisions. Each test creates a unique schema, runs tool calls scoped to
that schema, and drops it in teardown. Teardown is best-effort — if a test
crashes mid-run, you may want to manually drop leftover `mcp_test_*` schemas.
