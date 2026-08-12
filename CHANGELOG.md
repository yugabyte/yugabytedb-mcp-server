# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

_(No unreleased changes.)_

## [2.0.0] - Unreleased

### Added

- **`MCP_PORT` / `--port` config.** HTTP transport now honors a
  configurable port instead of the hardcoded `8000`. Env:
  `MCP_PORT`, default `8000`. (DB-22139 round 2.)
- **`YB_MCP_MAX_RESULT_BYTES` cap.** Cumulative byte budget applied
  while streaming rows from `run_read_only_query`. A wide-row query
  (e.g. `SELECT repeat('x', 1_000_000) FROM generate_series(1, 100)`)
  now truncates on bytes rather than materializing ~100 MiB before
  the row cap is checked. Default: 50 MiB. (DB-22159 round 2.)
- **`connect_timeout` on the pool conninfo.** Defaults to `10` when
  the operator's `YUGABYTEDB_URL` doesn't set one, so a network
  partition to the DB doesn't hang startup or pool checkouts
  indefinitely. (DB-22159 round 2.)
- **`YB_MCP_LEGACY_ACCEPT_ID_TOKENS` compat flag.** Restores
  pre-DB-22136 auth defaults (`YB_MCP_REQUIRE_ACCESS_TOKEN=false`,
  `YB_MCP_IDENTITY_CLAIM=email`) in one env var. (DB-22136 round 2.)

### Changed

- **BREAKING: default `YB_MCP_IDENTITY_CLAIM` is now `sub`.**
  Previously `email`, which is absent from Cognito access tokens. Set
  `YB_MCP_LEGACY_ACCEPT_ID_TOKENS=true` to keep the old default.
  (DB-22136 round 2.)
- **BREAKING: `YB_MCP_REQUIRE_ACCESS_TOKEN` now defaults to `true`.**
  ID tokens are rejected on `/mcp` by default; the compat flag above
  reverts. (DB-22136 round 2.)
- **All INSERT shapes now bounded by `SET LOCAL statement_timeout`**
  (`YB_MCP_STATEMENT_TIMEOUT_MS`) instead of a static row cap. Every
  write — INSERT VALUES, INSERT SELECT, INSERT DEFAULT VALUES, UPDATE,
  DELETE, DDL — runs under the timeout unconditionally, so a runaway
  statement is killed by the DB. (DB-22131 round 2.)
- **Pool sizing is validated at startup** — `pool_min_size >
  pool_max_size` raises a clean error before `pool.open`. (DB-22159
  round 2.)
- **`summarize_database` now runs under `SET LOCAL statement_timeout`**,
  matching the read / write tools. A slow `COUNT(*)` no longer
  holds a pool connection indefinitely. (DB-22159 round 2.)
- **HTTP transport fails closed when auth is off and no Origin
  allowlist is configured** — previously a warning; now a startup
  error, matching the auth-off + non-loopback guard. Same escape
  hatch: `MCP_ALLOW_UNAUTHENTICATED=true`. (DB-22139 round 2.)

### Removed

- **BREAKING: `YB_MCP_IDENTITY_TRANSFORM` removed.** Its only value
  (`strip_domain`) silently collapsed users across email domains
  (`alice@a.com` and `alice@b.com` both → role `alice`). Startup
  now fails if the env var is set, with a message pointing at
  `YB_MCP_IDENTITY_MAP`. The `strip_domain`-based tutorial
  (`examples/oidc-auth/`) is removed; use `examples/oidc-auth-mapping/`.
  (DB-22174 round 2.)
- **`YB_MCP_MAX_INSERT_ROWS` removed.** The static row cap was
  redundant now that every write goes through `SET LOCAL
  statement_timeout`. Setting the env var is a non-fatal warning at
  startup. (DB-22131 round 2.)

### Security

- **DB-22131 round 2: block `CREATE OR REPLACE FUNCTION` and
  `CREATE OR REPLACE PROCEDURE` on the write tool.** The
  keyword-pair matcher on `('CREATE', 'FUNCTION')` couldn't see the
  `OR REPLACE` form because sqlparse tokenizes it as one keyword;
  a dedicated scanner now catches the shape. `SECURITY DEFINER`
  variants are covered.
- **DB-22135 round 2: fail-closed on list-shaped claims at request
  time.** The startup guard only caught known list-claim names
  (`cognito:groups`, `realm_access.roles`, `groups`, dotted paths).
  A request-time check now fires whenever the claim actually
  resolves to a list AND no `YB_MCP_IDENTITY_MAP` is configured —
  independent of the claim's name.

### Added

- **OIDC v2 identity mapping + JWT audience validation (#10).** Maps
  OIDC access-token claims to PostgreSQL roles via a `pg_ident.conf`-style
  identity map. New env vars: `YB_MCP_IDENTITY_CLAIM`,
  `YB_MCP_IDENTITY_MAP`, `YB_MCP_IDENTITY_MAP_NAME`. `SET ROLE` is
  issued on each connection checkout using the mapped role, and
  connections are returned to the pool via `RESET ROLE` + `DISCARD ALL`.
  JWT audience is validated against the configured resource server.
- **Keycloak OIDC → Postgres role mapping tutorial (#8).** New example
  under `examples/oidc-auth-mapping/keycloak/` — realm export, docker
  compose, and step-by-step README covering end-to-end auth to YugabyteDB.

### Changed

- **`run_write_query` is opt-in (#7).** The write tool is no longer
  registered by default. Set `YB_MCP_ENABLE_WRITE_QUERY=true` (or
  `--enable-write-query`) to expose it. The read tools remain on by
  default.
- **`run_read_only_query` response shape (#9).** Now returns
  `{"columns": [...], "rows": [[...], ...]}` instead of a list of dicts.
  Duplicate output column names (e.g. `SELECT * FROM a, b` where both
  have `id`) previously collapsed silently; the new shape preserves
  every column.

### Security

- **Unified SQL guardrail across read and write tools (#9).** A single
  parsed-statement validator now runs on both `run_read_only_query` and
  `run_write_query`. Highlights:
  - Read tool blocks dangerous functions and catalog surfaces that can
    read files, execute shell, or reach across sessions
    (`pg_read_file`, `pg_read_binary_file`, `pg_write_file`, `pg_ls_dir`,
    `dblink*`, `lo_import`, `set_config`, `pg_sleep`, etc.) plus
    `SET search_path`.
  - Multi-statement input is rejected on both tools.
  - Optional `WHERE`-clause requirement on `UPDATE`/`DELETE` via
    `YB_MCP_REQUIRE_WHERE_ON_UPDATE` / `..._ON_DELETE`.
  - Statement timeout applied to reads to bound expensive queries.
- **Write-tool guardrail: block `CREATE FUNCTION` / `CREATE PROCEDURE`
  (and `ALTER FUNCTION` / `ALTER PROCEDURE`).** These can execute
  arbitrary PL code and, with `SECURITY DEFINER`, run as the
  function owner. Both the plain and `CREATE OR REPLACE` forms are
  caught (the latter is tokenized as one keyword by sqlparse and
  needs a dedicated scanner). `INSERT ... SELECT`, `CREATE TABLE ...
  AS SELECT`, and `SELECT ... INTO` are allowed — their runtime is
  bounded by `SET LOCAL statement_timeout`.

## [2.0.0rc2] - 2026-05-25

### Fixed

- Entry point for `uvx yugabytedb-mcp-server` invocation.

## [2.0.0rc1] - 2026-05-25

### Added

- **Claude Connector Directory submission readiness (#6).** Package renamed
  to `yugabytedb-mcp-server` on PyPI. Reworked packaging, metadata, and
  entry points so the server can be installed and launched by Claude
  Connector Directory clients.

## [1.0.2] - 2025-08-05

### Added

- **AWS Secrets Manager–based SSL root certificate support (#5).** New env
  vars `YB_AWS_SSL_ROOT_CERT_SECRET_ARN`,
  `YB_AWS_SSL_ROOT_CERT_SECRET_REGION`, and
  `YB_AWS_SSL_ROOT_CERT_SECRET_KEY` let the server fetch a YugabyteDB CA
  bundle from Secrets Manager at startup and materialize it for the
  psycopg driver.
- Groundwork for Claude Connector Directory submission.

### Fixed

- **JSON-RPC handling (#4).** Fixes a JSON-RPC framing bug that surfaced
  in some MCP client interactions.

---

Older releases (v1.0.0, v1.0.1) predate the public PyPI listing and are
not enumerated here — see git tags for the source snapshots.

[Unreleased]: https://github.com/yugabyte/yugabytedb-mcp-server/compare/v2.0.0...HEAD
[2.0.0]: https://github.com/yugabyte/yugabytedb-mcp-server/compare/v2.0.0rc2...HEAD
[2.0.0rc2]: https://github.com/yugabyte/yugabytedb-mcp-server/compare/v2.0.0rc1...v2.0.0rc2
[2.0.0rc1]: https://github.com/yugabyte/yugabytedb-mcp-server/compare/v1.0.2...v2.0.0rc1
[1.0.2]: https://github.com/yugabyte/yugabytedb-mcp-server/releases/tag/v1.0.2
