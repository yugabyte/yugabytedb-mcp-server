# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

_(No unreleased changes.)_

## [2.0.0] - Unreleased

### Added

- **OIDC v2 identity mapping + JWT audience validation (#10).** Maps
  OIDC access-token claims to PostgreSQL roles via a `pg_ident.conf`-style
  identity map. New env vars: `YB_MCP_IDENTITY_CLAIM`,
  `YB_MCP_IDENTITY_TRANSFORM`, `YB_MCP_IDENTITY_MAP`,
  `YB_MCP_IDENTITY_MAP_NAME`. `SET ROLE` is issued on each connection
  checkout using the mapped role, and connections are returned to the
  pool via `RESET ROLE` + `DISCARD ALL`. JWT audience is validated
  against the configured resource server.
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
  - `YB_MCP_MAX_INSERT_ROWS` caps `INSERT ... VALUES` row counts on the
    write tool.
  - Statement timeout applied to reads to bound expensive queries.
- **Write-tool guardrail: reject additional unbounded write shapes.**
  `YB_MCP_MAX_INSERT_ROWS` only caps `INSERT ... VALUES`. The following
  shapes bypassed the cap and are now rejected outright on the write
  path:
  - `INSERT ... SELECT ...` (unbounded row copy).
  - `CREATE FUNCTION`, `CREATE PROCEDURE`, `ALTER FUNCTION`, `ALTER PROCEDURE`
    (can execute arbitrary PL code and, with `SECURITY DEFINER`, run as the
    function owner).

  Safe shapes are unaffected: `INSERT ... VALUES (...)` (subject to
  `YB_MCP_MAX_INSERT_ROWS`) and `INSERT ... DEFAULT VALUES` continue to
  work, as do `CREATE TABLE ... AS SELECT` and `SELECT ... INTO` (also
  unbounded copies, but left allowed for their common
  materialize-a-snapshot use case).

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
