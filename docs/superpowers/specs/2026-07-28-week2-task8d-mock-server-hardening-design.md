# Week 2 Task 8D Mock Server Hardening Design

## Scope

Task 8D adds an operationally realistic, offline-only process smoke path for
the existing Week 2 mock API. It exercises the local TCP socket, ASGI routes,
readiness behavior, request timeout handling, process failure reporting, and
cleanup without changing the default API application or enabling real
providers.

The smoke path uses only the fixed synthetic mock service already implemented
by Task 8C. It never reads `.env`, credentials, private annotations, gold data,
dev or frozen splits, manifests, real queries, or raw external responses. It
does not call OpenAlex, Semantic Scholar, an LLM, or any other external
service. It does not compute Recall, Precision, F1, ranking quality, or formal
baseline metrics.

## Requirements Mapping

This design covers the PRD Week 2 Task 8 operational requirement that a local
server can expose health and search endpoints while preserving the Week 2
mock-only safety boundary:

- start a local server process on a loopback address and an ephemeral port;
- wait for a live readiness signal with a bounded timeout;
- call readiness and synthetic search through a real HTTP client;
- report process exit, request timeout, malformed request, and unknown route
  failures deterministically;
- terminate and reap every child process after each test;
- prove that no non-loopback network operation occurs;
- leave the default `paper_search.api.app:app` dependency-injected and
  unconfigured.

Task 8D is an operational smoke test for the mock contract. It is not a claim
that the PRD's formal Week 2 quality gate or real baseline has passed.

## Chosen Architecture

Add a dedicated `paper_search.api.mock_server` module with a small CLI. The
module constructs the existing `create_app` factory with:

1. `build_synthetic_search_service()` as the search service;
2. a fixed readiness probe reporting the two synthetic providers as ready;
3. a loopback host and a caller-provided port.

The module then calls `uvicorn.run` for the configured ASGI app. It owns no
provider clients, credential loading, persistent storage, prediction output,
or evaluation logic. The existing `paper_search.api.app:app` remains the
unconfigured application object used by the general API contract tests.

Before startup, the dedicated entry point installs a process-local
loopback-only socket guard. It permits the server's bind address and local
client connections, and raises a fixed runtime error for any non-loopback
connect attempt. The guard is scoped to the mock-server child process and does
not monkeypatch the parent test process or the default API module.

The process test launches the module with `sys.executable -m
paper_search.api.mock_server`, `--host 127.0.0.1`, and a dynamically selected
free port. The test harness communicates with it using `httpx.AsyncClient`
with explicit connect and read timeouts.

## Public CLI Contract

The CLI accepts only:

```text
python -m paper_search.api.mock_server [--host HOST] [--port PORT]
```

Defaults are `127.0.0.1` and `8000` for manual local use. The process smoke
test always supplies a free port and never relies on the default. Abbreviation
is disabled with `allow_abbrev=False`. Unknown options and attempts to provide
configuration, credentials, endpoints, input files, splits, metrics, or
concurrency options fail during argument parsing.

The CLI has no output artifact. It writes no response payload to stdout and
does not expose secrets or child-process internals in normal startup output.
Uvicorn access logging is disabled for the test process so synthetic request
content cannot be emitted to logs.

## Process Lifecycle

The integration harness follows a bounded state machine:

```text
spawn
  -> poll /health/live until 200 or startup timeout
  -> verify /health/ready is 200 and status=ready
  -> POST synthetic SearchRequest
  -> assert StructuredSearchResponse contract
  -> terminate
  -> wait/reap
```

The harness records only safe diagnostics: return code, timeout category,
HTTP status, and bounded stderr lines that contain no request or response
body. If the child exits before readiness, the failure includes the exit code
and a fixed diagnostic category. If a request exceeds its timeout, the harness
terminates the child and reports a request-timeout category.

Cleanup is idempotent. It checks whether the process is already exited before
terminating, waits for normal reaping, and escalates to a forced kill only if
the bounded cleanup timeout expires. The test never deletes files outside its
temporary directory and leaves no server process running after completion.

## Network Isolation

The child process receives a minimal environment containing only the project
module path and the loopback server settings. It is not started with an
environment-file option. The child-side loopback-only guard rejects any
non-loopback socket connection before it can leave the process. The test also
records the child's fixed guard diagnostic and asserts that the normal run
produces no guard violations. A guard violation is a process failure, not a
degraded API response.

The server's fixed mock service does not construct `httpx.AsyncClient`, provider
transports, LLM transports, or cache paths. This keeps the process smoke test
independent of external availability and credentials.

## HTTP Scenarios

The process integration suite covers:

1. `/health/live` returns `200` and the live contract.
2. `/health/ready` returns `200`, `status=ready`, and the fixed provider map.
3. A fixed synthetic request returns `200` and a valid
   `StructuredSearchResponse` with the expected synthetic stop semantics.
4. Malformed JSON or schema-invalid search input returns a non-success status
   without terminating the process.
5. An unknown route returns `404` without terminating the process.
6. A deliberately occupied port prevents startup and is reported as a
   startup failure.
7. A child that exits before readiness is reported as process failure.
8. A request timeout is bounded and the child is reaped.
9. The network guard records zero non-loopback attempts.

The request body contains only a fixed synthetic query and fixed safe values;
tests assert status and structured-field shape rather than printing the body.

## Files and Responsibilities

Production:

- Create `src/paper_search/api/mock_server.py`: CLI parser, fixed mock app
  factory, and `uvicorn.run` entry point.

Tests:

- Create `tests/api/test_mock_server.py`: parser and fixed-injection contract.
- Create `tests/integration/test_mock_server_process.py`: subprocess
  lifecycle, real loopback HTTP, failure categories, cleanup, and network
  isolation.

Documentation:

- Update `PRD.md` only if the repository's implementation status section is
  explicitly maintained for Task 8D; otherwise keep the source requirements
  unchanged.
- Add an implementation plan after this spec is reviewed.

The protected file
`docs/superpowers/specs/2026-07-15-task2-evaluation-design.md` is not touched.

## Verification

The focused verification is:

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/api/test_mock_server.py tests/integration/test_mock_server_process.py -q
```

The completion verification is:

```powershell
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest -q
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file ruff check .
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file mypy src
git diff --check
```

Before any commit, inspect unstaged and staged file lists, confirm that no
protected-data path is changed, and run a scoped credential-pattern scan. The
final review reports only safe aggregates: test counts, exit codes, statuses,
and branch state.

## Non-Goals

Task 8D does not:

- change the default API app's dependency behavior;
- call external APIs or load `.env` or credentials;
- read or validate labels, gold, dev, frozen split, manifest, or real queries;
- generate predictions, metrics, usage files, or formal baseline artifacts;
- add retries, production deployment files, TLS, authentication, or a public
  network bind;
- claim that the Week 1 Recall gate or Week 2 formal quality gate passed.
