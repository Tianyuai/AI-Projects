# Phase 4 Task 3 report

## RED baseline

Command run before production edits:

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/cli/test_serve.py tests/integration/test_serve_process.py tests/test_packaging.py -q
```

Result: **5 failed, 13 passed**.

- `test_serve_parser_is_replay_only_and_loopback_by_default` failed because
  `serve` was not a recognized CLI command.
- The two composition/cleanup tests failed because `CompositionRoot` did not
  provide `compose_server`.
- The canonical child-process test failed because the missing `serve` command
  terminated before HTTP readiness.
- The occupied-port test failed because stderr reported the parser error rather
  than the required sanitized serve startup error.

These failures are expected and are attributable to the absent server command
and composition lifecycle, not test setup errors.

## Expanded RED baseline

After adding tests for sanitized startup errors, interrupt handling, live bundle
isolation, and capture publication ordering, the same focused command produced
**9 failed, 13 passed**.

The additional expected failures were the absent `_run_serve` command runtime
helper and absent `_RequestLiveCaptureService`, alongside the original absent
`serve` parser/composition behavior. The process failures remain caused by the
missing command rather than an external dependency or network access.

## Corrected lifecycle RED baseline

The corrected lifecycle suite (typed replay lock and real success/failure result
fixtures, plus cancellation coverage) was run before completing command/runtime
production work. It produced **10 failed, 15 passed**. Nine failures remained
the expected absent command/composition/runtime capabilities; the cancellation
assertion initially demonstrated the intended distinction that cancellation has
no execution record, but must still fail the staging session and close the
request bundle. The assertion was corrected to encode that policy.

## Final verification

- Focused suite (configured offline environment): **25 passed**.
- `paper-search serve --help`: exit 0 and includes the required server options.
- Ruff: all checks passed.
- Mypy: success, no issues in the CLI or composition module.
- `git diff --check`: clean.

The prescribed focused command without `UV_PROJECT_ENVIRONMENT` could not spawn
`pytest` in this workspace (`program not found`); the same test targets passed
with the repository's configured offline virtual environment. The only adjacent
API change adds FastAPI lifespan injection so the server bundle owns shutdown
cleanup.

## Final lifecycle hardening

### RED

After the prior snapshot-source change, a tests-only regression was added that
atomically replaces `config.lock.yaml` between its pre-open metadata check and
the open. The targeted server suite produced **1 failed, 20 passed, 2 skipped**:
the replacement was accepted by the old resolve/check/`read_bytes()` sequence.
The two skips are Windows hosts without symlink privileges; they are the
pre-existing symlink-escape parametrizations.

### GREEN and verification

- The source reader now snapshots from one descriptor and validates the regular
  file identity `(st_dev, st_ino, st_size, st_mtime_ns)` before opening, on the
  opened descriptor, after the read, and after closing. POSIX additionally uses
  `O_NOFOLLOW`; Windows retains the same descriptor and post-close checks.
- The tests exercise real `CompositionRoot.compose_server().service_router`
  live requests using deterministic `httpx.MockTransport`: an LLM 401 produces
  `SearchFailure` plus `<run_id>.failed`, and cancellation after dispatch raises
  `CancelledError` while producing the same failed-artifact/session/client
  cleanup. Neither case leaves a final success directory or an incomplete
  staging directory.
- A fake-Uvicorn unit test verifies SIGTERM registration and restoration without
  signaling the pytest runner, and confirms explicit host passthrough.
- Serve-only GREEN: **21 passed, 2 skipped**.
- Required focused suite: **31 passed, 2 skipped**.
- Adjacent `tests/integration/test_smoke_cli.py -q`: **34 passed**.
- `paper-search serve --help`: exit 0.
- Ruff on the required serve paths: all checks passed.
- Mypy on `src/paper_search/cli.py` and
  `src/paper_search/application/composition.py`: success, no issues.
