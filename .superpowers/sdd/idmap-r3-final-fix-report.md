# Identifier-map R3 Final Fix Report

## Status

`DONE_WITH_CONCERNS`

All brief requirements are implemented and verified. Concerns are limited to
environment/tooling coverage: the current Windows account could not create a
real symlink, the POSIX runtime branch was type-checked but not executed on this
Windows host, no independent reviewer/subagent tool was available, and Git
reported an unrelated bad `refs/codex/turn-diffs/...` reference during its
post-commit geometric repack. The implementation commit itself is readable and
verified.

## Commits and changed files

Implementation commit:
`c9cefde9f9f3de020d2e13f5762c1299ee89b971`

Base commit:
`c0a314b10435eb0321024ad316903f4eb59f247a`

Changed files:

- `src/paper_search/evaluation/metrics.py`
- `src/paper_search/evaluation/runner.py`
- `tests/evaluation/test_cli.py`
- `tests/evaluation/test_runner.py`
- `docs/superpowers/specs/2026-07-29-identifier-map-r3-design.md`
- `docs/superpowers/plans/2026-07-29-identifier-map-r3.md`
- `.superpowers/sdd/idmap-r3-final-fix-report.md`

## TDD RED evidence

### Standalone metrics CLI

Command:

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/evaluation/test_cli.py::test_cli_uses_one_identifier_map_byte_snapshot_for_metrics_and_hash tests/evaluation/test_cli.py::test_cli_redacts_invalid_or_missing_identifier_map -q
```

Observed result before production edits: `3 failed`.

Expected failures observed:

- The snapshot regression expected the original-byte digest
  `sha256:c8274163c8bff0a2f250248be5ef49a48d9a6b65e315812708edceebadc0e7cd`
  but received the replacement-byte digest
  `sha256:bd5d1d38878bc25339d984fa54b1ca67fc8cb036d1c0aa12b888003459a7c22a`.
  Metrics still reflected the original map, proving the parse/hash split.
- The invalid-map case expected exactly
  `evaluation failed: identifier map is invalid`; the existing CLI instead
  emitted the map path and conflict identifier.
- The missing-map case expected exactly
  `evaluation failed: identifier map is unavailable`; the existing CLI instead
  emitted the operating-system error and full map path.

### Runner open-handle confinement

Command:

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/evaluation/test_runner.py::test_cli_rejects_opened_identifier_map_target_outside_data_before_provider tests/evaluation/test_runner.py::test_cli_rejects_identifier_map_when_final_handle_target_is_unavailable tests/evaluation/test_runner.py::test_normalize_windows_final_path_prefixes -q
```

Observed result before production edits: `4 failed`.

Expected failures observed:

- A real open handle redirected to an outside file passed preliminary path
  validation and reached provider construction:
  `Failed: provider constructed: ['api_key', 'cache', 'client']`.
- Simulated failure to determine the final handle target also reached provider
  construction with the same failure.
- Both Windows prefix cases failed with
  `AttributeError: module 'paper_search.evaluation.runner' has no attribute
  '_normalize_windows_final_path'`.

### Cross-platform type-check follow-up

Command:

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file mypy --platform win32 src/paper_search/evaluation/runner.py
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file mypy --platform linux src/paper_search/evaluation/runner.py
```

Observed result before the platform-guard production edit:

- win32: `Success: no issues found in 1 source file`
- linux: `3 errors` for unavailable POSIX type declarations of
  `ctypes.WinDLL`, `msvcrt.get_osfhandle`, and `ctypes.get_last_error`

The minimal fix conditionally defines the Windows implementation under
`sys.platform == "win32"` and leaves a fixed failure implementation on other
platforms.

## GREEN and full verification

Focused metrics:

```powershell
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/evaluation/test_cli.py tests/evaluation/test_metrics.py -q
```

Result: `21 passed in 1.72s`.

Focused runner after all edits:

```powershell
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/evaluation/test_runner.py -k 'id_map or identifier_map or artifact or cli_parser' -q
```

Result: `23 passed, 1 skipped, 62 deselected in 4.02s`. The skip is the
privilege-gated real symlink test; WinError 1314 denied symlink creation.

Combined focused metrics/runner verification:

```powershell
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/evaluation/test_cli.py tests/evaluation/test_metrics.py tests/evaluation/test_runner.py -k 'test_cli_uses_one_identifier_map_byte_snapshot_for_metrics_and_hash or test_cli_redacts_invalid_or_missing_identifier_map or id_map or identifier_map or artifact or cli_parser' -q
```

Result: `28 passed, 1 skipped, 78 deselected in 4.75s`.

Final full offline suite after the last production edit:

```powershell
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest -q
```

Result: `786 passed, 2 skipped in 64.27s`. Skips were the privilege-gated
Windows symlink test and the credential-gated live OpenAlex test.

Static verification:

```powershell
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file ruff check src/paper_search/evaluation/metrics.py src/paper_search/evaluation/runner.py tests/evaluation/test_cli.py tests/evaluation/test_runner.py
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file mypy src/paper_search/evaluation/metrics.py src/paper_search/evaluation/runner.py
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file mypy --platform win32 src/paper_search/evaluation/runner.py
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file mypy --platform linux src/paper_search/evaluation/runner.py
git diff --check
```

Results:

- Ruff: `All checks passed!`
- Production mypy: `Success: no issues found in 2 source files`
- win32 mypy: `Success: no issues found in 1 source file`
- linux mypy: `Success: no issues found in 1 source file`
- `git diff --check`: exit code 0

## Design decisions

### Standalone metrics

`_load_identifier_map_snapshot` reads the optional map exactly once with
`Path.read_bytes()`. `IdentifierMap.from_bytes()` parses that immutable
snapshot, and `hashlib.sha256()` hashes those exact bytes. A narrow private
exception maps read failures to `identifier map is unavailable` and validation
failures to `identifier map is invalid`; unrelated gold/prediction
`OSError`/`ValueError` diagnostics retain the previous detailed behavior.
No-map payload shape remains unchanged.

### Runner static and final-handle confinement

The existing absolute/traversal/static-link checks remain in
`_resolve_cli_id_map`. `_read_confined_identifier_map` captures the resolved
data root, invokes that static check, opens one binary handle, rejects
non-regular files using `fstat`, validates the final target represented by that
same handle, then reads bytes from it. `IdentifierMap.from_bytes()` and the
identity hash consume only the returned snapshot. The `with` block closes the
handle on success and every exception path.

On Windows, the code obtains the OS handle with `msvcrt.get_osfhandle`, calls
`GetFinalPathNameByHandleW` through `ctypes.WinDLL`, grows the buffer when
required, and normalizes both `\\?\C:\...` and
`\\?\UNC\server\share\...` into paths comparable with the resolved data root.

On POSIX, the code reads `/proc/self/fd/<fd>` or `/dev/fd/<fd>`, requires an
absolute target, normalizes it lexically without reopening it, and compares the
target's `(st_dev, st_ino)` with `os.fstat(fd)`. If neither descriptor path can
be used or identity cannot be verified, the map is rejected.

## Self-review and residual limitations

- Requirement-by-requirement review found no missing production behavior.
- Tests exercise real parsing, hashing, CLI behavior, provider-before-reject
  ordering, and on Windows the real final-handle API. The race test redirects
  the open operation to a real outside file and does not mock final-target
  validation.
- Existing no-map, replacement snapshot, absolute/traversal, missing,
  directory, invalid, partial, and provider-construction regressions pass.
- The privilege-gated real symlink test remains skipped on this Windows account;
  deterministic real-handle redirection covers the same final-target rejection
  boundary without requiring symlink privileges.
- The POSIX branch was not runtime-executed on this Windows host. Both linux and
  win32 mypy modes pass. POSIX systems without a supported descriptor path fail
  closed by design.
- The Codex session exposed no reviewer/subagent tool, so an independent
  reviewer could not be dispatched. Two explicit self-review passes and full
  verification were completed instead.
- Git reported an unrelated malformed Codex turn-diff ref during post-commit
  geometric repack. `git rev-parse HEAD` and `git show` both verified the
  implementation commit. No refs were modified or repaired.
- No `.env`, network service, private map content, or R1/R2/R3 run directory was
  accessed, and R3 was not rerun.
