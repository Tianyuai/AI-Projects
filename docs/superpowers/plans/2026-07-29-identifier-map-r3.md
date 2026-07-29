# Identifier-Map-Bound R3 Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bind the complete private Week1 identifier map into the formal evaluation identity and produce a verified nonzero mapped R3 development baseline.

**Architecture:** Extend `IdentifierMap` with a non-disclosing coverage query, then add an optional confined CLI map input that is validated before provider construction and passed to existing deduplication and metric code. Static confinement is followed by validation of the final target represented by the one open map handle; parsing and hashing consume the immutable bytes read from that same handle. Map-enabled artifacts bind the exact map hash, while no-map artifacts retain their existing shape. R3 runs from a new Git-external root with a fresh cache and an atomic execution receipt.

**Tech Stack:** Python 3.11+, Pydantic, pytest, httpx, SQLite snapshot cache, uv, PowerShell

## Global Constraints

- The accepted CLI form is `--id-map data/annotation_work/dev_identifier_map.v1.json`.
- The statically resolved map path and the final target represented by its open handle must remain beneath the resolved process `data/` root.
- Map parsing and hashing must use one immutable byte snapshot read from that validated handle.
- The private map body, entries, path, gold identifiers, query text, labels, raw provider responses, and credentials must not be printed or copied into formal metadata.
- All 141 unique development-gold identifiers must be explicitly covered before provider construction.
- The exact map hash is `sha256:` plus 64 lowercase hexadecimal characters and is recorded only when a map is supplied.
- No-map formal artifact shapes and behavior remain compatible.
- Offline tests and static checks use `--no-env-file`; only R3 uses relative `--env-file .env` from `D:\AI Projects\Projects`.
- R1 and R2 are immutable; R3 uses a new output directory and new SQLite cache.
- R3 is formal only if direct exit code is zero, mapped true positives are positive, and macro and micro Recall are both greater than zero.

---

### Task 1: Add Non-Disclosing Identifier-Map Coverage

**Files:**
- Modify: `src/paper_search/evaluation/dataset.py:253-324`
- Test: `tests/evaluation/test_dataset.py:256-340`

**Interfaces:**
- Consumes: normalized aliases already stored by `IdentifierMap.from_path`
- Produces: `IdentifierMap.covers(value: str) -> bool`

- [ ] **Step 1: Write the failing coverage test**

Add beside the existing identifier-map tests:

```python
def test_identifier_map_reports_explicit_alias_coverage(tmp_path: Path) -> None:
    path = tmp_path / "map.json"
    path.write_text(
        '{"arxiv:2501.10120":"openalex:W1"}',
        encoding="utf-8",
    )

    identifier_map = dataset_module.IdentifierMap.from_path(path)

    assert identifier_map.covers("https://arxiv.org/abs/2501.10120v2") is True
    assert identifier_map.covers("openalex:W1") is False
    assert identifier_map.covers("doi:10.2000/unmapped") is False
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/evaluation/test_dataset.py::test_identifier_map_reports_explicit_alias_coverage -q
```

Expected: FAIL with `AttributeError` because `covers` does not exist.

- [ ] **Step 3: Implement the minimal coverage method**

Add after `IdentifierMap.from_path` and before `resolve`:

```python
    def covers(self, value: str) -> bool:
        """Return whether value is an explicit normalized map alias."""
        return normalize_paper_id(value) in self._resolved
```

- [ ] **Step 4: Run focused dataset tests and verify GREEN**

Run:

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/evaluation/test_dataset.py -k identifier_map -q
```

Expected: all identifier-map tests PASS.

- [ ] **Step 5: Run static checks**

Run:

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file ruff check src/paper_search/evaluation/dataset.py tests/evaluation/test_dataset.py
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file mypy src/paper_search/evaluation/dataset.py
git diff --check
```

Expected: every command exits 0.

- [ ] **Step 6: Commit Task 1**

```powershell
git add -- src/paper_search/evaluation/dataset.py tests/evaluation/test_dataset.py
git commit -m "feat: expose identifier-map coverage"
```

### Task 2: Bind the Identifier Map into CLI, Identity, and Artifacts

**Files:**
- Modify: `src/paper_search/evaluation/runner.py:90-253`
- Modify: `src/paper_search/evaluation/runner.py:420-490`
- Modify: `src/paper_search/evaluation/runner.py:695-743`
- Test: `tests/evaluation/test_runner.py`

**Interfaces:**
- Consumes: `IdentifierMap.covers(value: str) -> bool` from Task 1
- Produces: optional CLI `--id-map PATH`
- Produces: `RunIdentity.id_map_sha256: str | None = None`
- Produces: `_resolve_cli_id_map(data_root: Path, raw_path: Path) -> Path`
- Produces: `_read_confined_identifier_map(data_root: Path, raw_path: Path) -> bytes`
- Produces: `_require_id_map_coverage(gold: Sequence[EvaluationQuery], id_map: IdentifierMap) -> None`
- Changes: `_run_cli_evaluation(..., id_map: IdentifierMap | None) -> None`

- [ ] **Step 1: Extend the test CLI helper**

Replace `_run_cli_from` with:

```python
def _run_cli_from(
    root: Path,
    *,
    split: str = "dev",
    id_map: str | None = None,
) -> int:
    argv = [
        "--config",
        str(CONFIG),
        "--split",
        split,
        "--output",
        "out",
    ]
    if id_map is not None:
        argv.extend(["--id-map", id_map])
    return runner_module.main(argv)
```

Update the parser-contract test to require three required options and one
optional option:

```python
def test_cli_parser_exposes_required_options_and_optional_id_map() -> None:
    parser = runner_module._build_parser()
    actions = {
        option: action
        for action in parser._actions
        for option in action.option_strings
        if option not in {"--help", "-h"}
    }

    assert set(actions) == {"--config", "--split", "--output", "--id-map"}
    assert all(actions[option].required for option in {"--config", "--split", "--output"})
    assert actions["--id-map"].required is False
```

- [ ] **Step 2: Write the failing successful-map CLI test**

Add a test that writes an arXiv gold record and a confined map, injects the
existing fake provider, and verifies mapped metrics and identity:

```python
def test_cli_binds_confined_identifier_map_into_metrics_and_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_cli_manifest(
        tmp_path,
        gold_content=(
            '{"query_id":"query-1","query":"graph retrieval",'
            '"relevant_paper_ids":["arxiv:2501.10120"]}\n'
        ),
    )
    map_path = tmp_path / "data" / "annotation_work" / "dev-map.json"
    map_path.parent.mkdir(parents=True)
    map_path.write_text(
        '{"arxiv:2501.10120":"openalex:W1"}',
        encoding="utf-8",
    )
    map_hash = _sha256(map_path.read_bytes())
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENALEX_API_KEY", SENTINEL_API_KEY)

    def fake_provider_factory(**kwargs: object) -> FakeProvider:
        del kwargs
        return FakeProvider(
            {
                "graph retrieval": _provider_result(
                    [_paper("openalex:W1", title="graph retrieval")],
                    calls=1,
                    latency_ms=1,
                )
            }
        )

    monkeypatch.setattr(
        runner_module,
        "OpenAlexProvider",
        fake_provider_factory,
        raising=False,
    )

    assert _run_cli_from(
        tmp_path,
        id_map="data/annotation_work/dev-map.json",
    ) == 0

    run = json.loads((tmp_path / "out" / "run.json").read_text(encoding="utf-8"))
    metrics = json.loads(
        (tmp_path / "out" / "metrics.json").read_text(encoding="utf-8")
    )
    assert run["identity"]["id_map_sha256"] == map_hash
    assert run["input_hashes"]["id_map"] == map_hash
    assert metrics["input_hashes"]["id_map"] == map_hash
    assert metrics["summary"]["macro_recall"] == 1.0
    assert metrics["summary"]["micro_recall"] == 1.0
```

- [ ] **Step 3: Run the successful-map test and verify RED**

Run:

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/evaluation/test_runner.py::test_cli_binds_confined_identifier_map_into_metrics_and_identity -q
```

Expected: FAIL because `--id-map` is not accepted.

- [ ] **Step 4: Write failing confinement and coverage tests**

Add:

```python
@pytest.mark.parametrize(
    "raw_path",
    ["../outside-map.json", "C:/outside-map.json"],
)
def test_cli_rejects_identifier_map_outside_data_before_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    raw_path: str,
) -> None:
    _write_cli_manifest(tmp_path)
    (tmp_path / "outside-map.json").write_text("{}", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENALEX_API_KEY", SENTINEL_API_KEY)
    monkeypatch.setattr(
        runner_module,
        "OpenAlexProvider",
        lambda **kwargs: pytest.fail(f"provider constructed: {sorted(kwargs)}"),
        raising=False,
    )

    assert _run_cli_from(tmp_path, id_map=raw_path) == 2
    assert "identifier map path must stay under data" in capsys.readouterr().err
```

```python
def test_cli_rejects_partial_identifier_map_before_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_cli_manifest(
        tmp_path,
        gold_content=(
            '{"query_id":"query-1","query":"graph retrieval",'
            '"relevant_paper_ids":["arxiv:2501.10120"]}\n'
        ),
    )
    map_path = tmp_path / "data" / "annotation_work" / "partial.json"
    map_path.parent.mkdir(parents=True)
    map_path.write_text(
        '{"arxiv:2501.99999":"openalex:W1"}',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENALEX_API_KEY", SENTINEL_API_KEY)
    monkeypatch.setattr(
        runner_module,
        "OpenAlexProvider",
        lambda **kwargs: pytest.fail(f"provider constructed: {sorted(kwargs)}"),
        raising=False,
    )

    assert _run_cli_from(
        tmp_path,
        id_map="data/annotation_work/partial.json",
    ) == 2
    captured = capsys.readouterr()
    assert "identifier map does not cover frozen gold identifiers" in captured.err
    assert "2501.10120" not in captured.out + captured.err
```

Add the missing-file case:

```python
def test_cli_rejects_missing_identifier_map_before_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_cli_manifest(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENALEX_API_KEY", SENTINEL_API_KEY)
    monkeypatch.setattr(
        runner_module,
        "OpenAlexProvider",
        lambda **kwargs: pytest.fail(f"provider constructed: {sorted(kwargs)}"),
        raising=False,
    )

    assert _run_cli_from(
        tmp_path,
        id_map="data/annotation_work/missing.json",
    ) == 2
    assert "identifier map file does not exist" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()
```

Add invalid/conflicting/cyclic cases:

```python
@pytest.mark.parametrize(
    "payload",
    [
        "not-json",
        (
            '{"arxiv:2501.10120":"openalex:W1",'
            '"https://arxiv.org/abs/2501.10120":"openalex:W2"}'
        ),
        (
            '{"arxiv:2501.10120":"openalex:W1",'
            '"openalex:W1":"arxiv:2501.10120"}'
        ),
    ],
)
def test_cli_redacts_invalid_identifier_map_before_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    payload: str,
) -> None:
    _write_cli_manifest(tmp_path)
    map_path = tmp_path / "data" / "annotation_work" / "invalid.json"
    map_path.parent.mkdir(parents=True)
    map_path.write_text(payload, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENALEX_API_KEY", SENTINEL_API_KEY)
    monkeypatch.setattr(
        runner_module,
        "OpenAlexProvider",
        lambda **kwargs: pytest.fail(f"provider constructed: {sorted(kwargs)}"),
        raising=False,
    )

    assert _run_cli_from(
        tmp_path,
        id_map="data/annotation_work/invalid.json",
    ) == 2
    captured = capsys.readouterr()
    assert captured.err.strip() == "evaluation failed"
    assert payload not in captured.out + captured.err
    assert not (tmp_path / "out").exists()
```

- [ ] **Step 5: Run the confinement and coverage tests and verify RED**

Run:

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/evaluation/test_runner.py -k "identifier_map_outside or partial_identifier_map" -q
```

Expected: FAIL because the option and preflight checks do not exist.

- [ ] **Step 6: Implement identity and path helpers**

Add the optional field:

```python
class RunIdentity(DomainModel):
    split: NonEmptyStr
    git_sha: NonEmptyStr
    gold_sha256: NonEmptyStr
    manifest_sha256: NonEmptyStr
    dataset_revision: NonEmptyStr
    zero_answer_policy: Literal["reject", "allow"]
    id_map_sha256: NonEmptyStr | None = None
```

Add the parser option:

```python
parser.add_argument("--id-map", type=Path)
```

Add:

```python
def _resolve_cli_id_map(data_root: Path, raw_path: Path) -> Path:
    if raw_path.is_absolute():
        raise _CliInputError("identifier map path must stay under data")
    resolved_root = data_root.resolve()
    resolved = raw_path.resolve()
    if not resolved.is_relative_to(resolved_root):
        raise _CliInputError("identifier map path must stay under data")
    if not resolved.is_file():
        raise _CliInputError("identifier map file does not exist")
    return resolved
```

Keep this static check, then add a single-handle read boundary:

```python
def _read_confined_identifier_map(data_root: Path, raw_path: Path) -> bytes:
    resolved_root = data_root.resolve()
    resolved_path = _resolve_cli_id_map(resolved_root, raw_path)
    with resolved_path.open("rb") as source:
        if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
            raise _CliInputError("identifier map file does not exist")
        final_path = _final_path_from_open_file(source)
        if not final_path.is_relative_to(resolved_root):
            raise _CliInputError("identifier map path must stay under data")
        return source.read()
```

`_final_path_from_open_file` uses `GetFinalPathNameByHandleW` on Windows,
normalizing both `\\?\C:\...` and `\\?\UNC\server\share\...` forms. On POSIX it
reads `/proc/self/fd/<fd>` or `/dev/fd/<fd>` and requires the resolved target's
device/inode to match `os.fstat(fd)`. Failure to determine or validate the final
target is rejected. The `with` statement closes the handle on success and every
failure path.

Add the unchanged coverage check:

```python
def _require_id_map_coverage(
    gold: Sequence[EvaluationQuery],
    id_map: IdentifierMap,
) -> None:
    identifiers = {
        identifier
        for record in gold
        for identifier in record.relevant_paper_ids
    }
    if any(not id_map.covers(identifier) for identifier in identifiers):
        raise _CliInputError(
            "identifier map does not cover frozen gold identifiers"
        )
```

- [ ] **Step 7: Implement conditional artifact identity**

Build hashes as:

```python
input_hashes = {
    "gold": result.identity.gold_sha256,
    "predictions": _sha256_bytes(prediction_bytes),
}
if result.identity.id_map_sha256 is not None:
    input_hashes["id_map"] = result.identity.id_map_sha256
```

Serialize run identity without absent optional fields:

```python
"identity": result.identity.model_dump(mode="json", exclude_none=True),
```

Update existing no-map artifact and frozen-split assertions to use
`model_dump(mode="json", exclude_none=True)` so their expected bytes and object
shape remain unchanged.

- [ ] **Step 8: Implement CLI loading before provider construction**

Extend `_run_cli_evaluation` with:

```python
id_map: IdentifierMap | None,
```

and pass `id_map=id_map` to `run_evaluation`.

In `main`, after resolving the frozen split and before loading the runtime
provider, use:

```python
identity = frozen_split.identity
id_map: IdentifierMap | None = None
if args.id_map is not None:
    id_map_bytes = _read_confined_identifier_map(Path("data"), args.id_map)
    id_map = IdentifierMap.from_bytes(id_map_bytes)
    _require_id_map_coverage(frozen_split.gold, id_map)
    identity = identity.model_copy(
        update={"id_map_sha256": _sha256_bytes(id_map_bytes)}
    )
```

Pass `identity=identity` and `id_map=id_map` into `_run_cli_evaluation`.

The standalone metrics CLI follows the same snapshot rule without runner path
confinement: read `args.id_map` once into bytes, parse with
`IdentifierMap.from_bytes`, and compute the namespaced SHA-256 from those bytes.
Map read and validation failures become fixed, value-free CLI errors; existing
gold and prediction diagnostics remain available.

- [ ] **Step 9: Run focused runner tests and verify GREEN**

Run:

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/evaluation/test_runner.py -k "id_map or identifier_map or artifact or cli_parser" -q
```

Expected: all selected tests PASS.

- [ ] **Step 10: Run full offline verification**

Run:

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file ruff check src/paper_search/evaluation/dataset.py src/paper_search/evaluation/runner.py tests/evaluation/test_dataset.py tests/evaluation/test_runner.py
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file mypy src/paper_search/evaluation/dataset.py src/paper_search/evaluation/runner.py
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest -q
git diff --check
```

Expected: every command exits 0; the live credential-gated test may remain
skipped.

- [ ] **Step 11: Commit Task 2**

```powershell
git add -- src/paper_search/evaluation/runner.py tests/evaluation/test_runner.py
git commit -m "feat: bind identifier map to evaluation runs"
```

### Task 3: Execute and Audit the Immutable R3 Baseline

**Files:**
- Create outside Git: `$runRoot`, computed from the committed Task 2 SHA
- Create outside Git: `D:\OpenAI\CodexHome\visualizations\2026\07\29\019facba-bf0b-79e2-bb99-2c17ddf9eca8\run_week3_r3.py`
- Create outside Git: `$runRoot\execution-receipt.json`
- Create outside Git: `$runRoot\baseline-dev\`
- Create outside Git: `$runRoot\.cache\openalex.sqlite3`
- Preserve: `D:\AI Projects\private-baseline-runs\week3-real-baseline-fb948c4-20260729\`
- Preserve: `D:\AI Projects\private-baseline-runs\week3-real-baseline-33f0cf4-20260729-r2\`

**Interfaces:**
- Consumes: committed map-enabled runner, frozen Week1 data, private map, process-level OpenAlex credential
- Produces: map-bound formal artifacts plus a secret-safe direct-exit receipt

- [ ] **Step 1: Establish the exact R3 root**

From the Week3 worktree:

```powershell
$fullSha=(git rev-parse HEAD).Trim()
if ($fullSha -notmatch '^[0-9a-f]{40}$') { throw 'invalid source SHA' }
$shortSha=$fullSha.Substring(0,8)
$runRoot="D:\AI Projects\private-baseline-runs\week3-real-baseline-$shortSha-20260729-r3"
if (Test-Path -LiteralPath $runRoot) { throw 'R3 run root already exists' }
New-Item -ItemType Directory -Path $runRoot | Out-Null
New-Item -ItemType Junction `
  -Path (Join-Path $runRoot 'data') `
  -Target 'D:\AI Projects\.worktrees\week1-collaboration\data' | Out-Null
if (Test-Path -LiteralPath (Join-Path $runRoot 'baseline-dev')) {
    throw 'R3 output must start absent'
}
if (Test-Path -LiteralPath (Join-Path $runRoot '.cache\openalex.sqlite3')) {
    throw 'R3 cache must start absent'
}
```

- [ ] **Step 2: Create the atomic foreground launcher**

Set:

```powershell
$launcherPath='D:\OpenAI\CodexHome\visualizations\2026\07\29\019facba-bf0b-79e2-bb99-2c17ddf9eca8\run_week3_r3.py'
```

Create that launcher with:

```python
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

WEEK3_ROOT = Path(r"D:\AI Projects\.worktrees\week3")
RUN_ROOT = Path(os.environ["BASELINE_RUN_ROOT"]).resolve()
OUTPUT = RUN_ROOT / "baseline-dev"
RECEIPT = RUN_ROOT / "execution-receipt.json"


def sha256_bytes(content: bytes) -> str:
    return f"sha256:{hashlib.sha256(content).hexdigest()}"


def write_receipt(payload: dict[str, object]) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        dir=RUN_ROOT,
        prefix=".execution-receipt.",
        suffix=".tmp",
        delete=False,
    ) as target:
        temporary = Path(target.name)
        json.dump(payload, target, sort_keys=True, indent=2)
        target.write("\n")
        target.flush()
        os.fsync(target.fileno())
    temporary.replace(RECEIPT)


git_dir = subprocess.run(
    ["git", "-C", str(WEEK3_ROOT), "rev-parse", "--absolute-git-dir"],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
source_sha = subprocess.run(
    ["git", "-C", str(WEEK3_ROOT), "rev-parse", "HEAD"],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
os.environ["GIT_DIR"] = git_dir
sys.path.insert(0, str(WEEK3_ROOT / "src"))
os.chdir(RUN_ROOT)

from paper_search.evaluation.runner import main

started_at = datetime.now(UTC)
exit_code = main(
    [
        "--config",
        str(WEEK3_ROOT / "configs" / "base.yaml"),
        "--split",
        "dev",
        "--output",
        str(OUTPUT),
        "--id-map",
        "data/annotation_work/dev_identifier_map.v1.json",
    ]
)
ended_at = datetime.now(UTC)
launcher_path = Path(__file__).resolve()
write_receipt(
    {
        "cache_path": ".cache/openalex.sqlite3",
        "ended_at": ended_at.isoformat(),
        "exit_code": exit_code,
        "launcher_sha256": sha256_bytes(launcher_path.read_bytes()),
        "output_path": "baseline-dev",
        "source_sha": source_sha,
        "started_at": started_at.isoformat(),
    }
)
raise SystemExit(exit_code)
```

- [ ] **Step 3: Run R3 in the foreground**

From `D:\AI Projects\Projects`, run with a command timeout of at least ten
minutes:

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
$env:BASELINE_RUN_ROOT=$runRoot
& 'D:\Dev\uv\uv.exe' run --no-sync --env-file .env python $launcherPath
$directExitCode=$LASTEXITCODE
if ($directExitCode -ne 0) { throw "R3 exited $directExitCode" }
```

Expected: foreground process exits 0, receipt exists with `exit_code` 0, and
seven formal top-level artifacts exist under `baseline-dev`.

- [ ] **Step 4: Validate map binding and structure**

Using a local offline validation script, assert without printing private values:

```text
receipt source_sha == git HEAD
receipt launcher_sha256 == recomputed launcher hash
receipt exit_code == 0
run.identity.git_sha == git HEAD
run.identity.id_map_sha256 == recomputed private-map hash
run.input_hashes.id_map == recomputed private-map hash
metrics.input_hashes.id_map == recomputed private-map hash
unique gold identifier count == 141
explicitly covered unique gold count == 141
unresolved arXiv count == 0
prediction/deduplication/filtering record counts == 60
snapshot manifest validation succeeds
snapshot response count > 0
queries with cache keys > 0
queries with predictions > 0
invalid_request count == 0
sum of per-query true_positive_count > 0
macro_recall > 0
micro_recall > 0
```

The script prints only aggregate counts, hashes, summary metrics, and safe
provider error-code counts.

- [ ] **Step 5: Validate privacy and immutability**

Read-only checks must confirm:

- R1 and R2 artifact hashes are unchanged from their retained reports;
- the exact private-map byte sequence is not present in any generated R3
  receipt, output, or cache file; the `data` junction is excluded because it
  intentionally references the authoritative private source file;
- `OPENALEX_API_KEY`, `LLM_API_KEY`, and other loaded credential values are not
  present in any R3 file, without printing those values;
- receipt, `run.json`, `metrics.json`, `usage.json`, snapshot manifest, and
  SQLite cache metadata contain neither the private map path nor serialized
  map-entry structure;
- individual provider identifiers in predictions or snapshots are allowed and
  are not reported as leaks.

- [ ] **Step 6: Compare R3 with R2 safely**

Report only aggregate differences:

```text
search API calls
elapsed milliseconds
snapshot count
queries with predictions
provider error-code counts
macro/micro Precision, Recall, and F1
Recall@5, Recall@10, and Recall@20
```

Label R2 metrics as invalid no-map diagnostic values. Do not use R2's zero
metrics as a performance baseline.

- [ ] **Step 7: Independent operational review**

Dispatch a read-only reviewer with the Task 3 brief, execution report, receipt,
formal artifact path, and source SHA. The reviewer may recompute aggregate
counts and hashes but must not make network calls or expose private values.

Expected: both structural/effectiveness compliance and operational quality are
approved before R3 is reported as formal.
