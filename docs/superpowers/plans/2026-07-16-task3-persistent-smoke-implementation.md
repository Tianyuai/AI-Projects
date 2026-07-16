# Task 3 Persistent OpenAlex Smoke Artifacts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the keyed Task 3 online pytest command leave a safe, repeatable, locally persistent OpenAlex smoke summary and immutable raw-response snapshots under `experiments/smoke/`.

**Architecture:** Keep smoke orchestration in `tests/integration/test_openalex_live.py`; do not add a production module or change provider/cache contracts. Each invocation creates a run-specific cache and snapshot directory, validates it, then atomically publishes `experiments/smoke/provider.json` as a pointer to the latest accepted run.

**Tech Stack:** Python 3.11, pathlib, uuid, httpx 0.28, pytest, existing `OpenAlexProvider`, existing `SQLiteResponseCache`, Ruff, mypy strict.

## Global Constraints

- Work only on `codex/task3-openalex`.
- Never read, print, log, or commit `.env` or `OPENALEX_API_KEY`; the authorized keyed command may load `.env` only into its child process.
- Never stage or modify the existing Task 2 design line-ending metadata change.
- Add no production provider behavior, new dependency, CLI, Semantic Scholar, citation, ranking, or Task 4 code.
- Preserve exact response bytes and validate every snapshot SHA-256 through the existing manifest contract.
- Store live artifacts locally under `experiments/smoke/` and exclude that directory from Git.
- Every code behavior follows RED -> verify RED -> minimal GREEN -> verify GREEN.
- Do not merge. Do not push until all acceptance checks and independent review pass and the user separately authorizes the integration step.

---

### Task 1: Persist Run-specific Smoke Artifacts

**Files:**
- Modify: `tests/integration/test_openalex_live.py`

**Interfaces:**
- Consumes: `OpenAlexProvider`, `SQLiteResponseCache.export_snapshot(cache_keys: Sequence[str], run_dir: Path) -> Path`, `validate_snapshot_manifest(path: Path) -> None`.
- Produces: `run_live_queries(api_key: str, smoke_root: Path, *, transport: httpx.AsyncBaseTransport | None = None) -> Path`, returning the published top-level `provider.json` path.
- Produces summary fields: `contract_version`, `run_id`, `manifest`, and `queries`.

- [ ] **Step 1: Write failing offline persistence tests**

Add safe fixture transport and two unmarked tests to `tests/integration/test_openalex_live.py`. The tests run during the default offline suite and never inspect process credentials.

```python
FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "openalex"


def fixture_transport(filename: str) -> httpx.MockTransport:
    payload = (FIXTURE_ROOT / filename).read_bytes()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=payload,
            headers={"content-type": "application/json"},
            request=request,
        )

    return httpx.MockTransport(handler)


def test_smoke_artifacts_are_persistent_versioned_and_secret_free(
    tmp_path: Path,
) -> None:
    smoke_root = tmp_path / "experiments" / "smoke"
    key = "sentinel-openalex-key"

    first_provider = asyncio.run(
        run_live_queries(
            key,
            smoke_root,
            transport=fixture_transport("works_page_1.json"),
        )
    )
    first_summary = json.loads(first_provider.read_text(encoding="utf-8"))
    first_manifest = smoke_root / first_summary["manifest"]
    validate_snapshot_manifest(first_manifest)

    second_provider = asyncio.run(
        run_live_queries(
            key,
            smoke_root,
            transport=fixture_transport("works_page_1.json"),
        )
    )
    second_summary = json.loads(second_provider.read_text(encoding="utf-8"))
    second_manifest = smoke_root / second_summary["manifest"]

    assert first_summary["contract_version"] == "openalex-smoke-v1"
    assert first_summary["run_id"] != second_summary["run_id"]
    assert first_manifest.exists()
    assert second_manifest.exists()
    assert len(second_summary["queries"]) == 3
    assert all(item["paper_count"] > 0 for item in second_summary["queries"])
    assert key.encode() not in b"".join(
        path.read_bytes() for path in smoke_root.rglob("*") if path.is_file()
    )


def test_failed_smoke_does_not_replace_last_accepted_summary(tmp_path: Path) -> None:
    smoke_root = tmp_path / "experiments" / "smoke"
    provider = asyncio.run(
        run_live_queries(
            "sentinel-openalex-key",
            smoke_root,
            transport=fixture_transport("works_page_1.json"),
        )
    )
    accepted = provider.read_bytes()

    with pytest.raises(AssertionError):
        asyncio.run(
            run_live_queries(
                "sentinel-openalex-key",
                smoke_root,
                transport=fixture_transport("works_empty.json"),
            )
        )

    assert provider.read_bytes() == accepted
```

- [ ] **Step 2: Run RED and confirm the missing interface**

Run:

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
uv run --no-sync --no-env-file pytest -m "not online" tests/integration/test_openalex_live.py -v
```

Expected: both new tests fail with `TypeError` because `run_live_queries` does not accept `transport` and does not publish a run-specific summary.

- [ ] **Step 3: Implement the minimal run-specific publisher**

Add `uuid4`, `os.replace`, constants, and atomic JSON publication in `tests/integration/test_openalex_live.py`:

```python
from uuid import uuid4


SMOKE_CONTRACT_VERSION = "openalex-smoke-v1"


def new_run_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{timestamp}-{uuid4().hex[:8]}"


def publish_summary(path: Path, value: dict[str, object]) -> None:
    content = json.dumps(value, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)
```

Change the runner to create a fresh run directory, accept an injected transport for offline contract tests, validate snapshots before publication, and return the top-level summary path:

```python
async def run_live_queries(
    api_key: str,
    smoke_root: Path,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> Path:
    run_id = new_run_id()
    run_dir = smoke_root / "runs" / run_id
    cache = SQLiteResponseCache(run_dir / "openalex-cache.sqlite3")
    summaries: list[dict[str, object]] = []
    ordered_keys: list[str] = []
    async with httpx.AsyncClient(
        base_url="https://api.openalex.org",
        timeout=httpx.Timeout(30.0),
        transport=transport,
    ) as client:
        provider = OpenAlexProvider(client=client, cache=cache, api_key=api_key)
        for index, query in enumerate(LIVE_QUERIES, start=1):
            result = await provider.search(query, {}, 3, reservation(index))
            assert result.data
            ordered_keys.extend(json.loads(result.provenance["cache_keys"]))
            summaries.append(
                {
                    "error_codes": [error.code for error in result.errors],
                    "latency_ms": result.latency_ms,
                    "paper_count": len(result.data),
                    "response_hash": result.provenance["response_hash"],
                }
            )
    manifest = cache.export_snapshot(list(dict.fromkeys(ordered_keys)), run_dir)
    validate_snapshot_manifest(manifest)
    summary: dict[str, object] = {
        "contract_version": SMOKE_CONTRACT_VERSION,
        "run_id": run_id,
        "manifest": manifest.relative_to(smoke_root).as_posix(),
        "queries": summaries,
    }
    provider_path = smoke_root / "provider.json"
    publish_summary(provider_path, summary)
    return provider_path
```

- [ ] **Step 4: Run GREEN for offline persistence**

Run:

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
uv run --no-sync --no-env-file pytest -m "not online" tests/integration/test_openalex_live.py -v
```

Expected: two offline persistence tests pass and the keyed online test is deselected.

- [ ] **Step 5: Run focused Task 3 regression tests**

Run:

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
uv run --no-sync --no-env-file pytest -m "not online" tests/unit/test_openalex.py tests/unit/test_cache.py tests/unit/test_normalize.py tests/integration/test_openalex_live.py -v
```

Expected: every selected Task 3 test passes; no online request is made.

- [ ] **Step 6: Commit the tested persistence behavior**

```powershell
git add -- tests/integration/test_openalex_live.py
git commit -m "test: persist OpenAlex smoke artifacts"
```

---

### Task 2: Route the Keyed Test to the PRD Smoke Directory

**Files:**
- Modify: `.gitignore`
- Modify: `tests/integration/test_openalex_live.py`

**Interfaces:**
- Consumes: `run_live_queries(...) -> Path` from Task 1.
- Produces: a keyed online test that writes to repository-local `experiments/smoke/` and validates the published manifest.

- [ ] **Step 1: Write the failing repository-output assertion**

Add constants and change the online test to use the repository-local smoke root:

```python
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = PROJECT_ROOT / "experiments" / "smoke"


@pytest.mark.online
def test_three_live_queries_produce_safe_snapshot() -> None:
    api_key = os.environ.get("OPENALEX_API_KEY")
    if not api_key:
        pytest.skip("OPENALEX_API_KEY is not set in the process environment")

    provider_path = asyncio.run(run_live_queries(api_key, SMOKE_ROOT))
    serialized = provider_path.read_text(encoding="utf-8")
    summary = json.loads(serialized)
    manifest = SMOKE_ROOT / summary["manifest"]

    assert provider_path == PROJECT_ROOT / "experiments" / "smoke" / "provider.json"
    assert api_key not in serialized
    assert len(summary["queries"]) == 3
    assert all(item["paper_count"] > 0 for item in summary["queries"])
    validate_snapshot_manifest(manifest)
```

Before implementation, run the keyed test only after Step 2 adds the ignore rule; the prior implementation cannot satisfy this contract because it writes to `tmp_path`.

- [ ] **Step 2: Ignore every local smoke artifact**

Append exactly this repository-root rule to `.gitignore`:

```gitignore

# Local provider smoke artifacts
/experiments/smoke/
```

Verify the rule without creating an artifact:

```powershell
git check-ignore -v --no-index experiments/smoke/provider.json
```

Expected: output identifies `.gitignore` and `/experiments/smoke/`.

- [ ] **Step 3: Run the authorized keyed online test**

Run from `D:\AI Projects\.worktrees\task2-evaluation`:

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
uv run --no-sync --env-file ../../Projects/.env pytest -m online tests/integration/test_openalex_live.py -v
```

Expected: `test_three_live_queries_produce_safe_snapshot PASSED`, `1 passed`, and no skipped test.

- [ ] **Step 4: Validate the persisted summary and manifest without printing secrets or raw responses**

Run a small Python validation that reads only generated smoke metadata and reports booleans/counts, never values from `.env`:

```powershell
uv run --no-sync --no-env-file python -c "import json; from pathlib import Path; from paper_search.storage.cache import validate_snapshot_manifest; root=Path('experiments/smoke'); summary=json.loads((root/'provider.json').read_text(encoding='utf-8')); manifest=root/summary['manifest']; validate_snapshot_manifest(manifest); print({'queries': len(summary['queries']), 'all_nonempty': all(item['paper_count'] > 0 for item in summary['queries']), 'manifest_valid': True})"
```

Expected:

```text
{'queries': 3, 'all_nonempty': True, 'manifest_valid': True}
```

Run:

```powershell
git status --short --ignored experiments/smoke
```

Expected: the smoke directory is shown only as ignored (`!!`), never staged or untracked (`??`).

- [ ] **Step 5: Run focused tests after live publication**

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
uv run --no-sync --no-env-file pytest -m "not online" tests/unit/test_openalex.py tests/unit/test_cache.py tests/unit/test_normalize.py tests/integration/test_openalex_live.py -v
```

Expected: all selected offline Task 3 tests pass.

- [ ] **Step 6: Commit only code and ignore behavior**

```powershell
git add -- .gitignore tests/integration/test_openalex_live.py
git diff --cached --check
git commit -m "test: retain OpenAlex live smoke evidence"
```

---

### Task 3: Complete Task 3 Acceptance and Record Truthful Status

**Files:**
- Modify: `PRD.md:808-810`

**Interfaces:**
- Consumes: the successful keyed run and validated persistent artifacts from Task 2.
- Produces: truthful checked Task 3 live-smoke and online-test acceptance items.

- [ ] **Step 1: Run the complete quality gate with `.env` disabled**

Run each command independently:

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
uv run --no-sync --no-env-file pytest -m "not online" tests/unit/test_openalex.py tests/unit/test_cache.py tests/unit/test_normalize.py tests/integration/test_openalex_live.py -v
uv run --no-sync --no-env-file pytest -q
uv run --no-sync --no-env-file ruff check .
uv run --no-sync --no-env-file mypy src
git diff --check
```

Expected: focused tests pass; full pytest passes with exactly the keyed online test skipped; Ruff reports `All checks passed!`; mypy reports no issues; `git diff --check` reports no errors.

- [ ] **Step 2: Perform secret and artifact boundary checks**

Search tracked content for common credential material without reading `.env`:

```powershell
git grep -n -I -E "(Authorization:|Bearer [A-Za-z0-9_-]{16,}|OPENALEX_API_KEY=.+)" -- . ":(exclude)docs/superpowers/plans/*" ":(exclude)docs/superpowers/specs/*"
```

Expected: no tracked credential value or literal authorization header. Variable-name-only declarations are allowed.

Run:

```powershell
git status --short --ignored
```

Expected: `.env` and `experiments/smoke/` are ignored; neither appears in the staged or untracked change set. The pre-existing Task 2 design metadata change remains unstaged.

- [ ] **Step 3: Request independent review of the Task 3 delta**

Review commits from `8b0bdf2` through `HEAD`, plus the current PRD-only status change, against:

- `PRD.md:797-812`;
- `docs/superpowers/specs/2026-07-16-task3-openalex-design.md`;
- `docs/superpowers/specs/2026-07-16-task3-persistent-smoke-design.md`.

The reviewer must report critical, important, and minor findings with exact file/line references. Fix all critical and important findings through a fresh RED/GREEN cycle before proceeding.

- [ ] **Step 4: Mark only the two evidenced PRD items complete**

Change:

```markdown
- [ ] 使用 3 条真实查询做冒烟测试，保存原始响应快照。
...
- [ ] 运行 `uv run pytest -m online tests/integration/test_openalex_live.py -v`；只有已配置 key 时执行，结果写入 `experiments/smoke/provider.json`。
```

to:

```markdown
- [x] 使用 3 条真实查询做冒烟测试，保存原始响应快照。
...
- [x] 运行 `uv run pytest -m online tests/integration/test_openalex_live.py -v`；只有已配置 key 时执行，结果写入 `experiments/smoke/provider.json`。
```

Do not change any Task 2, Task 4, collaborator, or first-week stage-gate checkbox.

- [ ] **Step 5: Commit the acceptance record**

```powershell
git add -- PRD.md
git diff --cached --check
git diff --cached --name-only
git commit -m "docs: record Task 3 live acceptance"
```

Expected staged file list: `PRD.md` only.

- [ ] **Step 6: Re-run the final acceptance evidence after the last commit**

Repeat the keyed online test, focused suite, full pytest, Ruff, mypy, manifest validator, secret scan, `git diff --check`, and `git status --short --branch`. Record exact pass counts and confirm there are no unresolved critical or important review findings.

Do not merge. Do not push unless the user separately asks after reviewing the final evidence.
