# OpenAlex Query Normalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make frozen natural-language queries valid OpenAlex requests and produce a verified real fresh-cache development baseline.

**Architecture:** Add a pure outbound-query normalizer at the OpenAlex provider boundary. Keep frozen input bytes and evaluation identity unchanged, while cache keys and provenance record the normalized request actually sent. Preserve the failed run and create a new Git-external run rooted at the fixed source revision.

**Tech Stack:** Python 3.11+, httpx, pytest, SQLite response snapshots, uv, PowerShell

## Global Constraints

- Replace every `?` and `*` in outbound OpenAlex search text with one space.
- Collapse consecutive whitespace to one ASCII space and reject an empty result.
- Never modify frozen Week1 data or reuse the failed run's cache or artifacts.
- Load credentials from `D:\AI Projects\Projects\.env` by running uv from that directory with relative `--env-file .env`.
- Never print or persist credentials.

---

### Task 1: Normalize Outbound OpenAlex Search Text

**Files:**
- Modify: `src/paper_search/retrieval/openalex.py`
- Test: `tests/unit/test_openalex.py`

**Interfaces:**
- Consumes: `OpenAlexProvider.search(query: str, filters: dict[str, object], limit: int, reservation: BudgetReservation)`
- Produces: `_normalize_search_query(query: str) -> str`, used only to build OpenAlex request parameters and cache keys

- [ ] **Step 1: Write failing normalization request test**

Add:

```python
def test_search_removes_openalex_wildcards_and_collapses_whitespace(
    tmp_path: Path,
) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=fixture_bytes("works_page_1.json"))

    asyncio.run(
        run_search(
            SQLiteResponseCache(tmp_path / "cache.sqlite3"),
            handler,
            query="  graph?   retrieval* methods  ",
        )
    )

    assert seen[0].url.params["search"] == "graph retrieval methods"
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/unit/test_openalex.py::test_search_removes_openalex_wildcards_and_collapses_whitespace -q
```

Expected: FAIL because the request still contains `?`, `*`, and repeated whitespace.

- [ ] **Step 3: Write failing wildcard-only rejection test**

Add:

```python
def test_search_rejects_query_that_is_empty_after_wildcard_normalization(
    tmp_path: Path,
) -> None:
    requested = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requested
        requested = True
        return httpx.Response(200, content=fixture_bytes("works_page_1.json"))

    with pytest.raises(ValueError, match="query must not be empty"):
        asyncio.run(
            run_search(
                SQLiteResponseCache(tmp_path / "cache.sqlite3"),
                handler,
                query=" ? * ",
            )
        )

    assert requested is False
```

- [ ] **Step 4: Run the rejection test and verify RED**

Run:

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/unit/test_openalex.py::test_search_rejects_query_that_is_empty_after_wildcard_normalization -q
```

Expected: FAIL because the current provider sends the wildcard-only query.

- [ ] **Step 5: Implement the minimal normalizer**

In `src/paper_search/retrieval/openalex.py`, add:

```python
def _normalize_search_query(query: str) -> str:
    return " ".join(query.replace("?", " ").replace("*", " ").split())
```

At the start of `OpenAlexProvider.search`, replace:

```python
normalized_query = query.strip()
```

with:

```python
normalized_query = _normalize_search_query(query)
```

Keep the existing empty-query validation immediately after it.

- [ ] **Step 6: Run focused tests and verify GREEN**

Run:

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/unit/test_openalex.py -q
```

Expected: all OpenAlex unit tests PASS, including the existing ordinary `RAG` request assertion.

- [ ] **Step 7: Run static and regression checks**

Run:

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file ruff check src/paper_search/retrieval/openalex.py tests/unit/test_openalex.py
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file mypy src/paper_search/retrieval/openalex.py
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest -q
git diff --check
```

Expected: every command exits 0.

- [ ] **Step 8: Commit the fix**

```powershell
git add -- src/paper_search/retrieval/openalex.py tests/unit/test_openalex.py
git commit -m "fix: normalize OpenAlex wildcard queries"
```

### Task 2: Run and Verify a New Real Fresh-Cache Baseline

**Files:**
- Create outside Git: `$runRoot`, computed as `D:\AI Projects\private-baseline-runs\week3-real-baseline-$shortSha-20260729-r2`
- Preserve: `D:\AI Projects\private-baseline-runs\week3-real-baseline-fb948c4-20260729\`

**Interfaces:**
- Consumes: committed `paper_search.evaluation.runner`, frozen Week1 `data/`, `configs/base.yaml`, and process-level `OPENALEX_API_KEY`
- Produces: immutable baseline artifacts, SQLite response snapshot, aggregate metrics, usage summary, and validation evidence

- [ ] **Step 1: Establish the new run identity and empty directories**

Resolve `git rev-parse HEAD`, assign its first eight characters to `$shortSha`,
assign the path above to `$runRoot`, create that Git-external run root, create a
directory junction named `data` to
`D:\AI Projects\.worktrees\week1-collaboration\data`, and verify that the new
`.cache\openalex.sqlite3` and `baseline-dev` output do not exist.

- [ ] **Step 2: Execute from the credential directory**

Run uv with working directory `D:\AI Projects\Projects`, relative
`--env-file .env`, and an absolute wrapper path. The wrapper must set its
working directory to the new run root, set `GIT_DIR` to the Week3 worktree Git
directory, prepend the Week3 `src` directory to `sys.path`, and call:

```python
import os
from pathlib import Path

main(
    [
        "--config",
        r"D:\AI Projects\.worktrees\week3\configs\base.yaml",
        "--split",
        "dev",
        "--output",
        str(Path(os.environ["BASELINE_RUN_ROOT"]) / "baseline-dev"),
    ]
)
```

Expected: exit code 0, no secret-bearing stdout/stderr, and seven formal
artifacts under `baseline-dev`.

- [ ] **Step 3: Validate structural evidence**

Validate `snapshot_manifest.json` with
`paper_search.storage.validate_snapshot_manifest`; verify 60 prediction,
deduplication, and filtering records; verify `run.json` binds the current
40-character Git SHA and frozen data hashes; and calculate SHA-256 for every
artifact.

- [ ] **Step 4: Validate effective retrieval**

Aggregate only public-safe counts from `usage.json` and artifacts. Require:

```text
invalid_request count == 0
search_api_calls > 0
snapshot manifest response count > 0
queries with non-empty cache_keys > 0
queries with at least one prediction > 0
```

If any condition fails, retain the run as diagnostic evidence and do not report
it as an effective baseline.

- [ ] **Step 5: Report the baseline**

Report the run root, source SHA, artifact hashes, macro/micro
Precision/Recall/F1, Recall@5/10/20, query count, provider error-code counts,
search-call count, elapsed time, cache snapshot count, and retrieval/prediction
coverage. Do not expose query text, gold labels, per-query metrics, credentials,
or raw provider responses.
