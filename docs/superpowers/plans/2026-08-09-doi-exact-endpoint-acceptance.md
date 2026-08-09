# DOI Exact-Endpoint Acceptance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make DOI exact-endpoint HTTP 200 responses available when they contain a valid OpenAlex Work ID, while preserving strict OpenAlex-ID matching and all existing safety contracts.

**Architecture:** Keep the change inside the aggregate diagnostic classifier. Treat the exact DOI endpoint plus a valid returned Work ID as the identity binding; keep response `doi` descriptive, and retain strict equality for OpenAlex-ID requests.

**Tech Stack:** Python 3.11, `httpx.MockTransport`, pytest, Ruff, mypy.

## Global Constraints

- Follow `docs/superpowers/specs/2026-08-09-doi-exact-endpoint-acceptance-design.md` exactly.
- Do not read `.env`, make network requests, rerun the availability probe, rebuild locks, run readiness, or perform capture/replay/validation.
- Do not change provider adapters, production retrieval, candidate generation, ranking, report schema, privacy rules, budget accounting, or historical diagnostic artifacts.
- Preserve untracked `data/budget_ledger.sqlite3` and `deliverables/`.

---

### Task 1: Classify DOI exact-endpoint responses by returned Work ID

**Files:**
- Modify: `tests/scripts/test_analyze_gold_bottlenecks.py:675`
- Modify: `scripts/analyze_gold_bottlenecks.py:844`

**Interfaces:**
- Consumes: `_classify_response(identifier: str, payload: object)` through `probe_openalex_exact(...)`.
- Produces: the existing `(AvailabilityStatus, IntegrityFailureReason | None)` tuple; no new type or schema.

+ [x] **Step 1: Replace the old DOI-field failure cases with contract tests**

Add synthetic `httpx.MockTransport` tests covering these exact outcomes:

```python
@pytest.mark.parametrize(
    "doi_value",
    [
        "https://doi.org/10.1000/a",
        None,
        "https://doi.org/10.1000/b",
        "not-an-identifier",
    ],
)
def test_doi_exact_200_accepts_valid_work_id_regardless_of_canonical_doi(
    doi_value: str | None,
) -> None:
    payload: dict[str, object] = {"id": "https://openalex.org/W1"}
    if doi_value is not None:
        payload["doi"] = doi_value

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, request=request, json=payload)

    async def execute() -> ProbeBatch:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await probe_openalex_exact(
                ["doi:10.1000/a"],
                client=client,
                keys=(SecretStr("k1"),),
                sleep=lambda seconds: asyncio.sleep(0),
                clock=lambda: datetime(2026, 8, 9, tzinfo=UTC),
            )

    result = asyncio.run(execute())
    assert result.status_by_work == {"doi:10.1000/a": "available"}
    assert result.integrity_reason_by_work == {}


@pytest.mark.parametrize(
    ("payload", "expected_reason"),
    [
        ({"doi": "https://doi.org/10.1000/a"}, "missing_expected_field"),
        ({"id": "not-an-identifier"}, "unparseable_identifier"),
        ({"id": "https://doi.org/10.1000/a"}, "unparseable_identifier"),
    ],
)
def test_doi_exact_200_rejects_missing_or_non_work_id(
    payload: dict[str, object],
    expected_reason: str,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, request=request, json=payload)

    async def execute() -> ProbeBatch:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await probe_openalex_exact(
                ["doi:10.1000/a"],
                client=client,
                keys=(SecretStr("k1"),),
                sleep=lambda seconds: asyncio.sleep(0),
                clock=lambda: datetime(2026, 8, 9, tzinfo=UTC),
            )

    result = asyncio.run(execute())
    assert result.status_by_work == {"doi:10.1000/a": "integrity_failure"}
    assert result.integrity_reason_by_work == {
        "doi:10.1000/a": expected_reason,
    }


@pytest.mark.parametrize(
    ("response_id", "expected_status", "expected_reason"),
    [
        ("https://openalex.org/W2", "available", None),
        (
            "https://openalex.org/W3",
            "integrity_failure",
            "canonical_mismatch",
        ),
    ],
)
def test_openalex_exact_200_keeps_strict_work_id_match(
    response_id: str,
    expected_status: str,
    expected_reason: str | None,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            request=request,
            json={"id": response_id},
        )

    async def execute() -> ProbeBatch:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await probe_openalex_exact(
                ["openalex:W2"],
                client=client,
                keys=(SecretStr("k1"),),
                sleep=lambda seconds: asyncio.sleep(0),
                clock=lambda: datetime(2026, 8, 9, tzinfo=UTC),
            )

    result = asyncio.run(execute())
    assert result.status_by_work == {"openalex:W2": expected_status}
    expected_reasons = (
        {} if expected_reason is None else {"openalex:W2": expected_reason}
    )
    assert result.integrity_reason_by_work == expected_reasons


def test_200_non_object_payload_remains_integrity_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, request=request, content=b"not-json")

    async def execute() -> ProbeBatch:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await probe_openalex_exact(
                ["doi:10.1000/a"],
                client=client,
                keys=(SecretStr("k1"),),
                sleep=lambda seconds: asyncio.sleep(0),
                clock=lambda: datetime(2026, 8, 9, tzinfo=UTC),
            )

    result = asyncio.run(execute())
    assert result.status_by_work == {"doi:10.1000/a": "integrity_failure"}
    assert result.integrity_reason_by_work == {
        "doi:10.1000/a": "missing_expected_field",
    }
```

Do not call `_classify_response` directly and do not use network access.

+ [x] **Step 2: Run the new focused tests and verify RED**

Run:

```powershell
$env:PYTHONPATH = (Get-Location).Path
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest tests/scripts/test_analyze_gold_bottlenecks.py -k "doi_exact_200 or openalex_exact_200 or non_object_payload" -q
```

Expected: DOI responses with a valid Work ID fail under the old top-level DOI equality rule; missing/non-Work IDs and OpenAlex mismatch retain their expected classifications.

+ [x] **Step 3: Implement the minimal classifier change**

Replace `_classify_response` with the following logic:

```python
def _classify_response(
    identifier: str,
    payload: object,
) -> tuple[AvailabilityStatus, IntegrityFailureReason | None]:
    if not isinstance(payload, dict):
        return "integrity_failure", "missing_expected_field"
    normalized = normalize_paper_id(identifier)
    response_identifier = payload.get("id")
    if not isinstance(response_identifier, str):
        return "integrity_failure", "missing_expected_field"
    try:
        response_normalized = normalize_paper_id(response_identifier)
    except ValueError:
        return "integrity_failure", "unparseable_identifier"
    if not response_normalized.startswith("openalex:"):
        return "integrity_failure", "unparseable_identifier"
    if normalized.startswith("doi:"):
        return "available", None
    if not normalized.startswith("openalex:"):
        return "integrity_failure", "unparseable_identifier"
    if response_normalized != normalized:
        return "integrity_failure", "canonical_mismatch"
    return "available", None
```

Do not add alias maps, new report fields, provider-field fallbacks, or logging.

+ [x] **Step 4: Verify GREEN and focused static checks**

Run:

```powershell
$env:PYTHONPATH = (Get-Location).Path
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest tests/scripts/test_analyze_gold_bottlenecks.py -q
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m ruff check scripts/analyze_gold_bottlenecks.py tests/scripts/test_analyze_gold_bottlenecks.py
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m mypy src scripts/analyze_gold_bottlenecks.py
```

Expected: all focused tests pass, Ruff reports no issues, and mypy reports no issues.

+ [x] **Step 5: Commit the behavior change**

```powershell
git add -- scripts/analyze_gold_bottlenecks.py tests/scripts/test_analyze_gold_bottlenecks.py
git commit -m "fix: accept DOI exact endpoint resolutions"
```

---

### Task 2: Record the offline contract and verify the repository

**Files:**
- Modify: `HANDOFF.md:22-53`
- Modify: `docs/retrieval-roadmap.md:25-38`

**Interfaces:**
- Consumes: the tested classifier behavior from Task 1.
- Produces: active project-state documents that distinguish the implemented offline contract from the unchanged historical online report.

+ [x] **Step 1: Update active project-state documentation**

Record only these facts:

- The DOI exact-endpoint acceptance contract is implemented and covered by synthetic offline tests.
- HTTP 200 plus a valid OpenAlex Work ID is available even when top-level DOI is missing, different, or unparseable.
- OpenAlex-ID requests remain strict.
- `docs/evidence/gold-bottleneck-attribution-2026-08-09.json` and its Markdown report remain historical evidence from the one approved probe and are not rewritten.
- No new diagnostic direction is selected without a separately authorized online rerun.

Remove the obsolete next-step wording that says the DOI acceptance contract is still undecided. Do not claim `diagnostic_complete=true`.

+ [x] **Step 2: Run full offline verification**

Run:

```powershell
$env:PYTHONPATH = (Get-Location).Path
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest -q
$trackedPython = @(git ls-files '*.py')
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m ruff check -- $trackedPython
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m mypy src scripts/analyze_gold_bottlenecks.py
git diff --check
```

Observed: `1923 passed / 36 skipped / 1 environment failure`; a process-local Git ownership compatibility setting removes the clone failure, while the remaining failure is Windows GBK-locale decoding of `uv build` UTF-8 output. Focused tests, Ruff, mypy (93 files), and `git diff --check` are clean. The full-suite green gate remains environment-blocked.

+ [x] **Step 3: Confirm forbidden side effects did not occur**

Run:

```powershell
git status --short
git diff -- docs/evidence/gold-bottleneck-attribution-2026-08-09.json docs/gold-bottleneck-attribution-2026-08-09.md runs/candidate.lock.yaml
```

Expected: the historical evidence and candidate lock have no diff; existing untracked `data/budget_ledger.sqlite3` and `deliverables/` remain untouched.

+ [x] **Step 4: Commit documentation and verification state**

```powershell
git add -- HANDOFF.md docs/retrieval-roadmap.md
git commit -m "docs: record DOI acceptance contract"
```
