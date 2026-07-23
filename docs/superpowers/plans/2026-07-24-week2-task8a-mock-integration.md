# Week 2 Task 8A Mock Integration Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete offline mock orchestration edge cases, adapt minimal results to the fixed structured response, and emit deterministic synthetic prediction JSONL.

**Architecture:** Keep external-call sequencing in `MockSearchOrchestrator`, add a pure response adapter in `pipeline/response.py`, and add a small prediction boundary in `evaluation/predictions.py`. Reuse the existing strict domain models, official adapter, and atomic JSONL writer.

**Tech Stack:** Python 3.11+, Pydantic 2, pytest, Ruff, mypy, uv.

## Global Constraints

- Run every Python command with `--no-env-file`.
- Do not call real OpenAlex, Semantic Scholar, or LLM endpoints.
- Do not read `.env`, private annotations, gold data, frozen splits, manifest state, real queries, or `docs/superpowers/specs/2026-07-15-task2-evaluation-design.md`.
- Do not add HTTP endpoints, API-server startup, scoring, ranking evidence, citation graphs, or new evaluation schemas.
- Use only synthetic in-memory fixtures and pytest temporary directories.
- Do not claim Week 1 or Week 2 gates, recall, F1, cost, or model-comparison targets.

## File Structure

- Modify `src/paper_search/pipeline/orchestrator.py`: settle known analyzer results, report structured analyzer errors, and fail closed on analyzer exceptions with unknown usage.
- Create `src/paper_search/pipeline/response.py`: pure `MinimalSearchResult` to `StructuredSearchResponse` conversion.
- Create `src/paper_search/evaluation/predictions.py`: response-to-prediction conversion, duplicate-query validation, and atomic JSONL writing.
- Modify `src/paper_search/evaluation/__init__.py`: expose the new prediction helpers.
- Modify `tests/integration/test_orchestrator.py`: cover structured analyzer failure, analyzer exception, all-empty retrieval, and soft stop.
- Create `tests/unit/test_response.py`: verify the response adapter and deliberate absence of fabricated data.
- Create `tests/evaluation/test_predictions.py`: verify deterministic bytes, empty output, duplicate rejection, strict parsing, and official-adapter round trip.

---

### Task 1: Mock Orchestration Failure and Stop Semantics

**Files:**
- Modify: `tests/integration/test_orchestrator.py`
- Modify: `src/paper_search/pipeline/orchestrator.py`

**Interfaces:**
- Consumes: `Analyzer`, `HardBudgetController`, `ProviderResult`, and `rule_fallback`.
- Produces: `MockSearchOrchestrator.run(...) -> MinimalSearchResult` with deterministic structured failure behavior.

- [ ] **Step 1: Add failing analyzer-error and analyzer-exception tests**

Add fakes whose complete behavior is explicit:

```python
class FailedAnalyzer:
    def __init__(self, events: list[str], *, raises: bool = False) -> None:
        self.events = events
        self.raises = raises

    async def __call__(
        self, query: str, _: object
    ) -> ProviderResult[dict[str, object]]:
        self.events.append("analyze")
        if self.raises:
            raise TimeoutError("synthetic analyzer timeout")
        return _result(
            "llm",
            {},
            UsageActual(llm_calls=1, cost_cny=0.1),
            failed=True,
        )
```

Add two tests:

```python
def test_orchestrator_uses_rule_fallback_for_structured_analyzer_error() -> None:
    events: list[str] = []
    orchestrator = _orchestrator(events, analyzer=FailedAnalyzer(events))

    result = asyncio.run(
        orchestrator.run("graph retrieval", max_provider_results=5)
    )

    assert result.query_analysis.query_spec.ambiguities == ["rules_only_fallback"]
    assert result.warnings[0] == "analysis: analyzer returned errors"
    assert result.is_partial is True
    assert "openalex" in events


def test_orchestrator_fails_closed_on_analyzer_exception_without_calling_provider() -> None:
    events: list[str] = []
    controller = HardBudgetController(_budget())
    orchestrator = _orchestrator(
        events,
        analyzer=FailedAnalyzer(events, raises=True),
        controller=controller,
    )

    result = asyncio.run(
        orchestrator.run("graph retrieval", max_provider_results=5)
    )

    assert events == ["analyze"]
    assert result.query_analysis.query_spec.ambiguities == ["rules_only_fallback"]
    assert result.stop_reason == "hard_stop"
    assert result.is_partial is True
    assert result.warnings == ["analysis: dependency failure"]
    assert controller.stop_status() == "hard_stop"
```

- [ ] **Step 2: Verify the new analyzer tests are RED**

Run:

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest `
  tests/integration/test_orchestrator.py `
  -k "structured_analyzer_error or analyzer_exception" -q
```

Expected: the structured error lacks its warning and the analyzer exception escapes.

- [ ] **Step 3: Implement minimal analyzer failure handling**

Replace the analyzer call/settlement block with:

```python
try:
    analysis_result = await self._analyzer(query, analysis_reservation)
except Exception:
    self._controller.fail_closed(analysis_reservation)
    return self._result(
        self._fallback(query),
        [],
        provider_results,
        trace,
        "hard_stop",
        True,
        ["analysis: dependency failure"],
    )
try:
    self._controller.settle(analysis_reservation, analysis_result.usage)
except ReservationError:
    self._controller.fail_closed(analysis_reservation)
    raise
if analysis_result.errors:
    warnings.append("analysis: analyzer returned errors")
analysis = await self._parser.parse(query, analysis_result)
```

Do not place settlement inside the broad dependency-exception handler: reservation
overruns remain programming/accounting errors and must still raise.

- [ ] **Step 4: Verify analyzer tests are GREEN**

Run the focused command from Step 2.

Expected: both tests pass.

- [ ] **Step 5: Add failing empty-result and soft-stop tests**

Extend `FakeProvider` with an `empty: bool = False` constructor option and return
`[]` when `empty` is true. Add:

```python
def test_orchestrator_treats_all_empty_provider_results_as_completed() -> None:
    events: list[str] = []
    orchestrator = _orchestrator(
        events,
        providers={"openalex": FakeProvider("openalex", events, empty=True)},
    )

    result = asyncio.run(
        orchestrator.run("graph retrieval", max_provider_results=5)
    )

    assert result.papers == []
    assert result.stop_reason == "completed"
    assert result.is_partial is False
    assert result.warnings == []


def test_orchestrator_soft_stop_prevents_provider_calls() -> None:
    events: list[str] = []
    orchestrator = _orchestrator(
        events,
        analysis_estimate=UsageEstimate(
            llm_calls=1, cost_cny=0.1, elapsed_ms=1_000
        ),
        analyzer=FakeAnalyzer(events, elapsed_ms=1_000),
    )

    result = asyncio.run(
        orchestrator.run("graph retrieval", max_provider_results=5)
    )

    assert events == ["analyze"]
    assert result.stop_reason == "soft_stop"
    assert result.is_partial is True
    assert result.warnings == ["openalex: budget unavailable"]
```

Refactor the existing fakes only enough to accept `elapsed_ms`, `empty`, and a
shared `_orchestrator(...)` test factory; do not add production-only test hooks.

- [ ] **Step 6: Verify empty and soft-stop tests**

Run:

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest `
  tests/integration/test_orchestrator.py -q
```

Expected: all orchestrator tests pass. Empty and soft-stop behavior may already
pass without another production change; their prior RED evidence is the absence
of the test contract, not a forced production failure.

- [ ] **Step 7: Run scoped quality checks and commit**

Run:

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file ruff check `
  src/paper_search/pipeline/orchestrator.py `
  tests/integration/test_orchestrator.py
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file mypy `
  src/paper_search/pipeline/orchestrator.py
git diff --check
```

Commit:

```powershell
git add src/paper_search/pipeline/orchestrator.py tests/integration/test_orchestrator.py
git commit -m "test: harden mock orchestration edges"
```

---

### Task 2: Structured Response Adapter

**Files:**
- Create: `tests/unit/test_response.py`
- Create: `src/paper_search/pipeline/response.py`

**Interfaces:**
- Consumes: `MinimalSearchResult`.
- Produces: `to_structured_response(result, *, query_id: str, git_sha: str) -> StructuredSearchResponse`.

- [ ] **Step 1: Write the failing pure-adapter test**

Create a synthetic `MinimalSearchResult` with two `Paper` objects and assert:

```python
response = to_structured_response(
    minimal_result,
    query_id="query-1",
    git_sha="abc1234",
)

assert response.query_id == "query-1"
assert response.selected_paper_ids == ["openalex:W1", "s2:S1"]
assert response.query_analysis == minimal_result.query_analysis
assert response.search_trace == minimal_result.trace
assert response.usage == minimal_result.usage
assert response.stop_reason == minimal_result.stop_reason
assert response.is_partial == minimal_result.is_partial
assert response.warnings == minimal_result.warnings
assert response.config_hash == minimal_result.config_hash
assert response.git_sha == "abc1234"
assert response.high_relevance == []
assert response.partial_relevance == []
assert response.citation_edges == []
```

Also assert Pydantic rejects blank `query_id` or `git_sha`; do not add duplicate
validation outside the existing model.

- [ ] **Step 2: Verify the adapter test is RED**

Run:

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest `
  tests/unit/test_response.py -q
```

Expected: import failure because `pipeline/response.py` does not exist.

- [ ] **Step 3: Implement the pure adapter**

Create:

```python
"""Pure conversion from mock orchestration output to the public response model."""

from paper_search.domain.models import StructuredSearchResponse
from paper_search.pipeline.orchestrator import MinimalSearchResult


def to_structured_response(
    result: MinimalSearchResult,
    *,
    query_id: str,
    git_sha: str,
) -> StructuredSearchResponse:
    """Preserve known result data without inventing ranking evidence."""
    return StructuredSearchResponse(
        query_id=query_id,
        query_analysis=result.query_analysis,
        selected_paper_ids=[paper.canonical_id for paper in result.papers],
        high_relevance=[],
        partial_relevance=[],
        citation_edges=[],
        search_trace=result.trace,
        usage=result.usage,
        stop_reason=result.stop_reason,
        is_partial=result.is_partial,
        warnings=result.warnings,
        config_hash=result.config_hash,
        git_sha=git_sha,
    )
```

- [ ] **Step 4: Verify adapter GREEN and commit**

Run:

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest `
  tests/unit/test_response.py tests/unit/test_models.py -q
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file ruff check `
  src/paper_search/pipeline/response.py tests/unit/test_response.py
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file mypy `
  src/paper_search/pipeline/response.py
git diff --check
```

Commit:

```powershell
git add src/paper_search/pipeline/response.py tests/unit/test_response.py
git commit -m "feat: adapt mock results to structured responses"
```

---

### Task 3: Synthetic Prediction JSONL Contract

**Files:**
- Create: `tests/evaluation/test_predictions.py`
- Create: `src/paper_search/evaluation/predictions.py`
- Modify: `src/paper_search/evaluation/__init__.py`

**Interfaces:**
- Consumes: `StructuredSearchResponse`.
- Produces: `prediction_from_response(response) -> InternalPredictionRecord`.
- Produces: `write_response_predictions(path, responses) -> list[InternalPredictionRecord]`.

- [ ] **Step 1: Write failing conversion and serialization tests**

Use synthetic `StructuredSearchResponse` instances only. Assert:

```python
records = write_response_predictions(
    output,
    [response_one, response_two_with_no_selected_ids],
)

assert records == [
    InternalPredictionRecord(
        query_id="q1",
        selected_paper_ids=["openalex:W1", "s2:S1"],
    ),
    InternalPredictionRecord(query_id="q2", selected_paper_ids=[]),
]
assert output.read_bytes() == (
    b'{"query_id":"q1","selected_paper_ids":["openalex:W1","s2:S1"]}\n'
    b'{"query_id":"q2","selected_paper_ids":[]}\n'
)
assert read_jsonl(output, InternalPredictionRecord) == records
assert [
    adapt_prediction_record(record) for record in records
] == [
    PredictionRecord(
        query_id="q1",
        predicted_paper_ids=["openalex:W1", "s2:S1"],
    ),
    PredictionRecord(query_id="q2", predicted_paper_ids=[]),
]
```

Add a second test that passes two responses with the same `query_id`, expects
`ValueError("duplicate query_id: q1")`, and asserts the output path was not
created.

- [ ] **Step 2: Verify prediction tests are RED**

Run:

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest `
  tests/evaluation/test_predictions.py -q
```

Expected: import failure because `evaluation/predictions.py` does not exist.

- [ ] **Step 3: Implement conversion, preflight, and atomic writing**

Create:

```python
"""Synthetic structured-response prediction serialization."""

from collections.abc import Sequence
from pathlib import Path

from paper_search.domain.models import StructuredSearchResponse
from paper_search.evaluation.dataset import write_jsonl_atomic
from paper_search.evaluation.official_adapter import InternalPredictionRecord


def prediction_from_response(
    response: StructuredSearchResponse,
) -> InternalPredictionRecord:
    """Copy the fixed prediction fields without scores or label access."""
    return InternalPredictionRecord(
        query_id=response.query_id,
        selected_paper_ids=response.selected_paper_ids,
    )


def write_response_predictions(
    path: Path,
    responses: Sequence[StructuredSearchResponse],
) -> list[InternalPredictionRecord]:
    """Validate a batch before atomically writing deterministic JSONL."""
    records = [prediction_from_response(response) for response in responses]
    seen: set[str] = set()
    for record in records:
        if record.query_id in seen:
            raise ValueError(f"duplicate query_id: {record.query_id}")
        seen.add(record.query_id)
    write_jsonl_atomic(path, records)
    return records
```

Export both functions from `paper_search.evaluation.__init__`.

- [ ] **Step 4: Verify prediction contract GREEN and commit**

Run:

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest `
  tests/evaluation/test_predictions.py `
  tests/evaluation/test_official_adapter.py `
  tests/evaluation/test_dataset.py -q
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file ruff check `
  src/paper_search/evaluation/predictions.py `
  src/paper_search/evaluation/__init__.py `
  tests/evaluation/test_predictions.py
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file mypy `
  src/paper_search/evaluation/predictions.py
git diff --check
```

Commit:

```powershell
git add src/paper_search/evaluation/predictions.py `
  src/paper_search/evaluation/__init__.py `
  tests/evaluation/test_predictions.py
git commit -m "feat: write synthetic response predictions"
```

---

### Task 4: Final Offline Verification and Scope Audit

**Files:**
- Verify all Task 8A changes; do not modify protected or data paths.

**Interfaces:**
- Consumes: all three Task 8A deliverables.
- Produces: fresh verification evidence and a review-ready clean branch.

- [ ] **Step 1: Run complete offline tests**

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest `
  -m "not online" -q
```

Expected: all collected offline tests pass and the online test is deselected.

- [ ] **Step 2: Run complete static checks**

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file ruff check .
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file mypy src
git diff --check
```

Expected: all commands exit zero.

- [ ] **Step 3: Audit scope and sensitive literals**

Compare against design commit `091f91a`:

```powershell
git diff --name-only 091f91a...HEAD
git status --short
rg -n --glob '!uv.lock' `
  'sk-[A-Za-z0-9_-]{16,}|OPENAI_API_KEY\s*=|SEMANTIC_SCHOLAR_API_KEY\s*=' `
  src tests docs/superpowers/plans `
  docs/superpowers/specs/2026-07-24-week2-task8a-mock-integration-design.md
```

Expected changed paths are limited to the files listed by this plan; status is
clean; credential-literal scan has no matches.

- [ ] **Step 4: Request independent review**

Review `091f91a...HEAD` against the design and this plan. Fix every Critical and
Important finding with a failing regression test first, rerun the complete
verification, and commit review fixes separately.

## Plan Self-Review

- Spec coverage: Tasks 1–3 map one-to-one to the three approved deliverables;
  Task 4 covers final verification and review.
- Placeholder scan: the plan contains no deferred implementation placeholders.
- Type consistency: response conversion consumes `MinimalSearchResult` and returns
  the existing `StructuredSearchResponse`; prediction conversion consumes that
  response and returns the existing `InternalPredictionRecord`.
- Scope: HTTP, real providers, real evaluation data, fabricated scores, and
  protected files remain excluded.
