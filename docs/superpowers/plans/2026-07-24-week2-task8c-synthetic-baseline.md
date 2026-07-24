# Week 2 Task 8C Synthetic Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic offline synthetic-query batch that runs through a fresh mock orchestrator per query, converts structured responses to fixed prediction records, isolates query failures, and writes byte-stable `predictions.jsonl`.

**Architecture:** Keep formal evaluation code untouched. Extend the existing prediction serializer with a record-level atomic writer, add a small batch core over the existing `SearchRequest`/`StructuredSearchResponse` service boundary, and place fixed analyzer/provider/orchestrator composition in a separate mock-only module. Expose the code-defined catalog through `python -m paper_search.evaluation.synthetic_baseline --output PATH`.

**Tech Stack:** Python 3.11, asyncio, Pydantic 2, existing `MockApiSearchService`, existing `MockSearchOrchestrator`, pytest, Ruff, mypy, uv.

## Global Constraints

- Use only fixed synthetic queries and injected mock dependencies.
- Emit only synthetic `predictions.jsonl`.
- Do not read annotations, gold data, a dev split, frozen partitions, manifests, labels, `.env`, credentials, or real queries.
- Do not call OpenAlex, Semantic Scholar, an LLM, HTTP, or any other external service.
- Do not calculate or report Recall, Precision, F1, ranking quality, or a formal gate.
- Preserve catalog order and exactly one prediction record per input query.
- A query-level exception becomes an empty prediction and must not terminate the batch.
- Catalog/preflight and filesystem failures remain batch-level failures.
- Run all commands with `--no-env-file`; do not run the online test.
- Follow strict RED-GREEN-REFACTOR for every production change.

## File Structure

- Modify `src/paper_search/evaluation/predictions.py`: validate and atomically write already-converted prediction records.
- Create `src/paper_search/evaluation/synthetic_baseline.py`: code-defined catalog, preflight, sequential batch runner, parser, and CLI entry point.
- Create `src/paper_search/evaluation/synthetic_mocks.py`: fixed mock analyzer/provider and fresh-orchestrator factory.
- Modify `src/paper_search/evaluation/__init__.py`: export only non-executable batch interfaces lazily, preserving warning-free `python -m` execution.
- Modify `tests/evaluation/test_predictions.py`: record-writer contract tests.
- Create `tests/evaluation/test_synthetic_baseline.py`: batch preflight and failure-isolation unit tests.
- Create `tests/integration/test_synthetic_baseline.py`: real mock stack, deterministic artifact, and fresh-controller tests.
- Create `tests/evaluation/test_synthetic_baseline_cli.py`: subprocess CLI boundary tests.

---

### Task 1: Deterministic Prediction Record Writer

**Files:**
- Modify: `src/paper_search/evaluation/predictions.py`
- Modify: `src/paper_search/evaluation/__init__.py`
- Modify: `tests/evaluation/test_predictions.py`

**Interfaces:**
- Consumes: `Sequence[InternalPredictionRecord]`, `Path`, and existing `write_jsonl_atomic`.
- Produces: `write_prediction_records(path: Path, records: Sequence[InternalPredictionRecord]) -> list[InternalPredictionRecord]`.
- Preserves: `write_response_predictions(path, responses)` behavior by delegating to the new record writer.

- [ ] **Step 1: Write failing record-writer tests**

Add the new import and these tests to `tests/evaluation/test_predictions.py`:

```python
from paper_search.evaluation.predictions import (
    write_prediction_records,
    write_response_predictions,
)


def test_write_prediction_records_preserves_order_and_bytes(tmp_path: Path) -> None:
    output = tmp_path / "predictions.jsonl"
    records = [
        InternalPredictionRecord(
            query_id="synthetic-q2",
            selected_paper_ids=["s2:S2"],
        ),
        InternalPredictionRecord(
            query_id="synthetic-q1",
            selected_paper_ids=[],
        ),
    ]

    written = write_prediction_records(output, records)

    assert written == records
    assert written is not records
    assert output.read_bytes() == (
        b'{"query_id":"synthetic-q2","selected_paper_ids":["s2:S2"]}\n'
        b'{"query_id":"synthetic-q1","selected_paper_ids":[]}\n'
    )


def test_write_prediction_records_rejects_duplicate_before_write(
    tmp_path: Path,
) -> None:
    output = tmp_path / "predictions.jsonl"
    output.write_bytes(b"preserve-me\n")

    with pytest.raises(
        ValueError,
        match=r"^duplicate query_id: synthetic-q1$",
    ):
        write_prediction_records(
            output,
            [
                InternalPredictionRecord(
                    query_id="synthetic-q1",
                    selected_paper_ids=[],
                ),
                InternalPredictionRecord(
                    query_id="synthetic-q1",
                    selected_paper_ids=["openalex:W1"],
                ),
            ],
        )

    assert output.read_bytes() == b"preserve-me\n"
```

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest `
  tests/evaluation/test_predictions.py -k prediction_records -v
```

Expected: collection fails because `write_prediction_records` is not defined.

- [ ] **Step 3: Implement the record writer and delegate response writing**

Replace `src/paper_search/evaluation/predictions.py` with:

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


def write_prediction_records(
    path: Path,
    records: Sequence[InternalPredictionRecord],
) -> list[InternalPredictionRecord]:
    """Validate and atomically write ordered deterministic prediction records."""
    ordered = list(records)
    seen: set[str] = set()
    for record in ordered:
        if record.query_id in seen:
            raise ValueError(f"duplicate query_id: {record.query_id}")
        seen.add(record.query_id)
    write_jsonl_atomic(path, ordered)
    return ordered


def write_response_predictions(
    path: Path,
    responses: Sequence[StructuredSearchResponse],
) -> list[InternalPredictionRecord]:
    """Convert structured responses and write deterministic predictions."""
    return write_prediction_records(
        path,
        [prediction_from_response(response) for response in responses],
    )
```

Add `write_prediction_records` to the eager imports and `__all__` in
`src/paper_search/evaluation/__init__.py`:

```python
from paper_search.evaluation.predictions import (
    prediction_from_response,
    write_prediction_records,
    write_response_predictions,
)
```

```python
    "write_prediction_records",
```

- [ ] **Step 4: Run focused prediction tests and verify GREEN**

Run:

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest `
  tests/evaluation/test_predictions.py -v
```

Expected: all tests in the file pass.

- [ ] **Step 5: Run focused static checks**

Run:

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file ruff check `
  src/paper_search/evaluation/predictions.py `
  src/paper_search/evaluation/__init__.py `
  tests/evaluation/test_predictions.py
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file mypy `
  src/paper_search/evaluation/predictions.py
```

Expected: Ruff and mypy pass.

- [ ] **Step 6: Commit Task 1**

```powershell
git add -- `
  src/paper_search/evaluation/predictions.py `
  src/paper_search/evaluation/__init__.py `
  tests/evaluation/test_predictions.py
git commit -m "feat: write deterministic prediction records"
```

---

### Task 2: Synthetic Catalog and Failure-Isolating Batch Core

**Files:**
- Create: `src/paper_search/evaluation/synthetic_baseline.py`
- Create: `tests/evaluation/test_synthetic_baseline.py`

**Interfaces:**
- Consumes: `SearchRequest`, an injected callable service, and `write_prediction_records`.
- Produces:
  - `SYNTHETIC_QUERIES: tuple[SearchRequest, ...]`
  - `validate_synthetic_requests(requests: Sequence[SearchRequest]) -> tuple[SearchRequest, ...]`
  - `run_synthetic_baseline(requests, *, search_service, output) -> list[InternalPredictionRecord]`
  - `SyntheticSearchService` protocol with `async __call__(SearchRequest) -> StructuredSearchResponse`.

- [ ] **Step 1: Write failing catalog, preflight, and batch tests**

Create `tests/evaluation/test_synthetic_baseline.py`:

```python
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from paper_search.api.contracts import SearchRequest
from paper_search.domain.models import (
    QueryAnalysisResult,
    QuerySpec,
    SearchPlan,
    StructuredSearchResponse,
    SubQuery,
    UsageActual,
)
from paper_search.evaluation.official_adapter import InternalPredictionRecord
from paper_search.evaluation.synthetic_baseline import (
    SYNTHETIC_QUERIES,
    run_synthetic_baseline,
    validate_synthetic_requests,
)


def _response(request: SearchRequest, paper_ids: list[str]) -> StructuredSearchResponse:
    spec = QuerySpec(
        original_query=request.query,
        research_goal="synthetic baseline",
    )
    return StructuredSearchResponse(
        query_id=request.query_id,
        query_analysis=QueryAnalysisResult(
            query_spec=spec,
            search_plan=SearchPlan(
                subqueries=[
                    SubQuery(
                        query_id=f"{request.query_id}-sq-1",
                        text=request.query,
                        query_type="exact",
                        target_constraints=[],
                        priority=1,
                        provider_hint="either",
                    )
                ],
                inherited_hard_filters={},
                rationale="synthetic",
            ),
        ),
        selected_paper_ids=paper_ids,
        high_relevance=[],
        partial_relevance=[],
        citation_edges=[],
        search_trace=[],
        usage=UsageActual(),
        stop_reason="completed",
        is_partial=False,
        warnings=[],
        config_hash="sha256:" + "a" * 64,
        git_sha="synthetic-task8c",
    )


class RecordingService:
    def __init__(self, failing_query_id: str | None = None) -> None:
        self.failing_query_id = failing_query_id
        self.calls: list[str] = []

    async def __call__(
        self,
        request: SearchRequest,
    ) -> StructuredSearchResponse:
        self.calls.append(request.query_id)
        if request.query_id == self.failing_query_id:
            raise TimeoutError("synthetic failure must not be persisted")
        return _response(request, [f"openalex:W{len(self.calls)}"])


def _requests() -> tuple[SearchRequest, ...]:
    return (
        SearchRequest(query_id="synthetic-q1", query="synthetic one"),
        SearchRequest(query_id="synthetic-q2", query="synthetic two"),
        SearchRequest(query_id="synthetic-q3", query="synthetic three"),
    )


def test_catalog_is_fixed_strict_and_unique() -> None:
    assert SYNTHETIC_QUERIES
    assert all(request.include_trace is False for request in SYNTHETIC_QUERIES)
    assert all("synthetic" in request.query.casefold() for request in SYNTHETIC_QUERIES)
    assert len({request.query_id for request in SYNTHETIC_QUERIES}) == len(
        SYNTHETIC_QUERIES
    )
    assert validate_synthetic_requests(SYNTHETIC_QUERIES) is SYNTHETIC_QUERIES


def test_validate_synthetic_requests_rejects_empty_and_duplicate() -> None:
    with pytest.raises(ValueError, match=r"^synthetic query catalog must not be empty$"):
        validate_synthetic_requests(())

    duplicate = SearchRequest(query_id="synthetic-q1", query="synthetic duplicate")
    with pytest.raises(ValueError, match=r"^duplicate query_id: synthetic-q1$"):
        validate_synthetic_requests((_requests()[0], duplicate))


def test_batch_keeps_order_and_continues_after_query_exception(
    tmp_path: Path,
) -> None:
    service = RecordingService(failing_query_id="synthetic-q2")
    output = tmp_path / "predictions.jsonl"

    records = asyncio.run(
        run_synthetic_baseline(
            _requests(),
            search_service=service,
            output=output,
        )
    )

    assert service.calls == ["synthetic-q1", "synthetic-q2", "synthetic-q3"]
    assert records == [
        InternalPredictionRecord(
            query_id="synthetic-q1",
            selected_paper_ids=["openalex:W1"],
        ),
        InternalPredictionRecord(
            query_id="synthetic-q2",
            selected_paper_ids=[],
        ),
        InternalPredictionRecord(
            query_id="synthetic-q3",
            selected_paper_ids=["openalex:W3"],
        ),
    ]
    assert output.read_bytes() == (
        b'{"query_id":"synthetic-q1","selected_paper_ids":["openalex:W1"]}\n'
        b'{"query_id":"synthetic-q2","selected_paper_ids":[]}\n'
        b'{"query_id":"synthetic-q3","selected_paper_ids":["openalex:W3"]}\n'
    )


def test_preflight_failure_does_not_call_service_or_replace_output(
    tmp_path: Path,
) -> None:
    service = RecordingService()
    output = tmp_path / "predictions.jsonl"
    output.write_bytes(b"preserve-me\n")
    duplicate = SearchRequest(query_id="synthetic-q1", query="synthetic duplicate")

    with pytest.raises(ValueError, match=r"^duplicate query_id: synthetic-q1$"):
        asyncio.run(
            run_synthetic_baseline(
                (_requests()[0], duplicate),
                search_service=service,
                output=output,
            )
        )

    assert service.calls == []
    assert output.read_bytes() == b"preserve-me\n"
```

- [ ] **Step 2: Run the new file and verify RED**

Run:

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest `
  tests/evaluation/test_synthetic_baseline.py -v
```

Expected: collection fails because
`paper_search.evaluation.synthetic_baseline` does not exist.

- [ ] **Step 3: Implement the catalog, protocol, preflight, and batch**

Create `src/paper_search/evaluation/synthetic_baseline.py` with:

```python
"""Deterministic offline synthetic baseline batch."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from paper_search.api.contracts import SearchRequest
from paper_search.domain.models import StructuredSearchResponse
from paper_search.evaluation.official_adapter import InternalPredictionRecord
from paper_search.evaluation.predictions import (
    prediction_from_response,
    write_prediction_records,
)


SYNTHETIC_QUERIES = (
    SearchRequest(
        query_id="synthetic-graph-retrieval",
        query="Synthetic graph retrieval research",
        budget_profile="low",
        include_trace=False,
    ),
    SearchRequest(
        query_id="synthetic-empty-result",
        query="Synthetic zero-result literature search",
        budget_profile="balanced",
        include_trace=False,
    ),
    SearchRequest(
        query_id="synthetic-budget-path",
        query="Synthetic budget-aware scholarly search",
        budget_profile="low",
        include_trace=False,
    ),
)


class SyntheticSearchService(Protocol):
    async def __call__(
        self,
        request: SearchRequest,
    ) -> StructuredSearchResponse: ...


def validate_synthetic_requests(
    requests: Sequence[SearchRequest],
) -> tuple[SearchRequest, ...]:
    """Return an immutable validated catalog before any query executes."""
    if not requests:
        raise ValueError("synthetic query catalog must not be empty")
    ordered = requests if isinstance(requests, tuple) else tuple(requests)
    seen: set[str] = set()
    for request in ordered:
        if request.query_id in seen:
            raise ValueError(f"duplicate query_id: {request.query_id}")
        seen.add(request.query_id)
    return ordered


async def run_synthetic_baseline(
    requests: Sequence[SearchRequest],
    *,
    search_service: SyntheticSearchService,
    output: Path,
) -> list[InternalPredictionRecord]:
    """Run an ordered synthetic batch and isolate query-level failures."""
    ordered = validate_synthetic_requests(requests)
    records: list[InternalPredictionRecord] = []
    for request in ordered:
        try:
            response = await search_service(request)
            record = prediction_from_response(response)
        except Exception:
            record = InternalPredictionRecord(
                query_id=request.query_id,
                selected_paper_ids=[],
            )
        records.append(record)
    return write_prediction_records(output, records)
```

- [ ] **Step 4: Run unit tests and verify GREEN**

Run:

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest `
  tests/evaluation/test_synthetic_baseline.py `
  tests/evaluation/test_predictions.py -v
```

Expected: all selected tests pass.

- [ ] **Step 5: Run focused static checks**

Run:

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file ruff check `
  src/paper_search/evaluation/synthetic_baseline.py `
  tests/evaluation/test_synthetic_baseline.py
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file mypy `
  src/paper_search/evaluation/synthetic_baseline.py
```

Expected: Ruff and mypy pass.

- [ ] **Step 6: Commit Task 2**

```powershell
git add -- `
  src/paper_search/evaluation/synthetic_baseline.py `
  tests/evaluation/test_synthetic_baseline.py
git commit -m "feat: isolate synthetic baseline queries"
```

---

### Task 3: Fixed Mock Composition and Real-Orchestrator Integration

**Files:**
- Create: `src/paper_search/evaluation/synthetic_mocks.py`
- Create: `tests/integration/test_synthetic_baseline.py`

**Interfaces:**
- Consumes: `BudgetProfile`, `HardBudgetController`, `MockSearchOrchestrator`, and `MockApiSearchService`.
- Produces:
  - `SyntheticOrchestratorFactory`
  - `build_synthetic_search_service() -> MockApiSearchService`
- Guarantees: fixed payloads and identities, no environment/config/network reads, and a fresh controller per factory call.

- [ ] **Step 1: Write failing real-stack integration tests**

Create `tests/integration/test_synthetic_baseline.py`:

```python
from __future__ import annotations

import asyncio
from pathlib import Path

from paper_search.evaluation.official_adapter import InternalPredictionRecord
from paper_search.evaluation.synthetic_baseline import (
    SYNTHETIC_QUERIES,
    run_synthetic_baseline,
)
from paper_search.evaluation.synthetic_mocks import (
    SyntheticOrchestratorFactory,
    build_synthetic_search_service,
)


def test_real_mock_stack_writes_only_ordered_synthetic_predictions(
    tmp_path: Path,
) -> None:
    output = tmp_path / "artifacts" / "predictions.jsonl"
    service = build_synthetic_search_service()

    records = asyncio.run(
        run_synthetic_baseline(
            SYNTHETIC_QUERIES,
            search_service=service,
            output=output,
        )
    )

    assert [record.query_id for record in records] == [
        request.query_id for request in SYNTHETIC_QUERIES
    ]
    assert records[0].selected_paper_ids == ["openalex:W100"]
    assert records[1] == InternalPredictionRecord(
        query_id="synthetic-empty-result",
        selected_paper_ids=[],
    )
    assert records[2].selected_paper_ids == ["openalex:W100"]
    assert [path.name for path in output.parent.iterdir()] == ["predictions.jsonl"]


def test_real_mock_stack_repeated_runs_are_byte_identical(
    tmp_path: Path,
) -> None:
    output = tmp_path / "predictions.jsonl"

    asyncio.run(
        run_synthetic_baseline(
            SYNTHETIC_QUERIES,
            search_service=build_synthetic_search_service(),
            output=output,
        )
    )
    first = output.read_bytes()
    asyncio.run(
        run_synthetic_baseline(
            SYNTHETIC_QUERIES,
            search_service=build_synthetic_search_service(),
            output=output,
        )
    )

    assert output.read_bytes() == first
    assert b"recall" not in first.casefold()
    assert b"f1" not in first.casefold()


def test_factory_creates_fresh_budget_controller_per_request(
    tmp_path: Path,
) -> None:
    factory = SyntheticOrchestratorFactory()
    service = build_synthetic_search_service(factory=factory)

    asyncio.run(
        run_synthetic_baseline(
            SYNTHETIC_QUERIES,
            search_service=service,
            output=tmp_path / "predictions.jsonl",
        )
    )

    assert len(factory.controllers) == len(SYNTHETIC_QUERIES)
    assert len({id(controller) for controller in factory.controllers}) == len(
        SYNTHETIC_QUERIES
    )
```

- [ ] **Step 2: Run the integration file and verify RED**

Run:

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest `
  tests/integration/test_synthetic_baseline.py -v
```

Expected: collection fails because
`paper_search.evaluation.synthetic_mocks` does not exist.

- [ ] **Step 3: Implement fixed analyzer, provider, factory, and service builder**

Create `src/paper_search/evaluation/synthetic_mocks.py`:

```python
"""Fixed offline dependencies for the Task 8C synthetic baseline."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TypeVar

from paper_search.api.contracts import BudgetProfile
from paper_search.api.service import MockApiSearchService
from paper_search.control.budget import HardBudgetController
from paper_search.domain.models import (
    BudgetReservation,
    Paper,
    ProviderResult,
    SearchBudget,
    UsageActual,
    UsageEstimate,
)
from paper_search.pipeline.orchestrator import MockSearchOrchestrator


_REQUESTED_AT = datetime(2026, 7, 24, tzinfo=UTC).isoformat()
_CONFIG_HASH = "sha256:" + "c" * 64
ResultT = TypeVar("ResultT")


def _result(
    provider: str,
    data: ResultT,
    usage: UsageActual,
) -> ProviderResult[ResultT]:
    return ProviderResult[ResultT](
        data=data,
        usage=usage,
        provenance={
            "provider": provider,
            "endpoint": "/synthetic",
            "model_id": "task8c-fixed-mock",
            "requested_at": _REQUESTED_AT,
            "response_hash": f"sha256:task8c-{provider}",
        },
        cache_hit=False,
        latency_ms=0,
        errors=[],
    )


class SyntheticAnalyzer:
    async def __call__(
        self,
        query: str,
        reservation: BudgetReservation,
    ) -> ProviderResult[dict[str, object]]:
        del reservation
        return _result(
            "llm",
            {
                "query_spec": {
                    "original_query": query,
                    "research_goal": "synthetic baseline",
                },
                "search_plan": {
                    "subqueries": [
                        {
                            "query_id": "synthetic-sq-1",
                            "text": query,
                            "query_type": "exact",
                            "target_constraints": [],
                            "priority": 1,
                            "provider_hint": "either",
                        }
                    ],
                    "inherited_hard_filters": {},
                    "rationale": "fixed synthetic plan",
                },
            },
            UsageActual(llm_calls=1),
        )


class SyntheticProvider:
    async def search(
        self,
        query: str,
        filters: dict[str, object],
        limit: int,
        reservation: BudgetReservation,
    ) -> ProviderResult[list[Paper]]:
        del filters, limit, reservation
        papers = (
            []
            if "zero-result" in query
            else [
                Paper(
                    canonical_id="openalex:W100",
                    title="Synthetic Graph Retrieval Paper",
                    openalex_id="W100",
                    sources=["openalex"],
                )
            ]
        )
        return _result(
            "openalex",
            papers,
            UsageActual(search_api_calls=1),
        )


def _budget(profile: BudgetProfile) -> SearchBudget:
    max_search_calls = 4 if profile == "balanced" else 3
    return SearchBudget(
        max_search_api_calls=max_search_calls,
        target_search_api_calls=1,
        max_llm_calls=2,
        target_llm_calls=1,
        max_total_tokens=100,
        max_cost_cny=1.0,
        max_elapsed_seconds=2,
        soft_deadline_seconds=1,
    )


class SyntheticOrchestratorFactory:
    def __init__(self) -> None:
        self.controllers: list[HardBudgetController] = []

    def __call__(
        self,
        profile: BudgetProfile,
    ) -> MockSearchOrchestrator:
        controller = HardBudgetController(_budget(profile))
        self.controllers.append(controller)
        return MockSearchOrchestrator(
            controller=controller,
            analyzer=SyntheticAnalyzer(),
            providers={"openalex": SyntheticProvider()},
            config_hash=_CONFIG_HASH,
            prompt_version="task8c-synthetic-v1",
            analysis_estimate=UsageEstimate(llm_calls=1),
            provider_estimate=UsageEstimate(search_api_calls=1),
        )


def build_synthetic_search_service(
    *,
    factory: SyntheticOrchestratorFactory | None = None,
) -> MockApiSearchService:
    """Build a service containing no real provider or environment boundary."""
    selected_factory = factory or SyntheticOrchestratorFactory()
    return MockApiSearchService(
        selected_factory,
        git_sha="synthetic-task8c",
        max_provider_results=5,
    )
```

- [ ] **Step 4: Run integration tests and fix only synthetic fixture assumptions**

Run:

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest `
  tests/integration/test_synthetic_baseline.py -v
```

Expected: all integration tests pass. If the real planner produces a different
fixed number of provider calls, adjust only the synthetic budget fixture and
expected mock usage; do not weaken one-record-per-query or byte-equality
assertions.

- [ ] **Step 5: Run focused static checks**

Run:

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file ruff check `
  src/paper_search/evaluation/synthetic_mocks.py `
  tests/integration/test_synthetic_baseline.py
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file mypy `
  src/paper_search/evaluation/synthetic_mocks.py
```

Expected: Ruff and mypy pass.

- [ ] **Step 6: Commit Task 3**

```powershell
git add -- `
  src/paper_search/evaluation/synthetic_mocks.py `
  tests/integration/test_synthetic_baseline.py
git commit -m "feat: compose fixed synthetic mock baseline"
```

---

### Task 4: Offline CLI and End-to-End Artifact Contract

**Files:**
- Modify: `src/paper_search/evaluation/synthetic_baseline.py`
- Modify: `src/paper_search/evaluation/__init__.py`
- Create: `tests/evaluation/test_synthetic_baseline_cli.py`

**Interfaces:**
- Consumes: `SYNTHETIC_QUERIES`, `build_synthetic_search_service`, and an output path.
- Produces:
  - `main(argv: Sequence[str] | None = None) -> int`
  - module execution via `python -m paper_search.evaluation.synthetic_baseline --output PATH`
- CLI accepts only `--output`.

- [ ] **Step 1: Write failing subprocess CLI tests**

Create `tests/evaluation/test_synthetic_baseline_cli.py`:

```python
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


SUBPROCESS_ENV = {**os.environ, "PYTHONPATH": str(Path("src").resolve())}


def _run_cli(output: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "paper_search.evaluation.synthetic_baseline",
            "--output",
            str(output),
            *extra,
        ],
        check=False,
        capture_output=True,
        env=SUBPROCESS_ENV,
        text=True,
    )


def test_cli_writes_only_byte_stable_synthetic_predictions(
    tmp_path: Path,
) -> None:
    output = tmp_path / "artifacts" / "predictions.jsonl"

    first = _run_cli(output)
    assert first.returncode == 0, first.stderr
    assert first.stderr == ""
    first_bytes = output.read_bytes()

    second = _run_cli(output)
    assert second.returncode == 0, second.stderr
    assert second.stderr == ""
    assert output.read_bytes() == first_bytes
    assert [path.name for path in output.parent.iterdir()] == ["predictions.jsonl"]
    assert b"recall" not in first_bytes.casefold()
    assert b"f1" not in first_bytes.casefold()


def test_cli_rejects_formal_evaluation_arguments(tmp_path: Path) -> None:
    for argument in (
        "--gold",
        "--split",
        "--metrics",
        "--manifest",
        "--api-key",
        "--endpoint",
    ):
        output = tmp_path / f"{argument[2:]}.jsonl"
        result = _run_cli(output, argument, "forbidden")
        assert result.returncode == 2
        assert not output.exists()


def test_package_batch_exports_are_lazy_and_warning_free() -> None:
    script = """
import sys
import paper_search.evaluation as evaluation

assert "paper_search.evaluation.synthetic_baseline" not in sys.modules
from paper_search.evaluation import synthetic_baseline

for name in (
    "SYNTHETIC_QUERIES",
    "run_synthetic_baseline",
    "validate_synthetic_requests",
):
    assert getattr(evaluation, name) is getattr(synthetic_baseline, name)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        env=SUBPROCESS_ENV,
        text=True,
    )
    assert result.returncode == 0, result.stderr
```

- [ ] **Step 2: Run CLI tests and verify RED**

Run:

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest `
  tests/evaluation/test_synthetic_baseline_cli.py -v
```

Expected: module execution fails because no CLI parser or `main` exists, and
package exports are absent.

- [ ] **Step 3: Add the strict CLI to the batch module**

Add these imports to `src/paper_search/evaluation/synthetic_baseline.py`:

```python
import argparse
import asyncio
from collections.abc import Sequence
from pathlib import Path

from paper_search.evaluation.synthetic_mocks import build_synthetic_search_service
```

Keep the existing `Sequence` and `Path` imports only once. Add:

```python
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write deterministic Task 8C synthetic predictions"
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        asyncio.run(
            run_synthetic_baseline(
                SYNTHETIC_QUERIES,
                search_service=build_synthetic_search_service(),
                output=args.output,
            )
        )
    except (OSError, ValueError):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Do not print query failures, dependency exceptions, paths other than argparse's
normal usage text, or synthetic quality claims.

- [ ] **Step 4: Add lazy package exports**

In `src/paper_search/evaluation/__init__.py`, add a type-checking import:

```python
    from paper_search.evaluation.synthetic_baseline import (
        SYNTHETIC_QUERIES,
        SyntheticSearchService,
        run_synthetic_baseline,
        validate_synthetic_requests,
    )
```

Add:

```python
_SYNTHETIC_BASELINE_EXPORTS = frozenset(
    {
        "SYNTHETIC_QUERIES",
        "SyntheticSearchService",
        "run_synthetic_baseline",
        "validate_synthetic_requests",
    }
)
```

Before the final `else` in `__getattr__`, add:

```python
    elif name in _SYNTHETIC_BASELINE_EXPORTS:
        from paper_search.evaluation.synthetic_baseline import (
            SYNTHETIC_QUERIES,
            SyntheticSearchService,
            run_synthetic_baseline,
            validate_synthetic_requests,
        )

        exports = {
            "SYNTHETIC_QUERIES": SYNTHETIC_QUERIES,
            "SyntheticSearchService": SyntheticSearchService,
            "run_synthetic_baseline": run_synthetic_baseline,
            "validate_synthetic_requests": validate_synthetic_requests,
        }
```

Add the four names to `__all__`.

- [ ] **Step 5: Run CLI and all Task 8C tests**

Run:

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest `
  tests/evaluation/test_predictions.py `
  tests/evaluation/test_synthetic_baseline.py `
  tests/evaluation/test_synthetic_baseline_cli.py `
  tests/integration/test_synthetic_baseline.py -v
```

Expected: all selected tests pass.

- [ ] **Step 6: Run the CLI twice and compare exact artifact hashes**

Run:

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
$task8cOutput = Join-Path $env:TEMP 'task8c-synthetic-predictions.jsonl'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file python -m `
  paper_search.evaluation.synthetic_baseline --output $task8cOutput
$firstHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $task8cOutput).Hash
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file python -m `
  paper_search.evaluation.synthetic_baseline --output $task8cOutput
$secondHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $task8cOutput).Hash
if ($firstHash -ne $secondHash) { throw "Task 8C output is not byte-stable" }
Get-Content -LiteralPath $task8cOutput -Encoding UTF8
```

Expected: both SHA-256 values match and output contains only ordered
`query_id`/`selected_paper_ids` records.

- [ ] **Step 7: Run final offline verification**

Run:

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest -q
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file ruff check .
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file mypy src
git diff --check
```

Expected:

- the complete offline suite passes with only the explicitly online live test
  skipped;
- Ruff reports `All checks passed!`;
- mypy reports no issues;
- `git diff --check` exits `0`.

- [ ] **Step 8: Audit scope and forbidden references**

Run:

```powershell
git diff --name-only 853dfbe..HEAD
rg -n -i `
  "api[_-]?key|secret|token|password|recall|precision|macro[_-]?f1|micro[_-]?f1" `
  src/paper_search/evaluation/synthetic_baseline.py `
  src/paper_search/evaluation/synthetic_mocks.py `
  tests/evaluation/test_synthetic_baseline.py `
  tests/evaluation/test_synthetic_baseline_cli.py `
  tests/integration/test_synthetic_baseline.py
```

Review every match. Allowed matches are negative assertions and forbidden CLI
argument tests only. Confirm changed production code has no file reads for
annotations, gold, dev, split, labels, manifests, `.env`, or credentials.

- [ ] **Step 9: Commit Task 4**

```powershell
git add -- `
  src/paper_search/evaluation/synthetic_baseline.py `
  src/paper_search/evaluation/__init__.py `
  tests/evaluation/test_synthetic_baseline_cli.py
git commit -m "feat: expose offline synthetic baseline cli"
```

- [ ] **Step 10: Request independent read-only review**

Ask the reviewer to inspect the complete range `853dfbe..HEAD` for:

- any path that could read protected or formal evaluation data;
- any external/network dependency;
- failure isolation that accidentally swallows preflight or filesystem errors;
- non-deterministic timestamps, ordering, identity, or serialization;
- any generated metric or quality claim;
- missing one-input/one-output guarantees.

Fix every Critical or Important finding with a failing regression test before
claiming completion.
