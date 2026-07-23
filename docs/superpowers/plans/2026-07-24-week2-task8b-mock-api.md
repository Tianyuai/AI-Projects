# Week 2 Task 8B Mock API Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic, dependency-injected FastAPI contract around the existing mock orchestrator without loading secrets or calling real services.

**Architecture:** Strict Pydantic contracts live in `api/contracts.py`; `MockApiSearchService` creates a fresh injected orchestrator per request; `api/app.py` owns only HTTP validation, health semantics, and safe transport errors. End-to-end tests use the real orchestrator and response adapter with synthetic analyzer/provider boundaries.

**Tech Stack:** Python 3.11, FastAPI, HTTPX ASGI transport, Pydantic 2, pytest, Ruff, mypy, uv.

## Global Constraints

- Run all Python commands with `--no-env-file`.
- Do not call real OpenAlex, Semantic Scholar, or LLM endpoints.
- Do not read `.env`, credentials, private annotations, gold data, frozen splits, manifest state, real queries, or the protected Task 2 evaluation design.
- Do not start a listening server; use HTTPX `ASGITransport`.
- Do not add UI behavior, real baseline execution, metrics, ranking evidence, citation graphs, or provider construction.
- Do not claim Week 1 or Week 2 gates, recall, F1, cost, or readiness of real providers.

## File Structure

- Create `src/paper_search/api/__init__.py`: public Task 8B API exports.
- Create `src/paper_search/api/contracts.py`: strict search and health models.
- Create `src/paper_search/api/service.py`: fresh-orchestrator request adapter.
- Create `src/paper_search/api/app.py`: application factory and default degraded app.
- Create `tests/api/test_contracts.py`: request and health schema tests.
- Create `tests/api/test_service.py`: service composition tests.
- Create `tests/api/test_app.py`: HTTP health, validation, trace, and safe-error tests.
- Create `tests/integration/test_api.py`: six real-orchestrator mock scenarios and cross-request budget isolation.

---

### Task 1: Strict HTTP Contracts

**Files:**
- Create: `tests/api/test_contracts.py`
- Create: `src/paper_search/api/contracts.py`
- Create: `src/paper_search/api/__init__.py`

**Interfaces:**
- Produces: `BudgetProfile`, `SearchRequest`, `LiveHealthResponse`, `ReadyHealthResponse`, and `ProviderHealthStatus`.

- [ ] **Step 1: Write failing contract tests**

Test the wished-for API:

```python
def test_search_request_has_prd_defaults_and_strips_identity() -> None:
    request = SearchRequest(query_id=" q1 ", query=" graph retrieval ")
    assert request.model_dump() == {
        "query_id": "q1",
        "query": "graph retrieval",
        "budget_profile": "balanced",
        "include_trace": True,
    }


@pytest.mark.parametrize(
    "payload",
    [
        {"query_id": "", "query": "valid"},
        {"query_id": "q1", "query": ""},
        {"query_id": "q1", "query": "valid", "budget_profile": "large"},
        {"query_id": "q1", "query": "valid", "extra": True},
    ],
)
def test_search_request_rejects_invalid_payload(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        SearchRequest.model_validate(payload)


def test_health_contracts_are_strict() -> None:
    assert LiveHealthResponse().model_dump() == {"status": "ok"}
    assert ReadyHealthResponse(
        status="degraded",
        providers={"openalex": "ready", "semantic_scholar": "degraded"},
    ).model_dump()["providers"] == {
        "openalex": "ready",
        "semantic_scholar": "degraded",
    }
```

- [ ] **Step 2: Verify RED**

Run:

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/api/test_contracts.py -q
```

Expected: collection fails because `paper_search.api.contracts` does not exist.

- [ ] **Step 3: Implement the strict models**

Create:

```python
"""Strict HTTP request and health contracts."""

from typing import Literal, TypeAlias

from paper_search.domain.models import DomainModel, NonEmptyStr

BudgetProfile: TypeAlias = Literal["low", "balanced"]
ProviderHealthStatus: TypeAlias = Literal["ready", "degraded"]


class SearchRequest(DomainModel):
    query_id: NonEmptyStr
    query: NonEmptyStr
    budget_profile: BudgetProfile = "balanced"
    include_trace: bool = True


class LiveHealthResponse(DomainModel):
    status: Literal["ok"] = "ok"


class ReadyHealthResponse(DomainModel):
    status: Literal["ready", "degraded"]
    providers: dict[NonEmptyStr, ProviderHealthStatus]
```

Export the four contracts from `paper_search.api.__init__`.

- [ ] **Step 4: Verify GREEN and commit**

Run:

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/api/test_contracts.py -q
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file ruff check `
  src/paper_search/api tests/api/test_contracts.py
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file mypy `
  src/paper_search/api/contracts.py
git diff --check
```

Commit:

```powershell
git add src/paper_search/api/__init__.py `
  src/paper_search/api/contracts.py tests/api/test_contracts.py
git commit -m "feat: define strict mock api contracts"
```

---

### Task 2: Fresh-Orchestrator Search Service

**Files:**
- Create: `tests/api/test_service.py`
- Create: `src/paper_search/api/service.py`
- Modify: `src/paper_search/api/__init__.py`

**Interfaces:**
- Consumes: `SearchRequest`, `MinimalSearchResult`, and `to_structured_response`.
- Produces: `SearchOrchestrator`, `OrchestratorFactory`, and `MockApiSearchService`.

- [ ] **Step 1: Write failing service tests**

Define a complete fake orchestrator that records its query/limit and returns a
fully valid `MinimalSearchResult`. Test:

```python
def test_service_forwards_profile_and_identity_and_suppresses_trace() -> None:
    factory = RecordingFactory()
    service = MockApiSearchService(
        factory,
        git_sha="abc1234",
        max_provider_results=7,
    )

    response = asyncio.run(
        service(
            SearchRequest(
                query_id="q1",
                query="graph retrieval",
                budget_profile="low",
                include_trace=False,
            )
        )
    )

    assert factory.profiles == ["low"]
    assert factory.instances[0].calls == [("graph retrieval", 7)]
    assert response.query_id == "q1"
    assert response.git_sha == "abc1234"
    assert response.search_trace == []
```

Call the service twice and assert two different orchestrator instances were
created. Parameterize constructor failures for blank `git_sha` and non-positive
`max_provider_results`.

- [ ] **Step 2: Verify RED**

Run:

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/api/test_service.py -q
```

Expected: collection fails because `paper_search.api.service` does not exist.

- [ ] **Step 3: Implement the service adapter**

Create:

```python
"""Request-scoped composition for the mock search orchestrator."""

from collections.abc import Callable
from typing import Protocol

from paper_search.api.contracts import BudgetProfile, SearchRequest
from paper_search.domain.models import StructuredSearchResponse
from paper_search.pipeline.orchestrator import MinimalSearchResult
from paper_search.pipeline.response import to_structured_response


class SearchOrchestrator(Protocol):
    async def run(
        self, query: str, *, max_provider_results: int
    ) -> MinimalSearchResult: ...


OrchestratorFactory = Callable[[BudgetProfile], SearchOrchestrator]


class MockApiSearchService:
    def __init__(
        self,
        orchestrator_factory: OrchestratorFactory,
        *,
        git_sha: str,
        max_provider_results: int,
    ) -> None:
        if not git_sha.strip():
            raise ValueError("git_sha must not be blank")
        if max_provider_results <= 0:
            raise ValueError("max_provider_results must be positive")
        self._orchestrator_factory = orchestrator_factory
        self._git_sha = git_sha.strip()
        self._max_provider_results = max_provider_results

    async def __call__(
        self, request: SearchRequest
    ) -> StructuredSearchResponse:
        orchestrator = self._orchestrator_factory(request.budget_profile)
        result = await orchestrator.run(
            request.query,
            max_provider_results=self._max_provider_results,
        )
        response = to_structured_response(
            result,
            query_id=request.query_id,
            git_sha=self._git_sha,
        )
        if not request.include_trace:
            return response.model_copy(update={"search_trace": []})
        return response
```

Export the service interfaces from `paper_search.api.__init__`.

- [ ] **Step 4: Verify GREEN and commit**

Run:

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest `
  tests/api/test_service.py tests/unit/test_response.py -q
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file ruff check `
  src/paper_search/api tests/api/test_service.py
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file mypy `
  src/paper_search/api/service.py
git diff --check
```

Commit:

```powershell
git add src/paper_search/api/__init__.py `
  src/paper_search/api/service.py tests/api/test_service.py
git commit -m "feat: adapt requests to fresh mock orchestrators"
```

---

### Task 3: FastAPI Routes and Six-Scenario End-to-End Contract

**Files:**
- Create: `tests/api/test_app.py`
- Create: `tests/integration/test_api.py`
- Create: `src/paper_search/api/app.py`
- Modify: `src/paper_search/api/__init__.py`

**Interfaces:**
- Consumes: injected `SearchService` and `ReadinessProbe`.
- Produces: `create_app(...) -> FastAPI`, `app`, `/health/live`, `/health/ready`, and `/v1/search`.

- [ ] **Step 1: Write failing HTTP contract tests**

Use HTTPX `ASGITransport` without opening a socket. Cover:

```python
def test_live_is_always_ok() -> None:
    response = request(create_app(), "GET", "/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_requires_service_and_all_nonempty_provider_states() -> None:
    ready = request(
        create_app(
            SuccessfulService(),
            readiness_probe=lambda: {
                "semantic_scholar": True,
                "openalex": True,
            },
        ),
        "GET",
        "/health/ready",
    )
    assert ready.status_code == 200
    assert ready.json() == {
        "status": "ready",
        "providers": {
            "openalex": "ready",
            "semantic_scholar": "ready",
        },
    }

    degraded = request(create_app(), "GET", "/health/ready")
    assert degraded.status_code == 503
    assert degraded.json() == {"status": "degraded", "providers": {}}
```

Also test one false provider, readiness-probe exception, default module app,
`422` invalid/extra request fields without service invocation, fixed safe `503`
for unavailable/raising service, response-model serialization, and
`include_trace=False`.

- [ ] **Step 2: Write failing real-orchestrator API integration tests**

Build synthetic analyzer/provider classes that return complete
`ProviderResult` objects. Parameterize the real
`MockSearchOrchestrator -> MockApiSearchService -> FastAPI` chain for:

```python
[
    ("success", 200, "completed", False, ["openalex:W1"]),
    ("empty", 200, "completed", False, []),
    ("provider_failure", 200, "completed", True, ["s2:S1"]),
    ("analysis_failure", 200, "hard_stop", True, []),
    ("budget_exhausted", 200, "hard_stop", True, []),
    ("soft_stop", 200, "soft_stop", True, []),
]
```

For every case assert the response validates as `StructuredSearchResponse`.
For provider, analysis, budget, and soft-stop cases assert the fake call events
prove no prohibited call occurred. Send two requests through one app and assert
the factory created two controllers and the second request did not inherit the
first request's usage.

- [ ] **Step 3: Verify all route and E2E tests are RED**

Run:

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest `
  tests/api/test_app.py tests/integration/test_api.py -q
```

Expected: collection fails because `paper_search.api.app` does not exist.

- [ ] **Step 4: Implement the application factory**

Create an injected `SearchService` protocol and `ReadinessProbe` type. Implement
`create_app(search_service=None, *, readiness_probe=None)` with:

```python
@application.get("/health/live", response_model=LiveHealthResponse)
async def live() -> LiveHealthResponse:
    return LiveHealthResponse()
```

The ready handler catches probe exceptions, sorts provider keys, maps booleans
to `ready`/`degraded`, and returns a `JSONResponse(status_code=503, ...)` unless
the service exists, providers are non-empty, and all are ready.

The search handler declares
`response_model=StructuredSearchResponse`. It returns the injected service
result; when no service exists or any `Exception` is raised, it returns exactly:

```python
JSONResponse(
    status_code=503,
    content={"detail": "search temporarily unavailable"},
)
```

Create the safe default:

```python
app = create_app()
```

Export `SearchService`, `ReadinessProbe`, `create_app`, and `app`.

- [ ] **Step 5: Verify GREEN and commit**

Run:

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest `
  tests/api tests/integration/test_api.py tests/integration/test_orchestrator.py -q
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file ruff check `
  src/paper_search/api tests/api tests/integration/test_api.py
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file mypy src/paper_search/api
git diff --check
```

Commit:

```powershell
git add src/paper_search/api tests/api tests/integration/test_api.py
git commit -m "feat: expose offline mock search api"
```

---

### Task 4: Final Offline Verification and Review

**Files:**
- Verify only the design, plan, API package, and API tests introduced by Task 8B.

**Interfaces:**
- Produces: a clean, review-ready Task 8B branch with fresh evidence.

- [ ] **Step 1: Run complete offline tests**

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest -m "not online" -q
```

Expected: every offline test passes and the online test is deselected.

- [ ] **Step 2: Run full static checks**

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file ruff check .
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file mypy src
git diff --check
```

Expected: every command exits zero.

- [ ] **Step 3: Audit scope and credential literals**

Compare against design commit `4c6d31f`:

```powershell
git diff --name-only 4c6d31f...HEAD
git status --short
rg -n --glob '!uv.lock' `
  'sk-[A-Za-z0-9_-]{16,}|OPENAI_API_KEY\s*=|SEMANTIC_SCHOLAR_API_KEY\s*=' `
  src/paper_search/api tests/api tests/integration/test_api.py `
  docs/superpowers/plans/2026-07-24-week2-task8b-mock-api.md `
  docs/superpowers/specs/2026-07-24-week2-task8b-mock-api-design.md
```

Expected: changed paths match this plan, status is clean, and the literal scan
has no matches.

- [ ] **Step 4: Request independent read-only review**

Review `f815f83..HEAD` against the design and plan. Fix every Critical and
Important issue with a failing regression test first, rerun the complete
verification, and commit review fixes separately.

## Plan Self-Review

- Spec coverage: contracts, service composition, three routes, all six scenarios,
  trace suppression, safety, isolation, verification, and review are mapped.
- Placeholder scan: no deferred implementation placeholders remain.
- Type consistency: `SearchRequest` feeds `MockApiSearchService`, which returns
  the existing `StructuredSearchResponse`; `create_app` consumes the same
  callable service contract.
- Scope: no configuration loading, environment access, real providers, sockets,
  labels, metrics, UI behavior, scoring, or graph work is introduced.
