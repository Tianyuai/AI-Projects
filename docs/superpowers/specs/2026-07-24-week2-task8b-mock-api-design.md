# Week 2 Task 8B Mock API Integration Design

## Scope

Task 8B adds the fixed Week 2 HTTP contract around the existing mock
orchestration stack:

- `GET /health/live`;
- `GET /health/ready`;
- `POST /v1/search`;
- in-memory end-to-end coverage for success, empty results, one provider
  failure, query-analysis failure, budget exhaustion, and soft deadline.

The work does not call real OpenAlex, Semantic Scholar, or LLM endpoints. It
does not load `.env`, credentials, private annotations, gold data, frozen
splits, manifest state, or real queries. It does not start a listening server,
add UI behavior, generate real baseline metrics, fabricate ranking evidence, or
claim a Week 1 or Week 2 gate.

## Chosen Architecture

Use an application factory with explicit injected boundaries.

- `api/contracts.py` owns strict HTTP request and health response models.
- `api/service.py` adapts one validated request to a fresh injected mock
  orchestrator, then converts its `MinimalSearchResult` with the existing
  `to_structured_response` function.
- `api/app.py` owns routing, response codes, safe transport errors, and the
  module-level degraded application.

The API layer does not construct provider transports, load configuration, read
Git state, or inspect environment variables. Composition code supplies a
factory, fixed Git SHA, maximum provider result count, and a readiness probe.

## HTTP Contracts

`SearchRequest` is a frozen, extra-forbidden Pydantic model with the PRD fields:

```python
query_id: NonEmptyStr
query: NonEmptyStr
budget_profile: Literal["low", "balanced"] = "balanced"
include_trace: bool = True
```

`POST /v1/search` returns the existing `StructuredSearchResponse` model.
Pydantic request failures return FastAPI's standard `422` response. Completed,
partial, soft-stop, and hard-stop domain results all return HTTP `200`; their
meaning remains in `stop_reason`, `is_partial`, and `warnings`.

An unavailable service or unexpected service exception returns HTTP `503` with
the fixed JSON body `{"detail":"search temporarily unavailable"}`. Raw
exceptions, credentials, provider payloads, and request headers never enter the
response.

`GET /health/live` always returns HTTP `200` and `{"status":"ok"}`.

`GET /health/ready` calls an injected zero-argument readiness probe that returns
a provider-to-boolean mapping. It returns:

- HTTP `200`, status `ready`, and provider values `ready` when a search service
  is injected, the provider mapping is non-empty, and every provider is ready;
- HTTP `503`, status `degraded`, and explicit provider values otherwise.

The response provider keys are sorted for deterministic output. A readiness
probe exception is converted to a degraded response without exposing the
exception. The module-level `app` has no service and therefore reports
degraded readiness.

## Search Service Composition

`MockApiSearchService` receives:

- `orchestrator_factory(budget_profile)`;
- a non-empty fixed `git_sha`;
- a positive fixed `max_provider_results`.

For every request it calls the factory exactly once, so budget reservations and
committed usage cannot leak across requests. It passes only `request.query` to
the orchestrator, then calls:

```python
to_structured_response(
    minimal_result,
    query_id=request.query_id,
    git_sha=git_sha,
)
```

When `include_trace` is false it returns a model copy with an empty
`search_trace`; every other response field remains unchanged. The service does
not reinterpret partial or stop states.

## Failure and Safety Behavior

- Invalid or extra request fields fail before the service is called.
- Query-analysis structured errors and provider errors remain domain data and
  return HTTP `200` structured responses.
- Query-analysis exceptions fail closed in the existing orchestrator and return
  HTTP `200` hard-stop partial responses.
- Budget exhaustion and soft deadlines prevent new provider calls and remain
  HTTP `200` structured responses.
- Only failures that prevent producing `StructuredSearchResponse` become the
  fixed HTTP `503`.
- Health checks never call the search service or an external transport.

## Testing Strategy

Implementation follows strict RED-GREEN-REFACTOR cycles.

- Contract tests cover strict `SearchRequest` validation and defaults.
- ASGI transport tests cover live, ready, degraded, safe `503`, `422`, response
  model serialization, and trace suppression without opening a socket.
- Service tests prove profile forwarding, fresh factory use per request, fixed
  identity injection, positive limit validation, and response preservation.
- End-to-end integration tests compose the real `MockSearchOrchestrator`,
  `MockApiSearchService`, and FastAPI app with synthetic analyzers/providers for
  all six required scenarios.
- Tests assert injected fakes are the only external-call boundary and repeated
  requests receive fresh budget controllers.

Final verification runs the complete offline suite with `--no-env-file`, Ruff,
mypy, `git diff --check`, a changed-path and credential-literal audit, and an
independent read-only code review.
