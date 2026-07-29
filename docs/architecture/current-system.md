# Current System Architecture

## Scope and Revision

This document describes the current source revision's offline, dependency-injected search composition. It is an implementation map and not a statement of retrieval quality, production readiness, or formal evaluation results. The runnable API described here is the loopback-only mock server; the default `paper_search.api.app:app` has no search service injected and therefore reports degraded readiness.

## QuerySpec and QueryPlanner

`QuerySpec` is the validated representation of the user query, its research goal, constraints, optional year range, venues, exclusions, and ambiguities. The injected analyzer returns a combined query-analysis payload. `QueryParser` validates that payload, normalizes the original query, and passes its plan to `QueryPlanner`.

`QueryPlanner` canonicalizes order, whitespace, duplicate subqueries, priorities, target constraints, and inherited hard filters. It keeps three to five distinct subqueries; if no usable model plan is available, it constructs a deterministic rules-only plan. The rules fallback extracts only explicit years, known venues, and a `without` exclusion before `QueryPlanner` produces the fallback plan.

## Provider Boundary

The orchestrator receives a mapping of `SearchProvider` implementations and works with their `ProviderResult` envelopes, including provenance, usage, cache status, latency, and sanitized error details. The repository contains provider adapters behind that boundary, but the executable mock server does not compose them.

`paper_search.api.mock_server` builds a synthetic, loopback-only service. Its fixed readiness probe names OpenAlex and Semantic Scholar as ready for the mock API contract. The synthetic orchestrator factory itself injects one deterministic OpenAlex-style provider and a deterministic analyzer; it has no real provider, environment, or network boundary. The mock-server audit hook rejects non-loopback socket and name-resolution targets.

## Cache and Budget Accounting

`SQLiteResponseCache` stores successful raw provider responses with a canonical non-secret request identity, response hash, expiry, safe response headers, and cooldown records. It can prepare and write immutable snapshot files with a manifest and verifies their hashes when validating a snapshot. The current synthetic mock composition does not inject this cache.

For every mock API request, `MockApiSearchService` creates a fresh `MockSearchOrchestrator` and its `HardBudgetController` from the requested budget profile. The controller reserves estimated usage before analysis and provider work, settles actual usage afterward, enforces hard limits, records committed usage, and can fail closed. The returned usage is the controller's committed usage for that request.

## Deduplication, Filtering, Fusion, and Ranking

After retrieval, `MockSearchOrchestrator` combines results by provider, deduplicates papers, applies hard filters, and fuses provider results with reciprocal-rank fusion. Only fused papers whose canonical IDs were accepted by filtering are returned. On a completed, trace-enabled retrieval path, the public trace records the analysis, retrieval, deduplication, filtering, and fusion stages. An unavailable analysis budget or an analyzer exception returns before the analysis trace entry, and `include_trace=false` clears the public trace.

Embedding ranking, citation expansion, and constraint reranking are optional injected stages. When provided, they run after fusion in that order and add trace or safe warning information. The synthetic mock factory does not inject any of those optional stages, so the demo path ends with the fused, filtered result set.

## Evaluation, Snapshots, and Reproducibility

Evaluation utilities and snapshot support are present as offline tooling. Provider snapshots preserve exact cached bytes plus a manifest. `StructuredSearchResponse` carries `config_hash` and `git_sha` as top-level fields. `prompt_version` is not a top-level response field: it appears in the successful `analyze` trace entry and is visible publicly only when tracing is enabled. These mechanisms make inputs and outputs inspectable, but this document does not report a formal evaluation.

R2 is retrieval diagnostic evidence only. It is not a relevance-performance conclusion, and no relevance metrics are included here. R3 is the later point at which formal evaluation artifacts may be considered; they are outside this document's scope.

## Single-Run Pipeline

1. The mock API accepts `POST /v1/search` with a query ID, query text, budget profile, and optional trace flag.
2. `MockApiSearchService` selects a fresh synthetic orchestrator for the profile.
3. On the normal completed path, the orchestrator reserves analysis budget, invokes the injected analyzer, parses and finalizes the plan, then reserves and invokes each eligible injected provider.
4. It then deduplicates, filters, fuses, and optionally applies the injected post-retrieval stages before returning a minimal result with trace, usage, stop reason, partial flag, and warnings. Analysis-budget and analyzer-failure paths instead return early with an empty trace.
5. The response converter produces the public structured response. If `include_trace` is false, the API clears only the public trace; the other response fields remain available.

The API also exposes `GET /health/live` and `GET /health/ready`. Liveness is independent of dependencies. Readiness requires both an injected search service and a nonempty all-ready provider map.

## Offline Adaptive Evolution Boundary

The adaptive `EvolutionCoordinator` is an offline component that wraps an injected single-round executor. It coordinates an initial round plan with injected coverage analysis, next-round generation, cost estimation, marginal-gain evaluation, and budget preflight. It returns copied round records, candidates, stopping decisions, warnings, and any failed round without mutating injected input objects.

The coordinator does not replace `MockSearchOrchestrator`. It is not enabled in API composition or in the runnable mock-server configuration. Its preflight budget check is advisory for prospective rounds; it does not take the orchestrator's reservation and settlement role.

## Components Not Enabled in Main Configuration

The runnable main demonstration configuration is the fixed synthetic mock composition, not a real-provider deployment. It does not enable the adaptive coordinator, real provider adapters, SQLite caching, embedding ranking, citation expansion, or constraint reranking. The default API module is likewise deliberately uncomposed.

Those boundaries are intentional deferred work. Adaptive behavior remains offline and injected until it is explicitly wired through a future runtime composition and supported by the required evaluation evidence.
