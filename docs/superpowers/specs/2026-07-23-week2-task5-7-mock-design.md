# Week 2 Task 5–7 Mock-Driven Design

## Scope

This design implements only the offline, deterministic engineering contracts for:

- Task 5: an OpenAI-compatible LLM adapter, query parsing, repair fallback, and deterministic planning;
- Task 6: a Semantic Scholar adapter and deterministic multi-source fusion;
- Task 7: atomic budget accounting, recoverable state, and a minimal orchestrator.

It does not use private annotations, gold data, frozen splits, real credentials, or network calls. It does not implement Task 8 HTTP endpoints, UI work, embeddings, reranking, citation graphs, or tuning.

## Chosen Architecture

Use thin external adapters around injected transports and keep the policy layer deterministic.

The LLM client converts a fake transport response into `ProviderResult[dict]`, records model and token metadata, redacts credentials, and classifies timeout, empty, and malformed responses. `QueryParser` validates a combined `QueryAnalysisResult`, permits one explicitly budgeted repair attempt, and then produces a rules-only `QuerySpec`. `QueryPlanner` canonicalizes, orders, deduplicates, validates hard-constraint inheritance, and clips the plan to a configured 3–5 subqueries.

`SearchProvider` is a protocol shared by OpenAlex, Semantic Scholar, and test fakes. The Semantic Scholar adapter maps search, batch detail, reference, and citation payloads to existing domain models. It retains Semantic Scholar IDs, endpoint provenance, raw edge identity, and structured errors. The fusion module accepts ordered provider results and supports configurable RRF or weighted rank fusion. Cross-source duplicates are merged through the existing deduplicator so DOI and provider IDs survive.

`HardBudgetController` remains the single budget authority. Task 7 adds a lock around state transitions, injected UTC clock support, expiration/release, soft and hard stop inspection, and JSON state export/import. Unknown actual cost remains `None` and is tracked separately from known cost. The orchestrator owns reservations for query analysis and every provider call, settles actual usage, and never invokes a provider without a reservation.

## Component Boundaries

- `llm/client.py`: transport protocol, response decoding, safe error classification, and `ProviderResult` construction.
- `query/parser.py`: schema validation, one repair attempt, and deterministic rule fallback.
- `query/planner.py`: stable ordering, hard-filter inheritance, deduplication, and 3–5 item clipping.
- `retrieval/base.py`: shared provider protocol and provider-call specification.
- `retrieval/semantic_scholar.py`: Semantic Scholar request/response contract only.
- `ranking/fusion.py`: pure deterministic rank fusion and source-preserving merge.
- `control/budget.py`: thread-safe reservation lifecycle and recoverable budget state.
- `storage/experiment.py`: immutable/minimal experiment record persistence.
- `pipeline/orchestrator.py`: sequencing and partial-result assembly only.

Existing models, OpenAlex mapping, cache, deduplication, filters, and lexical ranking are reused rather than reimplemented.

## Determinism and Failure Behavior

Stable IDs, explicit rank tie-breakers, sorted constraints, injected clocks, and fixture responses make repeated runs identical. A provider error is data, not an unhandled pipeline exception. One provider can fail while valid sibling results continue. Soft stop returns assembled partial results and prevents new optional work. Hard stop prevents every new external call. Reservation expiry and explicit release both restore capacity.

No log or exception includes authorization headers, API keys, full request headers, or raw sensitive payloads. Provenance records only safe endpoint/model/time/hash metadata.

## Testing Strategy

Each task follows RED → GREEN → REFACTOR and receives its own scoped commit.

- Task 5 tests valid/invalid/empty/timeout responses, one repair, failed repair fallback, usage metadata, deterministic planning, clipping, hard constraints, and credential redaction.
- Task 6 uses original synthetic fixtures for search, batch, references, citations, empty/missing/invalid records, rate limits, timeout, provider errors, RRF, weighted fusion, cross-source evidence, and one-provider failure.
- Task 7 tests API/LLM/token/cost dimensions, unknown cost, atomic reservations, settlement, release/expiry, soft/hard stops, state recovery, experiment records, pipeline ordering, provider bypass prevention, partial results, and deterministic failure modes.

Focused tests precede each commit. Final verification runs the complete offline pytest suite, Ruff, mypy, `git diff --check`, staged-file checks, and an independent review. Passing mock contracts do not claim Week 1 or Week 2 gates, real recall, F1, cost, or model-comparison targets.
