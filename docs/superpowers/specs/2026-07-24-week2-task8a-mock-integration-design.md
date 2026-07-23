# Week 2 Task 8A Mock Integration Hardening Design

## Scope

Task 8A advances three deterministic, offline-only parts of Week 2:

1. complete mock end-to-end coverage for query-analysis failure, empty retrieval, provider failure, budget exhaustion, and soft deadline behavior;
2. convert `MinimalSearchResult` into the existing `StructuredSearchResponse` contract without inventing ranking evidence;
3. generate deterministic prediction JSONL from synthetic structured responses and validate it through the existing official adapter.

The work does not add HTTP endpoints, start an API server, call real providers or
LLMs, load `.env`, or read private annotations, gold data, frozen splits,
manifest state, real queries, or the protected Task 2 evaluation design. It does
not claim the Week 1 or Week 2 gate.

## Chosen Architecture

Keep orchestration, response assembly, and prediction serialization as separate
units.

- `pipeline/orchestrator.py` continues to own reservations and external-call
  sequencing. It converts known analyzer failures into rule-based fallback and
  returns a structured partial result. If an analyzer exception leaves actual
  usage unknown, the budget fails closed and no provider is called.
- `pipeline/response.py` is a pure adapter from `MinimalSearchResult` to
  `StructuredSearchResponse`. Callers inject `query_id` and `git_sha`.
- `evaluation/predictions.py` converts structured responses into the existing
  strict `InternalPredictionRecord` format and delegates deterministic,
  atomic JSONL output to `write_jsonl_atomic`.

No API framework, network transport, provider implementation, scoring model, or
new evaluation schema is introduced.

## Response Contract

The adapter preserves these fields directly:

- query analysis;
- selected paper IDs in fused result order;
- search trace;
- committed usage;
- stop reason and partial flag;
- warnings;
- configuration hash.

`query_id` and `git_sha` are explicit non-empty inputs. Because
`MinimalSearchResult` contains no `CandidateEvidence`, the adapter returns empty
`high_relevance` and `partial_relevance` lists. It also returns no citation edges.
The adapter never fabricates scores, evidence, relevance labels, or graph data.

## Failure Semantics

- A structured analyzer error is settled using its reported usage, then parsed
  through the existing deterministic rule fallback. Retrieval may continue while
  budget remains.
- An analyzer exception with unknown actual usage triggers fail-closed budget
  handling. The orchestrator returns rule-based analysis, `hard_stop`,
  `is_partial=true`, an analysis warning, and no provider calls.
- Empty provider results without errors produce a completed, non-partial response
  with an empty paper list.
- A provider result containing errors remains data: valid sibling results are
  retained and the response is partial.
- A soft deadline prevents all new provider calls and returns any already
  assembled results with `soft_stop`.
- A hard stop prevents all new external calls.

Warnings expose only stable error categories and dependency names, never raw
exceptions, credentials, headers, or payloads.

## Prediction Contract

Each `StructuredSearchResponse` maps to one `InternalPredictionRecord`:

- `query_id` is copied unchanged;
- `selected_paper_ids` is copied in ranked order;
- empty results remain an empty list.

Batch serialization rejects duplicate query IDs before writing. Output uses the
existing atomic JSONL writer, UTF-8 encoding, deterministic key ordering, compact
JSON, and one trailing newline per record. Tests use only synthetic models and
temporary directories, then read the output through `InternalPredictionRecord`
and apply `adapt_prediction_record`.

## Testing Strategy

Implementation follows strict RED-GREEN-REFACTOR cycles.

- Integration tests exercise analyzer structured failure, analyzer exception,
  all-provider empty results, one-provider failure, soft stop, and hard budget
  exhaustion using injected fakes.
- Unit tests verify field preservation and the deliberate absence of fabricated
  relevance and citation data.
- Evaluation tests verify deterministic prediction bytes, empty predictions,
  duplicate-query rejection, strict re-reading, and official-adapter round trips.

Final verification runs the complete offline test suite with `--no-env-file`,
Ruff, mypy, `git diff --check`, a scope/secret scan, and an independent code
review. No online test or live readiness check is part of Task 8A.
