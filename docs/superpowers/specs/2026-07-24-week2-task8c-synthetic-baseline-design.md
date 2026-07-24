# Week 2 Task 8C Synthetic Baseline Design

## Scope

Task 8C adds a deterministic, offline-only synthetic baseline pipeline:

```text
versioned synthetic queries
  -> MockApiSearchService
  -> fresh MockSearchOrchestrator per query
  -> StructuredSearchResponse
  -> predictions.jsonl
```

The pipeline exists to harden integration boundaries before annotation work is
complete. It is independent of the collaborators' 90/40 annotations and does
not claim formal evaluation quality.

The implementation:

- uses only fixed synthetic queries and injected mock analyzer/provider
  dependencies;
- emits only a synthetic `predictions.jsonl`;
- proves that repeated runs produce byte-identical output;
- preserves one output record per input query;
- converts a query-level exception to an empty prediction and continues the
  batch;
- never reads annotations, gold data, a dev split, a manifest, `.env`,
  credentials, or real queries;
- never calls an external API or opens a listening HTTP server;
- never computes or reports Recall, Precision, F1, or any other formal metric.

## Chosen Architecture

Use a focused synthetic batch module that calls the existing service boundary
directly. Do not route the batch through HTTP and do not extend the formal
`evaluation.runner`.

The new `paper_search.evaluation.synthetic_baseline` module owns:

- the versioned, code-defined synthetic query catalog;
- batch preflight validation;
- sequential query execution;
- query-level exception isolation;
- deterministic prediction conversion;
- atomic `predictions.jsonl` creation;
- the offline CLI entry point.

The module receives a search service through an explicit protocol. Production
composition for this task supplies `MockApiSearchService`, whose factory creates
a fresh `MockSearchOrchestrator` and budget controller for every request.
Neither the batch layer nor its CLI constructs real transports or loads runtime
configuration.

This boundary keeps the formal evaluation runner unchanged. That runner may
read frozen partitions and gold data and may compute metrics, so reusing it
would violate Task 8C's narrower safety contract.

## Synthetic Query Catalog

The catalog is declared in code rather than loaded from a data path. Each entry
is an existing strict `SearchRequest` with:

- a stable synthetic `query_id`;
- an explicitly synthetic query string;
- a fixed budget profile;
- `include_trace=False`, because traces are not part of prediction output.

Catalog order is part of the contract. The CLI exposes no input-file, split,
gold, manifest, or metric argument, so a caller cannot accidentally turn this
tool into a formal evaluation run.

Before invoking the service, the batch validates the complete catalog:

- the sequence must be non-empty;
- every item must already satisfy `SearchRequest`;
- `query_id` values must be unique.

A preflight failure occurs before service execution and before any output write.

## Batch Data Flow

The batch accepts:

```python
async def run_synthetic_baseline(
    requests: Sequence[SearchRequest],
    *,
    search_service: SearchService,
    output: Path,
) -> list[InternalPredictionRecord]
```

It processes requests sequentially in their declared order. Concurrency is
deliberately excluded because it adds scheduling variability without serving
the integration goal.

For each request:

1. call `await search_service.search(request)`;
2. require a valid `StructuredSearchResponse`;
3. convert it with the existing `prediction_from_response` boundary;
4. append the resulting `InternalPredictionRecord`.

After every request has produced a record, write the complete ordered record
list atomically. The output uses the established prediction contract:

```json
{"query_id":"synthetic-q1","selected_paper_ids":["openalex:W1"]}
```

The batch returns the same records it writes. It does not return or persist
metrics, labels, scores, traces, usage summaries, or formal run identity.

## Failure Isolation

Completed, empty, partial, soft-stop, and hard-stop structured responses are all
valid results. They are converted without reinterpreting their domain state.

If a single service call cannot produce `StructuredSearchResponse`, the batch
catches the query-level exception and appends:

```json
{"query_id":"the-original-query-id","selected_paper_ids":[]}
```

It then continues with the next request. Raw exception text, exception types,
provider payloads, and credentials are not persisted.

The catch boundary covers only one service invocation and its response
conversion. Preflight validation and final artifact-writing failures remain
batch-level failures. This distinction prevents malformed input or filesystem
errors from being silently represented as search misses.

## Determinism and Artifact Safety

Byte determinism relies on the following fixed inputs and behaviors:

- stable catalog order and values;
- fixed mock analyzer/provider payloads;
- fixed mock timestamps, Git SHA, config hash, prompt version, budgets, and
  result limits;
- a fresh orchestrator and controller for each request;
- sequential execution;
- canonical JSON serialization with sorted keys, compact separators, UTF-8,
  and one trailing newline per record;
- atomic sibling-temp-file replacement.

Running the CLI twice with the same output path is allowed. The second run
atomically replaces the file with byte-identical content. Tests compare the
complete bytes from both runs rather than comparing parsed JSON.

No partial final artifact is exposed while queries are still running. If final
serialization or replacement fails, an existing `predictions.jsonl` remains
unchanged and temporary files are cleaned up.

## CLI Contract

The CLI accepts exactly one required data-affecting argument:

```text
--output PATH
```

It uses the code-defined synthetic catalog and fixed mock composition. It does
not accept configuration, input, split, gold, labels, manifest, endpoint,
credential, metric, or concurrency options.

On success it exits `0`. Invalid arguments, catalog preflight failures, or
artifact-writing failures exit nonzero without exposing secrets. Individual
query failures do not make the CLI fail because they are represented by ordered
empty predictions.

## Testing Strategy

Implementation follows RED-GREEN-REFACTOR cycles.

Unit tests cover:

- fixed catalog order, strict values, and unique identifiers;
- rejection of an empty catalog and duplicate identifiers before execution;
- conversion of successful structured responses through
  `prediction_from_response`;
- preservation of empty, partial, soft-stop, and hard-stop response outputs;
- a middle query exception producing an empty record while later queries still
  execute;
- ordered one-to-one input/output cardinality;
- atomic protection of an existing artifact on write failure.

Integration tests compose the real `MockApiSearchService` and
`MockSearchOrchestrator` with fixed synthetic analyzer/provider dependencies.
They prove:

- the full catalog reaches `StructuredSearchResponse` before prediction
  serialization;
- each request receives a fresh orchestrator and budget controller;
- two complete runs produce byte-identical `predictions.jsonl`;
- no network or HTTP transport is used;
- the output directory contains only `predictions.jsonl`;
- no metrics or formal run artifacts are generated.

CLI tests prove that `--output` works and that split, gold, metric, endpoint,
and credential arguments are unavailable.

Final verification runs the complete suite with `--no-env-file`, Ruff, mypy,
`git diff --check`, a changed-path audit, and a scoped scan for credential
literals and protected-data references.

## Non-Goals

Task 8C does not:

- consume or validate the collaborators' 90/40 annotations;
- read gold, dev, frozen split, manifest, label, or protected-data files;
- calculate real or synthetic Recall, Precision, F1, or ranking quality;
- call OpenAlex, Semantic Scholar, an LLM, or any other external service;
- exercise the FastAPI transport added in Task 8B;
- add concurrency, retries, resume state, checkpoints, or multiple artifacts;
- establish a Week 2 quality gate or replace a later formal baseline run.
