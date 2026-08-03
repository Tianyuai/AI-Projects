# Current System Architecture

## Scope and evidence boundary

This document describes the integrated runtime now present in source. It distinguishes verified offline behavior from real-data and real-network evidence that remains blocked or not run. It makes no retrieval-quality, cost, or production-readiness claim.

## One application boundary

`CompositionRoot` builds the canonical `SearchApplicationService`. The same service boundary is used by smoke runs, formal evaluation, FastAPI, and the browser UI. There is no second evaluation-only or UI-only search pipeline.

A request flows through query analysis and planning, bounded multi-source retrieval, deduplication, hard filtering, reciprocal-rank fusion, optional named stages, response conversion, and canonical business-result projection. Every request owns a `HardBudgetController`; reservations occur before dependency work and are settled against actual safe usage.

## Replay and live isolation

The server is always composed around a verified replay lock and immutable dependency snapshot manifest. Its replay service is process-bound and has no live dependency client. A replay request reads only content-addressed snapshot bytes and verifies request identity, response hash, manifest hash, and snapshot-set identity.

Live is request-scoped. It requires all three authorization keys:

1. the replay lineage lock permits live execution;
2. the operator starts `paper-search serve` with `--allow-live`;
3. the request explicitly sets `mode: live`.

If any authorization predicate is missing, live is rejected. A forbidden lock cannot start a live-capable server. Each authorized live request constructs isolated clients, budget state, capture store, and application service. A successful response is exposed only after the capture is recorded, snapshots are sealed, replay lineage is written, evidence is validated, and the directory is atomically published. Failure or cancellation publishes only failed evidence and never a complete capture.

## Providers, snapshots, and credentials

OpenAlex, Semantic Scholar, and the configured LLM are accessed through typed adapters returning data, safe provenance, usage, latency, cache state, snapshot references, and sanitized error codes. Live adapters receive credentials only from the authorized child process environment. Replay adapters receive a `DependencySnapshotReader` and cannot fall back to network.

Snapshot manifests contain canonical non-secret request identities and exact response hashes. Raw response bytes and per-query artifacts remain access-controlled. Public API errors never expose exception text, headers, credentials, local paths, or raw dependency payloads.

## API and browser UI

`paper-search serve` exposes:

- `GET /health/live` for process liveness;
- `GET /health/ready` for cached safe mode/snapshot/dependency state;
- `POST /v1/search` for the canonical typed request;
- `/` and packaged static assets for the browser UI.

The UI posts only to `/v1/search`. It does not accept filesystem paths, arbitrary snapshot selection, credentials, or provider URLs. It renders selected papers, evidence fields, safe diagnostics, usage, partial/fallback state, snapshot time, configuration hash, and run ID from the canonical response.

## Formal evaluation and evidence

`paper-search evaluate` adapts the canonical service result into ordered execution, prediction, failure, business-result, usage, metric, and gate records. `FormalRunWorkspace` writes incomplete evidence first, validates canonical bytes and bindings, then publishes complete or failed state atomically.

`paper-search verify-run` is the machine predicate for a formal run. It validates the exact input lock, experiment identity, optional-module flags, dataset and policy bindings, snapshot evidence, record ordering/cardinality, aggregate metrics, and terminal publication state. `paper-search compare-replay` compares canonical `BusinessResultRecord` bytes between capture and replay.

Validation attempts are content-addressed and irrevocable. Recovery must match the archived lock bytes and full manifest binding; interruption does not authorize another attempt.

## Experiment registry

The registry accepts only these exact identities:

| Identity | Constructed optional behavior |
|---|---|
| `main-baseline` | none; fixed one round |
| `embedding` | embedding ranking only |
| `citation-expansion` | citation expansion only |
| `llm-rerank` | constraint/LLM reranking only |
| `fixed-two-round` | fixed two-round coordinator only |
| `adaptive-evolution` | adaptive evolution coordinator only |

Baseline planning and bounded multi-source routing are mandatory composition behavior, not optional ablation flags. Optional Provider/LLM stages share the request budget and preserve capture/replay references. Protected execution and integrity failures are never degraded into ordinary optional-stage warnings.

`configs/base.yaml` keeps `main-baseline`; no optional identity is promoted by implementation alone.

## Verified and deferred states

Offline unit/integration/E2E coverage, synthetic formal capture/replay verification, and real-browser replay acceptance are complete. The current public Gate 0 report remains blocked, so production Gate 0, real provider capture, formal dev/validation evidence, live-browser acceptance, measured quality/cost claims, and Gate 6 promotion evidence remain deferred pending their explicit authority and prerequisites.

The source evidence checkpoint is `fcc0ff0` on 2026-08-03. Reproducible repository evidence is implemented in `tests/e2e/test_dual_mode_serve.py`, `tests/integration/test_serve_process.py`, and `tests/fixtures/formal_run/`. The browser acceptance record is intentionally stored outside source control; it is not available from a fresh clone.
