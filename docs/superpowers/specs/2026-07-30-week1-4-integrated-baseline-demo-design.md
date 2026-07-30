# Week 1–4 Integrated Baseline and Demo Design

Date: 2026-07-30
Status: Approved in section-by-section design review
Design starting revision: `0dfdbd47dcd746e25b7d98b86a1ef126ba5dee47`

## Purpose

Weeks 1–4 are present in one Git history, but they do not yet form one runnable
product and evaluation path. The repository currently has:

- frozen-data and evaluation infrastructure;
- real OpenAlex, Semantic Scholar, LLM, cache, and budget boundaries;
- a synthetic loopback API composition;
- a separate server-rendered UI composition;
- optional embedding, citation, constraint-reranking, and adaptive-evolution
  components;
- offline evaluation and reporting utilities.

The missing deliverable is a single composition that turns those components
into both:

1. a reproducible formal fixed-one-round baseline; and
2. an interactive real-search demonstration.

This design establishes that common integration spine. It does not merge weekly
branches again: Week 3 is already an ancestor of the Week 1–4 baseline, and
Week 4 is already merged into it.

## Terms and Completion Semantics

- **V2 freeze** means the operator-approved, access-controlled gold partitions
  and identifier map produced by the second human-annotation freeze. Its
  existence is an operator assertion until Gate 0 verifies its files, hashes,
  schema, and approval evidence.
- **Capture run** means an explicitly authorized live execution that stores exact
  successful dependency response bytes in an access-controlled snapshot set,
  stores only safe request identity and response metadata, and seals a manifest.
- **Replay run** means an execution whose LLM and Provider adapters read only one
  composition-bound sealed snapshot set. It cannot open a network connection.
- **Formal run** means a run directory that passes the artifact validator and is
  marked `complete`. Formal validity is separate from quality-Gate passage: a
  valid run can faithfully prove that a quality threshold failed.
- **Business-result projection** is the canonical JSON projection of query
  analysis, selected paper IDs, relevance/evidence groups, citation edges,
  partial/fallback state, safe warnings, and stop reason. It excludes run IDs,
  timestamps, request IDs, live-versus-cache usage differences, and other
  transport metadata. Replay equivalence means byte equality of this canonical
  projection.
- **Safe run report** means the sanitized `run.json` plus aggregate metrics,
  usage, and failure counts. Raw dependency snapshots, gated queries, gold
  labels, and per-query predictions remain access-controlled artifacts and are
  not implied to be public.
- **Promotion Gate** means the separate evidence review that may enable an
  optional module in the main configuration. Completing its implementation or
  ablation does not imply promotion.

## Verified Starting State

At the baseline revision:

- `paper_search.api.app:app` is deliberately uncomposed and reports degraded
  readiness;
- the runnable mock server uses synthetic analysis and retrieval rather than
  real Providers;
- `paper_search.ui.app:app` injects an unavailable service and consumes the
  evaluation-only `PipelineResult`, not the API response contract;
- `MockSearchOrchestrator` can accept real adapters and optional stages, but no
  formal composition root constructs that combination;
- the evaluation runner and the API use different execution paths;
- `configs/base.yaml` disables embedding, and the optional advanced stages are
  not part of a main runtime composition;
- `data/manifest.json` still says `waiting_for_human_label_freeze`, although the
  operator states that frozen V2 data and the identifier map are available;
- `data/provider_readiness.json` contains sanitized historical readiness
  evidence, but it is not a current authorization or health check.

The clean baseline verification for this design branch is:

```text
936 passed, 2 skipped
```

The two skips are the expected Windows symlink-privilege case and the
credential-gated OpenAlex live test.

This integration is not an adapter-only change. It requires explicit migrations
to the application result contract, dependency snapshots, cost accounting,
evaluation failure isolation, run publication, API/UI contracts, and freeze
manifest schema.

## Goals

- Provide one formal composition root for configuration, frozen data,
  request-scoped budgets, LLM planning, OpenAlex, Semantic Scholar, cache and
  snapshots, the fixed-one-round pipeline, and optional experimental stages.
- Provide one `SearchApplicationService` used by smoke tests, evaluation, API,
  and UI.
- Make snapshot replay the safe and reproducible default.
- Permit real LLM and Provider calls only through an explicit live
  authorization and hard budgets.
- Run a small smoke, a complete dev baseline, freeze the selected configuration,
  and then run validation without further tuning.
- Produce predictions, metrics, usage, failures, snapshot provenance, config
  hash, Git SHA, and a sanitized run report for every formal run.
- Wire advanced Week 3–4 modules into the composition while keeping them off by
  default and available only through explicit experiment configurations.
- Reconcile public documentation and manifests with the verified V2 freeze
  state without publishing gated labels or credentials.

## Non-Goals

The following do not block this integration release:

- promoting embedding, citation expansion, constraint reranking, or adaptive
  evolution into the default baseline;
- relationship-graph visualization;
- cloud production deployment;
- multi-tenant authentication and authorization;
- large-scale load testing;
- inventing missing metric, cost, or quality thresholds.

Existing authoritative PRD thresholds must be reused. Any threshold that is not
already authoritative must be approved before the dev run and included in the
frozen configuration; it must not be selected from validation results.

## Decision and Alternatives

### Selected: layered integration spine

Build one application boundary, pass it through offline replay, authorized live
smoke, dev evaluation, frozen validation, and API/UI delivery gates, and only
then run advanced-module ablations.

This was selected because it keeps the formal baseline and the demo on the same
code path while preserving a clear failure boundary at every stage.

### Rejected: big-bang composition

Composing data, live dependencies, evaluation, API, UI, and all advanced stages
at once would make budget, Provider, data, and metric failures difficult to
attribute.

### Rejected: parallel evaluation and demo stacks

Separate product and evaluation tracks could show a UI earlier, but they would
recreate the current drift: a successful demo would not prove that the formal
evaluation path is runnable or reproducible.

## Architectural Invariant

The formal baseline and the demo are two consumers of one application service,
not two implementations.

```text
Frozen V2 inputs ─┐
Replay snapshots ─┼─> CompositionRoot ─> SearchApplicationService
Authorized live ──┘          │                      │
                             │                      ├─> smoke command
                             │                      ├─> evaluation runner
                             │                      ├─> FastAPI /v1/search
                             │                      └─> browser UI via API
                             │
                             └─> config, budget, LLM, Providers, cache,
                                 fixed baseline, optional-stage registry
```

All consumers share:

- request and response contracts;
- one internal execution-envelope contract;
- query-planning and deterministic fallback rules;
- Provider adapters and cache keys;
- hard-budget reservation and settlement;
- retry, timeout, and partial-result semantics;
- snapshot and configuration identities;
- error codes and sanitization.

No consumer may directly construct a second pipeline.

## Proposed Package Boundaries

Add an application package rather than expanding the mock-specific service:

```text
src/paper_search/application/
├── __init__.py
├── composition.py
├── service.py
├── modes.py
├── artifacts.py
└── experiments.py
```

### `composition.py`

`CompositionRoot` loads and validates committed configuration, resolves the
selected execution mode, builds the LLM analyzer, Provider adapters, cache and
snapshot wrappers, and constructs a request-scoped orchestrator factory.

It returns an immutable application bundle containing:

- `SearchApplicationService`;
- a mode-aware readiness probe;
- validated configuration identity;
- an artifact factory;
- the enabled experiment identity.

Environment reads happen at composition time, not at module import. Secret
values are passed only to external clients and are never stored in the bundle.

### `service.py`

`SearchApplicationService` accepts the canonical `SearchRequest`. Its internal
`execute()` method returns a `SearchExecutionResult` containing:

- exactly one typed success or hard-failure outcome;
- safe dependency diagnostics and stable failure codes;
- per-dependency cache and snapshot references;
- prompt, model, Provider, latency, and actual-usage identities needed by
  evaluation and capture.

Its API-facing `__call__()` returns the public response for a success or raises
a typed application exception carrying the safe `SearchErrorResponse` for a hard
failure. The API handler may call `execute()` directly to map that outcome to
HTTP. Evaluation calls the same service's `execute()` method; it does not
reconstruct the orchestrator or infer discarded diagnostics from the public
model.

For every request the service:

1. validates replay/live authorization;
2. creates a fresh `HardBudgetController` for the selected profile;
3. creates the configured single-run orchestrator;
4. executes query planning and retrieval;
5. converts the result to the public response contract;
6. records structured provenance, failure, and actual-usage events;
7. removes public trace entries only when the request disables tracing.

The current `MockApiSearchService` remains available for mock tests during
migration, but it is not the formal integration service.

The current contracts require explicit extensions:

- `SearchRequest.mode: Literal["replay", "live"] = "replay"`;
- `StructuredSearchResponse` gains `run_id`, `execution_mode`,
  `snapshot_set_id`, `snapshot_captured_at`, `planner_fallback`, and safe
  dependency-status fields;
- API error responses gain stable safe codes instead of collapsing every
  failure to one generic 503;
- readiness gains the selected mode, bound snapshot identity, and last
  explicitly authorized probe time.

For replay, the composition root receives both a replay lock and an operator CLI
manifest path. The path must resolve under the configured access-controlled
artifact root, its bytes must match the lock's
`snapshot_manifest_sha256`, and its internal `snapshot_set_id` must match the
lock. Any mismatch is `config_mismatch`. The root then binds that one set for its
lifetime. A browser cannot provide an arbitrary filesystem path or switch the
server to another snapshot set.

### `modes.py`

Define two public modes:

- `replay`: the default. LLM and Provider results must resolve from a validated
  immutable snapshot set. A miss returns `snapshot_unavailable`; it never
  switches to live execution.
- `live`: performs authorized external calls and stages exact successful
  response bytes with sanitized metadata for an immutable snapshot. It requires
  both the composition setting
  `runtime.allow_live=true` and explicit `mode=live` selection. CLI execution
  additionally requires `--allow-network` for one-shot `smoke` and `evaluate`
  commands. For the long-running server, `--allow-live` is the command-level
  network authorization; each request must still select `mode=live`.

Live is therefore a two-key operation: a browser request alone cannot enable
network access, and starting a live-capable server alone does not make every
request live.

Every live evaluation or smoke run is a capture run. A successful capture seals
the snapshots and manifest before the run is marked complete. Replay equivalence
is then checked without network access.

Live input locks bind `dependency-snapshot-v2` and a capture-policy hash, not a
manifest that does not yet exist. The completed capture records the generated
manifest hash in `run.json` and emits the replay lock that binds it.

The existing `provider-snapshot-v1` exporter is not the replay implementation:
it has no read-only snapshot adapter, is Provider-only, and names exported files
as OpenAlex files. Introduce a versioned `dependency-snapshot-v2` contract and
store with:

- entries for LLM, OpenAlex, and Semantic Scholar;
- dependency kind/name, endpoint, model or adapter version, canonical safe
  request identity, exact response hash, capture time, and relative response
  path;
- a read-only key-to-response index used by replay adapters;
- a fail-closed miss with no reference to a live client;
- migration/validation support for existing Provider snapshots where their
  identity is sufficient.

LLM and Semantic Scholar capture must expose cache/snapshot keys just as
OpenAlex does. Exact successful response bytes are retained only in the
access-controlled run root. Authorization headers, secret request fields, and
unsafe error bodies are never persisted; public safe reports contain only
aggregate or hashed identities.

### `artifacts.py`

Write each run under a temporary, run-scoped directory and publish it atomically:

```text
runs/<run_id>/
├── run.json
├── config.lock.yaml
├── replay.lock.yaml
├── snapshot-manifest.json
├── predictions.jsonl
├── metrics.json
├── usage.json
└── failures.jsonl
```

`run.json` records:

- run ID and status (`incomplete`, `failed`, or `complete`);
- Gate result (`passed`, `failed`, or `not_applicable`) kept separate from
  artifact-completion status;
- replay/live mode;
- split and frozen-data identities;
- Git SHA and whether tracked source/config files were dirty;
- config hash and prompt version;
- snapshot-manifest hash;
- experiment name and advanced-stage flags;
- start/end timestamps;
- sanitized environment and dependency readiness summaries.

Failed and interrupted directories remain available for diagnosis but cannot be
mistaken for a complete formal result.

`replay.lock.yaml` is emitted only by a successful capture. It copies the
capture policy and adds the sealed `snapshot_set_id` and
`snapshot_manifest_sha256`. A replay run consumes that lock; it does not invent
or rediscover a snapshot identity.

The dev input lock and validation lock are distinct:

- a pre-dev `candidate.lock.yaml` identifies the one approved main-baseline
  candidate and its snapshot schema/capture-policy hash; it contains no
  snapshot-manifest hash before capture;
- after a successful dev review, a content-addressed
  `validation.lock.yaml` is promoted as a Git-external, access-controlled
  artifact and binds the already committed source SHA plus the same capture
  policy; it contains no validation snapshot hash before capture;
- each run copies its exact input lock to `config.lock.yaml`.

The validation lock is not committed after embedding its own Git SHA, avoiding a
self-referential commit identity. Before the first validation network dispatch,
the runner performs every offline preflight and reserves the run budget, then
atomically creates `validation-attempts/<lock-sha256>.claim` with exclusive
create-if-absent semantics. The claim records run ID, timestamp, and state.
States transition atomically from `claimed` to exactly one of `complete`,
`failed`, or `interrupted`; they never return to unclaimed. Creation of the claim
irrevocably consumes that lock's live validation attempt, even if the process
crashes before a successful response. A retry requires a human-approved
superseding lock hash and incident note. Replay verification of the sealed
capture is recorded separately and never creates a live-attempt claim.

### `experiments.py`

Map the existing ablation registry to injected optional stages:

- embedding ranking;
- citation expansion;
- constraint reranking;
- fixed two rounds;
- adaptive evolution.

The main baseline experiment is fixed one round with all optional stages off.
Advanced stages may be constructed only when an explicit experiment name
enables them. They must be compared with the same frozen data, budgets,
snapshots, and measurement rules before any promotion decision.

This requires more than mapping current booleans. The existing
`citation-expansion` and `llm-rerank` YAML cases must be corrected because they
currently leave their named flags false. Single-round stages need production
adapters with the correct async/budget behavior: local embedding, Provider-call
citation expansion, and LLM-call reranking. Fixed-two-round and adaptive
strategies remain `EvolutionCoordinator` strategies wrapping a shared
single-round executor; they are not inserted as post-retrieval orchestrator
stages.

## Baseline Composition

The one main-baseline candidate is fixed before the complete dev capture:

| Identity | Fixed value |
| --- | --- |
| LLM primary/fallback | `qwen3.7-plus` / `qwen3.6-flash` |
| Prompt | `configs/prompts/query_analyze.yaml`, `query-analyze-v1` |
| Retrieval | OpenAlex `/works` main source plus Semantic Scholar `/graph/v1/paper/search` supplement |
| Routing | OpenAlex 3–6 calls; Semantic Scholar only the top 1–2 high-priority or uncovered-constraint subqueries; never unconditional dual-source fan-out |
| Strategy | fixed one round |
| Planning | one combined analysis/plan call, normally 3–5 subqueries |
| Retrieval caps | 50 results per subquery, 300 raw candidates, 200 deduplicated candidates |
| Output cap | 50 papers |
| Processing | deterministic deduplication, hard filtering, reciprocal-rank fusion |
| HTTP timeout | connect 5 s, read 20 s, write 20 s, pool 5 s |
| Retry | at most 3 attempts per external operation, only for timeout/429/5xx, bounded by the same reservation; backoff `min(8, 2^retry_index) + jitter[0,1)` |
| Request budget | committed `configs/budget_balanced.yaml` |
| Optional modules | embedding, citation expansion, constraint reranking, fixed-two-round, and adaptive evolution all off |
| Frozen data | Gate-0-approved V2 manifest and identifier-map hashes |
| Dependency inputs | live locks bind capture-policy/schema identity; replay locks bind the exact sealed manifest hash |

The balanced request budget currently fixes 12 search calls, 5 LLM calls, 2
iterations, 6 maximum configured subqueries, 90 elapsed seconds, 80 soft-deadline
seconds, 24,000 total tokens, 50 output papers, and CNY 0.30 maximum cost.
The baseline lock records the complete file hash rather than relying only on
this summary.

The deterministic rules planner is a declared fallback, not a separate
baseline. A syntactically invalid LLM plan receives at most one bounded repair
attempt. If repair still fails, the frozen rules fallback may continue the
query, and the response records `planner_fallback=true`. Transport,
authentication, or unaccounted-usage failures do not masquerade as a valid LLM
plan.

Current clients do not provide sufficient formal cost accounting: Qwen and both
Provider adapters can report `cost_cny=None`. Gate 0 therefore adds an
operator-approved, versioned pricing policy with effective date, source
identity, model/Provider units, and deterministic rounding. Gate 2 cannot pass
until actual tokens/calls can be valued or an authoritative billed amount can be
reconciled.

Request budgets are nested under a run budget and a project ledger:

- dev live capture: at most 60 × CNY 0.30 = CNY 18.00;
- validation live capture: at most 30 × CNY 0.30 = CNY 9.00;
- the actual caps are recomputed from Gate 0's approved partition counts and
  recorded in each lock;
- the PRD project hard cap is CNY 200, and large repeated dev experiments stop
  when cumulative spend reaches CNY 160;
- replay runs consume no external-call budget and are the only repeatable mode
  after capture.

`HardBudgetController` continues to enforce each request. A new run-level ledger
must reserve before dispatching a query and settle its actual usage afterward;
it must not replace or weaken request-scoped accounting.

## Authoritative Quality and Integrity Gates

This integration does not invent thresholds. It makes the existing PRD and Week
1/R3 rules executable.

Let `Q` be the ordered set of every query ID authorized by the validated frozen
split and `N = len(Q)`. Every failed query remains in `Q`. Before a formal live
capture, "normal network" means a Gate-2-compatible authorized readiness probe
for the exact LLM/Provider configuration reports all three dependencies ready
within the preceding 15 minutes. If that predicate is false, the formal capture
does not start and no validation-attempt claim is created.

| Class | Measure and formula | Required rule | Applies |
| --- | --- | --- | --- |
| Formal validity | prediction query IDs versus `Q` | exactly one prediction entry for every member of `Q`, in order, with no extras or duplicates | every dev/validation run |
| Formal validity | hard-failure records | exactly one supplemental failure entry linked to each hard-failed query and none for other queries | every dev/validation run |
| Formal validity | hash, provenance, sanitization, unaccounted-usage, and artifact-validator failures | exactly 0 | every complete run |
| Formal validity | elapsed time and valued actual cost | every request/run/project ledger remains within its frozen hard cap | every live capture |
| Baseline quality | valid model-produced `QueryAnalysisResult` after initial call or one repair, without rules fallback / `N` | at least 99% | dev and validation |
| Baseline quality | correctly extracted audited strong-constraint field-value pairs / all audited strong-constraint pairs | at least 90% | frozen constraint audit |
| Baseline quality | queries with at least one parseable successful configured retrieval response and no pre-retrieval hard failure / `N` | at least 95% | normal-network dev and validation |
| Baseline quality | human-confirmed correct fuzzy merge decisions / all frozen audited fuzzy decisions | at least 98%; an empty audit is invalid | frozen dedup audit |
| Baseline quality | `Recall(dedup_input) - Recall(hard_filter_accepted)` over mapped relevant IDs | at most 0.02 absolute | dev and validation |
| Baseline quality | aggregate R3 identifier-map Recall | macro Recall > 0 and micro Recall > 0 | dev and validation |
| Reporting only | macro/micro Precision, Recall, F1, and Recall@5/10/20 | calculable and reported; no minimum Week 1 F1 is invented | dev and validation |
| Reporting only | hard-failure rate, partial-result rate, planner-fallback rate, P50/P95 latency, calls, tokens, valued cost, and cache-hit rate | report numerator, denominator, and value | dev and validation |

A dev capture may promote a validation lock only when it is formally valid and
every baseline-quality row passes. Validation Gate passage likewise requires
formal validity and every baseline-quality row applicable to validation. A
formally valid run that misses quality remains preserved with status
`complete` and Gate result `failed`; it is evidence, not a successful Gate.

Optional-module promotion keeps the PRD rule: across three same-configuration
dev runs, median macro-F1 delta must be at least `+0.01`, the 1,000-sample
bootstrap 95% lower bound must be at least `-0.005`, and validation macro F1
must not fall by more than `0.01`. An implemented ablation that misses this rule
remains default-off.

## Evaluation Adapter

Refactor the formal evaluation entry point to call
`SearchApplicationService.execute()`. It converts the execution envelope into
the existing prediction and metric models; it does not directly construct an
OpenAlex-only pipeline.

The adapter must preserve:

- every frozen query ID, including failed queries;
- canonical paper identifiers through the verified identifier map;
- per-query Provider diagnostics and cache references;
- actual usage and latency;
- partial, fallback, and stop-reason fields;
- the frozen run identity.

A query failure produces a prediction record and a failure record rather than
being dropped. The batch may continue to collect diagnostics, but its final
Gate result uses the authoritative table above.

This is a controlled rewrite of evaluation orchestration, not a thin converter.
The current runner aborts on several per-query exceptions and writes directly to
the final output directory. The new runner must isolate each query, maintain
ordered records for the full split, use the run-level budget ledger, stage all
artifacts under an incomplete directory, validate the entire run, and only then
publish it atomically.

## API and UI

The supported runtime starts from an explicit command that loads configuration
and injects the application bundle into `create_app`. Importing a module must not
implicitly read credentials or enable network access.

The API retains:

- `GET /health/live`;
- `GET /health/ready`;
- `POST /v1/search`.

The API contract migration is part of this scope. Expected domain/dependency
failures map to stable 4xx/5xx error codes and bodies; only unexpected internal
exceptions use the generic safe `internal_error` 500. The response converter
must populate the existing relevance/evidence groups from the fused/ranked
output instead of discarding RRF scores, and it must carry the new run,
snapshot, fallback, and safe dependency fields.

Readiness is mode-aware:

- replay readiness requires a valid snapshot manifest and usable cache;
- live readiness requires authorized configuration and reports the latest
  explicitly run dependency probe rather than generating billable traffic on
  every health request.

The UI stops consuming `evaluation.runner.PipelineResult`. It submits the
canonical API request and renders `StructuredSearchResponse`, including papers,
scores/evidence, Provider sources, actual usage, partial/fallback state, safe
warnings, snapshot time, config hash, and run ID. The UI does not receive a
pipeline object and cannot construct Providers. Its browser code posts to
`/v1/search`; there is no hidden server-side UI search composition.

## Normative Application Contracts

The implementation may split these models across modules, but it must preserve
their fields and semantics. Existing domain types such as `QueryAnalysisResult`,
`RankedPaper`, `ResolvedCitationEdge`, `UsageActual`, and `ErrorDetail` are
reused rather than duplicated.

```python
SearchMode = Literal["replay", "live"]
DependencyName = Literal["llm", "openalex", "semantic_scholar"]
DependencyState = Literal["ready", "replayed", "degraded", "failed"]
PlannerStatus = Literal["primary", "repaired", "rules_fallback"]
DependencyErrorCode = Literal[
    "timeout",
    "network_error",
    "rate_limited",
    "server_error",
    "authentication_error",
    "invalid_request",
    "invalid_response",
    "invalid_record",
    "missing_record",
    "empty_response",
    "invalid_json",
    "budget_exhausted",
    "provider_error",
]
SearchErrorCode = Literal[
    "invalid_request",
    "live_not_authorized",
    "config_mismatch",
    "snapshot_unavailable",
    "budget_exhausted",
    "dependency_failure",
    "integrity_failure",
    "validation_attempt_conflict",
    "internal_error",
]


class SearchRequest(DomainModel):
    query_id: NonEmptyStr
    query: NonEmptyStr
    budget_profile: Literal["low", "balanced"] = "balanced"
    include_trace: bool = True
    mode: SearchMode = "replay"


class DependencyStatus(DomainModel):
    dependency: DependencyName
    state: DependencyState
    cache_hit: bool
    error_codes: list[DependencyErrorCode]


class StructuredSearchResponse(DomainModel):
    run_id: NonEmptyStr
    query_id: NonEmptyStr
    execution_mode: SearchMode
    snapshot_set_id: NonEmptyStr
    snapshot_captured_at: datetime | None
    query_analysis: QueryAnalysisResult
    selected_paper_ids: list[NonEmptyStr]
    high_relevance: list[RankedPaper]
    partial_relevance: list[RankedPaper]
    citation_edges: list[ResolvedCitationEdge]
    search_trace: list[dict[str, object]]
    usage: UsageActual
    stop_reason: NonEmptyStr
    is_partial: bool
    planner_fallback: bool
    planner_status: PlannerStatus
    dependency_status: list[DependencyStatus]
    warnings: list[NonEmptyStr]
    prompt_version: NonEmptyStr
    config_hash: Sha256
    git_sha: NonEmptyStr


class SnapshotRef(DomainModel):
    entry_id: NonEmptyStr
    dependency: DependencyName
    cache_key: Sha256
    response_sha256: Sha256
    captured_at: datetime
    snapshot_path: SafeRelativePath


class DependencyDiagnostic(DomainModel):
    dependency: DependencyName
    endpoint: NonEmptyStr
    model_id: NonEmptyStr | None
    usage: UsageActual
    latency_ms: NonNegativeInt
    cache_hit: bool
    snapshot_refs: list[SnapshotRef]
    errors: list[ErrorDetail]


class SearchErrorResponse(DomainModel):
    code: SearchErrorCode
    detail: NonEmptyStr
    retryable: bool
    run_id: NonEmptyStr | None


class SearchSuccess(DomainModel):
    kind: Literal["success"] = "success"
    response: StructuredSearchResponse


class SearchFailure(DomainModel):
    kind: Literal["failure"] = "failure"
    query_id: NonEmptyStr
    run_id: NonEmptyStr
    error: SearchErrorResponse
    usage: UsageActual
    stop_reason: NonEmptyStr


SearchOutcome = Annotated[
    SearchSuccess | SearchFailure,
    Field(discriminator="kind"),
]


class SearchExecutionResult(DomainModel):
    outcome: SearchOutcome
    diagnostics: list[DependencyDiagnostic]
    business_result_sha256: Sha256 | None


class ReadyHealthResponse(DomainModel):
    status: Literal["ready", "degraded"]
    execution_mode: SearchMode
    snapshot_set_id: NonEmptyStr | None
    dependencies: list[DependencyStatus]
    last_authorized_probe_at: datetime | None
```

`Sha256` accepts only `sha256:` plus 64 lowercase hexadecimal characters.
`SafeRelativePath` is POSIX-normalized, non-absolute, contains no `..`, and must
resolve under the bound artifact root.

`dependency_status` is ordered `llm`, `openalex`, `semantic_scholar`. Public
warnings and `SearchErrorResponse.detail` come from fixed safe templates, not
external exception text. Internal diagnostics may contain only the existing
sanitized `ErrorDetail` model and safe endpoint/model identities.

Every public dependency error maps to `DependencyErrorCode`; an unknown
third-party code becomes `provider_error` rather than entering the public
contract. `planner_status="rules_fallback"` is represented by
`planner_fallback=true`, `is_partial=true`, and a fixed safe warning.
`planner_status="primary"` or `"repaired"` requires
`planner_fallback=false`.

Top-level HTTP behavior is:

| Code | HTTP status |
| --- | --- |
| `invalid_request` | 400 |
| `live_not_authorized` | 403 |
| `config_mismatch` | 409 |
| `validation_attempt_conflict` | 409 |
| `budget_exhausted` before useful work | 429 |
| `snapshot_unavailable` | 503 |
| `dependency_failure` before useful work | 503 |
| `integrity_failure` | 500 |
| `internal_error` | 500 |

One-Provider degradation, planner rules fallback, or budget exhaustion after
useful work returns HTTP 200 with `is_partial=true` and the corresponding status,
warning, and stop reason. `planner_fallback=true` is always explicit.

## Command Outcomes

Expose one stable `paper-search` console entry point. These command contracts are
part of the design:

```text
paper-search smoke \
  --lock <live-input-or-replay-lock> \
  --output-root <runs> \
  [--mode replay|live] \
  [--snapshot-manifest <manifest>] \
  [--allow-network]

paper-search evaluate \
  --lock <live-input-or-replay-lock> \
  --split dev|validation \
  --mode replay|live \
  --output-root <runs> \
  [--snapshot-manifest <manifest>] \
  [--allow-network]

paper-search serve \
  --lock <replay-lock> \
  --mode replay \
  --snapshot-manifest <manifest> \
  --capture-output-root <runs> \
  [--allow-live]

paper-search verify-run <run-directory>
paper-search compare-replay <capture-run> <replay-run>
```

Required behavior:

- `smoke` defaults to replay and validates the end-to-end application path;
- replay commands require a replay lock and `--snapshot-manifest`; a missing,
  out-of-root, or hash-mismatched manifest fails before composition;
- live `smoke` requires a live-capable lock and `--allow-network`, writes a
  capture run under `--output-root`, seals it, and emits a replay lock;
- `evaluate` requires a frozen input lock and explicit mode;
- live evaluation requires both `runtime.allow_live=true` in the lock and
  `--allow-network`;
- `evaluate --split validation --mode live` additionally consumes the
  validation-attempt ledger and refuses a second live attempt for the same lock;
- `serve` starts the unified API/UI application bound to one replay snapshot
  set. When `--allow-live` is present, every live request creates an atomic
  per-request capture under `--capture-output-root`; the API returns success only
  after that capture is sealed and validated;
- a server accepts a live API request only when started with `--allow-live`, the
  lock permits live, and `SearchRequest.mode` is `live`;
- `verify-run` is the machine predicate for formal-run validity;
- `compare-replay` compares the defined business-result projection;
- all commands print only safe summaries and the resulting run ID/path.

## Data and Freeze Contract

The operator states that frozen V2 data, the identifier map, and real
credentials are available. Gate 0 verifies that assertion before implementation
claims or formal runs.

The repository must expose one authoritative, non-secret manifest that records:

- V2 schema and dataset revision;
- dev and validation counts and hashes;
- gold and identifier-map hashes without publishing gated contents;
- partition immutability and zero-answer policy;
- annotation/freeze status;
- creation and approval provenance.

If the current private V2 freeze cannot be reconciled with the public manifest,
Gate 0 fails. The implementation must not simply change
`waiting_for_human_label_freeze` to a success label without validating the
referenced files and hashes.

The current freeze validator requires exact V1 field sets, so this is a
versioned schema migration, not a hand edit. Add a V2 manifest/approval schema,
an explicit V1-to-V2 migration path, and tests for exact fields, path
confinement, approval matching, and no-overwrite behavior. Existing V1 evidence
remains readable; only a fully validated V2 approval may authorize the formal
integrated run.

After the accepted dev run, the promoted `validation.lock.yaml` freezes:

- data and identifier-map identities;
- prompt and model identities;
- Provider endpoints and result limits;
- timeout and retry counts;
- planner repair/fallback rules;
- budget limits and cost assumptions;
- cache and snapshot schema versions;
- all optional-stage flags;
- all authoritative integrity, quality, and promotion thresholds;
- source Git SHA.

The validation run copies that file to its run-local `config.lock.yaml` and does
not write configuration changes back to it.

## Stage Gates

### Gate 0: evidence and status reconciliation

Verify V2 files, identifier map, partition hashes, approvals, and sanitized
credential-name readiness. Publish a single authoritative manifest, approve the
versioned pricing policy, verify the PRD threshold table, and correct stale
status documentation.

Pass evidence:

- validated manifest;
- partition and identifier-map hash report;
- sanitized readiness report;
- pricing-policy and threshold identities;
- documentation diff.

### Gate 1: replay integration smoke

Run a representative query through the new composition with network access
blocked. Gate 1 uses a committed fixture or previously authorized smoke snapshot;
its replay lock binds that manifest. It is an engineering Gate, not the formal
dev result.

Pass conditions:

- complete structured response;
- valid budget ledger;
- snapshot and config provenance;
- a second replay has the same canonical business-result projection;
- no external socket or name-resolution call.

### Gate 2: authorized live smoke and capture

Run a small representative set against Qwen, OpenAlex, and Semantic Scholar with
strict request, token, time, and cost ceilings.

Pass conditions:

- hard budgets are respected;
- actual usage is accounted;
- failures have stable safe codes;
- exact response bytes with safe metadata are sealed in the access-controlled
  snapshot set;
- the capture emits a manifest-bound replay lock;
- replay reproduces the captured business-result projections.

### Gate 3: complete dev baseline

Run one authorized fixed-one-round live capture over the full dev split using the
pre-dev candidate lock. This live capture is the authoritative scored dev run.
Immediately run replay as reproducibility evidence; the replay is not a second
scored experiment.

Review metrics, failures, cost, latency, and cache behavior. The main baseline
has one pre-approved candidate, not an open parameter search. If an integration
defect or predeclared policy choice requires a change, create a new candidate
lock and rerun dev; keep every earlier run. No validation result participates in
that decision.

Pass evidence:

- complete predictions, metrics, usage, and failure artifacts;
- no missing or silently removed query;
- explained identifier-map coverage;
- all authoritative Gate thresholds evaluated;
- a promoted, content-addressed validation lock.

### Gate 4: frozen validation

Execute one authorized live validation capture using the promoted validation
lock and attempt ledger. This live capture is the authoritative scored
validation result. Then replay its sealed snapshot once to verify the business
projection; replay is evidence, not a second formal tuning opportunity.

Pass conditions:

- the lock, source revision, and input hashes match before execution;
- no prior live validation attempt exists for the lock hash;
- no parameter is selected from validation output;
- all validation queries have a prediction/failure record;
- the formal artifact directory validates and is published atomically.

Replaying the sealed validation capture to verify determinism is allowed; using
that replay to change configuration and rerun the formal validation is not.

### Gate 5: dual-mode API/UI delivery

Start the unified service in replay mode and demonstrate the same structured
results through API and UI. Separately demonstrate an explicitly authorized,
hard-budgeted live request and its visible provenance/degradation state.

### Gate 6: optional-module ablations

Run advanced stages only after the baseline delivery. Every comparison uses the
same frozen inputs and budget rules. An advanced stage remains default-off
unless its separate promotion Gate is approved.

## Migration Dependency Order

The implementation plan must preserve these dependencies:

1. extend versioned request/response, execution-envelope, freeze, snapshot,
   pricing, lock, and artifact schemas;
2. implement read-only replay and live-capture adapters for LLM, OpenAlex, and
   Semantic Scholar;
3. add request/run/project budget accounting and the formal composition root;
4. migrate evaluation to the execution envelope and atomic run publisher;
5. migrate API errors/readiness and UI rendering to the canonical response;
6. wire optional single-round stages and multi-round strategies behind corrected
   named experiment configurations;
7. run Gates 0–5 before any Gate-6 promotion work.

Later layers may use fake implementations while earlier schema work is under
test, but no layer may introduce a second production composition.

## Error and Degradation Semantics

### Fail closed

Stop the current Gate and do not produce a successful report when:

- data, identifier-map, partition, config, source, or snapshot hashes differ;
- validation attempts to change frozen parameters;
- budget usage cannot be measured or reconciled;
- replay data is missing and code attempts an implicit live call;
- an artifact lacks required provenance;
- sanitization detects credentials, authorization headers, or unsafe payloads.

At query scope, the following produce a structured hard-failure record rather
than a partial success:

- a replay snapshot miss;
- planner transport or authentication failure;
- both retrieval Providers fail before any candidate commits;
- hard-budget exhaustion before any useful work commits.

The evaluation batch may continue to preserve diagnostics, but the failed query
remains in every denominator and affects the authoritative response-rate Gate.

### Explicit bounded degradation

The application may return a partial result when:

- one Provider times out or is rate-limited while the other succeeds;
- an invalid LLM plan uses the frozen bounded repair/fallback path;
- the live hard budget is reached after some valid work has committed;
- the evaluation batch continues after recording a hard query failure; the
  failed query itself is not relabeled as a partial success.

Every such response records:

- `is_partial`;
- stable failure and fallback codes;
- affected dependency;
- committed actual usage;
- cache/snapshot origin;
- stop reason.

Timeouts and retries are bounded and consume the same hard budget. Previously
committed usage is never erased. A retry or fallback rule is part of the frozen
configuration.

## Security and Operational Safety

- Credentials come from the process environment or an injected secret provider;
  configuration contains names, never values.
- For publication and logging, an unsafe payload is any authorization header,
  credential-shaped value, raw external error body, non-allowlisted request
  field, gated query/label, or unvalidated path.
- Live execution needs both server/operator authorization and per-command or
  per-request selection.
- Replay cache misses never trigger network access.
- Snapshot headers, errors, logs, and run reports pass through one sanitizer.
- Readiness endpoints do not expose credential state beyond safe capability
  names and do not perform recurring billable probes.
- Temporary run directories publish only after artifact validation.
- Formal runs require committed source/config state; private gated inputs are
  identified by hashes even when they are not tracked.

## Testing Strategy

Implementation follows red-green TDD.

### Unit tests

- mode and live-authorization validation;
- configuration locking and hash stability;
- planner repair/fallback classification;
- request/run/project budget preflight, reservation, settlement, pricing, and
  exhaustion;
- artifact atomicity and sanitization;
- experiment flags defaulting to off.

### Contract tests

- LLM, OpenAlex, and Semantic Scholar adapter envelopes;
- LLM/OpenAlex/Semantic Scholar replay/capture snapshot schema, key selection,
  exact-byte replay, and hash validation;
- `SearchRequest` and `StructuredSearchResponse`;
- `SearchExecutionResult` diagnostic preservation;
- stable public error and readiness codes.

### Integration tests

- one composition root constructs replay and fake-live services;
- replay cannot access the network;
- capture seals a snapshot before publishing a complete run;
- evaluation and API receive equivalent results from the same service;
- UI renders the API response rather than a separate pipeline model;
- V1/V2 freeze-schema migration preserves exact approval semantics;
- a second live validation attempt for one lock hash is rejected;
- optional stages are absent from the baseline and present only in named
  experiments.

### End-to-end and formal checks

- one-command replay smoke;
- bounded authorized live smoke;
- complete dev run and artifact validation;
- frozen validation run and artifact validation;
- capture/replay business-projection comparison;
- API/UI replay demonstration;
- API/UI authorized live demonstration;
- full pytest, Ruff, and mypy verification.

Online checks remain explicitly marked and are never part of the default
credential-free test suite.

## Acceptance Criteria

The integration is complete only when all of the following are true:

- replay smoke runs end to end with one command and no network access;
- repeated replay produces the same canonical business-result projection;
- the live smoke uses both real Providers and real LLM planning within hard
  limits and produces access-controlled, safe-metadata, replayable snapshots;
- full dev and frozen validation runs contain every query and all required
  artifacts;
- the formal dev/validation captures and their replay evidence use the exact
  mode chronology defined in Gates 3–4;
- metrics include relevance, cost, latency, hard failures, partial results, and
  cache behavior;
- every authoritative integrity and quality rule in the Gate table is evaluated
  with its numerator and denominator;
- every complete run records the snapshot manifest, config hash, prompt
  identity, source Git SHA, and safe report;
- API readiness is healthy for the selected validated mode;
- UI uses the structured API contract and displays provenance and degradation;
- README, architecture, limitations, runbook, and public manifest agree with the
  verified V2 state;
- all pre-existing and new tests pass, Ruff is clean, and mypy is clean;
- optional advanced modules remain default-off and test-protected.

`paper-search verify-run` and `paper-search compare-replay` must both exit zero
for every delivered complete capture/replay pair. Nonzero exits are the
machine-readable evidence that the integration is not complete.

At that point, Weeks 1–4 have moved from separately frozen component work to one
complete, reproducible, and demonstrable flow.

## Risks and Mitigations

### V2 freeze evidence may not match the stale public manifest

Gate 0 verifies files and hashes before changing status. A mismatch blocks
formal evaluation but does not block application-layer unit work.

### Live Provider behavior may have changed since the historical readiness file

Gate 2 performs a new, bounded, explicitly authorized probe and captures safe
version/timing evidence.

### Model or Provider usage may have unknown monetary cost

Unknown cost is not treated as zero. A live formal run fails closed unless its
frozen hard-budget policy can account for the operation.

### Evaluation and product contracts may expose different result detail

`StructuredSearchResponse` is the application result contract. Evaluation adds
an adapter for metrics, and UI renders that response; neither owns a second
search composition.

### Advanced modules may regress baseline behavior

They are constructed only by named experiment configurations, remain off in the
main lock, and are covered by negative/default-off tests.

## Documentation Updates Required During Implementation

- Replace the mock-only runtime description in
  `docs/architecture/current-system.md` after the integrated composition exists.
- Update `docs/demo/demo-runbook.md` with the three supported command paths and
  real replay/live evidence.
- Update `docs/deployment/new-environment-checklist.md` with replay-first and
  explicit-live acceptance steps.
- Update `docs/limitations-and-risks.md` only when Gate evidence resolves a
  stated limitation.
- Update the README and public data manifest only after Gate 0 verification.

No document may publish private labels, credentials, unsafe Provider payloads,
or fabricated performance conclusions.
