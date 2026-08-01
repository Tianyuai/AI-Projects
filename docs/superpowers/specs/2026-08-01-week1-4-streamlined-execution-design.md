# Week 1–4 Streamlined Execution Design

**Date:** 2026-08-01  
**Status:** Approved
**Supersedes:** Only the execution grouping, review cadence, and evidence timing
in the 2026-07-30 Week 1–4 integrated plans. All product, safety, data,
budget, replay, evaluation, API, UI, and experiment requirements remain
normative unless this document explicitly resolves a contradiction.

## 1. Goal

Complete the Week 1–4 integrated baseline with lower agent, review, and test
overhead while preserving every existing capability. The streamlined flow
must deliver a complete offline Replay experience first, then execute real
network, cost-bearing, one-attempt, and optional-promotion evidence as a
separate authorized track.

This design does not remove code or requirements. It changes four things:

1. adjacent implementation tasks become larger, independently testable
   delivery packages;
2. one package receives one combined specification and code-quality review;
3. full-repository verification moves from every small task to each phase exit;
4. real evidence operations move to a second track with their existing
   authorization boundaries intact.

## 2. Non-Negotiable Boundaries

The following capabilities and controls are not reduced:

- canonical application request, response, readiness, error, and execution
  contracts;
- exact candidate, validation, and replay locks;
- operator-supplied production pricing and authoritative quality policies;
- V2 freeze validation, migration, path confinement, file-identity binding,
  atomic publication, recovery, and sanitized reports;
- exact-byte capture and offline replay for LLM, OpenAlex, and Semantic
  Scholar dependencies;
- request, run, and project budgets, including CNY 160 and CNY 200 stops;
- `SearchApplicationService` as the sole production search boundary;
- ordered failure-safe formal evaluation and machine-verifiable artifacts;
- irrevocable one-attempt validation claims;
- typed API behavior, Replay-default browser UI, and live-mode authorization;
- optional modules remaining default-off until separately promoted;
- explicit authorization before every real network, cost-bearing, or
  one-attempt operation.

No public status may claim a Gate has passed without matching verified
evidence.

## 3. Current State

Phase 1 Tasks 1–4 are complete and reviewed. Task 5 has an implementation and
two review rounds, but remains open because the latest review identified path
binding, POSIX same-inode mutation, CLI sanitization, resource-lifetime, and
identifier-map wording issues. Task 6 remains conditional on a real Gate 0
pass.

The streamlined plan preserves the current Task 5 net code and test work. It
does not accept the current review findings as resolved and does not roll back
completed Task 1–4 behavior.

## 4. Delivery Packages

The remaining engineering track is organized into twelve packages.

| Package | Original tasks | Independently testable output |
| --- | --- | --- |
| P1-A | Phase 1 Task 5 plus Phase 1 engineering exit | Deterministic, read-only Gate 0 verifier and a truthful blocked-or-passed report |
| P2-A | Phase 2 Tasks 1–2 | Sealed dependency snapshot V2 plus priced LLM capture/replay |
| P2-B | Phase 2 Tasks 3–4 | Provider capture/replay, bounded routing, and persistent budget ledgers |
| P2-C | Phase 2 Tasks 5–6 | Evidence-preserving `SearchApplicationService` and one `CompositionRoot` |
| P2-D | Phase 2 Task 7 plus Phase 2 engineering exit | Stable smoke CLI, atomic capture, Replay Gate 1, and fake-live Gate 2 evidence |
| P3-A | Phase 3 Tasks 1–2 | Ordered execution/business records and authoritative metric/Gate evaluation |
| P3-B | Phase 3 Tasks 3–4 | Atomic run workspace and irrevocable validation-attempt claims |
| P3-C | Phase 3 Tasks 5–6 plus Phase 3 engineering exit | Service-based formal runner, run validator, formal commands, and fixture capture/replay proof |
| P4-A | Phase 4 Tasks 1–2 | Typed FastAPI behavior and browser UI through `/v1/search` |
| P4-B | Phase 4 Task 3 | Stable `paper-search serve` lifecycle |
| P4-C | Phase 4 Task 4 | Explicit experiment identities and default-off optional-stage wiring |
| P4-D | Phase 4 Task 5, Replay portion of Task 6, documentation skeleton from Task 7, and Phase 4 engineering exit | Dual-mode process E2E, Replay browser acceptance, evidence-ready documentation, and complete offline delivery |

Each package is implemented by one fresh implementation agent. Substeps retain
the original task order and interfaces; grouping does not authorize parallel
writes to the shared worktree or bypass dependencies.

## 5. Deferred Real-Evidence Track

The following operations remain required for their corresponding real Gate,
but are not part of the first offline engineering batch:

| Evidence package | Original scope | Required trigger |
| --- | --- | --- |
| E0 | Phase 1 Task 6 | Operator V2 freeze, production pricing, safe readiness evidence, and passing Gate 0 |
| E2 | Real-live portion of Phase 2 Task 7 | Passing Gate 0 plus explicit network and cost authorization |
| E3 | Phase 3 Task 7 | Explicit dev-capture authorization and frozen run cap |
| E4 | Phase 3 Task 8 | Explicit authorization to consume the validation lock's one live attempt |
| E5 | Live portion of Phase 4 Task 6 and final evidence claims in Task 7 | Matching prior Gates plus explicit live-browser authorization |
| E6 | Phase 4 Task 8 | Gates 0–5, required captures, and separate optional-module promotion approval |

Deferral is not acceptance. A deferred Gate remains `blocked` or `not run`,
never `passed`. Documentation created in P4-D may describe commands, schemas,
and pending evidence locations, but cannot state that E0–E6 evidence exists.

## 6. Task 5 Convergence Contract

Task 5 must close the current review findings with the following exact
boundary.

### 6.1 Identifier-map coverage

`IdentifierMap` maps paper identifiers, not business query identifiers.
Therefore the authoritative coverage requirement is:

> For every partition query record, every normalized identifier in
> `relevant_paper_ids` must be explicitly covered by the exact bound
> identifier map.

`query_id` remains required, nonempty, and unique within its partition, but is
not passed to `IdentifierMap`. This resolves the contradictory phrase
"identifier map coverage for every partition query ID" without removing any
paper-identity validation.

### 6.2 Verification boundary

For manifest, approval, partitions, identifier map, pricing, quality policy,
and readiness evidence, Gate 0 must:

1. reject symlink/reparse components in the operator-supplied lexical path;
2. open the artifact through a confined descriptor/handle;
3. read, hash, and parse the same byte snapshot;
4. retain every descriptor/handle through provisional report construction;
5. before confirming the decision, re-hash the same descriptor and compare it
   with the initial hash, then recheck pathname identity and lexical ancestors;
6. map any failure to the artifact's fixed sanitized reason code;
7. close every descriptor deterministically on success and every exception
   path.

The report binds the exact bytes observed at this decision boundary. A later
external mutation does not retroactively rewrite a generated report; any later
consumer must match the report hashes before treating the evidence as current.

### 6.3 CLI and report safety

- Invalid arguments, path failures, validation failures, clock failures, and
  report publication failures produce only fixed sanitized summaries.
- CLI output must not echo raw argv, private paths, exception text, queries,
  credentials, labels, or Provider/LLM payloads.
- `Gate0Report(passed=True)` is valid only with complete fixed public evidence
  identities and all required hashes/counts.
- Report publication strictly revalidates the model and remains atomic,
  no-overwrite, and exact-match idempotent.

### 6.4 Clean Task 5 history

Before Task 5 is accepted, rewrite only the local, unpushed Task 5 range after
`202b6f4` into a clean net implementation commit. The local SDD report remains
available in the ignored workspace but must not exist in the reachable branch
history. The rewrite must preserve the approved plan correction, the exact net
code/tests, all unrelated untracked files, and all Task 1–4 commits.

## 7. Test and Review Cadence

### 7.1 Package implementation

- Every new behavior or defect fix uses RED → GREEN → REFACTOR.
- Each package substep runs only its focused tests.
- At package completion, run package-level tests, Ruff and mypy for changed
  Python files, and `git diff --check`.
- Task 5 additionally runs the complete test suite because it changes shared
  security and policy loaders.

### 7.2 Review

- One independent reviewer evaluates the complete net package diff.
- The reviewer must issue separate `Spec compliance` and `Code quality`
  verdicts.
- Critical and Important findings are consolidated into one fix wave, followed
  by one complete package re-review.
- Minor findings are either fixed in the same wave when local and low-risk or
  recorded for the final whole-branch review.
- A reviewer does not rerun tests whose exact command and passing output are
  already present in the implementation report.
- The same reviewer is reused for re-review when its context remains
  available; otherwise a fresh reviewer receives the complete brief, report,
  and refreshed net diff package.

There is no numerical cap that permits accepting open Critical or Important
findings. Cost is controlled by batching findings and reviewing package-sized
net diffs, not by lowering the acceptance threshold.

### 7.3 Phase and final verification

Run full pytest, full Ruff, and full mypy at each phase exit and before final
integration. Do not repeat full-repository checks after every package unless a
package modifies a shared security boundary or the focused suite exposes a
cross-cutting regression.

## 8. Offline Engineering Acceptance

The first batch is complete only when all of the following are true:

- Gate 0 produces a deterministic truthful report against synthetic passing
  evidence and the current real repository remains truthfully blocked where
  operator inputs are absent.
- Replay smoke succeeds without any network attempt and repeated Replay has a
  byte-identical canonical business projection.
- Fake-live capture verifies snapshots, pricing, ledgers, atomic publication,
  and immediate Replay equivalence.
- Formal fixture capture and Replay pass `verify-run` and `compare-replay`.
- The API and browser UI use the same `SearchApplicationService` boundary.
- A real browser can complete the Replay flow and display results, provenance,
  usage, and degradation.
- Optional experiment modules remain default-off.
- Documentation distinguishes verified offline behavior from deferred real
  evidence.
- Phase-exit and final engineering checks pass.

## 9. Trade-offs

Package reviews cover larger diffs than the original task reviews, so defects
may be found later within a package. This is controlled by retaining ordered
TDD substeps, focused tests, exact interface boundaries, and one complete net
package review.

Deferring real evidence means the first batch cannot claim a production-ready
live baseline or completed Gates 0, 2, 3, 4, 5-live, or 6. It does, however,
produce a complete and demonstrable offline workflow sooner and without
spending network or one-attempt budgets.

## 10. Plan Revision Requirements

The implementation-plan revision must:

- preserve all original task requirements inside the owning package;
- replace per-original-task agent/review instructions with per-package
  instructions;
- separate E0, E2, E3, E4, E5, and E6 into a deferred evidence plan;
- update Gate and acceptance language so offline engineering success is not
  confused with real Gate evidence;
- include the Task 5 contradiction resolution and clean-history procedure;
- name exact focused, package, phase-exit, and final verification commands;
- retain the Subagent-Driven execution choice.
