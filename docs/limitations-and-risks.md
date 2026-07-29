# Limitations and Risks

## Current Evidence Boundary

- Identifier-map wiring and the R3 baseline are external dependencies and were
  not present at this task's starting revision.
- R2 is retrieval diagnostic evidence only.
- R2's zero relevance metrics are caused by an identifier namespace mismatch
  and are not a retrieval-performance conclusion.
- Seven `invalid_work` records require later quality analysis.
- Adaptive evolution has no real fixed-strategy comparison yet.
- Relationship visualization has not passed its stage gate.

These are limitations of the current evidence, not claims about final system
quality, deployment readiness, or comparative performance.

## Operational and Evaluation Risks

- A synthetic loopback mock server verifies composition and request handling,
  but it does not prove real-provider integration, cache behavior, or API
  readiness in an authorized target environment.
- Credential-gated provider checks, fresh-cache execution, and access to gated
  inputs can fail independently of the offline suite.
- `--no-env-file` prevents automatic dotenv loading but does not make a process
  network-isolated or remove inherited environment variables.
- An identifier namespace mismatch can turn relevance accounting into a false
  zero. It must be diagnosed at the mapping boundary before any metric is
  treated as evaluative evidence.
- The offline adaptive coordinator is injected and disabled in main runtime
  composition. Its behavior must not be represented as a deployed strategy.
- A relationship view must not be treated as validated until its stage gate
  confirms that nodes and edges are backed by the authorized evidence path.

## Mitigations and Gates

- Wait for R3 before publishing formal relevance, cost, or ablation
  conclusions.
- Preserve frozen inputs, ID manifests, configuration revisions, and snapshots
  so later results can be reproduced and audited.
- Resolve identifier-map wiring and analyze all seven `invalid_work` records
  before accepting R3 quality evidence.
- Compare fixed and adaptive strategies later with identical frozen data,
  budgets, configurations, and measurement rules.
- Keep adaptive behavior disabled in runtime and API composition until the
  authorized integration, evaluation, and safety gates pass.
- Keep relationship visualization gated until its evidence contract and stage
  acceptance are complete.
- Require explicit operator authorization and credentials for fresh-cache and
  real-provider checks; record only sanitized operational evidence.
