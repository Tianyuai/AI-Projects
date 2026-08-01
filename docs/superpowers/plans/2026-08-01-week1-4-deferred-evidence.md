# Week 1–4 Deferred Real-Evidence Plan

> **For agentic workers:** Execute only after the named preconditions and a
> fresh explicit user authorization for each package. This plan does not grant
> network, cost, one-attempt, or promotion authority.

**Goal:** Complete real Gate evidence after the offline engineering plan is
finished, without coupling expensive or irreversible operations to code
implementation.

**Architecture:** Six evidence packages consume the immutable offline
application, locks, policies, snapshots, runner, validator, API, and UI. Each
package records machine-verifiable outputs and never mutates source/config
during an authoritative run. Failed or unavailable evidence remains truthful
`blocked`, `failed`, or `not run` state.

**Tech Stack:** Existing `paper-search` CLI/server, access-controlled artifact
roots, verified locks/policies, real authorized dependency credentials,
browser acceptance, Git.

## Global Constraints

- Engineering prerequisite:
  `docs/superpowers/plans/2026-08-01-week1-4-streamlined-engineering.md`.
- Product and Gate authority remains the approved 2026-07-30 design and phase
  plans.
- Request explicit authorization immediately before each package. Approval of
  one package does not authorize another.
- E4 consumes exactly one validation attempt for one lock hash; interruption
  or failure does not restore it.
- Never fabricate production pricing, readiness, credentials, costs, run IDs,
  or Gate outcomes.
- Keep raw snapshots, queries, labels, predictions, failures, business
  results, validation claims, and private Gate 0 inputs access-controlled.
- No source/config edits during E2, E3, or E4 authoritative execution.
- Public documentation/status changes only after matching machine evidence.

---

### Evidence E0: Gate 0 and Public Freeze Status

**Authority required:** operator access-controlled V2 freeze root, production
pricing policy, safe readiness evidence, and approval to publish only the
sanitized projection.

- [ ] Verify every supplied path is beneath the approved private root.
- [ ] Run Gate 0:

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file python -m paper_search.evaluation.gate0 --data-root 'private-gate0' --manifest 'private-gate0/manifest.json' --pricing-policy 'private-gate0/pricing-policy-v1.yaml' --quality-gates 'configs/quality_gates_v1.yaml' --readiness 'private-gate0/provider-readiness.json' --report 'private-gate0/gate0-report.json'
```

Required: exit 0 and `"passed":true`. Otherwise retain the private report and
make no public change.

- [ ] Publish only the verified sanitized manifest projection and
  `data/gate0_evidence.json`; update README/data README/PRD claims to exact
  counts/hashes.
- [ ] Run the original Phase 1 Task 6 secret scan and full Phase 1 checks.

---

### Evidence E2: Authorized Real-Live Smoke and Replay

**Authority required:** passing E0, live-capable lock, readiness less than 15
minutes old, explicit network authorization, and an approved maximum cost.

- [ ] Verify clean tracked source/config, lock hashes, project/run budget, and
  all optional modules off.
- [ ] Run exactly one bounded live smoke using the Phase 2 `smoke` CLI with
  `--allow-network` and the approved lock/output root.
- [ ] Require successful capture sealing, manifest/replay-lock emission,
  artifact validation, known actual cost, and final publication.
- [ ] Immediately run Replay against the emitted lock/manifest with the network
  tripwire and compare canonical business bytes.
- [ ] Preserve and review safe aggregate usage/cost/failure evidence.

---

### Evidence E3: Gate 3 Dev Capture and Validation-Lock Promotion

**Authority required:** E0 and E2 passed, approved candidate lock, current
readiness, explicit live dev authorization, and approved maximum CNY cap.

- [ ] Verify clean tracked source/config and capture the exact SHA.
- [ ] Run:

```powershell
paper-search evaluate --lock candidate.lock.yaml --split dev --mode live --output-root runs --allow-network
paper-search verify-run runs/dev-capture
paper-search evaluate --lock runs/dev-capture/replay.lock.yaml --split dev --mode replay --output-root runs --snapshot-manifest runs/dev-capture/snapshot-manifest.json
paper-search verify-run runs/dev-replay
paper-search compare-replay runs/dev-capture runs/dev-replay
```

- [ ] Promote a content-addressed validation lock only when capture/replay,
  formal validity, and every baseline-quality Gate pass. Otherwise preserve
  evidence and create no validation lock.

---

### Evidence E4: Gate 4 One-Attempt Validation

**Authority required:** promoted validation lock, matching source/input hashes,
no existing claim, current readiness, reserved budget, and explicit approval
to consume the one live attempt with its maximum CNY cap.

- [ ] Run exactly one live validation command; claim creation occurs
  immediately before first network dispatch:

```powershell
paper-search evaluate --lock validation.lock.yaml --split validation --mode live --output-root runs --allow-network
```

- [ ] If capture is complete, run:

```powershell
paper-search verify-run runs/validation-capture
paper-search evaluate --lock runs/validation-capture/replay.lock.yaml --split validation --mode replay --output-root runs --snapshot-manifest runs/validation-capture/snapshot-manifest.json
paper-search verify-run runs/validation-replay
paper-search compare-replay runs/validation-capture runs/validation-replay
```

- [ ] Preserve the terminal claim and authoritative result regardless of
  success, failure, quality outcome, or interruption. Never retry the same
  validation lock hash.

---

### Evidence E5: Live Browser Acceptance and Final Gate Claims

**Authority required:** matching E0–E4 evidence and explicit authorization for
one bounded live browser request and its cost.

- [ ] Re-run Replay browser acceptance against the verified dev capture.
- [ ] Restart `paper-search serve` with `--allow-live`, select live mode in the
  browser, submit one bounded request, and verify the request-scoped capture:

```powershell
paper-search verify-run runs/live-api-capture
```

- [ ] Record only safe browser/server evidence: snapshot set, safe run IDs,
  visible-field checklist, console error count, request summary, and validator
  exit.
- [ ] Update final delivery documentation only with matching E0–E5 machine
  evidence, then run the original contradiction/secret scan and full
  engineering verification.

---

### Evidence E6: Optional-Module Ablations and Promotion

**Authority required:** E0–E5 complete, required captures available, any new
live calls separately authorized, and a separate promotion decision after
results exist.

- [ ] Keep `configs/base.yaml` unchanged while generating evidence.
- [ ] For each selected optional experiment, run three same-configuration dev
  comparisons and one selection-only validation comparison against identical
  frozen inputs, snapshots, budgets, and measurement policy.
- [ ] Apply exactly:

```text
median_dev_macro_f1_delta >= +0.01
bootstrap_samples = 1000
bootstrap_95_percent_lower_bound >= -0.005
validation_macro_f1_drop <= 0.01
```

- [ ] Keep the module default-off when evidence is missing or any rule fails.
- [ ] If all rules pass, present complete `PromotionEvidence` and request a
  separate user decision before changing baseline configuration or locks.

## Evidence Completion

The Week 1–4 integrated project has full real evidence only when every required
package E0, E2, E3, E4, and E5 has matching validated artifacts. E6 remains
optional and never blocks the fixed-one-round baseline.
