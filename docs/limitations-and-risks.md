# Limitations and Risks

## Current evidence boundary

Verified at the current revision:

- the integrated application, replay/live composition, API, UI, formal runner, validator, and experiment registry pass the offline engineering suite;
- fake-provider live-mode capture and replay lifecycle tests cover publication, failure, cancellation, lineage, and isolation;
- synthetic formal capture/replay fixtures pass `verify-run` and `compare-replay`;
- a real browser has exercised the loopback replay UI twice with stable visible business content and provenance;
- `main-baseline` constructs no optional stage, and named optional identities construct only their declared behavior.

Not established by that evidence:

- real data freeze or a passing public Gate 0;
- real provider availability, retrieval quality, measured cost, or production readiness;
- formal dev or validation capture/replay results;
- real live-browser acceptance;
- comparative benefit or promotion eligibility for any optional module.

A local safe Gate 0 report generated on 2026-07-30 was blocked by invalid/incomplete manifest evidence, missing approved pricing evidence, and invalid/missing readiness evidence. That blocked report stays outside tracked public paths, so a fresh clone can independently confirm only that `data/manifest.json` remains `waiting_for_human_label_freeze`; it cannot reproduce private Gate 0 causes without the separately controlled inputs.

## Primary risks

- **Evidence substitution:** synthetic fixtures can prove contracts and lifecycle behavior but cannot support claims about real retrieval quality, provider reliability, or cost.
- **Authorization drift:** live execution is unsafe if lock permission, server authorization, and request mode are treated as interchangeable. All three must remain mandatory.
- **Sensitive artifact disclosure:** snapshots, predictions, failures, business results, gold labels, queries, and validation claims may reveal protected data even when credentials are absent.
- **Validation retry bias:** allowing a new validation attempt after interruption or failure would invalidate the one-attempt policy. Attempt identity is bound to archived lock bytes.
- **Replay integrity drift:** a manifest without exact response bytes, request identity, policy/config binding, or canonical business comparison can create false reproducibility.
- **Optional-stage overclaiming:** implementation and offline tests do not demonstrate positive quality delta. Baseline defaults must not change without Gate 6 evidence and separate approval.
- **Platform test flakiness:** the subprocess test helper reserves and releases an OS-assigned port before the server binds it; another local process could win the low-probability race. This affects test stability, not the verified runtime contract.
- **Environment ambiguity:** `--no-env-file` prevents dotenv loading but neither removes inherited secrets nor blocks network.

## Mitigations and gates

- Keep Gate 0 truthfully blocked until the V2 freeze, identifier map, production pricing, quality policy, and safe readiness evidence all verify.
- Require explicit, scoped authority before each real-network, cost-bearing, dev, validation, live-browser, or promotion action.
- Run live work under hard budgets and request-local capture; publish success only after sealed evidence validates.
- Use `verify-run` as the validity predicate and `compare-replay` as the canonical capture/replay comparison.
- Preserve irrevocable validation-attempt claims and reject cross-lock recovery.
- Keep protected artifacts outside Git and ordinary chat/log channels; share only approved aggregates, hashes, safe error codes, and run IDs.
- Keep `configs/base.yaml` on `main-baseline`; treat all non-promoted experiment artifacts as evidence, not as authorization to enable a module.
- Prefer a future port-0 handshake for subprocess tests to remove the remaining port-allocation race.

## Reporting rule

Every report must label each gate as passed, blocked, failed, or not run. Do not describe deferred evidence as accepted, and do not infer performance or deployment claims from engineering test counts.
