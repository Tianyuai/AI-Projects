# New Environment Deployment and Acceptance Checklist

Use a fresh clone or worktree. Keep engineering verification, replay acceptance, and authorized live evidence as separate gates.

## Runtime and dependencies

- [ ] Confirm Python 3.11.x; the project supports `>=3.11,<3.12`.
- [ ] Confirm `uv` is available.
- [ ] Install exactly one profile: `uv sync --locked --extra cpu`, or the separately approved CUDA profile.
- [ ] Treat bare `uv sync` as core-only and insufficient for complete acceptance.
- [ ] Record command outcomes and the repository revision without machine-specific paths or package-index credentials.

## Secret boundary

- [ ] Verify only the required variable names through the approved secret manager: `OPENALEX_API_KEY`, `SEMANTIC_SCHOLAR_API_KEY`, `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL_PRIMARY`, `LLM_MODEL_FALLBACK`, and `HF_TOKEN`.
- [ ] Never print, copy, log, commit, screenshot, or paste values.
- [ ] Keep offline commands on `--no-env-file`; remember this does not clear inherited variables or enforce network isolation.
- [ ] Never inspect `.env` as part of acceptance.

## Engineering gate

```powershell
uv run --no-sync --no-env-file python -m paper_search.health
uv run --no-sync --no-env-file pytest -q
uv run --no-sync --no-env-file ruff check .
uv run --no-sync --no-env-file mypy src
uv run --no-sync --no-env-file paper-search --help
```

- [ ] All commands exit 0.
- [ ] Credential-gated online tests without credentials remain explicit skips, not successful provider checks.

## Gate 0 and data state

- [ ] Read the current safe Gate 0 report and confirm `passed: true` before any real provider or formal-data claim.
- [ ] Require a V2 frozen manifest, exact partition and identifier-map hashes, approved production pricing policy, quality-gate policy, and safe readiness evidence.
- [ ] If Gate 0 is blocked, stop the real evidence path and report the named blocking reasons. Do not manually change `data/manifest.json`.
- [ ] Keep raw data, gold, label files, real queries, and per-query evidence outside Git and ordinary logs.

The current repository state is intentionally blocked at this gate; synthetic fixtures may still exercise all engineering paths.

## Replay service gate

- [ ] Verify the selected capture and replay artifacts with `paper-search verify-run`.
- [ ] Verify the pair with `paper-search compare-replay`.
- [ ] Start `paper-search serve` with the verified replay lock and snapshot manifest, without `--allow-live`.
- [ ] Bind only loopback unless a separate deployment security review approves another interface.
- [ ] Check `/health/live`, `/health/ready`, the browser UI, and one direct `/v1/search` request.
- [ ] Confirm repeated replay preserves canonical business results and stable provenance.
- [ ] Confirm replay performs no external name resolution or socket connection.
- [ ] Stop cleanly and check for incomplete artifacts or held locks.

## Live authorization gate

All three technical authorization predicates are mandatory; they are not credentials and do not replace the operator's governance approval:

- [ ] the verified lineage lock has `runtime_allow_live: true`;
- [ ] the operator explicitly starts the server with `--allow-live`;
- [ ] the individual request explicitly sets `mode: live`.

- [ ] Obtain separate approval for providers, credential scope, query class, hard budget, capture root, and retention policy.
- [ ] Confirm one request receives an isolated live service, clients, budget, and capture session.
- [ ] Confirm a successful capture is sealed, verified, and atomically published before HTTP 200.
- [ ] Confirm failed or cancelled work cannot appear complete.
- [ ] Run `paper-search verify-run` on every published live capture.
- [ ] Record only safe hashes, run IDs, aggregate usage/cost, and sanitized error codes.

## Formal dev and validation gates

- [ ] Run authorized dev capture under a frozen run cap.
- [ ] Verify the capture, generate replay from the same snapshot set, verify replay, and compare canonical business results.
- [ ] Promote a validation lock only from complete passing dev evidence.
- [ ] Treat the validation lock hash as a single irreversible attempt identity.
- [ ] Run one authorized live validation attempt; interruption or failure does not authorize a replacement attempt.
- [ ] Verify and compare validation capture/replay before reporting aggregate results.
- [ ] Keep predictions, failures, business results, snapshots, gold labels, and validation claims access-controlled.

## Optional-module promotion gate

- [ ] Keep `configs/base.yaml` on `main-baseline` throughout evidence generation.
- [ ] Require Gates 0–5 before optional ablations.
- [ ] Run three same-configuration dev comparisons with identical frozen inputs, snapshots, budgets, and measurement policy.
- [ ] Use 1,000 bootstrap samples and the committed promotion thresholds.
- [ ] Run only the approved selection-only validation comparison.
- [ ] Keep the module default-off if evidence is incomplete or any threshold fails.
- [ ] Request a separate promotion decision before changing baseline defaults or a validation lock.

## Handoff record

- [ ] State separately which gates are passed, blocked, failed, or not run.
- [ ] Link only to access-appropriate evidence.
- [ ] Do not convert fixture success into real-data, real-provider, quality, cost, or production-readiness claims.
