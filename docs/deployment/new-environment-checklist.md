# New Environment Deployment and Acceptance Checklist

Use a fresh clone or worktree. This checklist separates evidence that can be
collected with the repository's offline, synthetic setup from later actions
that require an authorized operator, credentials, and a fresh provider cache.

## Python 3.11 and uv

- [ ] Confirm that the selected interpreter is Python 3.11.x. The project
  supports `>=3.11,<3.12`.
- [ ] Confirm that `uv` is available before provisioning the environment.
- [ ] Select one accelerator profile: CPU is the required portable acceptance
  profile; CUDA is opt-in. Do not install both extras together.

## Dependency Installation

- [ ] From the repository root, install the selected profile with
  `uv sync --locked --extra cpu` (or the explicitly approved CUDA profile).
- [ ] Treat a bare `uv sync` as core-only; it is insufficient for complete
  retrieval, health, and test acceptance.
- [ ] Record only the command outcome and lockfile revision. Do not include
  machine-specific paths, credentials, or package-index authentication in the
  handoff.

## Environment Variable Names

- [ ] Verify only the presence and intended use of these names: `OPENALEX_API_KEY`,
  `SEMANTIC_SCHOLAR_API_KEY`, `LLM_API_KEY`, `LLM_BASE_URL`,
  `LLM_MODEL_PRIMARY`, `LLM_MODEL_FALLBACK`, and `HF_TOKEN`.
- [ ] Check names through the operator's approved secret-management mechanism;
  never print, copy, log, commit, or paste their values.
- [ ] Keep offline commands on `--no-env-file`. This prevents automatic `.env`
  loading, but does not clear inherited process variables or create network
  isolation.

## Offline Test Gate

These checks can run now after the selected locked environment is prepared;
they require no credential or external-provider authorization.

- [ ] Run `uv run --no-sync --no-env-file python -m paper_search.health`.
- [ ] Run `uv run --no-sync --no-env-file pytest -q`.
- [ ] Run `uv run --no-sync --no-env-file ruff check .`.
- [ ] Run `uv run --no-sync --no-env-file mypy src`.
- [ ] Preserve any skipped credential-gated online test as a skip, not as a
  successful provider check.

## Mock Server Gate

This gate can run now and exercises only the synthetic, loopback-only service.

- [ ] Start `uv run --no-sync --no-env-file python -m paper_search.api.mock_server --host 127.0.0.1 --port 8000`.
- [ ] Check `http://127.0.0.1:8000/health/live` and
  `http://127.0.0.1:8000/health/ready` from a second local terminal.
- [ ] Submit one synthetic `POST /v1/search` request and confirm its response
  is identified as mock-composition evidence, not a provider-backed result.
- [ ] Stop the mock service cleanly. Do not expose it beyond loopback.

## API Readiness

This is a later, authorized gate. The default API is deliberately uncomposed,
and mock readiness is not production readiness.

- [ ] Obtain operator approval for the target environment and real-provider
  readiness check.
- [ ] Confirm the required credential names are available to the approved child
  process without revealing values.
- [ ] Verify injected real-provider composition, provider status, budget
  limits, and degraded behavior under the approved runbook.
- [ ] Record sanitized status, timestamps, configuration hash, and failure
  category only; exclude request headers, credential-bearing URLs, and raw
  provider responses unless separately approved for protected storage.

## Fresh-Cache Run Gate

This is a later, authorized credential and external-service gate.

- [ ] Receive explicit authorization for provider calls, dataset access, and
  cache creation before starting a fresh-cache run.
- [ ] Freeze and record the authorized input revision, split/ID manifest,
  configuration hash, random seed, and budget before execution.
- [ ] Create the fresh cache through the approved workflow and preserve its
  manifest and response hashes without storing secrets.
- [ ] Do not interpret R2 diagnostics or a fresh-cache smoke run as R3 formal
  evaluation evidence.

## Artifact Verification

- [ ] Verify snapshot and cache-manifest hashes against their recorded
  artifacts.
- [ ] Verify that configuration hash and git SHA identify the run revision.
- [ ] Verify frozen inputs and ID manifests before later comparison work.
- [ ] Keep future metric, cost, and ablation artifacts separate from this
  checklist until R3 authorizes formal evaluation.

## Secret-Handling Rules

- [ ] Never print or store secret values in terminal output, Markdown, test
  data, commits, reports, screenshots, issue trackers, or provider URLs.
- [ ] Never read, parse, copy, or commit `.env` content for this checklist.
- [ ] Use variable names only in documentation and diagnostics; a variable name
  must never be followed by a secret value.
- [ ] Redact credential-bearing headers and identifiers before sharing any
  operational evidence.
