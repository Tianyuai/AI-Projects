# Demonstration Runbook

## Evidence choice

Use a capture whose `replay.lock.yaml` and snapshot manifest have already passed the applicable verifier. The commands below are replay-only and do not load `.env`, contact providers, or incur provider cost.

For repository-only engineering acceptance, first verify the synthetic formal pair:

```powershell
uv run --no-sync --no-env-file paper-search verify-run tests/fixtures/formal_run/capture
uv run --no-sync --no-env-file paper-search verify-run tests/fixtures/formal_run/replay
uv run --no-sync --no-env-file paper-search compare-replay tests/fixtures/formal_run/capture tests/fixtures/formal_run/replay
```

Expected result: both runs are valid and the pair is equivalent. This validates the evidence machinery, not real retrieval quality.

A fresh clone does not contain an interactive service-ready capture. Before following the interactive sections, an authorized operator must supply one verified capture directory and an access-controlled runs root. For a fully repository-contained replay/service acceptance, run:

```powershell
uv run --no-sync --no-env-file pytest tests/e2e/test_dual_mode_serve.py tests/integration/test_serve_process.py -q
```

The E2E test installs socket and name-resolution tripwires, proves they reject non-loopback targets, and then exercises the real `paper-search serve` subprocess.

## Start the unified replay service

From the artifact root expected by the selected lock, start the canonical service on loopback:

```powershell
uv run --no-sync --no-env-file paper-search serve `
  --lock <capture>/replay.lock.yaml `
  --mode replay `
  --snapshot-manifest <capture>/snapshot-manifest.json `
  --capture-output-root <runs-root> `
  --host 127.0.0.1 `
  --port 8000
```

Do not add `--allow-live` for a replay demonstration. Keep this terminal open.

## Check health

In a second terminal:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health/live
Invoke-RestMethod http://127.0.0.1:8000/health/ready
```

Liveness should report `live`. Readiness should report `ready` or `degraded`, `execution_mode: replay`, the bound snapshot-set identity, and dependency states drawn from `ready`, `replayed`, `degraded`, or `failed`. It must not make a live dependency probe.

## Browser demonstration

Open `http://127.0.0.1:8000/`, retain the default Replay mode, and submit one representative query approved for the selected snapshot. Verify that the UI displays:

- selected paper IDs and ranked result evidence;
- execution mode and snapshot set/time;
- configuration hash and per-request run ID;
- usage, stop reason, partial state, planner state, and fallback state;
- dependency statuses, safe warnings, and citation edges.

Submit the same request a second time. The per-request run ID may change; canonical business content and stable provenance must not.

Inspect the browser console and network panel. There should be no JavaScript error, one `/v1/search` POST for each submit action, and no browser request to OpenAlex, Semantic Scholar, an LLM endpoint, or a local filesystem path.

## Direct API demonstration

The UI and direct API share the same boundary:

```powershell
$body = @{
  query_id = "demo-replay-1"
  query = "<approved replay query>"
  budget_profile = "balanced"
  mode = "replay"
  include_trace = $true
} | ConvertTo-Json

$response = Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/v1/search `
  -ContentType application/json `
  -Body $body

$response | ConvertTo-Json -Depth 12
```

Do not place a private query in committed documentation, screenshots, or ordinary logs.

## Authorized live demonstration

Live is optional and remains blocked until the operator separately authorizes the target providers, credentials, query class, hard budget, capture location, and disclosure boundary. A prior replay authorization is not live authorization.

When those prerequisites are satisfied, verify that the lineage lock permits live, start the same server with `--allow-live`, and explicitly select `mode: live` for one bounded request. The server flag alone does not make omitted-mode requests live. After HTTP 200, verify the newly published capture:

```powershell
uv run --no-sync --no-env-file paper-search verify-run <live-capture-directory>
```

Record only the safe run ID, aggregate status, hashes, bounded degradation codes, and verifier outcome. Do not record credentials, query text, raw snapshots, predictions, gold labels, or private paths.

## Stop and clean up

Return to the service terminal and press `Ctrl+C`. On Windows, confirm the port is closed and inspect only the chosen runs root for incomplete/lock markers:

```powershell
Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
Get-ChildItem -LiteralPath <runs-root> -Force | Where-Object { $_.Name -match 'incomplete|\.lock$|\.lck$' }
```

Expected result: both commands produce no matching record. Store approved screenshots and safe acceptance records outside source control unless a documentation-assets review explicitly authorizes committing them.

## Current project status

Replay browser acceptance and dual-mode fake-live lifecycle E2E are verified. Real live browser acceptance, real dev/validation captures, metric claims, cost claims, and optional-module promotion remain not run because the current public Gate 0 is blocked and no separate live hard-budget authorization is recorded here.
