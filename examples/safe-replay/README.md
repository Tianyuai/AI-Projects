# Safe zero-network replay sample

This synthetic sample contains one public demonstration query and deterministic mock
provider responses. It contains no API credentials, private training data, hidden test
queries, Gold labels, or external production receipts.

Run it from the repository or extracted evaluator runtime root:

```powershell
python scripts/run_evaluator_package.py `
  --queries examples/safe-replay/queries.jsonl `
  --output runs/safe-replay/predictions.jsonl `
  --lock examples/safe-replay/replay.lock.yaml `
  --snapshot-manifest examples/safe-replay/snapshots/smoke/snapshot-manifest.json `
  --artifact-root examples/safe-replay `
  --capture-output-root runs/safe-replay/captures `
  --mode replay
```

Replay is request-identity bound: changing the sample query text intentionally fails
closed instead of returning unrelated cached results.
