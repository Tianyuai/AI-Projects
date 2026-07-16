# Task 3 Persistent OpenAlex Smoke Artifacts Design

## Context

Task 3 already runs three fixed, non-sensitive OpenAlex queries and validates the
resulting snapshot manifest. The current online test writes its outputs to
pytest's `tmp_path`, so a successful run does not leave the PRD-required
`experiments/smoke/provider.json` or durable raw response snapshots.

This change closes only that acceptance gap. It does not change OpenAlex search,
normalization, caching, retry, budgeting, or public provider interfaces.

## Goals

- A successful keyed online test leaves `experiments/smoke/provider.json`.
- Each run preserves exact raw response bytes and a validated manifest in an
  immutable run-specific directory.
- Repeated smoke runs do not overwrite earlier snapshots.
- Smoke artifacts never contain the OpenAlex API key and are not committed.
- The three queries make real external requests rather than replaying an older
  smoke cache.

## Non-goals

- Adding a separate smoke-test CLI.
- Committing live OpenAlex responses or experiment summaries.
- Changing the provider, cache, snapshot, or manifest contracts.
- Adding Semantic Scholar, citation retrieval, ranking, or Task 4 behavior.

## Output Layout

Each successful run produces:

```text
experiments/smoke/
|-- provider.json
`-- runs/
    `-- <run_id>/
        |-- openalex-cache.sqlite3
        |-- snapshot_manifest.json
        `-- snapshots/
            |-- openalex-0001.json
            |-- openalex-0002.json
            `-- openalex-0003.json
```

`run_id` is generated for the invocation and is safe for Windows paths. Each run
gets a new SQLite cache, so a keyed smoke test exercises the live API. The cache
is retained with the run for diagnosis but is not referenced as a formal
artifact.

The top-level `provider.json` contains only:

- a smoke summary contract version;
- the run ID;
- the manifest path relative to `experiments/smoke`;
- per-query error codes, latency, paper count, and response hash.

It does not contain query text, request headers, authorization material, or an
API key.

## Data Flow

1. The online test resolves the repository root from the test file location and
   selects `experiments/smoke` as the smoke root.
2. It creates a new run ID and run directory.
3. It executes the existing three fixed queries with a fresh run-local cache.
4. Every query must return at least one normalized paper.
5. The cache exports the ordered response set into the run directory.
6. The test validates the generated snapshot manifest and verifies that the key
   is absent from the serialized summary and manifest.
7. Only after all checks pass, the summary is atomically published as the
   top-level `provider.json`.

If a query, export, or validation fails, `provider.json` is not updated. A
partially created run directory may remain for diagnosis, but it is ignored by
Git and is not treated as an accepted run.

## Immutability and Repeated Runs

Existing snapshot export behavior remains unchanged: differing files in the
same run directory cannot be overwritten. Repeated online tests create distinct
run directories, so historical raw responses remain immutable. The top-level
`provider.json` is a mutable pointer to the most recent fully validated run and
is written through a temporary file followed by atomic replacement.

## Version-control and Secret Safety

`/experiments/smoke/` is added to `.gitignore`. Only the ignore rule, test logic,
PRD status, design, and implementation plan may be committed. The acceptance
workflow may load `.env` into the child test process, but no implementation or
diagnostic command may print the file or its values.

Before commit, tracked files are scanned for credential-like literals and
authorization headers. Git status must confirm that smoke artifacts, cache
databases, and `.env` are absent from the change set.

## Testing Strategy

TDD adds an offline contract test before changing the online orchestration. The
test uses a safe fixture transport and a temporary repository root to prove:

- output is published at `experiments/smoke/provider.json`;
- the referenced manifest exists and validates;
- raw response snapshots live under `runs/<run_id>/snapshots`;
- two runs use different directories and preserve the first run;
- neither the summary nor manifest contains a supplied sentinel key;
- a failed query does not publish a new top-level summary.

After the offline test passes, acceptance runs:

```powershell
uv run --no-sync --env-file ../../Projects/.env pytest -m online tests/integration/test_openalex_live.py -v
uv run --no-sync --no-env-file pytest -m "not online" tests/unit/test_openalex.py tests/unit/test_cache.py tests/unit/test_normalize.py -v
uv run --no-sync --no-env-file pytest -q
uv run --no-sync --no-env-file ruff check .
uv run --no-sync --no-env-file mypy src
```

The keyed online command must pass and leave a validated local summary and
manifest before the two remaining Task 3 PRD checkboxes are marked complete.

## Acceptance

Task 3 may be accepted when all of the following are true:

- the three keyed live queries pass and each returns a non-empty result;
- `experiments/smoke/provider.json` exists and references a valid manifest;
- every referenced raw response exists and matches its SHA-256;
- the API key is absent from all smoke artifacts and tracked changes;
- focused tests, full pytest, Ruff, and mypy pass;
- an independent code review has no unresolved critical or important findings.

Acceptance does not authorize merging. Integration remains a separate decision.
