# Identifier-Map-Bound R3 Baseline Design

## Context

The R2 real fresh-cache run proved that outbound OpenAlex retrieval works after
wildcard normalization: all 60 queries produced snapshots, 58 produced
predictions, and no request failed with `invalid_request`. R2 is not a valid
relevance measurement because frozen gold identifiers and provider predictions
use different identifier namespaces.

The private Week1 map at
`data/annotation_work/dev_identifier_map.v1.json` contains 141 unique aliases.
Those aliases cover all 141 unique identifiers across 143 development-gold
references. Every gold identifier resolves to a DOI or OpenAlex identifier;
none remains in the arXiv namespace. The map body remains private.

## Runner Interface and Confinement

The evaluation CLI accepts an optional `--id-map PATH`. The path is relative to
the process working directory and is supplied in repository form, for example
`data/annotation_work/dev_identifier_map.v1.json`. Its resolved target must be
beneath the resolved `data/` root and must name an existing file. Absolute paths
and traversal outside `data/` are rejected before any network call. This
interface matches the existing Week1 identifier-map plan and also works when
`data/` is a junction to the frozen Week1 directory.

Static path checks are followed by an open-handle check that closes the
validation/open race. The runner opens the map once, determines the final target
represented by that handle, verifies that target is still beneath the resolved
`data/` root, and reads the bytes from the same handle. Windows resolves the
handle with `GetFinalPathNameByHandleW` and normalizes `\\?\` drive and UNC
forms. POSIX resolves a supported descriptor path and verifies that its
device/inode matches the open descriptor. If the final target cannot be
determined or validated, the map is rejected and the handle is closed.

The CLI parses the immutable byte snapshot with `IdentifierMap.from_bytes`,
hashes those same bytes, and checks explicit alias coverage for every unique gold
identifier. Missing coverage fails with a fixed, value-free error before
provider construction. The standalone metrics CLI uses the same one-read byte
snapshot for parsing and hashing; invalid or unavailable map input produces a
fixed, value-free error while non-map input diagnostics remain unchanged. The
runner never prints map entries or missing identifiers.

`IdentifierMap` exposes a boolean coverage method that normalizes the supplied
identifier and reports whether it is an explicit alias. It does not expose or
serialize the internal mapping.

## Evaluation and Identity

The same validated map instance is passed to:

- candidate deduplication through `process_candidates`;
- final metric evaluation through `evaluate`.

`RunIdentity` gains optional `id_map_sha256`. When a map is supplied:

- `run.json.identity.id_map_sha256` records the exact map hash;
- `run.json.input_hashes.id_map` records the same hash;
- `metrics.json.input_hashes.id_map` records the same hash.

Every recorded map hash uses the canonical `sha256:` prefix followed by 64
lowercase hexadecimal characters. Formal artifacts record neither the map path
nor any map entry.

When no map is supplied, `id_map_sha256` and the `id_map` input hash are omitted
from formal artifacts. Existing no-map artifact shapes and behavior remain
compatible.

Frozen gold bytes, predictions, query text, scoring, filtering, deduplication
rules, and provider request parameters are otherwise unchanged.

## Tests and Error Handling

Test-first coverage proves:

- a confined map resolves prediction and gold namespaces and produces nonzero
  mapped metrics;
- the map hash appears in run identity and both formal input-hash objects;
- an absent `--id-map` preserves existing artifact shapes;
- absolute, traversing, missing, invalid, partial, conflicting, cyclic,
  link-escaping, swapped, or unverifiable final-handle maps fail before any HTTP
  request;
- map parsing, metrics, and recorded hashes use one immutable byte snapshot;
- errors contain no map entries, gold identifiers, query text, or credentials.

Focused runner and dataset tests run before the complete offline Ruff, mypy, and
pytest suites. All test and static-analysis commands use `--no-env-file`.

## R3 Execution

R3 uses the committed implementation SHA, an eight-character short SHA in its
new Git-external directory name, a new output directory, and a new empty SQLite
cache. R1 and R2 remain immutable.

The real command runs in the foreground from `D:\AI Projects\Projects` with
relative `--env-file .env`. A secret-safe execution receipt outside the formal
artifact directory is written atomically after the runner's `main` function
returns, and the launcher exits with the same code. The receipt records:

- full source SHA;
- wrapper SHA-256;
- UTC start and end timestamps;
- direct process exit code;
- run-relative output and cache paths.

The receipt never records environment values, the private map path, or the map
body.

## R3 Acceptance

Before R3 is reported as the formal development baseline:

- direct exit code is zero;
- the snapshot manifest validates;
- all formal artifact hashes recompute;
- run identity binds the committed source, frozen manifest, gold, and map;
- the recomputed private-map hash matches every recorded map hash;
- map coverage remains complete;
- `invalid_request` is zero;
- search calls, snapshots, cache-key coverage, and prediction coverage are
  positive;
- mapped aggregate metrics are internally consistent, contain at least one true
  positive, and have macro and micro Recall greater than zero;
- aggregate R3 retrieval coverage, provider-error counts, and mapped metrics are
  compared with R2 without treating R2's no-map zero metrics as a performance
  result;
- a secret scan finds no credentials, private map path, complete serialized map,
  or map-entry structure in the receipt, formal metadata, or cache metadata.
  Individual DOI/OpenAlex identifiers may legitimately occur in provider
  snapshots and predictions and are not, by themselves, evidence of a map leak.

The seven R2 `invalid_work` records remain a separately reported quality concern
unless R3 evidence changes their aggregate count.
