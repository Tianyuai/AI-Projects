# Identifier Relation-Correctness Gate Design

**Date:** 2026-08-11
**Status:** Approved
**Scope:** Development identifier-map semantic promotion only

## Decision

The hard gate validates every required anchor and every provider relation enumerated by sealed evidence. Provider coverage remains visible as a diagnostic count; absence of an optional provider relation is not itself an unresolved relation.

For every development gold arXiv ID, the rebuilt map must contain its deterministic same-arXiv DataCite anchor. Every evidence ref is an actual provider relation candidate and must be audited before filtering. Only verified candidates may enter candidate-map bytes. Missing optional provider evidence creates no synthetic relation and does not fail the gate.

This amendment supersedes only the earlier requirements that every gold group must have provider metadata and that a provider-missing group must create an `unresolved` placeholder. Snapshot integrity, exact identity binding, mismatch, conflict, privacy, and validation boundaries remain unchanged.

## Evidence for the Change

The authorized capture bound to lock `f3d210e8...` completed with two verified Semantic Scholar snapshots. It produced provider identity evidence for 90 of 141 gold groups. The previous rebuild verified every proposed DOI relation and every deterministic anchor, but synthesized 51 `provider_identity_missing` rows solely because no optional provider alias was discovered for those groups.

Repeated capture is not a resolution strategy: it queries the same locked IDs, consumes budget, advances the ledger checkpoint, and does not create an identity relation the provider does not expose. The gate must distinguish an incorrect proposed relation from the absence of an optional relation.

## Alternatives Considered

1. **Relation-correctness gate - selected.** Verify all anchors and all evidence refs; report provider coverage separately. This preserves false-merge protection and permits a useful partial alias map.
2. **Keep 100% provider coverage as a hard gate - rejected for normal execution.** This blocks promotion despite no mismatches and cannot be solved by repeating the same request.
3. **Add another provider or manual annotation - policy fallback only.** Use this only if stakeholders require independent provider confirmation for every gold group.

## Gate Contract

### Required anchor relations

- Each normalized unique gold arXiv ID has exactly one same-arXiv DataCite anchor row.
- Every anchor row must be `verified` with `proof_kind=arxiv_datacite_exact`.

### Provider candidate universe

The sealed `identifier-identity-evidence-v1.evidence_refs` array is the authoritative and exhaustive provider-candidate enumeration for this builder. Each evidence ref must be consumed exactly once and creates exactly one candidate keyed by normalized `(arxiv_id, alias)`.

- The evidence-ref array must be canonical, sorted by that key, and contain no duplicate key.
- A malformed ref, duplicate key, ref outside the dev gold set, unconsumed ref, or ref that cannot be decoded from its bound snapshot is an input-integrity rejection; it is never ignored.
- Snapshot records without evidence refs are coverage observations only and cannot contribute aliases. Creating new evidence refs from such records belongs to the separately reviewed capture contract.
- Before conflict handling, `provider_candidate_count == len(evidence_refs)` and every candidate has one private audit row.
- Every candidate is classified by the existing pure semantic classifier before any filtering.
- An incomplete candidate lacks the exact positive evidence required for `verified` and therefore classifies as `unresolved`.
- A `semantic_mismatch` or `unresolved` candidate is excluded from candidate-map bytes and prevents publication of a formal map.

### Conflict conservation

If the same normalized alias occurs under multiple arXiv IDs, all contributing candidate rows remain present. Conflict handling changes each contributing row to `state=unresolved`, `proof_kind=null`, and `reason_code=alias_target_conflict`; it never deletes or merges rows. Candidate count and total relation count are invariant before and after conflict handling.

### Non-relations

A gold group with no evidence ref has no provider relation to classify. The builder must not synthesize an `alias=<arxiv_id>` unresolved row. Its required DataCite anchor remains subject to the normal exact-verification gate.

## Aggregate Audit v2

The public aggregate audit advances to `identifier-map-semantic-audit-v2`. `input_hashes` contains only source inputs; `artifact_hashes` contains hashes of derived canonical bytes.

The exact public schema is closed at every object level (`additionalProperties=false`):

| Field | Exact type/value |
|---|---|
| `schema_version` | literal `identifier-map-semantic-audit-v2` |
| `scope` | literal `dev` |
| `status` | `passed` or `failed` |
| `input_hashes` | object with exactly `dev_gold`, `identity_evidence`, `snapshot_manifest` |
| `artifact_hashes` | object with exactly `candidate_map`, `private_relation_audit` |
| `gold_group_count` | non-negative integer |
| `required_anchor_count` | non-negative integer |
| `verified_anchor_count` | non-negative integer |
| `provider_candidate_count` | non-negative integer |
| `provider_identity_group_count` | non-negative integer |
| `provider_identity_missing_group_count` | non-negative integer |
| `relation_count` | non-negative integer |
| `state_counts` | object with exactly `verified`, `semantic_mismatch`, `unresolved` |
| `proof_counts` | object with exactly `arxiv_datacite_exact`, `semantic_scholar_exact`, `openalex_location_exact` |
| `reason_counts` | object with exactly `arxiv_datacite_exact`, `arxiv_datacite_mismatch`, `semantic_scholar_exact`, `openalex_location_exact`, `openalex_location_mismatch`, `insufficient_identity_evidence`, `observation_binding_mismatch`, `provider_identity_missing`, `alias_target_conflict` |

Every hash is a lowercase `sha256:` prefix followed by exactly 64 lowercase hexadecimal characters. Every count value is a non-negative integer; booleans are not integers for schema purposes. The closed relation enums are: state = `verified | semantic_mismatch | unresolved`; proof kind = `arxiv_datacite_exact | semantic_scholar_exact | openalex_location_exact | null`; reason code = exactly the nine `reason_counts` keys above. `provider_identity_missing` remains a supported zero-valued diagnostic reason for compatibility, but this builder never creates such a relation.

The exact private artifact is a closed object with `schema_version=identifier-map-private-relation-audit-v2`, `scope=dev`, and `relations`. Each relation is a closed object containing exactly `relation_kind`, `arxiv_id`, `alias`, `terminal`, `state`, `proof_kind`, and `reason_code`. `relation_kind` is `required_anchor` or `provider_candidate`; `state`, `proof_kind`, and `reason_code` use the public schema enums, with `proof_kind` alone permitting `null`. `arxiv_id`, `alias`, and `terminal` are non-empty canonical identifiers: `arxiv_id` is normalized with `kind=arxiv`, `alias` with automatic kind detection, and `terminal` with `kind=doi`. Every row requires `terminal == arxiv_anchor(arxiv_id)`. Every `required_anchor` row additionally requires `alias == terminal`; its verified form requires `proof_kind=arxiv_datacite_exact` and `reason_code=arxiv_datacite_exact`, while its nonverified form requires `proof_kind=null` and `reason_code=arxiv_datacite_mismatch`. Rows are unique and sorted by `(relation_kind, arxiv_id, alias)` using the explicit kind order `required_anchor`, then `provider_candidate`. The relation kind prevents a sealed provider candidate whose alias equals its deterministic DataCite anchor from being confused with the required anchor row.

The candidate map remains a bare JSON object whose keys and values are normalized identifier strings. It equals exactly the union of `(required_anchor.arxiv_id -> terminal)` and `(verified provider_candidate.alias -> terminal)` pairs, omitting the redundant provider self-edge when `alias == terminal`. Duplicate raw or normalized keys, non-string values, chains, cycles, extra envelope fields, and noncanonical ordering are invalid for this generation; every value is the terminal deterministic DataCite anchor.

All three artifacts use UTF-8 without BOM and canonical JSON with recursively sorted object keys, `ensure_ascii=false`, separators `(',', ':')`, and exactly one trailing LF. Hashes cover those exact bytes.

Let `G` be the set of normalized unique gold arXiv IDs, `A` the anchor audit rows, `C` the provider candidate rows, and `P={c.arxiv_id | c in C}`. Audit v2 contains and enforces:

- `gold_group_count = |G|`;
- `required_anchor_count = |A| = |G|`, with exactly one anchor row per member of `G`;
- `verified_anchor_count` equals the number of distinct members of `G` whose unique anchor row has `state=verified` and `proof_kind=arxiv_datacite_exact`;
- `provider_candidate_count = |C| = len(evidence_refs)`;
- `provider_identity_group_count = |P|`, regardless of candidate state;
- `P` is a subset of `G`;
- `provider_identity_missing_group_count = |G - P|`;
- `relation_count = |A| + |C|`.

A group with verified and failed provider candidates is counted once in `provider_identity_group_count`. Every relation row has exactly one state and one reason code. Audit v2 serializes zero-valued keys for every supported state, proof kind, and reason code.

- `sum(state_counts.values()) == relation_count`;
- `sum(reason_counts.values()) == relation_count`;
- `sum(proof_counts.values())` equals the number of rows whose `proof_kind` is non-null.

For the current sealed evidence, the artifact-specific hashes are:

- `dev_gold=sha256:24009cf03ad069131793b9a190024e239082277bd0e48149a1efbbbb7978e215`;
- `identity_evidence=sha256:e4567d4b7641871ed538c18f5625cd7037e3014065e7311d3a76e81d4e4c61d4`;
- `snapshot_manifest=sha256:a0c0cd67543582e02365a2adfb3464a6f33fa96be5304d4ccac8dd031867943b`.

When all three match, the reconstruction invariant is 141 gold groups, 90 provider-covered groups, 51 provider-missing groups, 90 provider candidates, and 231 total relations. A mismatch is a decoder regression and stops before status calculation. Different sealed input hashes do not inherit the fixed value 90.

`status=passed` requires:

- `required_anchor_count == verified_anchor_count == gold_group_count`;
- every relation in the private audit is `verified`;
- `semantic_mismatch=0`, `unresolved=0`, and `alias_target_conflict=0` among actual relations;
- `input_hashes` exactly binds `dev_gold`, `identity_evidence`, and `snapshot_manifest`;
- `artifact_hashes.candidate_map` binds the canonical candidate-map bytes;
- `artifact_hashes.private_relation_audit` binds the canonical private-audit bytes;
- the public audit passes recursive privacy validation before any public write.

Provider coverage counts are computed before status but are not status predicates. `provider_identity_missing_group_count` must equal `gold_group_count - provider_identity_group_count` and remains visible so reduced coverage cannot be hidden.

## Map Construction and Publication

The builder consumes only dev gold, sealed identity evidence, and its bound snapshot manifest.

1. Resolve the formal-map, private-audit, public-audit, and `<public-audit>.lock` sibling targets to normalized absolute paths. Require all four paths to be pairwise distinct; require the three formal targets to be absent. Acquire the publication lock with exclusive create before reading inputs. A concurrent or stale lock refuses the run.
2. After locking, read dev gold, identity evidence, snapshot manifest, and each referenced response exactly once into immutable byte buffers. Hashing, schema validation, manifest verification, response verification, and decoding must all use those same buffers. A preflight hash outside the builder is informational only and never authorizes publication.
3. Build and verify one anchor relation per gold group.
4. Consume every evidence ref exactly once, reconstruct its bound provider observation, and classify it before filtering.
5. Resolve cross-group conflicts while retaining every contributing audit row.
6. Build deterministic candidate-map bytes from verified anchors and verified aliases and canonical private-audit bytes. Record their SHA-256 values as `artifact_hashes.candidate_map` and `artifact_hashes.private_relation_audit`.
7. Follow exactly one outcome path:
   - **Input-integrity rejection:** hashes, manifest, response, evidence-ref structure, membership, uniqueness, or conservation fails. Raise a value-free error, write none of the three formal outputs, and release the publication lock.
   - **Decoder-regression rejection:** the three sealed baseline hashes match but the 141/90/51/90/231 invariant fails. Raise a value-free error, write none of the three formal outputs, and release the publication lock.
   - **Semantic gate failure:** trusted inputs reconstruct successfully but at least one relation is not verified. Schema-validate the private audit. Before any public write, reject duplicate keys in the public audit, enforce the exact v2 schema, require canonical reserialization to equal its bytes, and run recursive privacy validation. Exclusively write the private audit, re-read and verify its canonical bytes and bound hash, then exclusively write the validated public failed audit as the final commit marker; do not write a formal map. Release the publication lock only after the marker is durable.
   - **Successful publication:** apply the same private schema validation and public duplicate-key, exact-v2, canonical-byte, and privacy gates. Exclusively write the private audit and formal map, re-read and verify both canonical byte streams and their bound hashes, then exclusively write the validated public passed audit last as the final commit marker. Release the publication lock only after the marker is durable.

Every formal write uses atomic no-replace publication; an existence check alone is not sufficient. Consumers recognize a generation only through the public audit at the requested output path. Before reading `status` or any hash, the loader rejects duplicate JSON keys, enforces the exact audit-v2 schema, canonically reserializes the parsed audit, requires byte-for-byte equality with the original public-audit bytes, and reruns recursive privacy validation. It then requires `status=passed`, verifies every bound input hash, verifies private-audit bytes against `artifact_hashes.private_relation_audit`, and verifies raw map bytes against `artifact_hashes.candidate_map` before parsing either private rows or map entries.

After raw-byte verification, the loader strictly parses the private audit and map with duplicate-key detection and exact schema enforcement, reserializes each with the canonical JSON encoder, and requires byte-for-byte equality with the original bytes. It reconstructs `G`, `A`, `C`, and `P`, rechecks every public count/state/proof/reason equation against private rows and sealed evidence refs, and requires the parsed map to equal the exact map reconstructed from verified private rows. Synchronously changing public, private, and map bytes and their hashes therefore cannot validate an unsupported relation set. Hashing a noncanonical representation and updating the audit hash does not make that representation valid. Recursive privacy validation applies only to the public audit; the private audit is intentionally validated by its closed schema, canonical bytes, hashes, and relation consistency.

Because targets are distinct and absent, the publication lock is exclusive, every artifact write is no-replace, and the public audit is written last, a partial run cannot expose an old or uncommitted map as current. If interruption leaves a publication lock, private audit, or map without a public marker, automation must not delete or reuse it. Human intervention verifies that no public marker exists, archives the residual generation and lock, and chooses fresh empty output targets before another run.

The builder must not accept predictions, query text, historical runs, validation data, network access, manually supplied aliases, override environment variables, or sidecar bypass files.

## Safety and Human Intervention

- No new capture, readiness, candidate lock, `.env` access, or provider request is permitted for this amendment or rebuild.
- Snapshot or evidence integrity failure writes no formal output, stops immediately, and requires technical investigation.
- A real semantic mismatch or alias-target conflict produces a failed audit, stops promotion, and requires manual relation review.
- The stakeholder requirement for second-provider proof is a code-external policy stop, not a runtime switch. If stakeholders require proof for all 141 groups, implementation stops and asks humans to choose a new provider or a separately designed annotation workflow.
- Manual review cannot directly edit the map or inject an alias. It may identify a code/data defect for a new reviewed implementation cycle, or authorize a separately designed and sealed evidence workflow.
- If the three sealed baseline hashes match but reconstruction violates 141/90/51/90/231, stop as a decoder regression. New evidence hashes do not use this fixed invariant.
- Task 4 starts only after the public audit passes duplicate-key rejection, exact v2 schema validation, canonical byte equality, and recursive privacy validation and declares `status=passed`. Its loader then verifies all three source-input hashes, the private-audit hash, and the raw map hash before parsing private rows or map entries; it finally rechecks all equations and exact map reconstruction.
- If Task 4 fails retrieval, ranking, integrity, or budget guards, stop this improvement direction. Do not return to repeated identity capture without new evidence or a human policy decision.

## Testing Strategy

TDD coverage must prove:

- a gold group with only a verified anchor creates no unresolved placeholder;
- every evidence ref is classified before filtering, and incomplete or mismatched candidates fail the gate;
- duplicate, malformed, outside-gold, unconsumed, or undecodable evidence refs cause input-integrity rejection with no formal outputs;
- every alias-conflict contributor remains in the private audit and counts remain conserved;
- all set/count equations are exact, booleans are rejected as counts, zero-valued enum keys are present, and state/reason totals equal `relation_count`;
- the current sealed evidence reconstructs 141 anchors, 90 provider-covered groups, 51 provider-missing groups, 90 provider candidates, and 231 total relations without network access;
- input-integrity and decoder-regression failures write no formal outputs;
- semantic failures write a schema-valid private audit and privacy-valid public audit, include all failed candidates privately, bind both candidate map and private audit, and write no formal map;
- before either failed or passed public-audit publication, the builder proves duplicate-key rejection, exact v2 schema, canonical byte equality, and recursive privacy validation;
- successful rebuilds bind both private audit and map, re-read both, and write the public passed audit as the final commit marker;
- normalized formal-output and publication-lock targets must all be pairwise distinct; existing targets, path aliases, concurrent publication locks, and atomic no-replace races are rejected;
- interruption before public-audit publication leaves no recognized generation and requires human archival to fresh targets rather than automatic cleanup;
- replacing any input between an attempted hash check and decode cannot change the byte buffers actually used by the builder; gold, evidence, manifest, and each distinct referenced response are opened once after publication-lock acquisition, while unreferenced responses are never opened;
- map and private-audit serialization are deterministic and canonical, while the public audit is additionally privacy-safe;
- the CLI and environment expose no predictions, query text, historical-run, validation, `.env`, network, manual-alias, override, or sidecar-bypass input;
- before reading any field, the Task 4 loader rejects public-audit duplicate keys, extra or missing schema fields, noncanonical public-audit bytes, and public-audit privacy violations;
- the loader rejects a non-v2 or non-passed audit; missing, malformed, or mismatched dev-gold, identity-evidence, snapshot-manifest, private-audit, or candidate-map hashes; and any raw map-byte change before parsing entries;
- after hash verification, the loader rejects duplicate JSON keys, extra structure, and hash-synchronized noncanonical JSON; it rejects any public/private/map combination whose rows, counts, evidence universe, or reconstructed map disagree even when all three hashes were changed together.

After focused tests, run the identifier semantics, capture, rebuild, ledger, and snapshot suites, followed by Ruff and mypy on affected source and scripts.

## Implementation Boundary

This amendment implements only the offline rebuild/audit contract, the strict verified-generation loader required by Task 4, and their tests. It may publish one aggregate public audit after the Gate passes. It does not implement or execute the deferred rescore, which remains a separate downstream task and must consume this loader rather than recreating its checks.

The capture collector, sealed snapshots, private evidence, shared snapshot contract, ledger, semantic classifier, historical runs, rescore report model, and OpenAlex behavior remain unchanged.

The amendment relies on `src/paper_search/evaluation/identifier_semantics.py` (`classify_relation`, `arxiv_anchor`, semantic states, and privacy scanner), `src/paper_search/storage/dependency_snapshot.py` (manifest and response verification), and `scripts/rebuild_dev_identifier_map.py` (canonical JSON hashing and atomic output publication).
