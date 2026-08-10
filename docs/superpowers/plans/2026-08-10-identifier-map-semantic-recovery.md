# Identifier Map Semantic Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore a semantically trustworthy, dev-only identifier map, bind its audit into formal evaluation inputs, and recompute the existing sealed baselines without changing retrieval behavior.

**Architecture:** A pure offline semantic core classifies exact arXiv/provider relations as `verified`, `semantic_mismatch`, or `unresolved`. A bounded capture command produces private identity snapshots; a separate offline builder emits a dev-only map plus an aggregate-safe audit. A split-scoped V3 freeze binds the map and audit while preserving all V2 artifacts, then an offline rescoring command recomputes existing sealed runs.

**Tech Stack:** Python 3.11, Pydantic v2, httpx, SQLite budget ledger, canonical JSON/JSONL, pytest, Ruff, mypy.

## Global Constraints

- Implement with TDD; every behavior change starts with a failing focused test.
- Use only existing dependencies from `pyproject.toml`; do not add packages.
- Do not modify production retrieval, ranking, filtering, prompts, or optional-module behavior.
- Do not modify `data/manifest.json`, the old identifier map, historical runs, snapshots, ledgers, or evidence.
- Scope all new identity data, maps, manifests, and scoring to `dev`; never open or score the validation partition.
- Keep per-paper identity evidence, map entries, and unresolved rows under ignored `data/annotation_work/`; public artifacts contain aggregate counts and hashes only.
- Do not read `.env` or make network requests during Tasks 1, 3, 4, and 5. Task 2 real capture requires a separate explicit authorization and `--allow-network`.
- A semantic audit passes only with zero mismatches, zero unresolved dev groups, complete provider checks, and all 12 directly identifiable sealed-baseline hits recovered.
- Do not build a candidate lock, run readiness, run live capture, or access validation in this plan.
- Preserve the existing user changes in `HANDOFF.md`, `docs/retrieval-roadmap.md`, `data/budget_ledger.sqlite3`, and `deliverables/` until Task 6 deliberately updates only the two tracked documents.

---

### Task 1: Pure semantic identity contract

**Files:**
- Create: `src/paper_search/evaluation/identifier_semantics.py`
- Modify: `src/paper_search/evaluation/dataset.py:253-339`
- Create: `tests/evaluation/test_identifier_semantics.py`
- Modify: `tests/evaluation/test_dataset.py:256-390`

**Interfaces:**
- Consumes: normalized identifiers from `normalize_paper_id`, exact map bytes, dev gold bytes, private evidence bytes, and sealed prediction bytes.
- Produces: `arxiv_anchor(arxiv_id: str) -> str`, `IdentityObservation`, `RelationAudit`, `IdentifierMapSemanticAudit`, and `audit_identifier_map_semantics(...) -> SemanticAuditBundle`.
- Produces for later tasks: `IdentifierMap.resolved_pairs() -> tuple[tuple[str, str], ...]`, sorted and in-memory only.

- [ ] **Step 1: Add failing map-introspection tests**

```python
def test_identifier_map_exposes_sorted_resolved_pairs_without_serializing() -> None:
    identifier_map = IdentifierMap.from_bytes(
        b'{"arxiv:2501.00002":"doi:10.48550/arxiv.2501.00002",'
        b'"doi:10.1000/example":"doi:10.48550/arxiv.2501.00002"}\n'
    )

    assert identifier_map.resolved_pairs() == (
        ("arxiv:2501.00002", "doi:10.48550/arxiv.2501.00002"),
        ("doi:10.1000/example", "doi:10.48550/arxiv.2501.00002"),
    )
```

- [ ] **Step 2: Run the new dataset test and confirm RED**

Run:

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest tests/evaluation/test_dataset.py -k resolved_pairs -q
```

Expected: FAIL because `IdentifierMap.resolved_pairs` does not exist.

- [ ] **Step 3: Add the minimal sorted read-only method**

```python
def resolved_pairs(self) -> tuple[tuple[str, str], ...]:
    """Return normalized resolved pairs for private in-process auditing only."""
    return tuple(sorted(self._resolved.items()))
```

- [ ] **Step 4: Write failing semantic-state tests**

```python
def test_datacite_arxiv_relation_requires_the_same_identifier() -> None:
    assert arxiv_anchor("arxiv:2501.10120") == "doi:10.48550/arxiv.2501.10120"
    assert classify_relation(
        alias="doi:10.48550/arxiv.2501.10120",
        arxiv_id="arxiv:2501.10120",
        observation=None,
    ).state == "verified"
    assert classify_relation(
        alias="doi:10.48550/arxiv.2409.00001",
        arxiv_id="arxiv:2501.10120",
        observation=None,
    ).state == "semantic_mismatch"


def test_s2_disagreement_without_exact_openalex_evidence_is_unresolved() -> None:
    observation = IdentityObservation(
        arxiv_id="arxiv:2501.10120",
        alias="doi:10.1000/example",
        lookup_complete=True,
        semantic_scholar_arxiv_paper_id="S2-A",
        semantic_scholar_alias_paper_id="S2-B",
        openalex_arxiv_ids=[],
        snapshot_sha256s=["sha256:" + "a" * 64],
    )

    assert classify_relation(
        alias=observation.alias,
        arxiv_id=observation.arxiv_id,
        observation=observation,
    ).state == "unresolved"
```

- [ ] **Step 5: Run semantic tests and confirm RED**

Run:

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest tests/evaluation/test_identifier_semantics.py -q
```

Expected: collection FAIL because `paper_search.evaluation.identifier_semantics` is absent.

- [ ] **Step 6: Implement the exact models and classifier**

Create strict frozen models with these signatures:

```python
SemanticState = Literal["verified", "semantic_mismatch", "unresolved"]
ProofKind = Literal[
    "arxiv_datacite_exact",
    "semantic_scholar_exact",
    "openalex_location_exact",
]


class IdentityObservation(DomainModel):
    arxiv_id: NonEmptyStr
    alias: NonEmptyStr
    lookup_complete: bool
    semantic_scholar_arxiv_paper_id: str | None = None
    semantic_scholar_alias_paper_id: str | None = None
    openalex_arxiv_ids: list[NonEmptyStr] = Field(default_factory=list)
    snapshot_sha256s: list[Sha256] = Field(min_length=1)


class RelationAudit(DomainModel):
    arxiv_id: NonEmptyStr
    alias: NonEmptyStr
    terminal: NonEmptyStr
    state: SemanticState
    proof_kind: ProofKind | None
    reason_code: NonEmptyStr


class SemanticAuditCounts(DomainModel):
    gold_group_count: NonNegativeInt
    relation_count: NonNegativeInt
    verified: NonNegativeInt
    semantic_mismatch: NonNegativeInt
    unresolved: NonNegativeInt


class DirectArxivSanity(DomainModel):
    prediction_count: NonNegativeInt
    hit_count: NonNegativeInt


class IdentifierMapSemanticAudit(DomainModel):
    schema_version: Literal["identifier-map-semantic-audit-v1"]
    scope: Literal["dev"]
    status: Literal["passed", "failed"]
    input_hashes: dict[NonEmptyStr, Sha256]
    counts: SemanticAuditCounts
    proof_counts: dict[NonEmptyStr, NonNegativeInt]
    direct_arxiv_sanity: DirectArxivSanity
    reason_codes: list[NonEmptyStr]


@dataclass(frozen=True)
class SemanticAuditBundle:
    report: IdentifierMapSemanticAudit
    private_relations: tuple[RelationAudit, ...]


def arxiv_anchor(arxiv_id: str) -> str:
    normalized = normalize_paper_id(arxiv_id, kind="arxiv")
    return f"doi:10.48550/arxiv.{normalized.removeprefix('arxiv:')}"


def classify_relation(
    *,
    alias: str,
    arxiv_id: str,
    observation: IdentityObservation | None,
) -> RelationAudit:
    """Apply only the three exact proof rules from the approved design."""
```

The function order is mandatory: normalize; accept same-ID DataCite DOI; reject different-ID DataCite DOI; accept equal non-empty S2 paper IDs; accept exact arXiv membership in OpenAlex IDs; reject an explicit different OpenAlex arXiv association; otherwise return unresolved. Error messages remain value-free.

- [ ] **Step 7: Add aggregate audit and privacy tests**

```python
def test_public_audit_contains_counts_and_hashes_but_no_identifiers() -> None:
    bundle = audit_identifier_map_semantics(
        map_bytes=DEV_MAP_BYTES,
        gold_bytes=DEV_GOLD_BYTES,
        evidence_bytes=PRIVATE_EVIDENCE_BYTES,
        baseline_predictions_bytes=PREDICTIONS_BYTES,
    )
    serialized = bundle.report.model_dump_json()

    assert bundle.report.schema_version == "identifier-map-semantic-audit-v1"
    assert bundle.report.scope == "dev"
    assert bundle.report.counts.semantic_mismatch == 0
    assert bundle.report.direct_arxiv_sanity.hit_count == 1
    assert "arxiv:" not in serialized
    assert "doi:" not in serialized
    assert "query-1" not in serialized
```

Implement:

```python
def audit_identifier_map_semantics(
    *,
    map_bytes: bytes,
    gold_bytes: bytes,
    evidence_bytes: bytes,
    baseline_predictions_bytes: bytes,
) -> SemanticAuditBundle:
    """Return private relation rows and one aggregate-safe public report."""
```

The public report has exactly `schema_version`, `scope`, `status`, `input_hashes`, `counts`, `proof_counts`, `direct_arxiv_sanity`, and `reason_codes`. `status="passed"` requires complete dev coverage, all relations verified, and the direct-hit count reconstructed from same-ID arXiv DOI predictions.

- [ ] **Step 8: Run focused tests and commit**

Run:

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest tests/evaluation/test_dataset.py tests/evaluation/test_identifier_semantics.py -q
```

Expected: all selected tests PASS.

Commit:

```powershell
git add src/paper_search/evaluation/dataset.py src/paper_search/evaluation/identifier_semantics.py tests/evaluation/test_dataset.py tests/evaluation/test_identifier_semantics.py
git commit -m "feat: add identifier semantic audit contract"
```

---

### Task 2: Bounded private identity capture

**Files:**
- Create: `src/paper_search/evaluation/identifier_identity_capture.py`
- Create: `scripts/capture_dev_identifier_identity.py`
- Create: `tests/evaluation/test_identifier_identity_capture.py`
- Create: `tests/scripts/test_capture_dev_identifier_identity.py`

**Interfaces:**
- Consumes: dev gold bytes, one immutable capture lock, an injected `httpx.AsyncClient`, `SQLiteBudgetLedger`, and `DependencyCaptureStore`.
- Produces: `IdentityCaptureLock`, `IdentityEvidenceSet`, `preflight_identity_capture(...)`, and `capture_identity_evidence(...)`.
- Real output: private canonical JSON and provider snapshots under `data/annotation_work/identifier_semantics/`; no public report.

- [ ] **Step 1: Write failing preflight and isolation tests**

```python
def test_preflight_locks_only_dev_arxiv_ids_and_makes_no_requests(tmp_path: Path) -> None:
    lock = preflight_identity_capture(
        gold_bytes=DEV_GOLD_BYTES,
        source_git_sha="a" * 40,
        project_checkpoint=(0, "sha256:" + "0" * 64),
    )

    assert lock.schema_version == "identifier-identity-capture-lock-v1"
    assert lock.scope == "dev"
    assert lock.gold_count == 2
    assert lock.semantic_scholar_batch_calls_max == 1
    assert lock.openalex_exact_calls_max == 2
    assert "query" not in lock.model_dump_json()
```

- [ ] **Step 2: Run tests and confirm RED**

Run:

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest tests/evaluation/test_identifier_identity_capture.py tests/scripts/test_capture_dev_identifier_identity.py -q
```

Expected: collection FAIL because the capture module and script do not exist.

- [ ] **Step 3: Implement the lock and zero-network preflight**

```python
class IdentityCaptureLock(DomainModel):
    schema_version: Literal["identifier-identity-capture-lock-v1"]
    scope: Literal["dev"]
    source_git_sha: NonEmptyStr
    gold_sha256: Sha256
    gold_count: PositiveInt
    semantic_scholar_batch_calls_max: Literal[1]
    openalex_exact_calls_max: PositiveInt
    attempts_per_request_max: Literal[3]
    timeout_seconds: Literal[1800]
    project_receipt_count: NonNegativeInt
    project_receipts_sha256: Sha256
    lock_sha256: Sha256


class IdentityEvidenceSet(DomainModel):
    schema_version: Literal["identifier-identity-evidence-v1"]
    scope: Literal["dev"]
    capture_lock_sha256: Sha256
    gold_sha256: Sha256
    observations: list[IdentityObservation]
    observation_count: NonNegativeInt
    snapshot_entry_count: NonNegativeInt
    snapshot_manifest_sha256: Sha256
    project_receipt_count: NonNegativeInt
    project_receipts_sha256: Sha256
    unsettled_receipt_count: Literal[0]


def preflight_identity_capture(
    *,
    gold_bytes: bytes,
    source_git_sha: str,
    project_checkpoint: tuple[int, str],
) -> IdentityCaptureLock:
    """Create a self-hashed dev-only lock without constructing an HTTP client."""
```

Preflight validates exactly 60 dev rows and 141 unique normalized arXiv IDs for the real dataset, but fixtures may use smaller positive counts. It never accepts a validation path, query text output, map path, or environment path.

- [ ] **Step 4: Write failing capture/replay/accounting tests**

```python
@pytest.mark.asyncio
async def test_capture_uses_exact_ids_and_seals_terminal_receipts(tmp_path: Path) -> None:
    evidence = await capture_identity_evidence(
        lock=LOCK,
        gold_bytes=DEV_GOLD_BYTES,
        client=mock_identity_client(),
        ledger=LEDGER,
        snapshot_store=STORE,
    )

    assert evidence.scope == "dev"
    assert evidence.observation_count == 2
    assert evidence.unsettled_receipt_count == 0
    assert evidence.snapshot_entry_count > 0
    assert all("query" not in request.body.decode("utf-8") for request in REQUESTS)
```

Also test HTTP 404, 429 retry exhaustion, malformed JSON, a missing S2 batch row, OpenAlex work without locations, cancellation, ledger settlement failure, lock/hash mismatch, and attempted `validation` scope. Every terminal path must settle or fail all reserved receipts and seal the observations already obtained.

- [ ] **Step 5: Implement bounded capture**

```python
async def capture_identity_evidence(
    *,
    lock: IdentityCaptureLock,
    gold_bytes: bytes,
    client: httpx.AsyncClient,
    ledger: SQLiteBudgetLedger,
    snapshot_store: DependencyCaptureStore,
) -> IdentityEvidenceSet:
    """Capture exact S2 batch and OpenAlex identity metadata under the lock."""
```

Send one Semantic Scholar `/graph/v1/paper/batch` request with frozen `ARXIV:<id>` values and fields `paperId,externalIds,title,year`. For each returned DOI, send one OpenAlex exact-work request; when S2 has no DOI, query the same-ID DataCite arXiv DOI. Preserve response bytes through `DependencyCaptureStore`, parse only after capture, and write canonical observations containing no query text. Use the lock's request and attempt caps; no title search or fuzzy fallback is allowed.

- [ ] **Step 6: Add the explicit CLI boundary**

```python
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subcommands = parser.add_subparsers(dest="command", required=True)
    preflight = subcommands.add_parser("preflight")
    preflight.add_argument("--gold", type=Path, required=True)
    preflight.add_argument("--ledger", type=Path, required=True)
    preflight.add_argument("--out-lock", type=Path, required=True)
    run = subcommands.add_parser("run")
    run.add_argument("--lock", type=Path, required=True)
    run.add_argument("--gold", type=Path, required=True)
    run.add_argument("--ledger", type=Path, required=True)
    run.add_argument("--snapshot-root", type=Path, required=True)
    run.add_argument("--out-private", type=Path, required=True)
    run.add_argument("--allow-network", action="store_true")
```

`run` returns exit code 2 before client construction unless `--allow-network` is present. Neither command loads `.env`; credentials come only from already-set process variables. Stdout contains counts, hashes, and status, never paper IDs.

- [ ] **Step 7: Run focused tests and commit the implementation only**

Run:

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest tests/evaluation/test_identifier_identity_capture.py tests/scripts/test_capture_dev_identifier_identity.py -q
```

Expected: all selected tests PASS with `httpx.MockTransport`; zero real requests.

Commit:

```powershell
git add src/paper_search/evaluation/identifier_identity_capture.py scripts/capture_dev_identifier_identity.py tests/evaluation/test_identifier_identity_capture.py tests/scripts/test_capture_dev_identifier_identity.py
git commit -m "feat: add bounded identifier identity capture"
```

Do not run the real `run` subcommand in this task.

---

### Task 3: Dev map rebuild and aggregate audit publication

**Files:**
- Create: `scripts/rebuild_dev_identifier_map.py`
- Create: `tests/scripts/test_rebuild_dev_identifier_map.py`
- Modify: `src/paper_search/evaluation/identifier_semantics.py`
- Modify: `tests/evaluation/test_identifier_semantics.py`

**Interfaces:**
- Consumes: dev gold bytes, private `IdentityEvidenceSet`, and the sealed baseline `predictions.jsonl` bytes.
- Produces: `build_verified_dev_map(...) -> DevMapBuildResult` and `publish_dev_semantic_map(...)`.
- Private outputs: canonical map and per-relation audit under `data/annotation_work/identifier_semantics/`.
- Public output: one aggregate `identifier-map-semantic-audit-v1` JSON under `docs/evidence/`.

- [ ] **Step 1: Write failing builder tests**

```python
def test_builder_uses_same_arxiv_anchor_and_only_verified_provider_aliases() -> None:
    result = build_verified_dev_map(
        gold_bytes=DEV_GOLD_BYTES,
        evidence_bytes=COMPLETE_EVIDENCE_BYTES,
        baseline_predictions_bytes=PREDICTIONS_BYTES,
    )

    assert result.mapping == {
        "arxiv:2501.10120": "doi:10.48550/arxiv.2501.10120",
        "doi:10.1000/example": "doi:10.48550/arxiv.2501.10120",
        "openalex:W1": "doi:10.48550/arxiv.2501.10120",
    }
    assert result.audit.status == "passed"


def test_builder_does_not_publish_formal_map_with_unresolved_gold() -> None:
    result = build_verified_dev_map(
        gold_bytes=DEV_GOLD_BYTES,
        evidence_bytes=INCOMPLETE_EVIDENCE_BYTES,
        baseline_predictions_bytes=PREDICTIONS_BYTES,
    )

    assert result.audit.status == "failed"
    assert result.mapping is None
```

- [ ] **Step 2: Run tests and confirm RED**

Run:

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest tests/scripts/test_rebuild_dev_identifier_map.py -q
```

Expected: FAIL because the builder does not exist.

- [ ] **Step 3: Implement deterministic grouping and conflict rejection**

```python
class DevMapBuildResult(DomainModel):
    mapping: dict[NonEmptyStr, NonEmptyStr] | None
    private_relations: list[RelationAudit]
    audit: IdentifierMapSemanticAudit


def build_verified_dev_map(
    *,
    gold_bytes: bytes,
    evidence_bytes: bytes,
    baseline_predictions_bytes: bytes,
) -> DevMapBuildResult:
    """Build a dev-only alias map; return no map unless every group passes."""
```

For each gold arXiv ID, add `gold -> same-ID DataCite anchor`. Add S2 DOI, OpenAlex DOI, and OpenAlex work ID only after `classify_relation` returns verified. If one provider alias is claimed by two anchors, mark both groups unresolved. Canonically sort keys and reject duplicate normalized keys before serialization.

- [ ] **Step 4: Add atomic publication and leakage tests**

```python
def test_publication_is_atomic_and_public_report_is_aggregate_only(tmp_path: Path) -> None:
    publish_dev_semantic_map(
        result=PASSED_RESULT,
        map_path=tmp_path / "private" / "dev-map.json",
        private_audit_path=tmp_path / "private" / "relations.json",
        public_audit_path=tmp_path / "public" / "audit.json",
    )

    public = (tmp_path / "public" / "audit.json").read_text(encoding="utf-8")
    assert "arxiv:" not in public
    assert "doi:" not in public
    assert "openalex:" not in public
    assert not list(tmp_path.rglob("*.tmp"))
```

`publish_dev_semantic_map` must write siblings, flush, replace atomically, reread all files, verify hashes, and run the prohibited-field scan. A failed result may write the private relation report and failed aggregate audit, but must not create or overwrite the formal map.

Implement the publisher with this exact boundary:

```python
def publish_dev_semantic_map(
    *,
    result: DevMapBuildResult,
    map_path: Path,
    private_audit_path: Path,
    public_audit_path: Path,
) -> None:
    """Atomically publish private rows and aggregate evidence without overwrites."""
```

- [ ] **Step 5: Add the offline CLI and run focused tests**

CLI arguments are exactly:

```text
--gold PATH --evidence PATH --baseline-predictions PATH
--out-map PATH --out-private-audit PATH --out-public-audit PATH
```

Run:

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest tests/evaluation/test_identifier_semantics.py tests/scripts/test_rebuild_dev_identifier_map.py -q
```

Expected: all selected tests PASS; no network fixture is constructed.

- [ ] **Step 6: Commit**

```powershell
git add src/paper_search/evaluation/identifier_semantics.py tests/evaluation/test_identifier_semantics.py scripts/rebuild_dev_identifier_map.py tests/scripts/test_rebuild_dev_identifier_map.py
git commit -m "feat: rebuild verified dev identifier map"
```

---

### Task 4: Split-scoped V3 freeze and formal binding

**Files:**
- Modify: `src/paper_search/evaluation/freeze_schema.py:140-245,1160-1320`
- Modify: `src/paper_search/application/locks.py:35-220`
- Modify: `src/paper_search/evaluation/gate0.py:45-190,320-570`
- Modify: `src/paper_search/evaluation/runner.py:140-290,1215-1255`
- Modify: `src/paper_search/application/artifacts.py:132-165,620-650`
- Modify: `src/paper_search/evaluation/validator.py:120-205,700-755`
- Create: `scripts/build_dev_semantic_freeze.py`
- Modify: `tests/evaluation/test_freeze_schema.py`
- Modify: `tests/application/test_locks.py`
- Modify: `tests/evaluation/test_gate0.py`
- Modify: `tests/evaluation/test_runner.py`
- Modify: `tests/evaluation/test_artifacts.py`
- Modify: `tests/evaluation/test_validator.py`
- Create: `tests/scripts/test_build_dev_semantic_freeze.py`

**Interfaces:**
- Consumes: the unchanged V2 manifest hash, dev partition binding, passed audit bytes, private dev map bytes, and private evidence-set hash.
- Produces: `FreezeManifestV3`, `FreezeApprovalReportV3`, optional semantic fields on `FrozenDataBinding`, and corresponding formal-run identity fields.
- Compatibility: V1/V2 manifests and integrated-lock-v1 historical artifacts continue parsing and verifying unchanged.

- [ ] **Step 1: Write failing V3 schema tests**

```python
def test_v3_manifest_binds_one_dev_partition_and_semantic_evidence() -> None:
    manifest = FreezeManifestV3.model_validate(V3_PAYLOAD, strict=True)

    assert manifest.schema_version == "paper-search-freeze-v3"
    assert manifest.partition.name == "dev"
    assert manifest.identifier_semantics.audit_sha256 == AUDIT_SHA
    assert manifest.identifier_semantics.private_evidence_sha256 == EVIDENCE_SHA


def test_v3_manifest_rejects_validation_scope() -> None:
    payload = copy.deepcopy(V3_PAYLOAD)
    payload["partition"]["name"] = "validation"
    with pytest.raises(ValueError, match="V3 manifest must be dev scoped"):
        FreezeManifestV3.model_validate(payload, strict=True)
```

- [ ] **Step 2: Run schema tests and confirm RED**

Run:

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest tests/evaluation/test_freeze_schema.py -k v3 -q
```

Expected: FAIL because V3 classes are absent.

- [ ] **Step 3: Add exact V3 models without changing V2**

```python
class IdentifierSemanticBindingV3(_FreezeModel):
    map: IdentifierMapBindingV2
    audit_sha256: Sha256
    private_evidence_sha256: Sha256


class FreezeApprovalReportV3(_FreezeModel):
    schema_version: Literal["freeze-approval-v3"]
    approved_at: datetime
    approver_ref: NonEmptyStr
    base_manifest_sha256: Sha256
    partition_sha256: Sha256
    identifier_map_sha256: Sha256
    semantic_audit_sha256: Sha256
    private_evidence_sha256: Sha256


class FreezeManifestV3(_FreezeModel):
    schema_version: Literal["paper-search-freeze-v3"]
    dataset_revision: NonEmptyStr
    created_at: datetime
    base_manifest_sha256: Sha256
    partition: FrozenPartitionV2
    identifier_semantics: IdentifierSemanticBindingV3
    partition_immutability: Literal["content_addressed"]
    approval: FreezeApprovalBindingV2
```

Add V3 to the discriminated `FreezeManifest` union. Its model validator requires `partition.name == "dev"` and internally consistent binding fields. `open_validated_freeze_evidence` performs the byte-level checks against the passed dev-scoped audit and V3 approval report because the Pydantic model does not perform file I/O.

- [ ] **Step 4: Write failing lock, Gate 0, and pre-provider tests**

```python
def test_v3_candidate_lock_requires_semantic_bindings() -> None:
    payload = copy.deepcopy(CANDIDATE_LOCK_PAYLOAD)
    payload["frozen_data"]["semantic_audit"] = AUDIT_BINDING
    payload["frozen_data"]["private_identity_evidence_sha256"] = EVIDENCE_SHA
    assert CandidateLock.model_validate(payload, strict=True).frozen_data.semantic_audit


def test_v3_gate_reads_only_dev_partition(v3_gate0: Gate0Fixture) -> None:
    report = v3_gate0.verify()
    assert report.passed
    assert [(item.identity, item.count) for item in report.partitions] == [("dev", 1)]
    assert "validation" not in report.model_dump_json()


def test_semantic_audit_failure_stops_before_provider_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(CompositionRoot, "compose", pytest.fail)
    with pytest.raises(ValueError, match="identifier semantic audit is invalid"):
        _load_formal_inputs(V3_REQUEST_WITH_FAILED_AUDIT)
```

- [ ] **Step 5: Extend bindings and validation**

Add to `FrozenDataBinding`:

```python
semantic_audit: ArtifactBinding | None = None
private_identity_evidence_sha256: Sha256 | None = None
```

Require the two fields together. Include `semantic_audit` in `_artifact_bindings` when present. V3 loading requires both; V2 loading requires both absent. V3 `Gate0Report` adds optional aggregate artifact evidence for the audit and its private evidence hash, while keeping `schema_version="gate0-report-v1"` for serialized backward compatibility.

Add optional `semantic_audit_path: Path | None = None` to `verify_gate0` and `--semantic-audit PATH` to its CLI. V3 requires the path and verifies its hash against the manifest; V2 rejects the argument so historical behavior remains explicit. The V3 manifest binds only the audit hash, while the lock binds the workspace-relative audit path and the same hash.

Add optional fields to `RunManifest`:

```python
identifier_semantic_audit_sha256: Sha256 | None = None
private_identity_evidence_sha256: Sha256 | None = None
```

The runner, validator, replay-lock inheritance check, and capture/replay comparison must require exact equality for both fields. Old runs with both fields absent remain valid.

- [ ] **Step 6: Add the offline V3 freeze builder**

```python
def build_dev_semantic_freeze(
    *,
    base_manifest_bytes: bytes,
    dev_gold_bytes: bytes,
    map_bytes: bytes,
    audit_bytes: bytes,
    private_evidence_sha256: str,
    approved_at: datetime,
    approver_ref: str,
) -> tuple[FreezeManifestV3, FreezeApprovalReportV3]:
    """Build a new dev-only identity without mutating the V2 manifest."""
```

The CLI accepts exactly `--base-manifest`, `--dev-gold`, `--id-map`, `--audit`, `--private-evidence`, `--out-manifest`, `--out-approval`, `--approval-ref`, and `--approved-at`. It uses `approved_at` as the V3 `created_at`, writes a new private manifest and approval report atomically under `data/annotation_work/identifier_semantics/freeze-v3/`, and binds the audit hash without copying public evidence into the private freeze. It rejects a non-passed audit, wrong hashes, any validation partition input, an existing output, or an approval timestamp without UTC offset.

- [ ] **Step 7: Run the affected contract suites**

Run:

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest tests/evaluation/test_freeze_schema.py tests/application/test_locks.py tests/evaluation/test_gate0.py tests/evaluation/test_runner.py tests/evaluation/test_artifacts.py tests/evaluation/test_validator.py tests/scripts/test_build_dev_semantic_freeze.py -q
```

Expected: all selected tests PASS, including unchanged V2 compatibility tests.

- [ ] **Step 8: Commit**

```powershell
git add src/paper_search/evaluation/freeze_schema.py src/paper_search/application/locks.py src/paper_search/evaluation/gate0.py src/paper_search/evaluation/runner.py src/paper_search/application/artifacts.py src/paper_search/evaluation/validator.py scripts/build_dev_semantic_freeze.py tests/evaluation/test_freeze_schema.py tests/application/test_locks.py tests/evaluation/test_gate0.py tests/evaluation/test_runner.py tests/evaluation/test_artifacts.py tests/evaluation/test_validator.py tests/scripts/test_build_dev_semantic_freeze.py
git commit -m "feat: bind semantic identity to dev evaluation"
```

---

### Task 5: Sealed-run offline rescoring and trustworthy funnel

**Files:**
- Create: `scripts/rescore_identifier_semantics.py`
- Create: `tests/scripts/test_rescore_identifier_semantics.py`
- Modify: `src/paper_search/evaluation/identifier_semantics.py`
- Modify: `tests/evaluation/test_identifier_semantics.py`

**Interfaces:**
- Consumes: passed audit, private verified map, dev gold, and one or more already sealed run directories.
- Produces: `SemanticRescoreReport` with one aggregate row per independently hashed run and an aggregate-only JSON/Markdown pair.
- Does not produce: new snapshots, predictions, business results, candidate locks, or ledger entries.

- [ ] **Step 1: Write failing rescoring tests**

```python
def test_rescore_requires_valid_sealed_run_and_recovers_direct_hits(tmp_path: Path) -> None:
    report = rescore_runs(
        gold_path=GOLD_PATH,
        map_path=VERIFIED_MAP_PATH,
        audit_path=PASSED_AUDIT_PATH,
        run_paths=[SEALED_BASELINE_PATH],
    )

    row = report.runs[0]
    assert row.run_id == "dev-sealed-baseline"
    assert row.direct_arxiv_hit_count == 12
    assert row.true_positive_count >= 12
    assert row.pipeline_stages.total_gold_associations == 143


def test_rescore_rejects_cross_run_stage_inputs() -> None:
    with pytest.raises(ValueError, match="sealed run binding mismatch"):
        rescore_runs(
            gold_path=GOLD_PATH,
            map_path=VERIFIED_MAP_PATH,
            audit_path=PASSED_AUDIT_PATH,
            run_paths=[RUN_WITH_SWAPPED_EXECUTIONS],
        )


def test_query_evolution_probe_uses_its_own_lock_outcomes_and_source_hashes() -> None:
    report = rescore_runs(
        gold_path=GOLD_PATH,
        map_path=VERIFIED_MAP_PATH,
        audit_path=PASSED_AUDIT_PATH,
        run_paths=[SEALED_SOURCE_RUN],
        query_evolution_probe_paths=[SEALED_PROBE_PATH],
    )

    assert report.runs[-1].source_kind == "query_evolution_probe"
    assert report.runs[-1].capture_replay_match == "matched"
```

- [ ] **Step 2: Run tests and confirm RED**

Run:

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest tests/scripts/test_rescore_identifier_semantics.py -q
```

Expected: FAIL because the rescoring script does not exist.

- [ ] **Step 3: Implement offline-only run loading and scoring**

```python
class LegacyRunBinding(DomainModel):
    run_path: Path
    evidence_path: Path


class PipelineStageCounts(DomainModel):
    total_gold_associations: NonNegativeInt
    not_retrieved: NonNegativeInt
    filtered_out: NonNegativeInt
    ranked_outside_top50: NonNegativeInt
    selected_top50: NonNegativeInt


class SemanticRescoreRun(DomainModel):
    source_kind: Literal["formal_run", "legacy_hash_bound_run", "query_evolution_probe"]
    run_id: NonEmptyStr
    input_hashes: dict[NonEmptyStr, Sha256]
    source_semantic_status: Literal["historical_semantics_unverified"]
    recomputed_semantic_status: Literal["semantic_audit_passed"]
    direct_arxiv_hit_count: NonNegativeInt
    true_positive_count: NonNegativeInt
    macro_f1: float
    macro_recall: float
    micro_recall: float
    pipeline_stages: PipelineStageCounts
    capture_replay_match: Literal["matched", "not_applicable"]


class SemanticRescoreReport(DomainModel):
    schema_version: Literal["identifier-semantic-rescore-v1"]
    scope: Literal["dev"]
    gold_sha256: Sha256
    identifier_map_sha256: Sha256
    semantic_audit_sha256: Sha256
    runs: list[SemanticRescoreRun]


def rescore_runs(
    *,
    gold_path: Path,
    map_path: Path,
    audit_path: Path,
    run_paths: Sequence[Path],
    query_evolution_probe_paths: Sequence[Path] = (),
    legacy_run_bindings: Sequence[LegacyRunBinding] = (),
) -> SemanticRescoreReport:
    """Verify, rescore, and attribute sealed runs without network access."""
```

For every formal run, call `validate_run_directory` first, then hash and load its own `run.json`, `predictions.jsonl`, `executions.jsonl`, and `business-results.jsonl`. Recompute metrics through `evaluate`, and derive `not_retrieved`, `filtered_out`, `ranked_outside_top50`, and `selected_top50` from that same run only. Bind every row to those file hashes, the verified map hash, audit hash, and dev gold hash.

For a Query Evolution probe, validate `probe.lock.json` self-hash and source hashes, require the sealed `result.json` to report `capture_replay_match="matched"`, hash `outcomes.jsonl`, reconstruct the frozen source through `reconstruct_frozen_baseline` and `merge_probe_results`, then call `evaluate_probe` with the verified map. Probe rows use a distinct `source_kind="query_evolution_probe"` and never pass through `validate_run_directory`.

`LegacyRunBinding(run_path: Path, evidence_path: Path)` is allowed only for the historical title run whose current verifier returns `artifact_invalid`. Validate the exact `business-results.jsonl` and `executions.jsonl` hashes listed in `docs/evidence/title-retention-offline-2026-08-09.json`, reconstruct selected predictions from those bound business results, and reject any missing or mismatched binding with `legacy evidence binding mismatch`. This adapter never treats the legacy run as formally valid and labels it `source_kind="legacy_hash_bound_run"`.

The CLI exposes repeatable `--formal-run PATH` and `--query-evolution-probe PATH`, plus paired `--legacy-run PATH --legacy-evidence PATH`. It rejects an unpaired legacy argument.

- [ ] **Step 4: Add aggregate-only publication and historical separation tests**

```python
def test_rescore_report_never_exposes_private_or_query_fields(tmp_path: Path) -> None:
    publish_rescore_report(REPORT, out_json=tmp_path / "report.json", out_md=tmp_path / "report.md")
    combined = (tmp_path / "report.json").read_text() + (tmp_path / "report.md").read_text()

    assert "arxiv:" not in combined
    assert "doi:" not in combined
    assert "query_id" not in combined
    assert "query" not in combined
```

The report must label V2 source runs `historical_semantics_unverified` and the new recomputation `semantic_audit_passed`; it must not overwrite source-run files or old evidence.

Implement the publisher with this exact boundary:

```python
def publish_rescore_report(
    report: SemanticRescoreReport,
    *,
    out_json: Path,
    out_md: Path,
) -> None:
    """Write aggregate-only canonical JSON and Markdown through atomic siblings."""
```

- [ ] **Step 5: Run focused tests and commit**

Run:

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest tests/evaluation/test_identifier_semantics.py tests/scripts/test_rescore_identifier_semantics.py -q
```

Expected: all selected tests PASS.

Commit:

```powershell
git add src/paper_search/evaluation/identifier_semantics.py tests/evaluation/test_identifier_semantics.py scripts/rescore_identifier_semantics.py tests/scripts/test_rescore_identifier_semantics.py
git commit -m "feat: rescore sealed runs with verified identities"
```

---

### Task 6: Verification, authorized recovery run, and status update

**Files:**
- Modify after real aggregate results exist: `HANDOFF.md`
- Modify after real aggregate results exist: `docs/retrieval-roadmap.md`
- Create after real aggregate results exist: `docs/evidence/identifier-map-semantic-audit-2026-08-10.json`
- Create after real aggregate results exist: `docs/identifier-map-semantic-rescore-2026-08-10.md`
- Create after real aggregate results exist: `docs/evidence/identifier-map-semantic-rescore-2026-08-10.json`
- Private ignored outputs: `data/annotation_work/identifier_semantics/`

**Interfaces:**
- Consumes: completed Tasks 1-5, explicit authorization for one identity metadata capture, and existing sealed runs.
- Produces: a passed or failed dev semantic audit, offline recomputed baselines, and updated project status.
- Stop boundary: this task never runs readiness, live evaluation capture, or validation.

- [ ] **Step 1: Run all offline quality checks before any real identity request**

Run:

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest -m "not online" -q
& 'D:\AI Projects\Projects\.venv\Scripts\ruff.exe' check src scripts tests
& 'D:\AI Projects\Projects\.venv\Scripts\mypy.exe' src scripts
git diff --check
```

Expected: pytest has zero failures; Ruff and mypy exit 0; `git diff --check` prints nothing. If the user-owned `deliverables/` remains outside Ruff's command, record that exclusion exactly and do not edit it.

- [ ] **Step 2: Build the zero-network identity capture lock**

Run:

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' scripts/capture_dev_identifier_identity.py preflight --gold data/dev/gold.jsonl --ledger data/budget_ledger.sqlite3 --out-lock data/annotation_work/identifier_semantics/capture.lock.json
```

Expected: status `preflight_complete`, scope `dev`, 141 unique IDs, fixed request caps, and zero new ledger receipts.

- [ ] **Step 3: Pause for explicit authorization, then perform exactly one bounded identity capture**

Do not execute this step until the user explicitly authorizes the identity metadata capture and temporary use of the required `.env` keys. After authorization, inject only `OPENALEX_API_KEY*` and `SEMANTIC_SCHOLAR_API_KEY` into that process, then run:

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' scripts/capture_dev_identifier_identity.py run --lock data/annotation_work/identifier_semantics/capture.lock.json --gold data/dev/gold.jsonl --ledger data/budget_ledger.sqlite3 --snapshot-root data/annotation_work/identifier_semantics/snapshots --out-private data/annotation_work/identifier_semantics/identity-evidence.json --allow-network
```

Expected: every locked request has a terminal receipt; snapshots are sealed; stdout contains only aggregate counts and hashes. Do not retry a failed locked capture without diagnosing its terminal reason and creating a new lock.

- [ ] **Step 4: Rebuild the map and publish the aggregate audit offline**

Run:

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' scripts/rebuild_dev_identifier_map.py --gold data/dev/gold.jsonl --evidence data/annotation_work/identifier_semantics/identity-evidence.json --baseline-predictions runs/dev-20260810T104256Z-d9e89476d484/predictions.jsonl --out-map data/annotation_work/identifier_semantics/dev-identifier-map.semantic-v1.json --out-private-audit data/annotation_work/identifier_semantics/relation-audit.json --out-public-audit docs/evidence/identifier-map-semantic-audit-2026-08-10.json
```

Expected for promotion: `status=passed`, 141 dev groups complete, zero mismatch, zero unresolved, and direct arXiv hit count 12. Otherwise stop, publish only the failed aggregate audit, and do not build V3 freeze or report a formal F1.

- [ ] **Step 5: Build the private dev V3 freeze only after the audit passes**

Use the current UTC timestamp and the fixed approved reference for this recovery:

```powershell
$semanticApprovedAt = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' scripts/build_dev_semantic_freeze.py --base-manifest data/manifest.json --dev-gold data/dev/gold.jsonl --id-map data/annotation_work/identifier_semantics/dev-identifier-map.semantic-v1.json --audit docs/evidence/identifier-map-semantic-audit-2026-08-10.json --private-evidence data/annotation_work/identifier_semantics/identity-evidence.json --out-manifest data/annotation_work/identifier_semantics/freeze-v3/manifest.json --out-approval data/annotation_work/identifier_semantics/freeze-v3/approval.json --approval-ref identifier-semantic-recovery-2026-08-10 --approved-at $semanticApprovedAt
```

Expected: V3 scope `dev`; old `data/manifest.json` hash unchanged; no validation file is opened.

- [ ] **Step 6: Rescore existing sealed evidence without network**

Verify the two formal source runs, then rescore the current baseline, the hash-bound legacy title run, and the sealed Prompt-v2 probe in one aggregate report:

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\paper-search.exe' verify-run runs/dev-20260810T104256Z-d9e89476d484
& 'D:\AI Projects\Projects\.venv\Scripts\paper-search.exe' verify-run runs/dev-20260809T061903Z-9bd861e90299
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' scripts/rescore_identifier_semantics.py --gold data/dev/gold.jsonl --id-map data/annotation_work/identifier_semantics/dev-identifier-map.semantic-v1.json --audit docs/evidence/identifier-map-semantic-audit-2026-08-10.json --formal-run runs/dev-20260810T104256Z-d9e89476d484 --formal-run runs/dev-20260809T061903Z-9bd861e90299 --legacy-run runs/dev-20260805T035209Z-7af4b103f6cc --legacy-evidence docs/evidence/title-retention-offline-2026-08-09.json --query-evolution-probe runs/_diag_query_evolution_query-evolution-prompt-v2-full-20260810 --out-json docs/evidence/identifier-map-semantic-rescore-2026-08-10.json --out-report docs/identifier-map-semantic-rescore-2026-08-10.md
```

Expected: both formal verify commands report `valid: true`; the title row is explicitly labeled legacy and accepted only through its evidence hashes; the Prompt-v2 row reports `capture_replay_match=matched`; at least the 12 direct hits score; no network or ledger mutation occurs.

- [ ] **Step 7: Update status documents from aggregate results only**

Update `HANDOFF.md` and `docs/retrieval-roadmap.md` to state:

- V2 capture/replay remains operationally valid but semantically superseded;
- the old `0.0038000670`, `134/134 available`, and old funnel are historical only;
- the new map/audit hashes and recomputed aggregate metrics;
- whether the trustworthy funnel selects retrieval expansion or Top-50 selection as the next single-variable experiment;
- no live capture or validation was run.

Do not include individual identifiers, queries, titles, authors, private paths beyond approved generic locations, or unresolved rows.

- [ ] **Step 8: Re-run final verification and commit only tracked implementation/public aggregate files**

Run:

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest -m "not online" -q
& 'D:\AI Projects\Projects\.venv\Scripts\ruff.exe' check src scripts tests
& 'D:\AI Projects\Projects\.venv\Scripts\mypy.exe' src scripts
git diff --check
git status --short
```

Expected: all checks pass; private map, identity evidence, snapshots, ledger, and `deliverables/` remain untracked/ignored and unstaged.

Commit public evidence and status updates only when the audit passed and rescoring completed:

```powershell
git add HANDOFF.md docs/retrieval-roadmap.md docs/evidence/identifier-map-semantic-audit-2026-08-10.json docs/identifier-map-semantic-rescore-2026-08-10.md docs/evidence/identifier-map-semantic-rescore-2026-08-10.json
git commit -m "docs: record verified identifier baseline"
```

If the audit failed, commit only the aggregate failure evidence and status wording; do not claim a restored baseline.

---

## Completion Boundary

This plan is complete when the implementation and offline checks pass, the dev semantic audit has an honest terminal result, and existing sealed evidence has either been credibly rescored or explicitly blocked by unresolved identity evidence. Algorithm improvement begins in a separate design cycle selected from the corrected funnel. A candidate lock, readiness, live capture, replay, compare, and validation are all outside this plan.
