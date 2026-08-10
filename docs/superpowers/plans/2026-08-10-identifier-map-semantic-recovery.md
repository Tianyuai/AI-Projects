# Identifier Map Semantic Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore a semantically trustworthy dev-only identifier map and recompute the existing sealed baselines without changing retrieval behavior or current lock schemas.

**Architecture:** A pure offline semantic core classifies exact arXiv/provider relationships as `verified`, `semantic_mismatch`, or `unresolved`. A separately authorized, bounded collector seals two-stage Semantic Scholar and exact OpenAlex identity evidence; an offline builder then emits a private dev map and aggregate audit. A final offline rescorer verifies each historical source independently and reconstructs the trustworthy funnel.

**Tech Stack:** Python 3.11, Pydantic v2, httpx, SQLite budget ledger, canonical JSON/JSONL, pytest, Ruff, mypy.

## Global Constraints

- Use TDD for every behavior change; focused tests must fail before implementation and pass afterward.
- Use only dependencies already present in `pyproject.toml`.
- Do not change production retrieval, ranking, filtering, prompts, optional modules, `integrated-lock-v1`, `FrozenDataBinding`, freeze schemas, Gate 0, runner, validator, or replay contracts.
- Do not modify `data/manifest.json`, the existing identifier map, or historical runs, snapshots, ledgers, and evidence.
- Scope every new input and output to `dev`; never open, infer, or score validation data.
- Keep per-paper evidence, the rebuilt map, relation audit, and unresolved rows under ignored `data/annotation_work/identifier_semantics/`.
- Public outputs may contain aggregate counts and hashes only. Recursively reject exact JSON keys `query_id` and `query_text`; scan string values and Markdown field boundaries for injected query sentinels and DOI/arXiv/OpenAlex identifier patterns.
- Tasks 1, 2, Task 3 Steps 1-4, and Task 4 are zero-network. Task 3 Step 5 may use the network only after explicit authorization of one exact lock hash, its endpoints, caps, credential names, and single run.
- Semantic Scholar capture is limited to two logical batches: locked `ARXIV:<id>` inputs followed by only the DOI values discovered and sealed from batch 1. Each batch permits one retry, so the total S2 HTTP-attempt cap is 4; no third logical batch is allowed. OpenAlex uses `GET /works` with an exact locked-ID filter and accepts exactly one response whose work ID matches the lock.
- A semantic audit consumes only dev gold, the candidate map, and private identity evidence. It never consumes historical predictions.
- The 12 direct same-arXiv hits are a rescoring acceptance check, not a map-audit condition.
- Do not build a candidate lock, run readiness, run live capture, or access validation in this plan.
- Preserve existing user changes in `HANDOFF.md`, `docs/retrieval-roadmap.md`, `data/budget_ledger.sqlite3`, and `deliverables/`. Task 5 may merge aggregate conclusions into the first two files without replacing unrelated content; the latter two remain unstaged.

---

### Task 1: Pure semantic contract and aggregate audit

**Files:**
- Create: `src/paper_search/evaluation/identifier_semantics.py`
- Modify: `src/paper_search/evaluation/dataset.py`
- Create: `tests/evaluation/test_identifier_semantics.py`
- Modify: `tests/evaluation/test_dataset.py`

**Interfaces:**
- Consumes: exact dev map bytes, dev gold bytes, and private identity-evidence bytes.
- Produces: `IdentifierMap.resolved_pairs()`, `arxiv_anchor()`, `classify_relation()`, and `audit_identifier_map_semantics()`.
- Public contract: `identifier-map-semantic-audit-v1`; no prediction hash or direct-hit field.

- [ ] **Step 1: Add a failing read-only map-introspection test**

```python
def test_identifier_map_exposes_sorted_resolved_pairs() -> None:
    identifier_map = IdentifierMap.from_bytes(
        b'{"arxiv:2501.00002":"doi:10.48550/arxiv.2501.00002",'
        b'"doi:10.1000/example":"doi:10.48550/arxiv.2501.00002"}\n'
    )

    assert identifier_map.resolved_pairs() == (
        ("arxiv:2501.00002", "doi:10.48550/arxiv.2501.00002"),
        ("doi:10.1000/example", "doi:10.48550/arxiv.2501.00002"),
    )
```

Run:

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest tests/evaluation/test_dataset.py -k resolved_pairs -q
```

Expected: FAIL because `IdentifierMap.resolved_pairs` is absent.

- [ ] **Step 2: Add the minimal sorted accessor and make the focused test pass**

```python
def resolved_pairs(self) -> tuple[tuple[str, str], ...]:
    """Return normalized resolved pairs for private in-process auditing."""
    return tuple(sorted(self._resolved.items()))
```

Run the Step 1 command again. Expected: PASS.

- [ ] **Step 3: Add failing semantic classification tests**

```python
def test_same_datacite_arxiv_id_verifies_and_different_id_mismatches() -> None:
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


def test_s2_requires_both_exact_lookups_to_resolve_to_one_paper() -> None:
    complete = IdentityObservation(
        arxiv_id="arxiv:2501.10120",
        alias="doi:10.1000/example",
        semantic_scholar_arxiv_paper_id="S2-A",
        semantic_scholar_doi_paper_id="S2-A",
        semantic_scholar_arxiv_external_ids={
            "ArXiv": "2501.10120", "DOI": "10.1000/example"
        },
        semantic_scholar_doi_external_ids={
            "ArXiv": "2501.10120", "DOI": "10.1000/example"
        },
        semantic_scholar_arxiv_complete=True,
        semantic_scholar_doi_complete=True,
        openalex_complete=False,
        openalex_arxiv_ids=[],
        snapshot_sha256s=["sha256:" + "a" * 64, "sha256:" + "b" * 64],
    )
    missing_doi_side = complete.model_copy(
        update={"semantic_scholar_doi_paper_id": None, "semantic_scholar_doi_complete": False}
    )

    assert classify_relation(alias=complete.alias, arxiv_id=complete.arxiv_id, observation=complete).state == "verified"
    assert classify_relation(alias=missing_doi_side.alias, arxiv_id=missing_doi_side.arxiv_id, observation=missing_doi_side).state == "unresolved"


def test_s2_same_paper_without_stage_one_target_doi_is_unresolved() -> None:
    observation = _observation(
        s2_arxiv="S2-A",
        s2_doi="S2-A",
        arxiv_external_ids={"ArXiv": "2501.10120"},
        doi_external_ids={"ArXiv": "2501.10120", "DOI": "10.1000/example"},
    )
    assert classify_relation(
        alias=observation.alias,
        arxiv_id=observation.arxiv_id,
        observation=observation,
    ).state == "unresolved"


def test_s2_disagreement_alone_is_unresolved() -> None:
    observation = _observation(s2_arxiv="S2-A", s2_doi="S2-B")
    assert classify_relation(alias=observation.alias, arxiv_id=observation.arxiv_id, observation=observation).state == "unresolved"
```

Run:

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest tests/evaluation/test_identifier_semantics.py -q
```

Expected: collection FAIL because the module is absent.

- [ ] **Step 4: Implement the strict models and exact classifier**

Use these public signatures and fields:

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
    semantic_scholar_arxiv_paper_id: NonEmptyStr | None = None
    semantic_scholar_doi_paper_id: NonEmptyStr | None = None
    semantic_scholar_arxiv_external_ids: dict[NonEmptyStr, NonEmptyStr] = Field(default_factory=dict)
    semantic_scholar_doi_external_ids: dict[NonEmptyStr, NonEmptyStr] = Field(default_factory=dict)
    semantic_scholar_arxiv_complete: bool
    semantic_scholar_doi_complete: bool
    openalex_complete: bool
    openalex_arxiv_ids: list[NonEmptyStr] = Field(default_factory=list)
    snapshot_sha256s: list[Sha256] = Field(min_length=1)


class RelationAudit(DomainModel):
    arxiv_id: NonEmptyStr
    alias: NonEmptyStr
    terminal: NonEmptyStr
    state: SemanticState
    proof_kind: ProofKind | None
    reason_code: NonEmptyStr


class IdentifierMapSemanticAudit(DomainModel):
    schema_version: Literal["identifier-map-semantic-audit-v1"]
    scope: Literal["dev"]
    status: Literal["passed", "failed"]
    input_hashes: dict[NonEmptyStr, Sha256]
    gold_group_count: NonNegativeInt
    relation_count: NonNegativeInt
    state_counts: dict[NonEmptyStr, NonNegativeInt]
    proof_counts: dict[NonEmptyStr, NonNegativeInt]
    reason_counts: dict[NonEmptyStr, NonNegativeInt]


@dataclass(frozen=True)
class SemanticAuditBundle:
    report: IdentifierMapSemanticAudit
    private_relations: tuple[RelationAudit, ...]


def arxiv_anchor(arxiv_id: str) -> str:
    normalized = normalize_paper_id(arxiv_id, kind="arxiv")
    return f"doi:10.48550/arxiv.{normalized.removeprefix('arxiv:')}"


def classify_relation(
    *, alias: str, arxiv_id: str, observation: IdentityObservation | None
) -> RelationAudit:
    """Apply only exact DataCite, two-sided S2, and OpenAlex proof rules."""


def audit_identifier_map_semantics(
    *, map_bytes: bytes, gold_bytes: bytes, evidence_bytes: bytes, snapshot_root: Path
) -> SemanticAuditBundle:
    """Rebuild observations from sealed snapshots and audit offline."""


def assert_public_json_safe(content: bytes) -> None:
    """Reject exact private keys and identifier/query values, not aggregate names."""


def assert_public_markdown_safe(content: str) -> None:
    """Reject private field tokens and identifier/query values at field boundaries."""
```

Classification order is fixed: normalize; verify same-ID DataCite; reject different-ID DataCite;
verify equal non-empty S2 IDs only when both lookup-complete flags are true, the arXiv-side external
IDs contain the candidate DOI, and normalized external IDs do not conflict; verify exact OpenAlex
arXiv membership; reject explicit different OpenAlex membership only when OpenAlex completed;
otherwise return `unresolved`. Exceptions and reason codes must not contain identifier values.

Add this external-ID conflict test:

```python
def test_equal_s2_paper_id_with_conflicting_external_ids_is_not_proof() -> None:
    observation = _observation(
        s2_arxiv="S2-A",
        s2_doi="S2-A",
        arxiv_external_ids={"ArXiv": "2501.10120"},
        doi_external_ids={"ArXiv": "2409.00001"},
    )

    assert classify_relation(
        alias=observation.alias,
        arxiv_id=observation.arxiv_id,
        observation=observation,
    ).state == "unresolved"
```

- [ ] **Step 5: Add audit-independence and privacy tests**

```python
def test_semantic_audit_has_no_prediction_dependency() -> None:
    signature = inspect.signature(audit_identifier_map_semantics)
    assert tuple(signature.parameters) == ("map_bytes", "gold_bytes", "evidence_bytes", "snapshot_root")
    assert "predictions" not in IdentifierMapSemanticAudit.model_fields


def test_public_audit_contains_only_aggregate_safe_data() -> None:
    bundle = audit_identifier_map_semantics(
        map_bytes=DEV_MAP_BYTES,
        gold_bytes=DEV_GOLD_BYTES,
        evidence_bytes=PRIVATE_EVIDENCE_BYTES,
        snapshot_root=SEALED_SNAPSHOT_ROOT,
    )
    serialized = bundle.report.model_dump_json().encode("utf-8")

    assert bundle.report.input_hashes.keys() == {
        "map", "dev_gold", "identity_evidence", "snapshot_manifest"
    }
    assert_public_json_safe(serialized)


def test_privacy_scan_allows_aggregate_field_names() -> None:
    assert_public_json_safe(
        b'{"query_count":60,"query_identity_count":141,"doi_count":128,'
        b'"arxiv_count":141,"openalex_request_count":6}'
    )


def test_audit_rejects_tampered_raw_snapshot(tmp_path: Path) -> None:
    root = copy_snapshot_fixture(tmp_path)
    next((root / "responses").rglob("*.bin")).write_bytes(b"tampered")

    with pytest.raises(ValueError, match="identity snapshot is invalid"):
        audit_identifier_map_semantics(
            map_bytes=DEV_MAP_BYTES,
            gold_bytes=DEV_GOLD_BYTES,
            evidence_bytes=PRIVATE_EVIDENCE_BYTES,
            snapshot_root=root,
        )
```

Run:

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest tests/evaluation/test_dataset.py tests/evaluation/test_identifier_semantics.py -q
```

Expected: all selected tests PASS.

- [ ] **Step 6: Commit Task 1**

```powershell
git add src/paper_search/evaluation/dataset.py src/paper_search/evaluation/identifier_semantics.py tests/evaluation/test_dataset.py tests/evaluation/test_identifier_semantics.py
git commit -m "feat: enforce identifier semantic contract"
```

---

### Task 2: Bounded two-stage identity capture

**Files:**
- Create: `scripts/capture_dev_identifier_identity.py`
- Create: `tests/scripts/test_capture_dev_identifier_identity.py`

**Interfaces:**
- Offline inventory step consumes only dev gold and the current candidate map; historical predictions are excluded.
- Capture step consumes only: the resulting normalized identifier inventory, the project budget ledger, and explicit `--allow-network`; it never reads queries, labels, ranks, or run directories.
- Produces privately: a preflight lock, batch-1 snapshots, a derived batch-2 DOI lock, batch-2 snapshots, OpenAlex snapshots, and `identity-evidence.json`.
- Network boundary: two logical Semantic Scholar batches, at most four S2 HTTP attempts including retries, and the preflight-locked OpenAlex attempt maximum.

- [ ] **Step 1: Add failing lock and sequencing tests**

```python
def test_preflight_locks_dev_inputs_without_network_or_ledger_mutation(tmp_path: Path) -> None:
    before = ledger_checkpoint(LEDGER_PATH)
    lock = build_capture_lock(
        inventory_path=IDENTIFIER_INVENTORY_PATH,
        ledger_path=LEDGER_PATH,
    )

    assert lock.schema_version == "identifier-identity-capture-lock-v2"
    assert lock.scope == "dev"
    assert lock.semantic_scholar_batch_max == 2
    assert lock.semantic_scholar_arxiv_ids == sorted(set(lock.semantic_scholar_arxiv_ids))
    assert ledger_checkpoint(LEDGER_PATH) == before


def test_capture_uses_arxiv_batch_then_only_discovered_doi_batch() -> None:
    transport = RecordingTransport(
        arxiv_response=S2_ARXIV_BATCH,
        doi_response=S2_DOI_BATCH,
        openalex_responses=OPENALEX_RESPONSES,
    )
    result = capture_identity(LOCK, RUNTIME.with_transport(transport))

    assert transport.semantic_scholar_batches == [
        ["ARXIV:2501.00001", "ARXIV:2501.00002"],
        ["DOI:10.1000/a", "DOI:10.1000/b"],
    ]
    assert result.derived_doi_lock.ids == ["DOI:10.1000/a", "DOI:10.1000/b"]
    assert result.semantic_scholar_batch_count == 2


def test_capture_never_queries_an_unsealed_or_hand_added_doi() -> None:
    runtime = RUNTIME.with_manual_doi_addition("DOI:10.1000/not-discovered")
    with pytest.raises(ValueError, match="derived DOI lock mismatch"):
        capture_identity(LOCK, runtime)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("semantic_scholar_base_url", "https://example.invalid"),
        ("openalex_endpoint_template", "/search"),
        ("semantic_scholar_api_key_env", "OTHER_KEY"),
        ("output_root", "data/annotation_work/other"),
    ],
)
def test_capture_rejects_runtime_scope_outside_locked_authorization(
    field: str, value: str
) -> None:
    with pytest.raises(ValueError, match="identity capture authorization mismatch"):
        capture_identity(LOCK, RUNTIME.model_copy(update={field: value}))
```

Run:

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest tests/scripts/test_capture_dev_identifier_identity.py -q
```

Expected: collection FAIL because the script is absent.

- [ ] **Step 2: Implement lock, derived lock, and private evidence models**

```python
class IdentityCaptureLock(DomainModel):
    schema_version: Literal["identifier-identity-capture-lock-v2"]
    scope: Literal["dev"]
    input_hashes: dict[NonEmptyStr, Sha256]
    semantic_scholar_arxiv_ids: list[NonEmptyStr]
    semantic_scholar_batch_max: Literal[2] = 2
    semantic_scholar_http_attempt_max: Literal[4] = 4
    semantic_scholar_base_url: Literal["https://api.semanticscholar.org"]
    semantic_scholar_endpoint: Literal["/graph/v1/paper/batch"]
    semantic_scholar_api_key_env: Literal["SEMANTIC_SCHOLAR_API_KEY"]
    openalex_exact_ids: list[NonEmptyStr]
    openalex_request_max: NonNegativeInt
    openalex_base_url: Literal["https://api.openalex.org"]
    openalex_endpoint_template: Literal["/works"]
    openalex_api_key_env: Literal["OPENALEX_API_KEY"]
    output_root: SafeRelativePath
    retry_max: Literal[1] = 1
    ledger_checkpoint_sha256: Sha256


class IdentifierInventory(DomainModel):
    schema_version: Literal["identifier-identity-inventory-v1"]
    scope: Literal["dev"]
    source_hashes: dict[NonEmptyStr, Sha256]
    arxiv_ids: list[NonEmptyStr]
    candidate_aliases: list[NonEmptyStr]


class DerivedDoiLock(DomainModel):
    schema_version: Literal["identifier-identity-derived-doi-lock-v1"]
    parent_lock_sha256: Sha256
    arxiv_batch_snapshot_sha256: Sha256
    ids: list[NonEmptyStr]


class IdentityEvidenceRef(DomainModel):
    arxiv_id: NonEmptyStr
    alias: NonEmptyStr
    semantic_scholar_arxiv_entry_id: NonEmptyStr | None = None
    semantic_scholar_doi_entry_id: NonEmptyStr | None = None
    openalex_entry_ids: list[NonEmptyStr] = Field(default_factory=list)


class IdentityCaptureResult(DomainModel):
    schema_version: Literal["identifier-identity-evidence-v1"]
    scope: Literal["dev"]
    capture_lock_sha256: Sha256
    derived_doi_lock: DerivedDoiLock
    semantic_scholar_batch_count: NonNegativeInt
    semantic_scholar_http_attempt_count: NonNegativeInt
    openalex_request_count: NonNegativeInt
    snapshot_manifest_sha256: Sha256
    evidence_refs: list[IdentityEvidenceRef]
```

The offline `inventory` subcommand verifies dev gold and the current map hashes, extracts and normalizes
their identifiers, and writes only `IdentifierInventory`; it never opens predictions, query IDs, query
text, labels, ranks, or titles.
`IdentityEvidenceRef` contains only the arXiv/alias relationship and immutable snapshot entry references;
it does not contain trusted derived observations. `build_capture_lock()` reads only the inventory, sorts and deduplicates identifiers, fixes all caps,
and writes canonical JSON atomically with exclusive-create semantics. `capture_identity()`
must verify the lock and ledger checkpoint before reserving requests. It seals the ARXIV batch before
deriving the DOI list, atomically writes `DerivedDoiLock`, then performs the DOI batch. Zero discovered
DOIs is valid and skips batch 2; it never produces an empty network request. Provider clients,
credential lookup, and output paths are constructed from the locked constants; runtime overrides are
rejected before ledger reservation or network access.

- [ ] **Step 3: Implement exact adapters and fail-closed accounting**

Use Semantic Scholar `POST /graph/v1/paper/batch` with only the fields needed to compare `paperId`
and `externalIds`; preserve both sides' external IDs in the sealed raw responses. Use OpenAlex
`GET /works` with an exact locked-ID filter and `per_page=1`; require exactly one result and an exact
work-ID match, and do not use title search. This private lock is v2; reject v1 rather than migrating it. Reuse
`SQLiteBudgetLedger` and the dependency snapshot writer. Every reserved operation must end settled
or failed, even on HTTP, decode, or sealing errors. A retry consumes the fixed `retry_max=1` allowance;
two logical batches may therefore make at most four HTTP attempts, and there is no third stage or
ad-hoc rerun.

Add tests for HTTP failure, malformed JSON, missing S2 side, OpenAlex mismatch, request-cap overflow,
snapshot tampering, ledger-checkpoint drift, and value-free errors. Run:

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest tests/scripts/test_capture_dev_identifier_identity.py tests/unit/test_budget_ledger.py tests/unit/test_dependency_snapshot.py -q
```

Expected: all selected tests PASS with mocked transports only.

- [ ] **Step 4: Commit Task 2 without performing real capture**

```powershell
git add scripts/capture_dev_identifier_identity.py tests/scripts/test_capture_dev_identifier_identity.py
git commit -m "feat: add bounded identifier identity capture"
```

---

### Task 3: Verified dev map rebuild and authorized recovery checkpoint

**Files:**
- Create: `scripts/rebuild_dev_identifier_map.py`
- Create: `tests/scripts/test_rebuild_dev_identifier_map.py`
- Modify: `src/paper_search/evaluation/identifier_semantics.py`
- Modify: `tests/evaluation/test_identifier_semantics.py`
- Private outputs: `data/annotation_work/identifier_semantics/`
- Public output after a real audit: `docs/evidence/identifier-map-semantic-audit-2026-08-10.json`

**Interfaces:**
- Consumes: dev gold and sealed `identifier-identity-evidence-v1` only.
- Produces: a private same-arXiv-anchored map, private per-relation audit, and aggregate public audit.
- Stop boundary: any mismatch or unresolved gold group prevents a passed map and prevents rescoring.

- [ ] **Step 1: Add failing deterministic rebuild tests**

```python
def test_rebuild_anchors_each_verified_group_to_its_own_arxiv_id() -> None:
    result = rebuild_dev_map(
        gold_bytes=GOLD_BYTES,
        evidence_bytes=VERIFIED_EVIDENCE_BYTES,
        snapshot_root=VERIFIED_SNAPSHOT_ROOT,
    )

    assert result.audit.status == "passed"
    assert result.map_payload["arxiv:2501.00001"] == "doi:10.48550/arxiv.2501.00001"
    assert result.map_payload["doi:10.1000/a"] == "doi:10.48550/arxiv.2501.00001"


def test_unresolved_group_stops_without_a_passed_map() -> None:
    with pytest.raises(SemanticAuditFailure, match="identifier semantic audit failed") as caught:
        rebuild_dev_map(
            gold_bytes=GOLD_BYTES,
            evidence_bytes=UNRESOLVED_EVIDENCE_BYTES,
            snapshot_root=UNRESOLVED_SNAPSHOT_ROOT,
        )

    assert caught.value.public_audit.status == "failed"
    assert caught.value.public_audit.state_counts["unresolved"] == 1


def test_rebuild_does_not_accept_predictions_input() -> None:
    assert tuple(inspect.signature(rebuild_dev_map).parameters) == (
        "gold_bytes", "evidence_bytes", "snapshot_root"
    )
```

Run:

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest tests/scripts/test_rebuild_dev_identifier_map.py -q
```

Expected: collection FAIL because the script is absent.

- [ ] **Step 2: Implement the private rebuild and aggregate publisher**

```python
@dataclass(frozen=True)
class RebuiltDevMap:
    map_payload: dict[str, str]
    private_relations: tuple[RelationAudit, ...]
    audit: IdentifierMapSemanticAudit


def rebuild_dev_map(
    *, gold_bytes: bytes, evidence_bytes: bytes, snapshot_root: Path
) -> RebuiltDevMap:
    """Rebuild observations from sealed snapshots, then build the dev map."""


def publish_semantic_audit(
    audit: IdentifierMapSemanticAudit, *, output_path: Path
) -> None:
    """Atomically write aggregate canonical JSON after privacy validation."""
```

The builder first verifies the snapshot manifest and every referenced raw response, re-decodes
`paperId`, `externalIds`, and OpenAlex locations, and then calls the pure classifier. It creates the
same-ID DataCite anchor for every dev gold arXiv ID; adds a DOI/OpenAlex/S2
alias only after exact verification; omits mismatches; keeps unresolved rows private; and returns a
failed aggregate audit if any gold group remains unresolved. Serialization order is deterministic.
The CLI accepts exactly `--gold`, `--evidence`, `--snapshot-root`, `--out-map`,
`--out-private-audit`, and `--out-public-audit`; it has no predictions argument.

Privacy validation parses JSON and recursively rejects exact keys `query_id` and `query_text`; it scans
string values for an injected test sentinel and compiled DOI/arXiv/OpenAlex value patterns. Add tests
that explicitly allow `query_count`, `query_identity_count`, `doi_count`, `arxiv_count`, and
`openalex_request_count`.

- [ ] **Step 3: Run the offline implementation gate and commit**

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest tests/evaluation/test_identifier_semantics.py tests/scripts/test_rebuild_dev_identifier_map.py -q
& 'D:\AI Projects\Projects\.venv\Scripts\ruff.exe' check src/paper_search/evaluation/identifier_semantics.py scripts/capture_dev_identifier_identity.py scripts/rebuild_dev_identifier_map.py tests/evaluation/test_identifier_semantics.py tests/scripts/test_capture_dev_identifier_identity.py tests/scripts/test_rebuild_dev_identifier_map.py
& 'D:\AI Projects\Projects\.venv\Scripts\mypy.exe' src/paper_search/evaluation/identifier_semantics.py scripts/capture_dev_identifier_identity.py scripts/rebuild_dev_identifier_map.py
```

Expected: all commands exit 0.

```powershell
git add src/paper_search/evaluation/identifier_semantics.py tests/evaluation/test_identifier_semantics.py scripts/rebuild_dev_identifier_map.py tests/scripts/test_rebuild_dev_identifier_map.py
git commit -m "feat: rebuild verified dev identifier map"
```

- [ ] **Step 4: Build a zero-network preflight lock, then pause for authorization**

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' scripts/capture_dev_identifier_identity.py inventory --gold data/dev/gold.jsonl --candidate-map data/identifier-map.json --out data/annotation_work/identifier_semantics/identifier-inventory.json
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' scripts/capture_dev_identifier_identity.py preflight --inventory data/annotation_work/identifier_semantics/identifier-inventory.json --ledger data/budget_ledger.sqlite3 --output-root data/annotation_work/identifier_semantics --out-lock data/annotation_work/identifier_semantics/capture.lock.json
```

Expected: scope `dev`, fixed hashes and caps, `semantic_scholar_batch_max=2`,
`semantic_scholar_http_attempt_max=4`, and no network or ledger mutation. Present only the exact lock
hash, aggregate counts, endpoints, request/attempt caps, retry cap, credential variable names, and the
single output root. Then obtain explicit authorization for that one lock and one run, including
temporary use of the required `.env` keys. Authorization does not cover a changed lock, retries beyond
the lock, live evaluation capture, or validation.

- [ ] **Step 5: After authorization, run one bounded capture and rebuild offline**

Load only the required OpenAlex/Semantic Scholar keys from the user-authorized `.env` into this one
process; never print values. Run:

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' scripts/capture_dev_identifier_identity.py run --lock data/annotation_work/identifier_semantics/capture.lock.json --ledger data/budget_ledger.sqlite3 --snapshot-root data/annotation_work/identifier_semantics/snapshots --out-private data/annotation_work/identifier_semantics/identity-evidence.json --allow-network
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' scripts/rebuild_dev_identifier_map.py --gold data/dev/gold.jsonl --evidence data/annotation_work/identifier_semantics/identity-evidence.json --snapshot-root data/annotation_work/identifier_semantics/snapshots --out-map data/annotation_work/identifier_semantics/dev-identifier-map.semantic-v1.json --out-private-audit data/annotation_work/identifier_semantics/relation-audit.json --out-public-audit docs/evidence/identifier-map-semantic-audit-2026-08-10.json
```

Promotion requires `status=passed`, complete dev coverage, `semantic_mismatch=0`, and `unresolved=0`.
On failure, stop here, retain details privately, publish only aggregate failure evidence, and do not
report a restored F1.

---

### Task 4: Sealed-run offline rescoring and trustworthy funnel

**Files:**
- Create: `scripts/rescore_identifier_semantics.py`
- Create: `tests/scripts/test_rescore_identifier_semantics.py`
- Modify: `src/paper_search/evaluation/identifier_semantics.py`
- Modify: `tests/evaluation/test_identifier_semantics.py`
- Public outputs after successful audit: `docs/evidence/identifier-map-semantic-rescore-2026-08-10.json`, `docs/identifier-map-semantic-rescore-2026-08-10.md`

**Interfaces:**
- Consumes: a passed audit, private verified map, dev gold, and independently bound sealed sources.
- Produces: one aggregate result row per source and a trustworthy pipeline funnel.
- Does not produce: snapshots, predictions, candidate locks, ledger entries, or network requests.

- [ ] **Step 1: Add failing source-binding and direct-hit tests**

```python
def test_formal_rescore_verifies_source_and_recovers_all_direct_hits() -> None:
    report = rescore_runs(
        gold_path=GOLD_PATH,
        map_path=VERIFIED_MAP_PATH,
        audit_path=PASSED_AUDIT_PATH,
        formal_run_paths=[CURRENT_BASELINE_PATH],
    )

    row = report.runs[0]
    assert row.source_kind == "formal_run"
    assert row.direct_arxiv_hit_count == 12
    assert row.true_positive_count >= 12
    assert row.pipeline_stages.total_gold_associations == 143


def test_rescore_rejects_cross_run_stage_inputs() -> None:
    with pytest.raises(ValueError, match="sealed run binding mismatch"):
        rescore_runs(
            gold_path=GOLD_PATH,
            map_path=VERIFIED_MAP_PATH,
            audit_path=PASSED_AUDIT_PATH,
            formal_run_paths=[RUN_WITH_SWAPPED_EXECUTIONS],
        )


def test_rescore_refuses_a_failed_or_prediction_bound_audit() -> None:
    with pytest.raises(ValueError, match="identifier semantic audit is invalid"):
        rescore_runs(
            gold_path=GOLD_PATH,
            map_path=VERIFIED_MAP_PATH,
            audit_path=PREDICTION_BOUND_AUDIT_PATH,
            formal_run_paths=[CURRENT_BASELINE_PATH],
        )
```

Run:

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest tests/scripts/test_rescore_identifier_semantics.py -q
```

Expected: collection FAIL because the script is absent.

- [ ] **Step 2: Implement strict aggregate report models and offline source loading**

```python
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
    formal_run_paths: Sequence[Path],
    legacy_title_binding: tuple[Path, Path] | None = None,
    query_evolution_probe_paths: Sequence[Path] = (),
) -> SemanticRescoreReport:
    """Verify, rescore, and attribute sealed dev sources without network access."""
```

For formal runs, call `validate_run_directory()` first and hash that run's own `run.json`,
`predictions.jsonl`, `executions.jsonl`, and `business-results.jsonl`. Recompute metrics through
`evaluate()` and derive every pipeline stage from the same source.

The legacy title run `runs/dev-20260805T035209Z-7af4b103f6cc` is accepted only through the exact
`business-results.jsonl` and `executions.jsonl` hashes in
`docs/evidence/title-retention-offline-2026-08-09.json`; label it `legacy_hash_bound_run`, never
formally valid. For the Prompt-v2 probe, verify its lock self-hash and source hashes, require sealed
`capture_replay_match="matched"`, then use `reconstruct_frozen_baseline()`, `merge_probe_results()`,
and `evaluate_probe()` with the verified map.

- [ ] **Step 3: Add aggregate publication privacy tests**

```python
def test_published_rescore_has_no_private_values(tmp_path: Path) -> None:
    publish_rescore_report(
        REPORT,
        out_json=tmp_path / "report.json",
        out_md=tmp_path / "report.md",
    )
    assert_public_json_safe((tmp_path / "report.json").read_bytes())
    assert_public_markdown_safe((tmp_path / "report.md").read_text())
```

Implement:

```python
def publish_rescore_report(
    report: SemanticRescoreReport, *, out_json: Path, out_md: Path
) -> None:
    """Atomically write aggregate JSON and Markdown after privacy scanning."""
```

The publisher must not overwrite historical source files. Run:

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest tests/evaluation/test_identifier_semantics.py tests/scripts/test_rescore_identifier_semantics.py -q
```

Expected: all selected tests PASS.

- [ ] **Step 4: Commit Task 4**

```powershell
git add src/paper_search/evaluation/identifier_semantics.py tests/evaluation/test_identifier_semantics.py scripts/rescore_identifier_semantics.py tests/scripts/test_rescore_identifier_semantics.py
git commit -m "feat: rescore sealed runs with verified identities"
```

- [ ] **Step 5: After a passed semantic audit, execute the three offline comparisons**

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\paper-search.exe' verify-run runs/dev-20260810T104256Z-d9e89476d484
& 'D:\AI Projects\Projects\.venv\Scripts\paper-search.exe' verify-run runs/dev-20260809T061903Z-9bd861e90299
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' scripts/rescore_identifier_semantics.py --gold data/dev/gold.jsonl --id-map data/annotation_work/identifier_semantics/dev-identifier-map.semantic-v1.json --audit docs/evidence/identifier-map-semantic-audit-2026-08-10.json --formal-run runs/dev-20260810T104256Z-d9e89476d484 --formal-run runs/dev-20260809T061903Z-9bd861e90299 --legacy-run runs/dev-20260805T035209Z-7af4b103f6cc --legacy-evidence docs/evidence/title-retention-offline-2026-08-09.json --query-evolution-probe runs/_diag_query_evolution_query-evolution-prompt-v2-full-20260810 --out-json docs/evidence/identifier-map-semantic-rescore-2026-08-10.json --out-report docs/identifier-map-semantic-rescore-2026-08-10.md
```

Expected: both formal verifications are valid; the legacy row is explicitly hash-bound; the probe
reports `capture_replay_match=matched`; all 12 direct hits are counted. If any condition fails, stop
without selecting the next experiment.

---

### Task 5: Full verification and project status

**Files:**
- Modify after aggregate results exist: `HANDOFF.md`
- Modify after aggregate results exist: `docs/retrieval-roadmap.md`
- Public evidence from Tasks 3-4: `docs/evidence/identifier-map-semantic-audit-2026-08-10.json`, `docs/evidence/identifier-map-semantic-rescore-2026-08-10.json`, `docs/identifier-map-semantic-rescore-2026-08-10.md`

**Interfaces:**
- Consumes: implementation, terminal semantic audit, and optional successful rescore report.
- Produces: verified code plus an honest aggregate project state.
- Stop boundary: no lock rebuild, readiness, live capture, replay, compare, or validation.

- [ ] **Step 1: Run the full offline quality gate**

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest -m "not online" -q
& 'D:\AI Projects\Projects\.venv\Scripts\ruff.exe' check src scripts tests
& 'D:\AI Projects\Projects\.venv\Scripts\mypy.exe' src scripts
git diff --check
```

Expected: pytest has zero failures; Ruff and mypy exit 0; `git diff --check` prints nothing.

- [ ] **Step 2: Update status documents without overwriting unrelated edits**

Merge only aggregate conclusions into `HANDOFF.md` and `docs/retrieval-roadmap.md`:

- V2 capture/replay remains operationally valid but its old semantic quality conclusion is superseded;
- `0.0038000670`, `134/134 available`, and the old funnel remain historical values only;
- record the new map, audit, and rescore hashes and aggregate metrics when available;
- if the audit failed, state the aggregate failure reason and do not publish a new F1;
- if rescoring passed, select exactly one next single-variable direction from the corrected funnel;
- state explicitly that no current lock schema, readiness, live capture, or validation was changed or run.

Do not include individual identifiers, queries, titles, authors, private paths, or unresolved rows.

- [ ] **Step 3: Re-run verification and commit only intended public files**

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest -m "not online" -q
& 'D:\AI Projects\Projects\.venv\Scripts\ruff.exe' check src scripts tests
& 'D:\AI Projects\Projects\.venv\Scripts\mypy.exe' src scripts
git diff --check
git status --short
```

Expected: all checks pass. `data/budget_ledger.sqlite3`, private identity artifacts, and `deliverables/`
remain unstaged. Stage only implementation, approved public aggregate evidence, and the two carefully
merged status documents. If the audit failed, commit the aggregate failure evidence and status only;
do not claim baseline recovery.

---

## Completion Boundary

This plan ends when the semantic implementation passes offline checks and the dev audit has an honest
terminal result; if the audit passes, it also requires all three sealed comparisons and recovery of the
12 direct hits. Algorithm improvement begins in a separate design cycle selected from the corrected
funnel.

`integrated-lock-v1` remains unchanged. Only after a trustworthy funnel selects a future formal live
experiment should a separate plan introduce an explicit `integrated-lock-v2` / freeze-v3 discriminated
union and bind the semantic evidence. Candidate-lock rebuild, readiness, live capture, replay, compare,
and validation are outside this plan.
