import inspect
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pytest

from paper_search.evaluation.identifier_semantics import (
    IdentifierMapSemanticAudit,
    IdentityObservation,
    assert_public_json_safe,
    audit_identifier_map_semantics,
    classify_relation,
)
from paper_search.storage.dependency_snapshot import (
    DependencyCaptureStore,
    DependencyRequestIdentity,
)


def _observation(
    *,
    s2_arxiv: str | None = "S2-A",
    s2_doi: str | None = "S2-A",
    arxiv_external_ids: dict[str, str] | None = None,
    doi_external_ids: dict[str, str] | None = None,
) -> IdentityObservation:
    return IdentityObservation(
        arxiv_id="arxiv:2501.10120",
        alias="doi:10.1000/example",
        semantic_scholar_arxiv_paper_id=s2_arxiv,
        semantic_scholar_doi_paper_id=s2_doi,
        semantic_scholar_arxiv_external_ids=arxiv_external_ids
        or {"ArXiv": "2501.10120", "DOI": "10.1000/example"},
        semantic_scholar_doi_external_ids=doi_external_ids
        or {"ArXiv": "2501.10120", "DOI": "10.1000/example"},
        semantic_scholar_arxiv_complete=True,
        semantic_scholar_doi_complete=True,
        openalex_complete=False,
        openalex_arxiv_ids=[],
        snapshot_sha256s=["sha256:" + "a" * 64, "sha256:" + "b" * 64],
    )


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
    complete = _observation()
    missing_doi_side = complete.model_copy(
        update={
            "semantic_scholar_doi_paper_id": None,
            "semantic_scholar_doi_complete": False,
        }
    )

    assert (
        classify_relation(
            alias=complete.alias,
            arxiv_id=complete.arxiv_id,
            observation=complete,
        ).state
        == "verified"
    )
    assert (
        classify_relation(
            alias=missing_doi_side.alias,
            arxiv_id=missing_doi_side.arxiv_id,
            observation=missing_doi_side,
        ).state
        == "unresolved"
    )


def test_s2_same_paper_without_stage_one_target_doi_is_unresolved() -> None:
    observation = _observation(
        arxiv_external_ids={"ArXiv": "2501.10120"},
        doi_external_ids={"ArXiv": "2501.10120", "DOI": "10.1000/example"},
    )

    assert (
        classify_relation(
            alias=observation.alias,
            arxiv_id=observation.arxiv_id,
            observation=observation,
        ).state
        == "unresolved"
    )


def test_s2_disagreement_alone_is_unresolved() -> None:
    observation = _observation(s2_arxiv="S2-A", s2_doi="S2-B")

    assert (
        classify_relation(
            alias=observation.alias,
            arxiv_id=observation.arxiv_id,
            observation=observation,
        ).state
        == "unresolved"
    )


def test_equal_s2_paper_id_with_conflicting_external_ids_is_not_proof() -> None:
    observation = _observation(
        arxiv_external_ids={"ArXiv": "2501.10120"},
        doi_external_ids={"ArXiv": "2409.00001"},
    )

    assert (
        classify_relation(
            alias=observation.alias,
            arxiv_id=observation.arxiv_id,
            observation=observation,
        ).state
        == "unresolved"
    )


def _audit_fixture(tmp_path: Path) -> tuple[bytes, bytes, bytes, Path]:
    snapshot_root = tmp_path / "snapshots"
    capture = DependencyCaptureStore(
        snapshot_root,
        clock=lambda: datetime(2026, 8, 10, tzinfo=UTC),
    )
    identity = DependencyRequestIdentity.from_canonical_request(
        dependency="openalex",
        operation="search",
        method="GET",
        endpoint="/works",
        model_or_adapter="openalex-identity-v1",
        canonical_request={"filter": "doi:10.1000/example"},
    )
    capture.stage_success(
        identity,
        response_bytes=b"{}",
        safe_headers={},
        captured_at=datetime(2026, 8, 10, tzinfo=UTC),
    )
    capture.seal()
    evidence = {
        "schema_version": "identifier-identity-evidence-v1",
        "scope": "dev",
        "capture_lock_sha256": "sha256:" + "c" * 64,
        "derived_doi_lock": {
            "schema_version": "identifier-identity-derived-doi-lock-v1",
            "parent_lock_sha256": "sha256:" + "c" * 64,
            "arxiv_batch_snapshot_sha256": "sha256:" + "d" * 64,
            "ids": [],
        },
        "semantic_scholar_batch_count": 0,
        "semantic_scholar_http_attempt_count": 0,
        "openalex_request_count": 1,
        "snapshot_manifest_sha256": capture.manifest_sha256,
        "evidence_refs": [],
    }
    return (
        b'{"arxiv:2501.10120":"doi:10.48550/arxiv.2501.10120"}\n',
        b'{"query_id":"q1","query":"example","relevant_paper_ids":["arxiv:2501.10120"]}\n',
        json.dumps(evidence, sort_keys=True).encode("utf-8"),
        snapshot_root,
    )


def test_semantic_audit_has_no_prediction_dependency() -> None:
    signature = inspect.signature(audit_identifier_map_semantics)
    assert tuple(signature.parameters) == (
        "map_bytes",
        "gold_bytes",
        "evidence_bytes",
        "snapshot_root",
    )
    assert "predictions" not in IdentifierMapSemanticAudit.model_fields


def test_public_audit_contains_only_aggregate_safe_data(tmp_path: Path) -> None:
    map_bytes, gold_bytes, evidence_bytes, snapshot_root = _audit_fixture(tmp_path)

    bundle = audit_identifier_map_semantics(
        map_bytes=map_bytes,
        gold_bytes=gold_bytes,
        evidence_bytes=evidence_bytes,
        snapshot_root=snapshot_root,
    )
    serialized = bundle.report.model_dump_json().encode("utf-8")

    assert bundle.report.input_hashes.keys() == {
        "map",
        "dev_gold",
        "identity_evidence",
        "snapshot_manifest",
    }
    assert_public_json_safe(serialized)


def test_audit_verifies_exact_openalex_location_from_sealed_snapshot(
    tmp_path: Path,
) -> None:
    snapshot_root = tmp_path / "snapshots"
    capture = DependencyCaptureStore(
        snapshot_root,
        clock=lambda: datetime(2026, 8, 10, tzinfo=UTC),
    )
    identity = DependencyRequestIdentity.from_canonical_request(
        dependency="openalex",
        operation="search",
        method="GET",
        endpoint="/works",
        model_or_adapter="openalex-identity-v1",
        canonical_request={"filter": "openalex:W1"},
    )
    ref = capture.stage_success(
        identity,
        response_bytes=(
            b'{"locations":[{"landing_page_url":'
            b'"https://arxiv.org/abs/2501.10120"}]}'
        ),
        safe_headers={},
        captured_at=datetime(2026, 8, 10, tzinfo=UTC),
    )
    capture.seal()
    evidence = {
        "schema_version": "identifier-identity-evidence-v1",
        "scope": "dev",
        "snapshot_manifest_sha256": capture.manifest_sha256,
        "evidence_refs": [
            {
                "arxiv_id": "arxiv:2501.10120",
                "alias": "openalex:W1",
                "openalex_entry_ids": [ref.entry_id],
            }
        ],
    }

    bundle = audit_identifier_map_semantics(
        map_bytes=(
            b'{"arxiv:2501.10120":"doi:10.48550/arxiv.2501.10120",'
            b'"openalex:W1":"doi:10.48550/arxiv.2501.10120"}\n'
        ),
        gold_bytes=(
            b'{"query_id":"q1","query":"example",'
            b'"relevant_paper_ids":["arxiv:2501.10120"]}\n'
        ),
        evidence_bytes=json.dumps(evidence, sort_keys=True).encode("utf-8"),
        snapshot_root=snapshot_root,
    )

    assert bundle.report.status == "passed"
    assert bundle.report.proof_counts == {"openalex_location_exact": 1, "arxiv_datacite_exact": 1}


def test_privacy_scan_allows_aggregate_field_names() -> None:
    assert_public_json_safe(
        b'{"query_count":60,"query_identity_count":141,"doi_count":128,'
        b'"arxiv_count":141,"openalex_request_count":6}'
    )


def test_audit_rejects_tampered_raw_snapshot(tmp_path: Path) -> None:
    map_bytes, gold_bytes, evidence_bytes, snapshot_root = _audit_fixture(tmp_path / "source")
    root = tmp_path / "tampered"
    shutil.copytree(snapshot_root, root)
    next((root / "responses").rglob("*.bin")).write_bytes(b"tampered")

    with pytest.raises(ValueError, match="identity snapshot is invalid"):
        audit_identifier_map_semantics(
            map_bytes=map_bytes,
            gold_bytes=gold_bytes,
            evidence_bytes=evidence_bytes,
            snapshot_root=root,
        )


def test_audit_rejects_dangling_private_snapshot_reference(tmp_path: Path) -> None:
    map_bytes, gold_bytes, evidence_bytes, snapshot_root = _audit_fixture(tmp_path)
    evidence = json.loads(evidence_bytes)
    evidence["evidence_refs"] = [
        {
            "arxiv_id": "arxiv:2501.10120",
            "alias": "doi:10.1000/example",
            "semantic_scholar_arxiv_entry_id": "missing-entry",
        }
    ]

    with pytest.raises(ValueError, match="identity snapshot is invalid"):
        audit_identifier_map_semantics(
            map_bytes=map_bytes,
            gold_bytes=gold_bytes,
            evidence_bytes=json.dumps(evidence, sort_keys=True).encode("utf-8"),
            snapshot_root=snapshot_root,
        )
