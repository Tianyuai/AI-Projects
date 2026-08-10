from __future__ import annotations

import inspect
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from paper_search.storage.dependency_snapshot import (
    DependencyCaptureStore,
    DependencyRequestIdentity,
)
from scripts.rebuild_dev_identifier_map import (
    SemanticAuditFailure,
    _parser,
    publish_semantic_audit,
    rebuild_dev_map,
)


GOLD_BYTES = (
    b'{"query_id":"q1","query":"private query sentinel",'
    b'"relevant_paper_ids":["arxiv:2501.00001"]}\n'
)


def _sealed_evidence(
    tmp_path: Path,
    *,
    arxiv_paper_id: str = "S2-A",
    doi_paper_id: str = "S2-A",
    arxiv_external_ids: dict[str, str] | None = None,
    doi_external_ids: dict[str, str] | None = None,
) -> tuple[bytes, Path]:
    snapshot_root = tmp_path / "snapshots"
    store = DependencyCaptureStore(
        snapshot_root,
        clock=lambda: datetime(2026, 8, 10, tzinfo=UTC),
    )
    arxiv_identity = DependencyRequestIdentity.from_canonical_request(
        dependency="semantic_scholar",
        operation="batch",
        method="POST",
        endpoint="/paper/batch",
        model_or_adapter="semantic-scholar-identity-arxiv-v1",
        canonical_request={
            "fields": "paperId,externalIds",
            "ids": ["ARXIV:2501.00001"],
        },
    )
    doi_identity = DependencyRequestIdentity.from_canonical_request(
        dependency="semantic_scholar",
        operation="batch",
        method="POST",
        endpoint="/paper/batch",
        model_or_adapter="semantic-scholar-identity-doi-v1",
        canonical_request={
            "fields": "paperId,externalIds",
            "ids": ["DOI:10.1000/a"],
        },
    )
    arxiv_ref = store.stage_success(
        arxiv_identity,
        response_bytes=json.dumps(
            [
                {
                    "paperId": arxiv_paper_id,
                    "externalIds": arxiv_external_ids
                    or {"ArXiv": "2501.00001", "DOI": "10.1000/a"},
                }
            ],
            separators=(",", ":"),
        ).encode("utf-8"),
        safe_headers={},
        captured_at=datetime(2026, 8, 10, tzinfo=UTC),
    )
    doi_ref = store.stage_success(
        doi_identity,
        response_bytes=json.dumps(
            [
                {
                    "paperId": doi_paper_id,
                    "externalIds": doi_external_ids or {"DOI": "10.1000/a"},
                }
            ],
            separators=(",", ":"),
        ).encode("utf-8"),
        safe_headers={},
        captured_at=datetime(2026, 8, 10, tzinfo=UTC),
    )
    store.seal()
    evidence = {
        "schema_version": "identifier-identity-evidence-v1",
        "scope": "dev",
        "snapshot_manifest_sha256": store.manifest_sha256,
        "evidence_refs": [
            {
                "arxiv_id": "arxiv:2501.00001",
                "alias": "doi:10.1000/a",
                "semantic_scholar_arxiv_entry_id": arxiv_ref.entry_id,
                "semantic_scholar_arxiv_item_index": 0,
                "semantic_scholar_doi_entry_id": doi_ref.entry_id,
                "semantic_scholar_doi_item_index": 0,
            }
        ],
    }
    return json.dumps(evidence, sort_keys=True).encode("utf-8"), snapshot_root


def test_rebuild_anchors_each_verified_group_to_its_own_arxiv_id(
    tmp_path: Path,
) -> None:
    evidence_bytes, snapshot_root = _sealed_evidence(tmp_path)

    result = rebuild_dev_map(
        gold_bytes=GOLD_BYTES,
        evidence_bytes=evidence_bytes,
        snapshot_root=snapshot_root,
    )

    assert result.audit.status == "passed"
    assert result.map_payload == {
        "arxiv:2501.00001": "doi:10.48550/arxiv.2501.00001",
        "doi:10.1000/a": "doi:10.48550/arxiv.2501.00001",
    }
    assert [relation.state for relation in result.private_relations] == [
        "verified",
        "verified",
    ]


def test_unresolved_group_stops_without_a_passed_map(tmp_path: Path) -> None:
    evidence_bytes, snapshot_root = _sealed_evidence(
        tmp_path,
        doi_paper_id="S2-B",
    )

    with pytest.raises(
        SemanticAuditFailure, match="identifier semantic audit failed"
    ) as caught:
        rebuild_dev_map(
            gold_bytes=GOLD_BYTES,
            evidence_bytes=evidence_bytes,
            snapshot_root=snapshot_root,
        )

    assert caught.value.public_audit.status == "failed"
    assert caught.value.public_audit.state_counts["unresolved"] == 1


def test_missing_provider_relation_stops_the_gold_group(tmp_path: Path) -> None:
    evidence_bytes, snapshot_root = _sealed_evidence(tmp_path)
    evidence = json.loads(evidence_bytes)
    evidence["evidence_refs"] = []

    with pytest.raises(SemanticAuditFailure) as caught:
        rebuild_dev_map(
            gold_bytes=GOLD_BYTES,
            evidence_bytes=json.dumps(evidence, sort_keys=True).encode("utf-8"),
            snapshot_root=snapshot_root,
        )

    assert caught.value.public_audit.status == "failed"
    assert caught.value.public_audit.reason_counts == {
        "arxiv_datacite_exact": 1,
        "provider_identity_missing": 1,
    }


def test_duplicate_evidence_relation_is_rejected(tmp_path: Path) -> None:
    evidence_bytes, snapshot_root = _sealed_evidence(tmp_path)
    evidence = json.loads(evidence_bytes)
    evidence["evidence_refs"].append(dict(evidence["evidence_refs"][0]))

    with pytest.raises(
        ValueError, match="identifier semantic audit inputs are invalid"
    ):
        rebuild_dev_map(
            gold_bytes=GOLD_BYTES,
            evidence_bytes=json.dumps(evidence, sort_keys=True).encode("utf-8"),
            snapshot_root=snapshot_root,
        )


def test_rebuild_does_not_accept_predictions_input() -> None:
    assert tuple(inspect.signature(rebuild_dev_map).parameters) == (
        "gold_bytes",
        "evidence_bytes",
        "snapshot_root",
    )


def test_public_audit_publication_is_deterministic_and_private_value_free(
    tmp_path: Path,
) -> None:
    evidence_bytes, snapshot_root = _sealed_evidence(tmp_path / "source")
    result = rebuild_dev_map(
        gold_bytes=GOLD_BYTES,
        evidence_bytes=evidence_bytes,
        snapshot_root=snapshot_root,
    )
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    publish_semantic_audit(result.audit, output_path=first)
    publish_semantic_audit(result.audit, output_path=second)

    assert first.read_bytes() == second.read_bytes()
    serialized = first.read_text(encoding="utf-8")
    assert "private query sentinel" not in serialized
    assert "arxiv:2501.00001" not in serialized
    assert "doi:10.1000/a" not in serialized


def test_cli_exposes_only_the_six_authorized_paths() -> None:
    parser = _parser()
    option_strings = {
        option
        for action in parser._actions
        for option in action.option_strings
        if option not in {"-h", "--help"}
    }

    assert option_strings == {
        "--gold",
        "--evidence",
        "--snapshot-root",
        "--out-map",
        "--out-private-audit",
        "--out-public-audit",
    }
