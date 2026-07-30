from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest
from pydantic import ValidationError

import paper_search.evaluation.freeze_schema as freeze_schema
from paper_search.evaluation.freeze_schema import (
    FreezeManifestV1,
    FreezeManifestV2,
    FrozenPartitionV2,
    canonical_gold_set_sha256,
    load_freeze_manifest,
    open_confined_artifact,
    publish_confined_bytes_no_overwrite,
)


FIXTURE_ROOT = Path("tests/fixtures/evaluation/freeze_v2")


def _sha256(content: bytes) -> str:
    return "sha256:" + hashlib.sha256(content).hexdigest()


def _canonical_document(payload: object) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode()


def _write_bound_v2_tree(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    data_root = tmp_path / "data"
    data_root.mkdir()
    dev_bytes = b'{"query_id":"dev-1","query":"synthetic","relevant_paper_ids":["p1"]}\n'
    validation_bytes = (
        b'{"query_id":"validation-1","query":"synthetic","relevant_paper_ids":[]}\n'
    )
    identifier_map_bytes = b'{"legacy:1":"canonical:1"}'
    for relative, content in (
        ("dev/gold.jsonl", dev_bytes),
        ("validation/gold.jsonl", validation_bytes),
        ("identifier-map.json", identifier_map_bytes),
    ):
        path = data_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    hashes = {
        "dev": _sha256(dev_bytes),
        "validation": _sha256(validation_bytes),
        "identifier_map": _sha256(identifier_map_bytes),
    }
    approval = {
        "schema_version": "freeze-approval-v2",
        "approval_requested": True,
        "approved_at": "2026-07-30T00:00:00Z",
        "approver_ref": "fixture-operator",
        "audit_sha256": _sha256(b"legacy-audit"),
        "partition_hashes": {"dev": hashes["dev"], "validation": hashes["validation"]},
        "identifier_map_sha256": hashes["identifier_map"],
    }
    approval_bytes = _canonical_document(approval)
    approval_path = data_root / "freeze_reports" / "approval.json"
    approval_path.parent.mkdir()
    approval_path.write_bytes(approval_bytes)
    manifest = {
        "schema_version": "paper-search-freeze-v2",
        "dataset_revision": "fixture-revision",
        "created_at": "2026-07-30T00:00:00Z",
        "annotation_status": "frozen",
        "freeze_status": "approved",
        "partitions": [
            {
                "name": "dev",
                "path": "dev/gold.jsonl",
                "query_count": 1,
                "sha256": hashes["dev"],
                "zero_answer_policy": "forbid",
            },
            {
                "name": "validation",
                "path": "validation/gold.jsonl",
                "query_count": 1,
                "sha256": hashes["validation"],
                "zero_answer_policy": "allow",
            },
        ],
        "gold_sha256": canonical_gold_set_sha256(hashes["dev"], hashes["validation"]),
        "identifier_map": {
            "path": "identifier-map.json",
            "sha256": hashes["identifier_map"],
            "entry_count": 1,
        },
        "partition_immutability": "content_addressed",
        "approval": {
            "report_path": "freeze_reports/approval.json",
            "report_sha256": _sha256(approval_bytes),
            "approved_at": "2026-07-30T00:00:00Z",
            "approver_ref": "fixture-operator",
        },
    }
    manifest_path = data_root / "manifest.json"
    manifest_path.write_bytes(_canonical_document(manifest))
    return data_root, manifest


def _legacy_manifest() -> dict[str, object]:
    digest = _sha256(b"evidence")
    return {
        "repo_id": "CarlanLark/pasa-dataset",
        "revision": "232428b0c867268c3b8ded90db4d98c1b30501d6",
        "license": "CC-BY-NC-SA-4.0",
        "access": "gated-hugging-face-dataset",
        "random_seed": 20260714,
        "sampling_algorithm": "answer-count-largest-remainder-v1",
        "status": "frozen",
        "source_files": [
            {
                "path": path,
                "raw_path": f"raw/{path}",
                "row_count": row_count,
                "byte_count": 1,
                "sha256": digest,
            }
            for path, row_count in (
                ("AutoScholarQuery/dev.jsonl", 1000),
                ("AutoScholarQuery/test.jsonl", 1000),
                ("RealScholarQuery/test.jsonl", 50),
            )
        ],
        "partitions": {
            name: {
                "count": {"dev": 60, "validation": 30, "simulated_test": 50}[name],
                "gold_path": f"{name}/gold.jsonl",
                "gold_sha256": digest,
                "ids_path": f"splits/{name}.ids.json",
                "ids_sha256": digest,
                "zero_answer_policy": "reject",
                "labels_complete": True,
            }
            for name in ("dev", "validation", "simulated_test")
        },
        "work_package_sampling": "answer-count-largest-remainder-v1-seeded-offsets",
        "work_packages": {
            "type_domain": {
                "count": 90,
                "source_path": "annotation_work/type_domain_source.jsonl",
                "source_sha256": digest,
                "ids_path": "splits/type_domain_annotation.ids.json",
                "ids_sha256": digest,
            },
            "constraints": {
                "count": 40,
                "source_path": "annotation_work/constraints_source.jsonl",
                "source_sha256": digest,
                "ids_path": "splits/constraint_annotation.ids.json",
                "ids_sha256": digest,
            },
            "overlap": {
                "count": 20,
                "ids_path": "splits/overlap_annotation.ids.json",
                "ids_sha256": digest,
            },
        },
        "prepared_manifest_sha256": digest,
        "freeze_report_path": "freeze_reports/legacy.json",
        "freeze_report_sha256": digest,
    }


def test_loads_only_approved_legacy_manifest_with_in_memory_discriminator(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    manifest_path = data_root / "manifest.json"
    payload = _legacy_manifest()
    original = json.dumps(payload, sort_keys=True, indent=2).encode() + b"\n"
    manifest_path.write_bytes(original)

    manifest = load_freeze_manifest(manifest_path, data_root=data_root)

    assert isinstance(manifest, FreezeManifestV1)
    assert manifest.schema_version == "paper-search-freeze-v1"
    assert manifest_path.read_bytes() == original


def test_rejects_unapproved_unversioned_legacy_manifest(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    payload = _legacy_manifest()
    payload["status"] = "waiting_for_human_label_freeze"
    path = data_root / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="freeze manifest is invalid"):
        load_freeze_manifest(path, data_root=data_root)


def test_rejects_legacy_discriminator_written_to_source_bytes(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    payload = _legacy_manifest()
    payload["schema_version"] = "paper-search-freeze-v1"
    path = data_root / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="freeze manifest is invalid"):
        load_freeze_manifest(path, data_root=data_root)


def test_v2_normative_models_are_strict_and_require_unique_ordered_partitions() -> None:
    with pytest.raises(ValidationError):
        FrozenPartitionV2(
            name="dev",
            path="dev/gold.jsonl",
            query_count="1",
            sha256=_sha256(b"dev"),
            zero_answer_policy="forbid",
        )

    digest = _sha256(b"same")
    with pytest.raises(ValidationError):
        FreezeManifestV2(
            schema_version="paper-search-freeze-v2",
            dataset_revision="revision-1",
            created_at="2026-07-30T00:00:00Z",
            annotation_status="frozen",
            freeze_status="approved",
            partitions=[
                FrozenPartitionV2(
                    name="validation",
                    path="validation/gold.jsonl",
                    query_count=1,
                    sha256=digest,
                    zero_answer_policy="forbid",
                ),
                FrozenPartitionV2(
                    name="dev",
                    path="dev/gold.jsonl",
                    query_count=1,
                    sha256=digest,
                    zero_answer_policy="forbid",
                ),
            ],
            gold_sha256=digest,
            identifier_map={"path": "ids/map.json", "sha256": digest, "entry_count": 1},
            partition_immutability="content_addressed",
            approval={
                "report_path": "freeze_reports/v2.json",
                "report_sha256": digest,
                "approved_at": "2026-07-30T00:00:00Z",
                "approver_ref": "operator-1",
            },
        )


def test_canonical_gold_identity_uses_fixed_partition_order() -> None:
    dev = _sha256(b"dev")
    validation = _sha256(b"validation")
    expected = _sha256(
        (
            '{"partitions":[{"name":"dev","sha256":"'
            + dev
            + '"},{"name":"validation","sha256":"'
            + validation
            + '"}],"schema_version":"paper-search-gold-set-v1"}'
        ).encode()
    )

    assert canonical_gold_set_sha256(dev, validation) == expected


def test_v2_loader_rejects_missing_bound_approval_report(tmp_path: Path) -> None:
    data_root, _ = _write_bound_v2_tree(tmp_path)
    (data_root / "freeze_reports" / "approval.json").unlink()

    with pytest.raises(ValueError, match="freeze manifest is invalid"):
        load_freeze_manifest(data_root / "manifest.json", data_root=data_root)


def test_v2_loader_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    data_root, _ = _write_bound_v2_tree(tmp_path)
    (data_root / "manifest.json").write_bytes(
        b'{"schema_version":"paper-search-freeze-v2","schema_version":"paper-search-freeze-v2"}'
    )

    with pytest.raises(ValueError, match="freeze manifest is invalid"):
        load_freeze_manifest(data_root / "manifest.json", data_root=data_root)


def test_loads_fully_bound_synthetic_versioned_v2_fixture(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    shutil.copytree(FIXTURE_ROOT, data_root)

    manifest = load_freeze_manifest(data_root / "manifest.json", data_root=data_root)

    assert isinstance(manifest, FreezeManifestV2)
    assert manifest.partitions[0].name == "dev"


def test_confined_report_publication_never_overwrites_different_existing_bytes(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()

    assert (
        publish_confined_bytes_no_overwrite(
            data_root, "freeze_reports/approval.json", b"first"
        )
        == "created"
    )
    assert (
        publish_confined_bytes_no_overwrite(
            data_root, "freeze_reports/approval.json", b"first"
        )
        == "matched"
    )
    with pytest.raises(FileExistsError):
        publish_confined_bytes_no_overwrite(
            data_root, "freeze_reports/approval.json", b"different"
        )
    assert (data_root / "freeze_reports" / "approval.json").read_bytes() == b"first"


def test_confined_report_publication_leaves_no_final_file_on_write_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed report write cannot reserve a partially written authority file."""
    data_root = tmp_path / "data"
    data_root.mkdir()

    def fail_after_partial_write(descriptor: int, content: bytes) -> None:
        os.write(descriptor, content[:1])
        raise OSError("synthetic write failure")

    monkeypatch.setattr(freeze_schema, "_write_descriptor", fail_after_partial_write)

    with pytest.raises(OSError, match="synthetic write failure"):
        publish_confined_bytes_no_overwrite(
            data_root, "freeze_reports/approval.json", b"complete-report"
        )

    assert not (data_root / "freeze_reports" / "approval.json").exists()


def test_v2_loader_rechecks_every_bound_artifact_after_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The loader forms one held-descriptor snapshot before it authorizes V2."""
    data_root, _ = _write_bound_v2_tree(tmp_path)
    original = freeze_schema.BoundArtifact.verify_path_identity
    checked: list[str] = []

    def record_check(artifact: freeze_schema.BoundArtifact) -> None:
        checked.append(str(artifact.relative_path))
        original(artifact)

    monkeypatch.setattr(freeze_schema.BoundArtifact, "verify_path_identity", record_check)

    load_freeze_manifest(data_root / "manifest.json", data_root=data_root)

    assert checked == [
        "manifest.json",
        "freeze_reports/approval.json",
        "dev/gold.jsonl",
        "validation/gold.jsonl",
        "identifier-map.json",
    ]


@pytest.mark.skipif(os.name != "nt", reason="Windows sharing contract")
def test_only_current_manifest_may_opt_in_to_delete_sharing_for_guarded_replace(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    manifest = data_root / "manifest.json"
    manifest.write_bytes(b"current")
    backup = data_root / "backup.json"

    with open_confined_artifact(
        data_root, "manifest.json", replaceable_manifest=True
    ) as artifact:
        os.replace(manifest, backup)
        assert artifact.content == b"current"

    assert backup.read_bytes() == b"current"


@pytest.mark.skipif(os.name != "nt", reason="Windows sharing contract")
def test_normal_evidence_cannot_be_replaced_while_bound(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    parent = data_root / "evidence"
    parent.mkdir()
    evidence = parent / "evidence.json"
    evidence.write_bytes(b"current")
    replacement = parent / "replacement.tmp"
    replacement.write_bytes(b"different")

    with open_confined_artifact(data_root, "evidence/evidence.json"):
        with pytest.raises(PermissionError):
            os.replace(replacement, evidence)
        with pytest.raises(PermissionError):
            os.replace(parent, data_root / "evidence-renamed")
        with pytest.raises(PermissionError):
            os.replace(data_root, tmp_path / "data-renamed")

    assert evidence.read_bytes() == b"current"
