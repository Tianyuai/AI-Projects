from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from paper_search.evaluation.freeze_schema import (
    FreezeManifestV1,
    FreezeManifestV2,
    FrozenPartitionV2,
    canonical_gold_set_sha256,
    load_freeze_manifest,
)


FIXTURE_ROOT = Path("tests/fixtures/evaluation/freeze_v2")


def _sha256(content: bytes) -> str:
    return "sha256:" + hashlib.sha256(content).hexdigest()


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


def test_loads_synthetic_versioned_v2_fixture(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    path = data_root / "manifest.json"
    path.write_bytes((FIXTURE_ROOT / "manifest.json").read_bytes())

    manifest = load_freeze_manifest(path, data_root=data_root)

    assert isinstance(manifest, FreezeManifestV2)
    assert manifest.partitions[0].name == "dev"
