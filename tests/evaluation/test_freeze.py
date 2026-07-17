from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import cast

import pytest

from paper_search.evaluation.freeze import audit_freeze_candidate


def _load_preparation_module() -> ModuleType:
    module_path = Path("scripts/prepare_task2_data.py")
    spec = importlib.util.spec_from_file_location("freeze_prepare_task2_data", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


preparation = _load_preparation_module()


def _source_bytes(prefix: str) -> bytes:
    rows = [
        {
            "qid": f"{prefix}-q1",
            "question": f"Synthetic question one for {prefix}",
            "answer": ["Synthetic Paper One"],
            "answer_arxiv_id": ["2501.10120"],
            "source_meta": {"published_time": "2025-01-01"},
        },
        {
            "qid": f"{prefix}-q2",
            "question": f"Synthetic question two for {prefix}",
            "answer": ["Synthetic Paper Two"],
            "answer_arxiv_id": ["1706.03762"],
            "source_meta": {"published_time": "2024-01-01"},
        },
    ]
    return ("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n").encode()


FIXTURE_FILES = {
    "AutoScholarQuery/dev.jsonl": _source_bytes("auto-dev"),
    "AutoScholarQuery/test.jsonl": _source_bytes("auto-test"),
    "RealScholarQuery/test.jsonl": _source_bytes("real-test"),
}


@dataclass(frozen=True)
class PreparedFixture:
    data_root: Path
    type_domain_labels: Path
    constraint_labels: Path
    overlap_labels: Path


def _jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) for row in rows
    )
    path.write_text(content + "\n", encoding="utf-8")


def _prepared_tree(tmp_path: Path) -> PreparedFixture:
    data_root = tmp_path / "data"

    def downloader(repo_id: str, revision: str, path: str, token: str) -> bytes:
        assert repo_id == preparation.PASA_REPO_ID
        assert revision == preparation.PASA_REVISION
        assert token == "test-token"
        return FIXTURE_FILES[path]

    manifest = preparation.prepare(
        output_root=data_root,
        token="test-token",
        downloader=downloader,
        expected_counts={path: 2 for path in FIXTURE_FILES},
        dev_size=1,
        validation_size=1,
        simulated_test_size=2,
        constraint_annotation_size=1,
        overlap_annotation_size=1,
    )
    work_packages = manifest["work_packages"]
    assert isinstance(work_packages, dict)

    type_domain_ids = json.loads(
        (data_root / work_packages["type_domain"]["ids_path"]).read_text(
            encoding="utf-8"
        )
    )
    constraint_ids = json.loads(
        (data_root / work_packages["constraints"]["ids_path"]).read_text(
            encoding="utf-8"
        )
    )
    overlap_ids = json.loads(
        (data_root / work_packages["overlap"]["ids_path"]).read_text(encoding="utf-8")
    )

    private_root = tmp_path / "private"
    type_domain_labels = private_root / "type_domain.jsonl"
    constraint_labels = private_root / "constraints.jsonl"
    overlap_labels = private_root / "overlap.jsonl"
    _jsonl(
        type_domain_labels,
        [
            {
                "query_id": query_id,
                "query_type": "method",
                "domain": "information-retrieval",
                "annotator": "member-a",
            }
            for query_id in type_domain_ids
        ],
    )
    constraint_rows = [
        {
            "query_id": query_id,
            "research_goal": "Synthetic goal",
            "must_have": ["retrieval"],
            "should_have": [],
            "exclusions": [],
            "year_from": None,
            "year_to": None,
            "venues": [],
            "query_type": "method",
            "domain": "information-retrieval",
            "annotator": "member-a",
        }
        for query_id in constraint_ids
    ]
    _jsonl(constraint_labels, constraint_rows)
    overlap_set = set(overlap_ids)
    _jsonl(
        overlap_labels,
        [
            {**row, "annotator": "member-b"}
            for row in constraint_rows
            if row["query_id"] in overlap_set
        ],
    )
    return PreparedFixture(
        data_root=data_root,
        type_domain_labels=type_domain_labels,
        constraint_labels=constraint_labels,
        overlap_labels=overlap_labels,
    )


def _manifest(fixture: PreparedFixture) -> dict[str, object]:
    payload = json.loads(
        (fixture.data_root / "manifest.json").read_text(encoding="utf-8")
    )
    assert isinstance(payload, dict)
    return cast(dict[str, object], payload)


def _write_manifest(fixture: PreparedFixture, payload: dict[str, object]) -> None:
    content = (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode()
    (fixture.data_root / "manifest.json").write_bytes(content)


def _source_entry(manifest: dict[str, object]) -> dict[str, object]:
    source_files = manifest["source_files"]
    assert isinstance(source_files, list)
    entry = source_files[0]
    assert isinstance(entry, dict)
    return cast(dict[str, object], entry)


def _partition(manifest: dict[str, object], name: str) -> dict[str, object]:
    partitions = manifest["partitions"]
    assert isinstance(partitions, dict)
    partition = partitions[name]
    assert isinstance(partition, dict)
    return cast(dict[str, object], partition)


def _work_package(manifest: dict[str, object], name: str) -> dict[str, object]:
    packages = manifest["work_packages"]
    assert isinstance(packages, dict)
    package = packages[name]
    assert isinstance(package, dict)
    return cast(dict[str, object], package)


def _sha256(content: bytes) -> str:
    return "sha256:" + hashlib.sha256(content).hexdigest()


def _audit(fixture: PreparedFixture) -> object:
    return audit_freeze_candidate(
        data_root=fixture.data_root,
        type_domain_labels_path=fixture.type_domain_labels,
        constraint_labels_path=fixture.constraint_labels,
        overlap_labels_path=fixture.overlap_labels,
        policies={"dev": "reject", "validation": "reject", "simulated_test": "allow"},
    )

def test_audit_candidate_builds_safe_result_without_writing(tmp_path: Path) -> None:
    fixture = _prepared_tree(tmp_path)
    original_manifest_bytes = (fixture.data_root / "manifest.json").read_bytes()

    result = audit_freeze_candidate(
        data_root=fixture.data_root,
        type_domain_labels_path=fixture.type_domain_labels,
        constraint_labels_path=fixture.constraint_labels,
        overlap_labels_path=fixture.overlap_labels,
        policies={"dev": "reject", "validation": "reject", "simulated_test": "allow"},
    )

    dev = result.report.partitions["dev"]
    assert dev.labels_complete is True
    assert dev.gold_sha256.startswith("sha256:")
    assert result.report.approval_requested is False
    assert result.report.prepared_manifest_sha256 == (
        "sha256:" + hashlib.sha256(original_manifest_bytes).hexdigest()
    )
    assert (fixture.data_root / "manifest.json").read_bytes() == original_manifest_bytes


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("status", "frozen"),
        ("repo_id", "different/repository"),
        ("revision", "different-revision"),
        ("random_seed", 1),
        ("sampling_algorithm", "different-algorithm"),
    ],
)
def test_manifest_rejects_invalid_prepared_identity(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    fixture = _prepared_tree(tmp_path)
    manifest = _manifest(fixture)
    manifest[field] = value
    _write_manifest(fixture, manifest)

    with pytest.raises(ValueError, match="prepared data is invalid"):
        _audit(fixture)


@pytest.mark.parametrize("raw_path", ["../outside.jsonl", "raw/missing.jsonl"])
def test_path_rejects_escape_and_missing_source(
    tmp_path: Path,
    raw_path: str,
) -> None:
    fixture = _prepared_tree(tmp_path)
    manifest = _manifest(fixture)
    _source_entry(manifest)["raw_path"] = raw_path
    _write_manifest(fixture, manifest)

    with pytest.raises(ValueError, match="prepared data is invalid"):
        _audit(fixture)


@pytest.mark.parametrize("field", ["byte_count", "sha256"])
def test_manifest_rejects_raw_source_identity_mismatch(
    tmp_path: Path,
    field: str,
) -> None:
    fixture = _prepared_tree(tmp_path)
    manifest = _manifest(fixture)
    _source_entry(manifest)[field] = 1 if field == "byte_count" else "sha256:bad"
    _write_manifest(fixture, manifest)

    with pytest.raises(ValueError, match="prepared data is invalid"):
        _audit(fixture)


def test_partition_rejects_empty_gold(tmp_path: Path) -> None:
    fixture = _prepared_tree(tmp_path)
    (fixture.data_root / "dev" / "gold.jsonl").write_bytes(b"")

    with pytest.raises(ValueError, match="prepared data is invalid"):
        _audit(fixture)


def test_partition_rejects_declared_count_mismatch(tmp_path: Path) -> None:
    fixture = _prepared_tree(tmp_path)
    manifest = _manifest(fixture)
    _partition(manifest, "dev")["count"] = 2
    _write_manifest(fixture, manifest)

    with pytest.raises(ValueError, match="prepared data is invalid"):
        _audit(fixture)


def test_partition_rejects_duplicate_ids(tmp_path: Path) -> None:
    fixture = _prepared_tree(tmp_path)
    manifest = _manifest(fixture)
    partition = _partition(manifest, "simulated_test")
    ids_path = fixture.data_root / cast(str, partition["ids_path"])
    identifiers = json.loads(ids_path.read_text(encoding="utf-8"))
    assert isinstance(identifiers, list) and len(identifiers) == 2
    duplicate_content = (
        json.dumps([identifiers[0], identifiers[0]], indent=2) + "\n"
    ).encode()
    ids_path.write_bytes(duplicate_content)
    partition["ids_sha256"] = _sha256(duplicate_content)
    _write_manifest(fixture, manifest)

    with pytest.raises(ValueError, match="prepared data is invalid"):
        _audit(fixture)


def test_partition_rejects_ordered_id_mismatch(tmp_path: Path) -> None:
    fixture = _prepared_tree(tmp_path)
    manifest = _manifest(fixture)
    partition = _partition(manifest, "simulated_test")
    ids_path = fixture.data_root / cast(str, partition["ids_path"])
    identifiers = json.loads(ids_path.read_text(encoding="utf-8"))
    assert isinstance(identifiers, list)
    reversed_content = (
        json.dumps(list(reversed(identifiers)), indent=2) + "\n"
    ).encode()
    ids_path.write_bytes(reversed_content)
    partition["ids_sha256"] = _sha256(reversed_content)
    _write_manifest(fixture, manifest)

    with pytest.raises(ValueError, match="prepared data is invalid"):
        _audit(fixture)


def test_partition_rejects_existing_gold_hash_mismatch(tmp_path: Path) -> None:
    fixture = _prepared_tree(tmp_path)
    manifest = _manifest(fixture)
    _partition(manifest, "dev")["gold_sha256"] = "sha256:bad"
    _write_manifest(fixture, manifest)

    with pytest.raises(ValueError, match="prepared data is invalid"):
        _audit(fixture)


def test_work_package_rejects_invalid_overlap_subset(tmp_path: Path) -> None:
    fixture = _prepared_tree(tmp_path)
    manifest = _manifest(fixture)
    overlap = _work_package(manifest, "overlap")
    validation_ids = json.loads(
        (fixture.data_root / "splits" / "validation.ids.json").read_text(
            encoding="utf-8"
        )
    )
    assert isinstance(validation_ids, list) and len(validation_ids) == 1
    overlap_path = fixture.data_root / cast(str, overlap["ids_path"])
    overlap_content = (json.dumps(validation_ids, indent=2) + "\n").encode()
    overlap_path.write_bytes(overlap_content)
    overlap["ids_sha256"] = _sha256(overlap_content)
    _write_manifest(fixture, manifest)

    with pytest.raises(ValueError, match="prepared data is invalid"):
        _audit(fixture)


def test_work_package_rejects_source_hash_mismatch(tmp_path: Path) -> None:
    fixture = _prepared_tree(tmp_path)
    manifest = _manifest(fixture)
    _work_package(manifest, "constraints")["source_sha256"] = "sha256:bad"
    _write_manifest(fixture, manifest)

    with pytest.raises(ValueError, match="prepared data is invalid"):
        _audit(fixture)