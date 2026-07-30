from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Iterator, cast

import pytest

import paper_search.evaluation.freeze as freeze_module
import paper_search.evaluation.freeze_schema as freeze_schema_module

from paper_search.evaluation.freeze import (
    FreezeApprovalPlan,
    OFFICIAL_EXPECTATIONS,
    approve_freeze,
    audit_freeze_candidate,
    build_approval_plan,
    migrate_v1_to_v2,
    main,
    parse_zero_answer_policies,
)
from paper_search.evaluation.freeze_schema import (
    FreezeApprovalReportV2,
    IdentifierMapBindingV2,
    FreezeManifestV1,
    FreezeManifestV2,
    load_freeze_manifest,
    open_confined_artifact,
)


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


def _source_bytes(prefix: str, count: int) -> bytes:
    rows = [
        {
            "qid": f"{prefix}-q{index:04d}",
            "question": f"Synthetic question {index} for {prefix}",
            "answer": [f"Synthetic Paper {index}"],
            "answer_arxiv_id": ["2501.10120" if index % 2 == 0 else "1706.03762"],
            "source_meta": {"published_time": "2025-01-01"},
        }
        for index in range(count)
    ]
    return ("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n").encode()


FIXTURE_FILES = {
    "AutoScholarQuery/dev.jsonl": _source_bytes("auto-dev", 1000),
    "AutoScholarQuery/test.jsonl": _source_bytes("auto-test", 1000),
    "RealScholarQuery/test.jsonl": _source_bytes("real-test", 50),
}

REDUCED_FIXTURE_FILES = {path: _source_bytes(path.replace("/", "-"), 2) for path in FIXTURE_FILES}


@dataclass(frozen=True)
class PreparedFixture:
    data_root: Path
    type_domain_labels: Path
    constraint_labels: Path
    overlap_labels: Path


@dataclass(frozen=True)
class MigrationFixture:
    prepared: PreparedFixture
    legacy: FreezeManifestV1
    approval: FreezeApprovalReportV2
    identifier_map: IdentifierMapBindingV2
    dataset_revision: str
    approval_report_path: str
    legacy_manifest_bytes: bytes


def _jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(json.dumps(row, sort_keys=True, separators=(",", ":")) for row in rows)
    path.write_text(content + "\n", encoding="utf-8")


def _prepared_tree(
    tmp_path: Path,
    *,
    reduced: bool = False,
) -> PreparedFixture:
    data_root = tmp_path / "data"
    fixture_files = REDUCED_FIXTURE_FILES if reduced else FIXTURE_FILES

    def downloader(repo_id: str, revision: str, path: str, token: str) -> bytes:
        assert repo_id == preparation.PASA_REPO_ID
        assert revision == preparation.PASA_REVISION
        assert token == "test-token"
        return fixture_files[path]

    manifest = preparation.prepare(
        output_root=data_root,
        token="test-token",
        downloader=downloader,
        expected_counts={
            path: len(content.splitlines()) for path, content in fixture_files.items()
        },
        dev_size=1 if reduced else 60,
        validation_size=1 if reduced else 30,
        simulated_test_size=2 if reduced else 50,
        constraint_annotation_size=1 if reduced else 40,
        overlap_annotation_size=1 if reduced else 20,
    )
    work_packages = manifest["work_packages"]
    assert isinstance(work_packages, dict)

    type_domain_ids = json.loads(
        (data_root / work_packages["type_domain"]["ids_path"]).read_text(encoding="utf-8")
    )
    constraint_ids = json.loads(
        (data_root / work_packages["constraints"]["ids_path"]).read_text(encoding="utf-8")
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
    payload = json.loads((fixture.data_root / "manifest.json").read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return cast(dict[str, object], payload)


def _write_manifest(fixture: PreparedFixture, payload: dict[str, object]) -> None:
    content = (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
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


def _label_path(fixture: PreparedFixture, name: str) -> Path:
    return {
        "type_domain": fixture.type_domain_labels,
        "constraints": fixture.constraint_labels,
        "overlap": fixture.overlap_labels,
    }[name]


def _label_rows(path: Path) -> list[dict[str, object]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert all(isinstance(row, dict) for row in rows)
    return cast(list[dict[str, object]], rows)


def _run_cli(args: list[str]) -> int:
    return main(args)


def _cli_args(fixture: PreparedFixture) -> list[str]:
    return [
        "--data-root",
        str(fixture.data_root),
        "--type-domain-labels",
        str(fixture.type_domain_labels),
        "--constraint-labels",
        str(fixture.constraint_labels),
        "--overlap-labels",
        str(fixture.overlap_labels),
        "--zero-answer-policy",
        "dev=reject",
        "--zero-answer-policy",
        "validation=reject",
        "--zero-answer-policy",
        "simulated_test=allow",
    ]


def _approval_plan(fixture: PreparedFixture) -> FreezeApprovalPlan:
    audit = _audit(fixture)
    return build_approval_plan(
        audit,
        report_relative_path="freeze_reports/synthetic-freeze.json",
    )


def _migration_fixture(tmp_path: Path) -> MigrationFixture:
    fixture = _prepared_tree(tmp_path)
    plan = _approval_plan(fixture)
    assert approve_freeze(data_root=fixture.data_root, plan=plan) == "created"
    legacy_manifest_bytes = (fixture.data_root / "manifest.json").read_bytes()
    legacy = load_freeze_manifest(
        fixture.data_root / "manifest.json",
        data_root=fixture.data_root,
    )
    assert isinstance(legacy, FreezeManifestV1)
    identifier_map_path = fixture.data_root / "identifier-map.json"
    identifier_map_bytes = b'{"legacy:1":"canonical:1"}'
    identifier_map_path.write_bytes(identifier_map_bytes)
    partitions = legacy.partitions
    approval = FreezeApprovalReportV2(
        schema_version="freeze-approval-v2",
        approval_requested=True,
        approved_at=datetime(2026, 7, 30, tzinfo=UTC),
        approver_ref="operator-1",
        audit_sha256=_sha256(plan.report_bytes),
        partition_hashes={
            "dev": partitions["dev"].gold_sha256,
            "validation": partitions["validation"].gold_sha256,
        },
        identifier_map_sha256=_sha256(identifier_map_bytes),
    )
    return MigrationFixture(
        prepared=fixture,
        legacy=legacy,
        approval=approval,
        identifier_map=IdentifierMapBindingV2(
            path="identifier-map.json",
            sha256=_sha256(identifier_map_bytes),
            entry_count=1,
        ),
        dataset_revision="v2-revision-1",
        approval_report_path="freeze_reports/v2-approval.json",
        legacy_manifest_bytes=legacy_manifest_bytes,
    )


def _migrate(case: MigrationFixture) -> FreezeManifestV2:
    return migrate_v1_to_v2(
        case.legacy,
        data_root=case.prepared.data_root,
        approval=case.approval,
        identifier_map=case.identifier_map,
        dataset_revision=case.dataset_revision,
        approval_report_path=case.approval_report_path,
    )


def _native_replace_windows(source: Path, target: Path) -> None:
    import ctypes

    class FileRenameInfoEx(ctypes.Structure):
        _fields_ = [
            ("flags", ctypes.c_uint32),
            ("root_directory", ctypes.c_void_p),
            ("file_name_length", ctypes.c_uint32),
            ("file_name", ctypes.c_wchar * 1),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.restype = ctypes.c_void_p
    handle = kernel32.CreateFileW(
        str(source),
        0x00010000,
        0x00000001 | 0x00000002 | 0x00000004,
        None,
        3,
        0,
        None,
    )
    invalid = ctypes.c_void_p(-1).value
    if handle == invalid:
        pytest.fail(f"CreateFileW failed: {ctypes.get_last_error()}")
    target_bytes = (str(target) + "\0").encode("utf-16-le")
    buffer = ctypes.create_string_buffer(
        ctypes.sizeof(FileRenameInfoEx)
        - ctypes.sizeof(ctypes.c_wchar)
        + len(target_bytes)
    )
    request = FileRenameInfoEx.from_buffer(buffer)
    request.flags = 0x00000001 | 0x00000002
    request.root_directory = None
    request.file_name_length = len(target_bytes) - 2
    ctypes.memmove(
        ctypes.addressof(buffer) + FileRenameInfoEx.file_name.offset,
        target_bytes,
        len(target_bytes),
    )
    try:
        result = kernel32.SetFileInformationByHandle(
            ctypes.c_void_p(handle), 22, ctypes.byref(request), len(buffer)
        )
        if not result:
            error = ctypes.get_last_error()
            if error in {1, 50, 87, 120}:
                pytest.skip(f"FileRenameInfoEx unavailable: WinError {error}")
            pytest.fail(f"FileRenameInfoEx failed: WinError {error}")
    finally:
        kernel32.CloseHandle(ctypes.c_void_p(handle))


@pytest.mark.skipif(os.name != "nt", reason="Windows sharing contract")
def test_windows_lock_stabilizes_root_during_replaceable_manifest_transition(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    manifest = data_root / "manifest.json"
    manifest.write_bytes(b"current")

    with freeze_module._exclusive_freeze_lock(data_root):
        with open_confined_artifact(
            data_root, "manifest.json", replaceable_manifest=True
        ):
            os.replace(manifest, data_root / "manifest.backup.json")
            with pytest.raises(PermissionError):
                os.replace(data_root, tmp_path / "data-renamed")


def test_migrate_approved_v1_freeze_to_v2_with_bound_evidence(tmp_path: Path) -> None:
    case = _migration_fixture(tmp_path)
    migrated = _migrate(case)

    assert isinstance(migrated, FreezeManifestV2)
    assert [partition.name for partition in migrated.partitions] == ["dev", "validation"]
    assert [partition.zero_answer_policy for partition in migrated.partitions] == [
        "forbid",
        "forbid",
    ]
    assert (
        case.prepared.data_root / "freeze_reports" / "v2-approval.json"
    ).is_file()
    assert load_freeze_manifest(
        case.prepared.data_root / "manifest.json",
        data_root=case.prepared.data_root,
    ) == migrated
    assert _migrate(case) == migrated


@pytest.mark.parametrize("artifact", ["manifest", "report", "partition", "identifier_map"])
@pytest.mark.parametrize("mutation", ["rewrite", "replace"])
def test_idempotent_migration_rejects_artifact_mutation_matrix(
    tmp_path: Path,
    artifact: str,
    mutation: str,
) -> None:
    case = _migration_fixture(tmp_path)
    _migrate(case)
    root = case.prepared.data_root
    target = {
        "manifest": root / "manifest.json",
        "report": root / case.approval_report_path,
        "partition": root / "dev" / "gold.jsonl",
        "identifier_map": root / case.identifier_map.path,
    }[artifact]
    if mutation == "rewrite":
        target.write_bytes(b"mutated-evidence")
    else:
        replacement = target.with_name(f"{target.name}.replacement")
        replacement.write_bytes(b"replacement-evidence")
        os.replace(replacement, target)

    with pytest.raises(ValueError, match="freeze migration failed"):
        _migrate(case)


@pytest.mark.parametrize("report_state", ["created", "matched"])
def test_migration_manifest_failure_preserves_v1_and_exact_retry_succeeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    report_state: str,
) -> None:
    case = _migration_fixture(tmp_path)
    root = case.prepared.data_root
    report_path = root / case.approval_report_path
    expected_report = freeze_module._json_bytes(
        case.approval.model_dump(mode="json")
    )
    if report_state == "matched":
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_bytes(expected_report)
    original_publish = freeze_module._replace_manifest_guarded

    def fail_manifest_publish(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise RuntimeError("synthetic manifest publisher failure")

    monkeypatch.setattr(
        freeze_module,
        "_replace_manifest_guarded",
        fail_manifest_publish,
    )
    with pytest.raises(RuntimeError, match="synthetic manifest publisher failure"):
        _migrate(case)

    assert (root / "manifest.json").read_bytes() == case.legacy_manifest_bytes
    assert report_path.read_bytes() == expected_report

    monkeypatch.setattr(
        freeze_module,
        "_replace_manifest_guarded",
        original_publish,
    )
    assert isinstance(_migrate(case), FreezeManifestV2)


def test_migration_rejects_different_orphan_report_and_preserves_v1(
    tmp_path: Path,
) -> None:
    case = _migration_fixture(tmp_path)
    root = case.prepared.data_root
    report_path = root / case.approval_report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_bytes(b"different-orphan-report")

    with pytest.raises(RuntimeError, match="freeze migration failed"):
        _migrate(case)

    assert (root / "manifest.json").read_bytes() == case.legacy_manifest_bytes
    assert report_path.read_bytes() == b"different-orphan-report"


@pytest.mark.skipif(os.name != "nt", reason="Windows FILE_ID publisher contract")
def test_windows_migration_rejects_final_report_native_replacement_and_keeps_v1(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _migration_fixture(tmp_path)
    root = case.prepared.data_root
    final_path = root / case.approval_report_path
    attacker = root / "attacker-report.tmp"
    approval_bytes = freeze_module._json_bytes(case.approval.model_dump(mode="json"))
    attacker.write_bytes(approval_bytes)
    original_open = freeze_schema_module.open_confined_artifact
    with original_open(root, "attacker-report.tmp") as attacker_artifact:
        attacker_id = attacker_artifact.windows_file_id
    attacked = False

    @contextmanager
    def replace_final_before_verification(
        data_root: Path,
        relative_path: str,
        **kwargs: object,
    ) -> Iterator[freeze_schema_module.BoundArtifact]:
        nonlocal attacked
        if (
            relative_path == case.approval_report_path
            and kwargs.get("allow_target_write_share") is True
            and not attacked
        ):
            _native_replace_windows(attacker, final_path)
            attacked = True
        with original_open(data_root, relative_path, **kwargs) as artifact:
            yield artifact

    monkeypatch.setattr(
        freeze_schema_module,
        "open_confined_artifact",
        replace_final_before_verification,
    )

    with pytest.raises(RuntimeError, match="freeze migration failed"):
        _migrate(case)

    assert attacked
    assert (root / "manifest.json").read_bytes() == case.legacy_manifest_bytes
    with original_open(root, case.approval_report_path) as final_artifact:
        assert final_artifact.windows_file_id == attacker_id
        assert final_artifact.content == approval_bytes


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
    assert isinstance(identifiers, list) and len(identifiers) >= 2
    identifiers[1] = identifiers[0]
    duplicate_content = (json.dumps(identifiers, indent=2) + "\n").encode()
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
    reversed_content = (json.dumps(list(reversed(identifiers)), indent=2) + "\n").encode()
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
        (fixture.data_root / "splits" / "validation.ids.json").read_text(encoding="utf-8")
    )
    assert isinstance(validation_ids, list) and len(validation_ids) >= 20
    overlap_path = fixture.data_root / cast(str, overlap["ids_path"])
    overlap_content = (json.dumps(validation_ids[:20], indent=2) + "\n").encode()
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


@pytest.mark.parametrize("label_name", ["type_domain", "constraints", "overlap"])
@pytest.mark.parametrize("mutation", ["missing", "duplicate", "extra", "wrong-set"])
def test_human_labels_require_exact_unique_annotation_alignment(
    tmp_path: Path,
    label_name: str,
    mutation: str,
) -> None:
    fixture = _prepared_tree(tmp_path)
    path = _label_path(fixture, label_name)
    rows = _label_rows(path)
    if mutation == "missing":
        changed = rows[:-1]
    elif mutation == "duplicate":
        changed = [*rows, rows[0]]
    elif mutation == "extra":
        changed = [*rows, {**rows[0], "query_id": "extra-query"}]
    else:
        changed = [{**rows[0], "query_id": "wrong-query"}, *rows[1:]]
    _jsonl(path, changed)

    with pytest.raises(ValueError, match="private annotations are invalid") as error:
        _audit(fixture)

    assert "query" not in str(error.value)
    assert str(path) not in str(error.value)


def test_annotation_alignment_rejects_low_agreement_without_details(
    tmp_path: Path,
) -> None:
    fixture = _prepared_tree(tmp_path)
    rows = _label_rows(fixture.overlap_labels)
    rows[0]["query_type"] = "topic"
    rows[0]["domain"] = "computer-vision"
    _jsonl(fixture.overlap_labels, rows)

    with pytest.raises(
        ValueError,
        match="human annotation agreement is below threshold",
    ) as error:
        _audit(fixture)

    assert rows[0]["query_id"] not in str(error.value)
    assert "computer-vision" not in str(error.value)


def test_zero_answer_policy_parser_requires_one_explicit_policy_per_partition() -> None:
    assert parse_zero_answer_policies(
        ["dev=reject", "validation=allow"],
        {"dev", "validation"},
    ) == {"dev": "reject", "validation": "allow"}


@pytest.mark.parametrize(
    "values",
    [
        ["dev=reject"],
        ["dev=reject", "dev=allow", "validation=reject"],
        ["dev=reject", "validation=reject", "unknown=allow"],
        ["dev", "validation=reject"],
        ["dev=maybe", "validation=reject"],
    ],
)
def test_zero_answer_policy_parser_rejects_unsafe_input(values: list[str]) -> None:
    with pytest.raises(ValueError, match="zero-answer policies are invalid"):
        parse_zero_answer_policies(values, {"dev", "validation"})


def test_partition_policy_controls_zero_answer_acceptance(tmp_path: Path) -> None:
    fixture = _prepared_tree(tmp_path)
    gold_path = fixture.data_root / "simulated_test" / "gold.jsonl"
    rows = _label_rows(gold_path)
    rows[0]["relevant_paper_ids"] = []
    _jsonl(gold_path, rows)

    audit_freeze_candidate(
        data_root=fixture.data_root,
        type_domain_labels_path=fixture.type_domain_labels,
        constraint_labels_path=fixture.constraint_labels,
        overlap_labels_path=fixture.overlap_labels,
        policies={"dev": "reject", "validation": "reject", "simulated_test": "allow"},
    )
    with pytest.raises(ValueError, match="prepared data is invalid"):
        audit_freeze_candidate(
            data_root=fixture.data_root,
            type_domain_labels_path=fixture.type_domain_labels,
            constraint_labels_path=fixture.constraint_labels,
            overlap_labels_path=fixture.overlap_labels,
            policies={
                "dev": "reject",
                "validation": "reject",
                "simulated_test": "reject",
            },
        )


def test_audit_report_is_content_safe(tmp_path: Path) -> None:
    fixture = _prepared_tree(tmp_path)
    sentinel = "PRIVATE-SENTINEL-DO-NOT-EMIT"
    for path, annotator in (
        (fixture.type_domain_labels, f"{sentinel}-first"),
        (fixture.constraint_labels, f"{sentinel}-first"),
        (fixture.overlap_labels, f"{sentinel}-second"),
    ):
        rows = _label_rows(path)
        for row in rows:
            row["annotator"] = annotator
            if "research_goal" in row:
                row["research_goal"] = sentinel
        _jsonl(path, rows)

    result = _audit(fixture)
    report_text = json.dumps(result.report.model_dump(mode="json"), sort_keys=True)

    assert sentinel not in report_text
    assert "Synthetic question" not in report_text
    assert str(fixture.type_domain_labels) not in report_text


def test_cli_audit_only_prints_safe_report_without_writing(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = _prepared_tree(tmp_path)
    original_manifest = (fixture.data_root / "manifest.json").read_bytes()

    exit_code = _run_cli(_cli_args(fixture))

    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert exit_code == 0
    assert captured.err == ""
    assert report["approval_requested"] is False
    assert (fixture.data_root / "manifest.json").read_bytes() == original_manifest
    assert not (fixture.data_root / "freeze_reports").exists()


@pytest.mark.parametrize(
    "extra_args",
    [
        ["--approve"],
        ["--report", "data/freeze_reports/report.json"],
    ],
)
def test_cli_rejects_unpaired_approval_arguments(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    extra_args: list[str],
) -> None:
    fixture = _prepared_tree(tmp_path)

    exit_code = _run_cli([*_cli_args(fixture), *extra_args])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert captured.err.rstrip() == "freeze approval failed"


def test_cli_requires_all_explicit_inputs(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert _run_cli([]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.rstrip() == "freeze approval failed"


def test_cli_redacts_private_validation_failures(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = _prepared_tree(tmp_path)
    sentinel = "PRIVATE-PATH-SENTINEL"
    private_path = tmp_path / sentinel / "invalid.jsonl"
    private_path.parent.mkdir()
    private_path.write_text("not-json", encoding="utf-8")
    args = _cli_args(fixture)
    args[args.index(str(fixture.overlap_labels))] = str(private_path)

    exit_code = _run_cli(args)

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert captured.err.rstrip() == "freeze audit failed: private annotations are invalid"
    assert sentinel not in captured.err


def test_cli_rejects_report_path_outside_freeze_reports(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = _prepared_tree(tmp_path)
    outside_report = tmp_path / "outside-report.json"

    exit_code = _run_cli(
        [
            *_cli_args(fixture),
            "--approve",
            "--report",
            str(outside_report),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert captured.err.rstrip() == "freeze approval failed"
    assert not outside_report.exists()


def test_build_approval_plan_binds_complete_report_without_mutating_audit(
    tmp_path: Path,
) -> None:
    fixture = _prepared_tree(tmp_path)
    audit = _audit(fixture)

    plan = build_approval_plan(
        audit,
        report_relative_path="freeze_reports/synthetic-freeze.json",
    )

    frozen = json.loads(plan.frozen_manifest_bytes)
    assert audit.report.approval_requested is False
    assert plan.report.approval_requested is True
    assert plan.report_bytes[-1:] == bytes([10])
    assert frozen["status"] == "frozen"
    assert frozen["freeze_report_path"] == "freeze_reports/synthetic-freeze.json"
    assert frozen["freeze_report_sha256"] == _sha256(plan.report_bytes)


def test_build_approval_plan_rejects_report_path_escape(tmp_path: Path) -> None:
    fixture = _prepared_tree(tmp_path)
    audit = _audit(fixture)

    with pytest.raises(ValueError, match="freeze approval failed"):
        build_approval_plan(audit, report_relative_path="../outside.json")


def test_approve_writes_report_then_manifest_and_is_idempotent(tmp_path: Path) -> None:
    fixture = _prepared_tree(tmp_path)
    plan = _approval_plan(fixture)
    report_path = fixture.data_root / "freeze_reports" / "synthetic-freeze.json"

    assert approve_freeze(data_root=fixture.data_root, plan=plan) == "created"
    assert report_path.read_bytes() == plan.report_bytes
    assert (fixture.data_root / "manifest.json").read_bytes() == plan.frozen_manifest_bytes
    assert not list(fixture.data_root.rglob("*.tmp"))
    assert approve_freeze(data_root=fixture.data_root, plan=plan) == "matched"


def test_approve_rejects_manifest_changed_after_audit(tmp_path: Path) -> None:
    fixture = _prepared_tree(tmp_path)
    plan = _approval_plan(fixture)
    (fixture.data_root / "manifest.json").write_bytes(b"changed")

    with pytest.raises(RuntimeError, match="freeze approval failed"):
        approve_freeze(data_root=fixture.data_root, plan=plan)

    assert not (fixture.data_root / "freeze_reports").exists()


def test_approve_rejects_different_existing_report(tmp_path: Path) -> None:
    fixture = _prepared_tree(tmp_path)
    plan = _approval_plan(fixture)
    report_path = fixture.data_root / "freeze_reports" / "synthetic-freeze.json"
    report_path.parent.mkdir()
    report_path.write_bytes(b"different")

    with pytest.raises(FileExistsError):
        approve_freeze(data_root=fixture.data_root, plan=plan)

    assert (fixture.data_root / "manifest.json").read_bytes() == (plan.prepared_manifest_bytes)


def test_approve_reuses_identical_orphan_report(tmp_path: Path) -> None:
    fixture = _prepared_tree(tmp_path)
    plan = _approval_plan(fixture)
    report_path = fixture.data_root / "freeze_reports" / "synthetic-freeze.json"
    report_path.parent.mkdir()
    report_path.write_bytes(plan.report_bytes)

    assert approve_freeze(data_root=fixture.data_root, plan=plan) == "created"
    assert report_path.read_bytes() == plan.report_bytes


def test_approve_rejects_mutation_at_pre_replace_boundary(tmp_path: Path) -> None:
    fixture = _prepared_tree(tmp_path)
    plan = _approval_plan(fixture)
    manifest_path = fixture.data_root / "manifest.json"

    with pytest.raises(RuntimeError, match="freeze approval failed"):
        approve_freeze(
            data_root=fixture.data_root,
            plan=plan,
            before_manifest_replace=lambda: manifest_path.write_bytes(b"changed"),
        )

    report_path = fixture.data_root / "freeze_reports" / "synthetic-freeze.json"
    assert report_path.read_bytes() == plan.report_bytes
    assert manifest_path.read_bytes() == b"changed"
    assert not list(fixture.data_root.rglob("*.tmp"))


def test_approve_leaves_complete_report_if_manifest_replace_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _prepared_tree(tmp_path)
    plan = _approval_plan(fixture)

    def fail_replace(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise OSError("synthetic replace failure")

    monkeypatch.setattr(freeze_module, "_replace_manifest_guarded", fail_replace)
    with pytest.raises(OSError, match="synthetic replace failure"):
        approve_freeze(data_root=fixture.data_root, plan=plan)

    report_path = fixture.data_root / "freeze_reports" / "synthetic-freeze.json"
    assert report_path.read_bytes() == plan.report_bytes
    assert (fixture.data_root / "manifest.json").read_bytes() == (plan.prepared_manifest_bytes)


def test_cli_approve_writes_bound_report_and_frozen_manifest(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = _prepared_tree(tmp_path)
    report_path = fixture.data_root / "freeze_reports" / "cli-freeze.json"

    exit_code = _run_cli([*_cli_args(fixture), "--approve", "--report", str(report_path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    assert json.loads(captured.out)["approval_requested"] is True
    manifest = json.loads((fixture.data_root / "manifest.json").read_bytes())
    assert manifest["status"] == "frozen"
    assert report_path.is_file()


def test_approve_report_write_failure_leaves_manifest_prepared(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _prepared_tree(tmp_path)
    plan = _approval_plan(fixture)

    def fail_report_write(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise OSError("synthetic report failure")

    monkeypatch.setattr(freeze_module, "write_frozen_bytes", fail_report_write)
    with pytest.raises(OSError, match="synthetic report failure"):
        approve_freeze(data_root=fixture.data_root, plan=plan)

    assert (fixture.data_root / "manifest.json").read_bytes() == (plan.prepared_manifest_bytes)
    assert not (fixture.data_root / "freeze_reports").exists()


def test_approve_rejects_different_plan_after_manifest_is_frozen(
    tmp_path: Path,
) -> None:
    fixture = _prepared_tree(tmp_path)
    audit = _audit(fixture)
    first = build_approval_plan(
        audit,
        report_relative_path="freeze_reports/first.json",
    )
    different = build_approval_plan(
        audit,
        report_relative_path="freeze_reports/different.json",
    )
    approve_freeze(data_root=fixture.data_root, plan=first)

    with pytest.raises(RuntimeError, match="freeze approval failed"):
        approve_freeze(data_root=fixture.data_root, plan=different)

    assert not (fixture.data_root / "freeze_reports" / "different.json").exists()


def test_official_contract_rejects_reduced_self_consistent_tree(
    tmp_path: Path,
) -> None:
    fixture = _prepared_tree(tmp_path, reduced=True)

    with pytest.raises(ValueError, match="prepared data is invalid"):
        audit_freeze_candidate(
            data_root=fixture.data_root,
            type_domain_labels_path=fixture.type_domain_labels,
            constraint_labels_path=fixture.constraint_labels,
            overlap_labels_path=fixture.overlap_labels,
            policies={
                "dev": "reject",
                "validation": "reject",
                "simulated_test": "allow",
            },
        )


def test_small_contract_rejects_missing_source_entry(tmp_path: Path) -> None:
    fixture = _prepared_tree(tmp_path)
    manifest = _manifest(fixture)
    source_files = manifest["source_files"]
    assert isinstance(source_files, list)
    source_files.pop()
    _write_manifest(fixture, manifest)

    with pytest.raises(ValueError, match="prepared data is invalid"):
        _audit(fixture)


def test_audit_rejects_same_annotator_for_overlap(tmp_path: Path) -> None:
    fixture = _prepared_tree(tmp_path)
    rows = _label_rows(fixture.overlap_labels)
    for row in rows:
        row["annotator"] = "member-a"
    _jsonl(fixture.overlap_labels, rows)

    with pytest.raises(ValueError, match="private annotations are invalid"):
        _audit(fixture)


def test_audit_rejects_constraint_labels_that_disagree_with_type_domain(
    tmp_path: Path,
) -> None:
    fixture = _prepared_tree(tmp_path)
    overlap_ids = {row["query_id"] for row in _label_rows(fixture.overlap_labels)}
    rows = _label_rows(fixture.constraint_labels)
    target = next(row for row in rows if row["query_id"] not in overlap_ids)
    target["query_type"] = "topic"
    target["domain"] = "computer-vision"
    _jsonl(fixture.constraint_labels, rows)

    with pytest.raises(ValueError, match="private annotations are invalid"):
        _audit(fixture)


@pytest.mark.parametrize("label_name", ["type_domain", "constraints", "overlap"])
def test_audit_rejects_unstable_annotator_within_each_private_file(
    tmp_path: Path,
    label_name: str,
) -> None:
    fixture = _prepared_tree(tmp_path)
    path = _label_path(fixture, label_name)
    rows = _label_rows(path)
    rows[0]["annotator"] = "member-c"
    _jsonl(path, rows)

    with pytest.raises(ValueError, match="private annotations are invalid"):
        _audit(fixture)


def test_audit_rejects_different_annotators_for_type_domain_and_constraints(
    tmp_path: Path,
) -> None:
    fixture = _prepared_tree(tmp_path)
    rows = _label_rows(fixture.constraint_labels)
    for row in rows:
        row["annotator"] = "member-c"
    _jsonl(fixture.constraint_labels, rows)

    with pytest.raises(ValueError, match="private annotations are invalid"):
        _audit(fixture)

@pytest.mark.parametrize("location", ["top", "source", "work_package"])
def test_audit_rejects_unknown_manifest_fields(
    tmp_path: Path,
    location: str,
) -> None:
    fixture = _prepared_tree(tmp_path)
    manifest = _manifest(fixture)
    if location == "top":
        target = manifest
    elif location == "source":
        target = _source_entry(manifest)
    else:
        target = _work_package(manifest, "constraints")
    target["PRIVATE-SENTINEL"] = "must-not-survive"
    _write_manifest(fixture, manifest)

    with pytest.raises(ValueError, match="prepared data is invalid"):
        _audit(fixture)


def test_approve_rejects_gold_changed_after_audit(tmp_path: Path) -> None:
    fixture = _prepared_tree(tmp_path)
    plan = _approval_plan(fixture)
    gold_path = fixture.data_root / "dev" / "gold.jsonl"
    gold_path.write_bytes(gold_path.read_bytes() + b"\n")

    with pytest.raises(RuntimeError, match="freeze approval failed"):
        approve_freeze(data_root=fixture.data_root, plan=plan)

    assert not (fixture.data_root / "freeze_reports").exists()


@pytest.mark.parametrize("label_name", ["type_domain", "constraints", "overlap"])
def test_approve_rejects_private_labels_changed_after_audit(
    tmp_path: Path,
    label_name: str,
) -> None:
    fixture = _prepared_tree(tmp_path)
    plan = _approval_plan(fixture)
    label_path = _label_path(fixture, label_name)
    label_path.write_bytes(label_path.read_bytes() + b"\n")

    with pytest.raises(RuntimeError, match="freeze approval failed"):
        approve_freeze(data_root=fixture.data_root, plan=plan)

    assert not (fixture.data_root / "freeze_reports").exists()


def test_approve_rejects_evidence_mutation_at_pre_replace_boundary(
    tmp_path: Path,
) -> None:
    fixture = _prepared_tree(tmp_path)
    plan = _approval_plan(fixture)

    with pytest.raises(RuntimeError, match="freeze approval failed"):
        approve_freeze(
            data_root=fixture.data_root,
            plan=plan,
            before_manifest_replace=lambda: fixture.overlap_labels.write_bytes(b"changed"),
        )

    assert (fixture.data_root / "manifest.json").read_bytes() == (plan.prepared_manifest_bytes)
    assert not list(fixture.data_root.rglob("*.tmp"))


def test_cli_approve_is_idempotent_across_invocations(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = _prepared_tree(tmp_path)
    report_path = fixture.data_root / "freeze_reports" / "cli-freeze.json"
    args = [*_cli_args(fixture), "--approve", "--report", str(report_path)]

    assert _run_cli(args) == 0
    first = capsys.readouterr()
    assert first.err == ""
    assert _run_cli(args) == 0
    second = capsys.readouterr()
    assert second.err == ""
    assert json.loads(second.out)["approval_requested"] is True


def test_cli_redacts_unknown_argument(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = _prepared_tree(tmp_path)
    sentinel = "PRIVATE-ARG-SENTINEL"

    exit_code = _run_cli([*_cli_args(fixture), f"--{sentinel}"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert captured.err.rstrip() == "freeze approval failed"
    assert sentinel not in captured.err


def test_approve_rejects_existing_freeze_lock(tmp_path: Path) -> None:
    fixture = _prepared_tree(tmp_path)
    plan = _approval_plan(fixture)
    lock_path = fixture.data_root / ".task2-freeze.lock"
    lock_path.write_text("busy", encoding="utf-8")

    with pytest.raises(RuntimeError, match="freeze approval failed"):
        approve_freeze(data_root=fixture.data_root, plan=plan)

    assert not (fixture.data_root / "freeze_reports").exists()
    assert lock_path.read_text(encoding="utf-8") == "busy"


def test_cli_approve_accepts_relative_data_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = _prepared_tree(tmp_path)
    monkeypatch.chdir(tmp_path)
    args = _cli_args(fixture)
    args[args.index(str(fixture.data_root))] = "data"
    report_path = Path("data/freeze_reports/relative-freeze.json")

    exit_code = _run_cli([*args, "--approve", "--report", str(report_path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    assert report_path.is_file()


def test_production_main_rejects_expectation_override(tmp_path: Path) -> None:
    fixture = _prepared_tree(tmp_path)

    with pytest.raises(TypeError):
        main(_cli_args(fixture), expectations=OFFICIAL_EXPECTATIONS)


def test_approve_never_writes_through_swapped_report_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _prepared_tree(tmp_path)
    plan = _approval_plan(fixture)
    report_root = fixture.data_root / "freeze_reports"
    moved_root = fixture.data_root / "freeze_reports-original"
    outside_root = tmp_path / "outside"
    report_root.mkdir()
    outside_root.mkdir()

    def swap_then_write(path: Path, content: bytes) -> None:
        report_root.rename(moved_root)
        (outside_root / path.name).write_bytes(content)

    monkeypatch.setattr(freeze_module, "write_frozen_bytes", swap_then_write)
    with pytest.raises((OSError, RuntimeError)):
        approve_freeze(data_root=fixture.data_root, plan=plan)

    assert not (outside_root / "synthetic-freeze.json").exists()


def test_approve_never_overwrites_manifest_inserted_at_publish_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _prepared_tree(tmp_path)
    plan = _approval_plan(fixture)
    manifest_path = fixture.data_root / "manifest.json"
    boundary_bytes = b"changed-at-publish-boundary"
    original_replace = freeze_module.os.replace
    original_link = freeze_module.os.link
    injected = False

    def inject() -> None:
        nonlocal injected
        if not injected:
            manifest_path.write_bytes(boundary_bytes)
            injected = True

    def replace(source: object, target: object) -> None:
        if Path(target) == manifest_path:
            inject()
        original_replace(source, target)

    def link(source: object, target: object) -> None:
        if Path(target) == manifest_path:
            inject()
        original_link(source, target)

    monkeypatch.setattr(freeze_module.os, "replace", replace)
    monkeypatch.setattr(freeze_module.os, "link", link)
    with pytest.raises(RuntimeError, match="freeze approval failed"):
        approve_freeze(data_root=fixture.data_root, plan=plan)

    assert manifest_path.read_bytes() == boundary_bytes


def test_approve_locks_evidence_through_manifest_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _prepared_tree(tmp_path)
    plan = _approval_plan(fixture)
    manifest_path = fixture.data_root / "manifest.json"
    original_link = freeze_module.os.link

    def mutate_then_link(source: object, target: object) -> None:
        if Path(target) == manifest_path:
            fixture.overlap_labels.write_bytes(b"changed-at-publish-boundary")
        original_link(source, target)

    monkeypatch.setattr(freeze_module.os, "link", mutate_then_link)
    with pytest.raises(RuntimeError, match="freeze approval failed"):
        approve_freeze(data_root=fixture.data_root, plan=plan)

    assert manifest_path.read_bytes() == plan.prepared_manifest_bytes


def test_audit_binds_agreement_threshold_to_point_eight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _prepared_tree(tmp_path)
    original_compare = freeze_module.compare_annotations

    def lowered_threshold(*args: object, **kwargs: object) -> object:
        report = original_compare(*args, **kwargs)
        fields = {
            name: field.model_copy(update={"threshold": 0.7, "accepted": True})
            for name, field in report.fields.items()
        }
        return report.model_copy(update={"fields": fields})

    monkeypatch.setattr(freeze_module, "compare_annotations", lowered_threshold)
    with pytest.raises(ValueError, match="human annotation agreement is below threshold"):
        _audit(fixture)


def test_backup_cleanup_failure_does_not_misreport_committed_freeze(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _prepared_tree(tmp_path)
    plan = _approval_plan(fixture)
    original_unlink = Path.unlink

    def fail_backup_unlink(
        path: Path,
        missing_ok: bool = False,
    ) -> None:
        if ".manifest.json.prepared." in path.name:
            raise OSError("synthetic backup cleanup failure")
        original_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", fail_backup_unlink)

    assert approve_freeze(data_root=fixture.data_root, plan=plan) == "created"
    assert (fixture.data_root / "manifest.json").read_bytes() == plan.frozen_manifest_bytes
    monkeypatch.setattr(Path, "unlink", original_unlink)
    assert approve_freeze(data_root=fixture.data_root, plan=plan) == "matched"
    remaining = list(fixture.data_root.glob(".manifest.json.prepared.*.tmp"))
    assert (not remaining) if os.name == "nt" else len(remaining) == 1


def test_frozen_temporary_cleanup_failure_is_retried_on_matched_freeze(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _prepared_tree(tmp_path)
    plan = _approval_plan(fixture)
    original_unlink = Path.unlink

    def fail_frozen_temporary_unlink(
        path: Path,
        missing_ok: bool = False,
    ) -> None:
        if (
            path.name.startswith(".manifest.json.")
            and ".manifest.json.prepared." not in path.name
        ):
            raise OSError("synthetic frozen temporary cleanup failure")
        original_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", fail_frozen_temporary_unlink)

    assert approve_freeze(data_root=fixture.data_root, plan=plan) == "created"
    assert (fixture.data_root / "manifest.json").read_bytes() == plan.frozen_manifest_bytes
    monkeypatch.setattr(Path, "unlink", original_unlink)
    frozen_temporaries = [
        candidate
        for candidate in fixture.data_root.glob(".manifest.json.*.tmp")
        if ".manifest.json.prepared." not in candidate.name
    ]
    assert len(frozen_temporaries) == 1
    assert frozen_temporaries[0].samefile(fixture.data_root / "manifest.json")

    assert approve_freeze(data_root=fixture.data_root, plan=plan) == "matched"
    remaining = [
        candidate
        for candidate in fixture.data_root.glob(".manifest.json.*.tmp")
        if ".manifest.json.prepared." not in candidate.name
    ]
    assert (not remaining) if os.name == "nt" else len(remaining) == 1


def test_matched_freeze_preserves_nonmatching_manifest_temporary(
    tmp_path: Path,
) -> None:
    fixture = _prepared_tree(tmp_path)
    plan = _approval_plan(fixture)
    assert approve_freeze(data_root=fixture.data_root, plan=plan) == "created"
    unrelated = fixture.data_root / ".manifest.json.unrelated.tmp"
    unrelated.write_bytes(b"not the frozen manifest")

    assert approve_freeze(data_root=fixture.data_root, plan=plan) == "matched"
    assert unrelated.read_bytes() == b"not the frozen manifest"


@pytest.mark.skipif(os.name != "nt", reason="Windows handle sharing semantics")
def test_matched_freeze_does_not_delete_candidate_replaced_after_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _prepared_tree(tmp_path)
    plan = _approval_plan(fixture)
    assert approve_freeze(data_root=fixture.data_root, plan=plan) == "created"
    candidate = fixture.data_root / ".manifest.json.race.tmp"
    candidate.write_bytes(plan.frozen_manifest_bytes)
    replacement = fixture.data_root / "replacement.tmp"
    replacement.write_bytes(b"replacement must survive")
    expected_sha256 = freeze_module._sha256_bytes(plan.frozen_manifest_bytes)
    original_sha256_bytes = freeze_module._sha256_bytes
    original_sha256_descriptor = freeze_module._sha256_descriptor
    frozen_byte_hashes = 0
    replacement_attempted = False

    def attempt_replacement() -> None:
        nonlocal replacement_attempted
        replacement_attempted = True
        try:
            os.replace(replacement, candidate)
        except OSError:
            pass

    def sha256_bytes_with_replacement(content: bytes) -> str:
        nonlocal frozen_byte_hashes
        result = original_sha256_bytes(content)
        if content == plan.frozen_manifest_bytes:
            frozen_byte_hashes += 1
            if frozen_byte_hashes == 2:
                attempt_replacement()
        return result

    def sha256_descriptor_with_replacement(descriptor: int) -> str:
        result = original_sha256_descriptor(descriptor)
        if result == expected_sha256 and not replacement_attempted:
            attempt_replacement()
        return result

    monkeypatch.setattr(freeze_module, "_sha256_bytes", sha256_bytes_with_replacement)
    monkeypatch.setattr(
        freeze_module,
        "_sha256_descriptor",
        sha256_descriptor_with_replacement,
    )

    assert approve_freeze(data_root=fixture.data_root, plan=plan) == "matched"
    assert replacement_attempted
    assert replacement.read_bytes() == b"replacement must survive"
    assert not candidate.exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows handle cleanup semantics")
def test_matched_freeze_ignores_cleanup_descriptor_close_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _prepared_tree(tmp_path)
    plan = _approval_plan(fixture)
    assert approve_freeze(data_root=fixture.data_root, plan=plan) == "created"
    candidate = fixture.data_root / ".manifest.json.close-failure.tmp"
    candidate.write_bytes(plan.frozen_manifest_bytes)
    original_close = freeze_module.os.close

    def close_then_fail(descriptor: int) -> None:
        original_close(descriptor)
        raise OSError("synthetic descriptor close failure")

    monkeypatch.setattr(freeze_module.os, "close", close_then_fail)

    assert approve_freeze(data_root=fixture.data_root, plan=plan) == "matched"
    assert not candidate.exists()


def test_matched_freeze_preserves_cross_category_manifest_hashes(
    tmp_path: Path,
) -> None:
    fixture = _prepared_tree(tmp_path)
    plan = _approval_plan(fixture)
    assert approve_freeze(data_root=fixture.data_root, plan=plan) == "created"
    prepared_name_with_frozen_bytes = (
        fixture.data_root / ".manifest.json.prepared.cross.tmp"
    )
    frozen_name_with_prepared_bytes = fixture.data_root / ".manifest.json.cross.tmp"
    prepared_name_with_frozen_bytes.write_bytes(plan.frozen_manifest_bytes)
    frozen_name_with_prepared_bytes.write_bytes(plan.prepared_manifest_bytes)

    assert approve_freeze(data_root=fixture.data_root, plan=plan) == "matched"
    assert prepared_name_with_frozen_bytes.read_bytes() == plan.frozen_manifest_bytes
    assert frozen_name_with_prepared_bytes.read_bytes() == plan.prepared_manifest_bytes


def test_matched_freeze_rejects_oversized_temporary_before_hashing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _prepared_tree(tmp_path)
    plan = _approval_plan(fixture)
    assert approve_freeze(data_root=fixture.data_root, plan=plan) == "created"
    oversized = fixture.data_root / ".manifest.json.oversized.tmp"
    oversized.write_bytes(b"x" * (1024 * 1024 + 1))

    def fail_hash(_: int) -> str:
        raise AssertionError("oversized temporary must not be hashed")

    monkeypatch.setattr(freeze_module, "_sha256_descriptor", fail_hash)
    assert approve_freeze(data_root=fixture.data_root, plan=plan) == "matched"
    assert oversized.stat().st_size == 1024 * 1024 + 1


if os.name == "nt":

    def test_matched_freeze_preserves_reparse_candidate(tmp_path: Path) -> None:
        fixture = _prepared_tree(tmp_path)
        plan = _approval_plan(fixture)
        assert approve_freeze(data_root=fixture.data_root, plan=plan) == "created"
        target = tmp_path / "outside-directory"
        target.mkdir()
        marker = target / "manifest.json"
        marker.write_bytes(plan.frozen_manifest_bytes)
        candidate = fixture.data_root / ".manifest.json.reparse.tmp"
        created = subprocess.run(
            ["cmd.exe", "/d", "/c", "mklink", "/J", str(candidate), str(target)],
            capture_output=True,
            check=False,
        )
        assert created.returncode == 0
        try:
            assert approve_freeze(data_root=fixture.data_root, plan=plan) == "matched"
            assert candidate.is_dir()
            assert marker.read_bytes() == plan.frozen_manifest_bytes
        finally:
            candidate.rmdir()


def test_failed_publish_and_restore_preserve_prepared_recovery_backup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _prepared_tree(tmp_path)
    plan = _approval_plan(fixture)
    manifest_path = fixture.data_root / "manifest.json"
    original_link = freeze_module.os.link

    def fail_manifest_links(source: object, target: object) -> None:
        if Path(target) == manifest_path:
            raise OSError("synthetic manifest link failure")
        original_link(source, target)

    monkeypatch.setattr(freeze_module.os, "link", fail_manifest_links)
    with pytest.raises(RuntimeError, match="freeze approval failed"):
        approve_freeze(data_root=fixture.data_root, plan=plan)

    recovery = list(fixture.data_root.glob(".manifest.json.prepared.*.tmp"))
    assert not manifest_path.exists()
    assert len(recovery) == 1
    assert recovery[0].read_bytes() == plan.prepared_manifest_bytes
