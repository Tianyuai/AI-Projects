from __future__ import annotations

import hashlib
import json
import os
import shutil
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

import paper_search.evaluation.freeze_schema as freeze_schema
from paper_search.evaluation.gate0 import verify_gate0
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


def _write_v2_manifest(data_root: Path, manifest: dict[str, object]) -> None:
    (data_root / "manifest.json").write_bytes(_canonical_document(manifest))


def _v2_report(
    data_root: Path,
    manifest: dict[str, object],
) -> tuple[Path, dict[str, object]]:
    approval_binding = manifest["approval"]
    assert isinstance(approval_binding, dict)
    report_path = data_root / str(approval_binding["report_path"])
    report = json.loads(report_path.read_bytes())
    assert isinstance(report, dict)
    return report_path, report


def _bind_v2_report(
    data_root: Path,
    manifest: dict[str, object],
    report: dict[str, object],
    *,
    content: bytes | None = None,
) -> None:
    report_path, _ = _v2_report(data_root, manifest)
    report_bytes = _canonical_document(report) if content is None else content
    report_path.write_bytes(report_bytes)
    approval_binding = manifest["approval"]
    assert isinstance(approval_binding, dict)
    approval_binding["report_sha256"] = _sha256(report_bytes)


def _bind_v2_partition(
    data_root: Path,
    manifest: dict[str, object],
    name: str,
    content: bytes,
    *,
    query_count: int,
) -> None:
    partitions = manifest["partitions"]
    assert isinstance(partitions, list)
    partition = next(
        item for item in partitions if isinstance(item, dict) and item["name"] == name
    )
    partition_path = data_root / str(partition["path"])
    partition_path.write_bytes(content)
    partition_hash = _sha256(content)
    partition["sha256"] = partition_hash
    partition["query_count"] = query_count
    _, report = _v2_report(data_root, manifest)
    partition_hashes = report["partition_hashes"]
    assert isinstance(partition_hashes, dict)
    partition_hashes[name] = partition_hash
    _bind_v2_report(data_root, manifest, report)
    hashes = {
        str(item["name"]): str(item["sha256"])
        for item in partitions
        if isinstance(item, dict)
    }
    manifest["gold_sha256"] = canonical_gold_set_sha256(
        hashes["dev"], hashes["validation"]
    )


def _bind_v2_identifier_map(
    data_root: Path,
    manifest: dict[str, object],
    content: bytes,
    *,
    entry_count: int,
) -> None:
    binding = manifest["identifier_map"]
    assert isinstance(binding, dict)
    (data_root / str(binding["path"])).write_bytes(content)
    identity = _sha256(content)
    binding["sha256"] = identity
    binding["entry_count"] = entry_count
    _, report = _v2_report(data_root, manifest)
    report["identifier_map_sha256"] = identity
    _bind_v2_report(data_root, manifest, report)


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


@pytest.fixture
def readable_v1_fixture(tmp_path: Path) -> tuple[Path, Path]:
    data_root = tmp_path / "readable-v1"
    data_root.mkdir()
    manifest_path = data_root / "manifest.json"
    manifest_path.write_bytes(_canonical_document(_legacy_manifest()))
    return data_root, manifest_path


def test_readable_v1_fixture_loads_but_gate0_rejects(
    readable_v1_fixture: tuple[Path, Path],
) -> None:
    data_root, manifest_path = readable_v1_fixture

    manifest = load_freeze_manifest(manifest_path, data_root=data_root)
    report = verify_gate0(
        data_root=data_root,
        manifest_path=manifest_path,
        pricing_policy_path=data_root / "missing-pricing.yaml",
        quality_gates_path=data_root / "missing-quality.yaml",
        readiness_report_path=data_root / "missing-readiness.json",
        clock=lambda: datetime(2026, 7, 30, tzinfo=UTC),
    )

    assert isinstance(manifest, FreezeManifestV1)
    assert report.passed is False
    assert "manifest_invalid" in report.blocking_reasons


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


def test_bound_evidence_context_retains_exact_v2_artifacts_and_partition_rows(
    tmp_path: Path,
) -> None:
    data_root, _ = _write_bound_v2_tree(tmp_path)
    manifest_bytes = (data_root / "manifest.json").read_bytes()

    with freeze_schema.open_validated_freeze_evidence(
        data_root / "manifest.json",
        data_root=data_root,
    ) as evidence:
        assert isinstance(evidence.manifest, FreezeManifestV2)
        assert evidence.manifest_artifact.content == manifest_bytes
        assert evidence.approval_artifact is not None
        assert evidence.approval_artifact.relative_path == "freeze_reports/approval.json"
        assert [artifact.relative_path for artifact in evidence.partition_artifacts] == [
            "dev/gold.jsonl",
            "validation/gold.jsonl",
        ]
        assert evidence.identifier_map_artifact is not None
        assert evidence.identifier_map_artifact.relative_path == "identifier-map.json"
        assert evidence.partition_rows == (
            (
                "dev",
                (
                    {
                        "query_id": "dev-1",
                        "query": "synthetic",
                        "relevant_paper_ids": ["p1"],
                    },
                ),
            ),
            (
                "validation",
                (
                    {
                        "query_id": "validation-1",
                        "query": "synthetic",
                        "relevant_paper_ids": [],
                    },
                ),
            ),
        )
        for artifact in evidence.artifacts:
            os.fstat(artifact.descriptor)


@pytest.mark.parametrize(
    ("mutation", "expected_reason"),
    [
        ("approval_missing", "approval_invalid"),
        ("partition_hash", "partition_hash_mismatch"),
        ("partition_count", "partition_count_mismatch"),
        ("identifier_missing", "identifier_map_missing"),
        ("identifier_hash", "identifier_map_hash_mismatch"),
    ],
)
def test_bound_evidence_reports_structured_validation_reason(
    tmp_path: Path,
    mutation: str,
    expected_reason: str,
) -> None:
    data_root, manifest = _write_bound_v2_tree(tmp_path)
    partitions = manifest["partitions"]
    identifier_map = manifest["identifier_map"]
    assert isinstance(partitions, list)
    assert isinstance(identifier_map, dict)
    dev = next(
        item for item in partitions if isinstance(item, dict) and item["name"] == "dev"
    )

    if mutation == "approval_missing":
        (data_root / "freeze_reports" / "approval.json").unlink()
    elif mutation == "partition_hash":
        dev["sha256"] = _sha256(b"different-partition")
        _write_v2_manifest(data_root, manifest)
    elif mutation == "partition_count":
        dev["query_count"] = 2
        _write_v2_manifest(data_root, manifest)
    elif mutation == "identifier_missing":
        (data_root / str(identifier_map["path"])).unlink()
    else:
        (data_root / str(identifier_map["path"])).write_bytes(
            b'{"legacy:2":"canonical:2"}'
        )

    with pytest.raises(ValueError) as error:
        with freeze_schema.open_validated_freeze_evidence(
            data_root / "manifest.json",
            data_root=data_root,
        ):
            pass

    assert getattr(error.value, "reason", None) == expected_reason
    assert str(error.value) == "freeze manifest is invalid"


@pytest.mark.parametrize(
    ("relative_path", "expected_reason"),
    [
        ("manifest.json", "manifest_invalid"),
        ("freeze_reports/approval.json", "approval_invalid"),
        ("dev/gold.jsonl", "partition_hash_mismatch"),
        ("identifier-map.json", "identifier_map_hash_mismatch"),
    ],
)
@pytest.mark.parametrize("error_type", [ValueError, OSError, RuntimeError])
def test_bound_evidence_exit_identity_failures_preserve_artifact_reason(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative_path: str,
    expected_reason: str,
    error_type: type[Exception],
) -> None:
    data_root, _ = _write_bound_v2_tree(tmp_path)
    original = freeze_schema.BoundArtifact.verify_path_identity

    def fail_selected(artifact: freeze_schema.BoundArtifact) -> None:
        if artifact.relative_path == relative_path:
            raise error_type("PRIVATE_PATH_SENTINEL")
        original(artifact)

    monkeypatch.setattr(
        freeze_schema.BoundArtifact,
        "verify_path_identity",
        fail_selected,
    )

    with pytest.raises(freeze_schema.FreezeEvidenceError) as error:
        with freeze_schema.open_validated_freeze_evidence(
            data_root / "manifest.json",
            data_root=data_root,
        ):
            pass

    assert error.value.reasons == (expected_reason,)
    assert error.value.reason == expected_reason
    assert str(error.value) == "freeze manifest is invalid"
    assert "PRIVATE_PATH_SENTINEL" not in str(error.value)


def test_bound_evidence_exit_collects_all_artifact_identity_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root, _ = _write_bound_v2_tree(tmp_path)
    original = freeze_schema.BoundArtifact.verify_path_identity

    def fail_selected(artifact: freeze_schema.BoundArtifact) -> None:
        if artifact.relative_path in {
            "freeze_reports/approval.json",
            "identifier-map.json",
        }:
            raise ValueError("PRIVATE_PATH_SENTINEL")
        original(artifact)

    monkeypatch.setattr(
        freeze_schema.BoundArtifact,
        "verify_path_identity",
        fail_selected,
    )

    with pytest.raises(freeze_schema.FreezeEvidenceError) as error:
        with freeze_schema.open_validated_freeze_evidence(
            data_root / "manifest.json",
            data_root=data_root,
        ):
            pass

    assert error.value.reasons == (
        "approval_invalid",
        "identifier_map_hash_mismatch",
    )


def test_public_manifest_loader_delegates_to_bound_evidence_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root, _ = _write_bound_v2_tree(tmp_path)
    original = freeze_schema.open_validated_freeze_evidence
    entered = False
    exited = False

    @contextmanager
    def record_delegation(
        path: Path,
        *,
        data_root: Path,
    ) -> object:
        nonlocal entered, exited
        with original(path, data_root=data_root) as evidence:
            entered = True
            yield evidence
        exited = True

    monkeypatch.setattr(
        freeze_schema,
        "open_validated_freeze_evidence",
        record_delegation,
    )

    manifest = load_freeze_manifest(data_root / "manifest.json", data_root=data_root)

    assert isinstance(manifest, FreezeManifestV2)
    assert entered is True
    assert exited is True


def test_public_manifest_loader_accepts_cli_style_relative_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_bound_v2_tree(tmp_path)
    monkeypatch.chdir(tmp_path)

    manifest = load_freeze_manifest(
        Path("data/manifest.json"),
        data_root=Path("data"),
    )

    assert isinstance(manifest, FreezeManifestV2)


@pytest.mark.parametrize("method_name", ["resolve", "absolute"])
@pytest.mark.parametrize("error_type", [OSError, RuntimeError])
def test_bound_evidence_wraps_path_normalization_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    method_name: str,
    error_type: type[Exception],
) -> None:
    data_root, _ = _write_bound_v2_tree(tmp_path)
    manifest_path = data_root / "manifest.json"
    target = data_root if method_name == "resolve" else manifest_path
    original = getattr(Path, method_name)

    def fail_target(candidate: Path, *args: object, **kwargs: object) -> Path:
        if candidate == target:
            raise error_type("PRIVATE_PATH_SENTINEL")
        return original(candidate, *args, **kwargs)

    monkeypatch.setattr(Path, method_name, fail_target)

    with pytest.raises(ValueError) as error:
        with freeze_schema.open_validated_freeze_evidence(
            manifest_path,
            data_root=data_root,
        ):
            pass

    assert getattr(error.value, "reason", None) == "manifest_invalid"
    assert str(error.value) == "freeze manifest is invalid"
    assert "PRIVATE_PATH_SENTINEL" not in str(error.value)


@pytest.mark.parametrize(
    "mutation",
    [
        "report_noncanonical",
        "report_hash",
        "report_timestamp_binding",
        "report_approver",
        "report_partition_binding",
        "report_identifier_binding",
        "partition_hash",
        "partition_count",
        "partition_blank",
        "partition_duplicate",
        "partition_order",
        "identifier_hash",
        "identifier_count",
        "identifier_duplicate_key",
        "identifier_invalid_key",
        "identifier_invalid_value",
        "gold_mismatch",
    ],
)
def test_v2_loader_rejects_complete_binding_mutation_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    data_root, manifest = _write_bound_v2_tree(tmp_path)
    _, report = _v2_report(data_root, manifest)
    partitions = manifest["partitions"]
    identifier_map = manifest["identifier_map"]
    approval_binding = manifest["approval"]
    assert isinstance(partitions, list)
    assert isinstance(identifier_map, dict)
    assert isinstance(approval_binding, dict)
    dev = next(
        item for item in partitions if isinstance(item, dict) and item["name"] == "dev"
    )
    report_validated = False
    original_validate_report = freeze_schema._validated_report_bytes

    def record_validated_report(content: bytes) -> object:
        nonlocal report_validated
        validated = original_validate_report(content)
        report_validated = True
        return validated

    monkeypatch.setattr(
        freeze_schema,
        "_validated_report_bytes",
        record_validated_report,
    )

    if mutation == "report_noncanonical":
        _bind_v2_report(
            data_root,
            manifest,
            report,
            content=json.dumps(report, sort_keys=True, separators=(",", ":")).encode(),
        )
    elif mutation == "report_hash":
        approval_binding["report_sha256"] = _sha256(b"different-report")
    elif mutation == "report_timestamp_binding":
        report["approved_at"] = "2026-07-30T00:00:01Z"
        _bind_v2_report(data_root, manifest, report)
    elif mutation == "report_approver":
        report["approver_ref"] = "different-operator"
        _bind_v2_report(data_root, manifest, report)
    elif mutation == "report_partition_binding":
        hashes = report["partition_hashes"]
        assert isinstance(hashes, dict)
        hashes["dev"] = _sha256(b"different-partition")
        _bind_v2_report(data_root, manifest, report)
    elif mutation == "report_identifier_binding":
        report["identifier_map_sha256"] = _sha256(b"different-map")
        _bind_v2_report(data_root, manifest, report)
    elif mutation == "partition_hash":
        dev["sha256"] = _sha256(b"different-partition")
    elif mutation == "partition_count":
        dev["query_count"] = 2
    elif mutation == "partition_blank":
        _bind_v2_partition(
            data_root,
            manifest,
            "dev",
            (data_root / str(dev["path"])).read_bytes() + b"\n",
            query_count=1,
        )
    elif mutation == "partition_duplicate":
        row = (data_root / str(dev["path"])).read_bytes()
        _bind_v2_partition(
            data_root,
            manifest,
            "dev",
            row + row,
            query_count=2,
        )
    elif mutation == "partition_order":
        partitions.reverse()
    elif mutation == "identifier_hash":
        identifier_map["sha256"] = _sha256(b"different-map")
    elif mutation == "identifier_count":
        identifier_map["entry_count"] = 2
    elif mutation == "identifier_duplicate_key":
        _bind_v2_identifier_map(
            data_root,
            manifest,
            b'{"legacy:1":"canonical:1","legacy:1":"canonical:2"}',
            entry_count=1,
        )
    elif mutation == "identifier_invalid_key":
        _bind_v2_identifier_map(
            data_root,
            manifest,
            b'{"":"canonical:1"}',
            entry_count=1,
        )
    elif mutation == "identifier_invalid_value":
        _bind_v2_identifier_map(
            data_root,
            manifest,
            b'{"legacy:1":""}',
            entry_count=1,
        )
    else:
        manifest["gold_sha256"] = _sha256(b"different-gold-set")

    _write_v2_manifest(data_root, manifest)

    with pytest.raises(ValueError, match="freeze manifest is invalid"):
        load_freeze_manifest(data_root / "manifest.json", data_root=data_root)
    if mutation in {
        "report_timestamp_binding",
        "report_approver",
        "report_partition_binding",
        "report_identifier_binding",
    }:
        assert report_validated


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


@pytest.mark.skipif(os.name == "nt", reason="POSIX publisher cleanup contract")
@pytest.mark.parametrize("failure_stage", ["fstat", "write", "fsync", "close"])
def test_posix_report_publisher_cleans_owned_temp_at_earliest_failure_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    report_parent = data_root / "freeze_reports"
    original_open = freeze_schema.os.open
    original_fstat = freeze_schema.os.fstat
    original_fsync = freeze_schema.os.fsync
    original_close = freeze_schema.os.close
    owned_descriptor: int | None = None
    owned_path: Path | None = None
    concurrent_owner = report_parent / "concurrent-owner.tmp"
    concurrent_owner.write_bytes(b"concurrent-owner") if report_parent.exists() else None
    parent_fsynced = False

    def replace_owned_path() -> None:
        nonlocal owned_path
        if owned_path is None:
            candidates = list(report_parent.glob(".approval.json.*.tmp"))
            assert len(candidates) == 1
            owned_path = candidates[0]
        if not concurrent_owner.exists():
            concurrent_owner.write_bytes(b"concurrent-owner")
        os.replace(concurrent_owner, owned_path)

    def capture_open(
        path: object,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal owned_descriptor
        descriptor = original_open(path, flags, mode, dir_fd=dir_fd)
        if isinstance(path, str) and path.startswith(".approval.json.") and path.endswith(
            ".tmp"
        ):
            owned_descriptor = descriptor
        return descriptor

    def fail_fstat(descriptor: int) -> os.stat_result:
        if failure_stage == "fstat" and descriptor == owned_descriptor:
            replace_owned_path()
            raise OSError("synthetic fstat failure")
        return original_fstat(descriptor)

    def fail_write(descriptor: int, content: bytes) -> None:
        if failure_stage == "write":
            replace_owned_path()
            raise OSError("synthetic write failure")
        freeze_schema._write_descriptor(descriptor, content)

    def fail_fsync(descriptor: int) -> None:
        nonlocal parent_fsynced
        if descriptor == owned_descriptor and failure_stage == "fsync":
            replace_owned_path()
            raise OSError("synthetic fsync failure")
        if descriptor != owned_descriptor:
            parent_fsynced = True
        original_fsync(descriptor)

    def fail_close(descriptor: int) -> None:
        if descriptor == owned_descriptor and failure_stage == "close":
            replace_owned_path()
            original_close(descriptor)
            raise OSError("synthetic close failure")
        original_close(descriptor)

    monkeypatch.setattr(freeze_schema.os, "open", capture_open)
    monkeypatch.setattr(freeze_schema.os, "fstat", fail_fstat)
    if failure_stage == "write":
        monkeypatch.setattr(freeze_schema, "_write_descriptor", fail_write)
    monkeypatch.setattr(freeze_schema.os, "fsync", fail_fsync)
    monkeypatch.setattr(freeze_schema.os, "close", fail_close)

    with pytest.raises(OSError, match=f"synthetic {failure_stage} failure"):
        freeze_schema.publish_confined_bytes_no_overwrite(
            data_root,
            "freeze_reports/approval.json",
            b"complete-report",
        )

    assert not (report_parent / "approval.json").exists()
    assert owned_path is not None
    assert owned_path.read_bytes() == b"concurrent-owner"
    assert parent_fsynced


@pytest.mark.skipif(os.name == "nt", reason="POSIX initial fstat cleanup contract")
def test_posix_initial_fstat_failure_may_retain_unowned_non_authoritative_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    report_parent = data_root / "freeze_reports"
    original_open = freeze_schema.os.open
    original_fstat = freeze_schema.os.fstat
    temporary_descriptor: int | None = None

    def capture_open(
        path: object,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal temporary_descriptor
        descriptor = original_open(path, flags, mode, dir_fd=dir_fd)
        if isinstance(path, str) and path.startswith(".approval.json."):
            temporary_descriptor = descriptor
        return descriptor

    def fail_initial_fstat(descriptor: int) -> os.stat_result:
        if descriptor == temporary_descriptor:
            raise OSError("synthetic initial fstat failure")
        return original_fstat(descriptor)

    monkeypatch.setattr(freeze_schema.os, "open", capture_open)
    monkeypatch.setattr(freeze_schema.os, "fstat", fail_initial_fstat)

    with pytest.raises(OSError, match="synthetic initial fstat failure"):
        publish_confined_bytes_no_overwrite(
            data_root,
            "freeze_reports/approval.json",
            b"complete-report",
        )

    assert not (report_parent / "approval.json").exists()
    retained = list(report_parent.glob(".approval.json.*.tmp"))
    assert len(retained) == 1
    assert retained[0].read_bytes() == b""


@pytest.mark.skipif(os.name == "nt", reason="POSIX publisher durability contract")
def test_posix_report_publisher_fsyncs_parent_after_partial_temp_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    calls: list[int] = []
    original_fsync = freeze_schema.os.fsync

    def record_fsync(descriptor: int) -> None:
        calls.append(descriptor)
        original_fsync(descriptor)

    monkeypatch.setattr(freeze_schema.os, "fsync", record_fsync)

    assert (
        publish_confined_bytes_no_overwrite(
            data_root,
            "freeze_reports/approval.json",
            b"complete-report",
        )
        == "created"
    )
    assert len(calls) >= 2


@pytest.mark.skipif(os.name != "nt", reason="Windows FILE_ID publisher contract")
def test_windows_report_publisher_rejects_replaced_temp_path_and_preserves_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    report_parent = data_root / "freeze_reports"
    attacker = data_root / "attacker.tmp"
    attacker.write_bytes(b"complete-report")
    original_file_id = freeze_schema._windows_file_id
    replaced_path: Path | None = None
    replacement_done = False

    def replace_after_retaining_id(handle: int) -> tuple[int, bytes]:
        nonlocal replaced_path, replacement_done
        identity = original_file_id(handle)
        if not replacement_done:
            candidates = list(report_parent.glob(".approval.json.*.tmp"))
            assert len(candidates) == 1
            replaced_path = candidates[0]
            _native_replace_windows(attacker, replaced_path)
            replacement_done = True
        return identity

    monkeypatch.setattr(freeze_schema, "_windows_file_id", replace_after_retaining_id)

    with pytest.raises(ValueError, match="freeze manifest is invalid"):
        publish_confined_bytes_no_overwrite(
            data_root,
            "freeze_reports/approval.json",
            b"complete-report",
        )

    assert replacement_done
    assert replaced_path is not None
    assert replaced_path.read_bytes() == b"complete-report"
    assert not (report_parent / "approval.json").exists()


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
def test_replaceable_manifest_denies_native_write_but_allows_rename(
    tmp_path: Path,
) -> None:
    import ctypes

    data_root = tmp_path / "data"
    data_root.mkdir()
    manifest = data_root / "manifest.json"
    manifest.write_bytes(b"current")
    backup = data_root / "manifest.backup.json"
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.restype = ctypes.c_void_p
    invalid = ctypes.c_void_p(-1).value

    with open_confined_artifact(
        data_root, "manifest.json", replaceable_manifest=True
    ) as artifact:
        write_handle = create_file(
            str(manifest),
            0x40000000,
            0x00000001 | 0x00000002 | 0x00000004,
            None,
            3,
            0,
            None,
        )
        if write_handle != invalid:
            kernel32.CloseHandle(ctypes.c_void_p(write_handle))
        assert write_handle == invalid
        assert ctypes.get_last_error() == 32
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


@pytest.mark.skipif(os.name == "nt", reason="POSIX descriptor ownership")
def test_posix_bound_artifacts_own_ancestor_descriptors_until_closed(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    (data_root / "one").mkdir()
    (data_root / "two").mkdir()
    (data_root / "one" / "evidence.json").write_bytes(b"one")
    (data_root / "two" / "evidence.json").write_bytes(b"two")

    with open_confined_artifact(data_root, "one/evidence.json") as first:
        with open_confined_artifact(data_root, "two/evidence.json") as second:
            assert first.ancestor_descriptors
            assert second.ancestor_descriptors
            for descriptor in [*first.ancestor_descriptors, *second.ancestor_descriptors]:
                os.fstat(descriptor)
            first.close()
            os.fstat(second.descriptor)
            for descriptor in second.ancestor_descriptors:
                os.fstat(descriptor)
            second.close()

    reused = os.open(data_root / "one" / "evidence.json", os.O_RDONLY)
    try:
        first.close()
        second.close()
        os.fstat(reused)
    finally:
        os.close(reused)


@pytest.mark.skipif(os.name != "nt", reason="Windows FILE_ID_INFO")
def test_windows_file_id_rejects_native_replacement_of_current_path(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    manifest = data_root / "manifest.json"
    manifest.write_bytes(b"current")
    replacement = data_root / "replacement.tmp"
    replacement.write_bytes(b"replacement")
    with open_confined_artifact(
        data_root, "manifest.json", replaceable_manifest=True
    ) as artifact:
        _native_replace_windows(replacement, manifest)
        with pytest.raises(ValueError, match="freeze manifest is invalid"):
            artifact.verify_path_identity()
