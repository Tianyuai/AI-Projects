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


def _manifest_recoveries_with_bytes(root: Path, content: bytes) -> list[Path]:
    return [
        candidate
        for candidate in root.glob(".manifest.json.*")
        if candidate.is_file() and candidate.read_bytes() == content
    ]


def _leave_exact_v1_recovery(
    case: MigrationFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    root = case.prepared.data_root
    manifest_path = root / "manifest.json"
    original_link = freeze_module.os.link

    def fail_manifest_publish(
        source: object,
        target: object,
        *args: object,
        **kwargs: object,
    ) -> None:
        source_path = Path(source)
        if (
            Path(target) == manifest_path
            and source_path.name.startswith(".manifest.json.")
            and ".prepared." not in source_path.name
        ):
            raise OSError("synthetic manifest publish failure")
        original_link(source, target, *args, **kwargs)

    monkeypatch.setattr(freeze_module.os, "link", fail_manifest_publish)
    with pytest.raises(RuntimeError, match="freeze approval failed"):
        _migrate(case)
    monkeypatch.setattr(freeze_module.os, "link", original_link)

    assert not manifest_path.exists()
    recoveries = [
        path
        for path in _manifest_recoveries_with_bytes(root, case.legacy_manifest_bytes)
        if path.name.startswith(".manifest.json.prepared.")
    ]
    assert len(recoveries) == 1
    return recoveries[0]


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


@pytest.mark.skipif(os.name == "nt", reason="POSIX permits held-path mutation")
@pytest.mark.parametrize("artifact", ["manifest", "report", "partition", "identifier_map"])
@pytest.mark.parametrize("mutation", ["rewrite", "replace"])
def test_idempotent_migration_rejects_mutation_between_stable_pre_and_post_checks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
    original_stable = freeze_module._stable_evidence_files
    injected = False

    @contextmanager
    def inject_after_precheck(
        evidence: object,
    ) -> Iterator[None]:
        nonlocal injected
        with original_stable(evidence):
            yield
            if mutation == "rewrite":
                target.write_bytes(b"post-check-mutation")
            else:
                replacement = target.with_name(f"{target.name}.post-check")
                replacement.write_bytes(target.read_bytes())
                os.replace(replacement, target)
            injected = True

    monkeypatch.setattr(
        freeze_module,
        "_stable_evidence_files",
        inject_after_precheck,
    )

    with pytest.raises(RuntimeError, match="freeze approval failed"):
        _migrate(case)
    assert injected


@pytest.mark.skipif(os.name != "nt", reason="Windows FILE_ID post-check contract")
def test_windows_idempotent_migration_rejects_native_manifest_replace_after_precheck(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _migration_fixture(tmp_path)
    _migrate(case)
    root = case.prepared.data_root
    manifest_path = root / "manifest.json"
    original_stable = freeze_module._stable_evidence_files
    injected = False

    @contextmanager
    def inject_after_precheck(
        evidence: object,
    ) -> Iterator[None]:
        nonlocal injected
        with original_stable(evidence):
            yield
            replacement = root / "manifest.post-check.json"
            replacement.write_bytes(manifest_path.read_bytes())
            _native_replace_windows(replacement, manifest_path)
            injected = True

    monkeypatch.setattr(
        freeze_module,
        "_stable_evidence_files",
        inject_after_precheck,
    )

    with pytest.raises(RuntimeError, match="freeze approval failed"):
        _migrate(case)
    assert injected


def test_idempotent_migration_rejects_rebound_approval_audit(
    tmp_path: Path,
) -> None:
    case = _migration_fixture(tmp_path)
    _migrate(case)
    root = case.prepared.data_root
    report_path = root / case.approval_report_path
    report = json.loads(report_path.read_bytes())
    assert isinstance(report, dict)
    report["audit_sha256"] = _sha256(b"different-legacy-audit")
    report_bytes = freeze_module._json_bytes(report)
    report_path.write_bytes(report_bytes)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    assert isinstance(manifest, dict)
    approval_binding = manifest["approval"]
    assert isinstance(approval_binding, dict)
    approval_binding["report_sha256"] = _sha256(report_bytes)
    manifest_path.write_bytes(freeze_module._json_bytes(manifest))

    assert isinstance(
        load_freeze_manifest(manifest_path, data_root=root),
        FreezeManifestV2,
    )
    with pytest.raises(ValueError, match="freeze migration failed"):
        _migrate(case)


def test_v1_migration_rejects_wrong_approval_audit_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _migration_fixture(tmp_path)
    wrong_approval = case.approval.model_copy(
        update={"audit_sha256": _sha256(b"different-legacy-audit")}
    )
    migration_manifest_called = False
    original_migration_manifest = freeze_module._migration_manifest

    def record_migration_manifest(*args: object, **kwargs: object) -> object:
        nonlocal migration_manifest_called
        migration_manifest_called = True
        return original_migration_manifest(*args, **kwargs)

    monkeypatch.setattr(
        freeze_module,
        "_migration_manifest",
        record_migration_manifest,
    )

    with pytest.raises(ValueError, match="freeze migration failed"):
        migrate_v1_to_v2(
            case.legacy,
            data_root=case.prepared.data_root,
            approval=wrong_approval,
            identifier_map=case.identifier_map,
            dataset_revision=case.dataset_revision,
            approval_report_path=case.approval_report_path,
        )

    assert not migration_manifest_called
    assert (
        case.prepared.data_root / "manifest.json"
    ).read_bytes() == case.legacy_manifest_bytes
    assert not (case.prepared.data_root / case.approval_report_path).exists()


@pytest.mark.parametrize("report_state", ["created", "matched"])
@pytest.mark.parametrize("failure_boundary", ["after_backup", "before_final_link"])
def test_migration_guarded_failure_preserves_v1_recovery_for_operator_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    report_state: str,
    failure_boundary: str,
) -> None:
    case = _migration_fixture(tmp_path)
    root = case.prepared.data_root
    manifest_path = root / "manifest.json"
    legacy_state = freeze_module._probe_file_state(manifest_path)
    assert legacy_state is not None and legacy_state.regular
    report_path = root / case.approval_report_path
    expected_report = freeze_module._json_bytes(
        case.approval.model_dump(mode="json")
    )
    if report_state == "matched":
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_bytes(expected_report)
    original_rename_no_overwrite = freeze_module._rename_no_overwrite
    original_link = freeze_module.os.link

    def fail_after_backup(source: Path, target: Path) -> str:
        outcome = original_rename_no_overwrite(source, target)
        if (
            source == manifest_path
            and target.name.startswith(".manifest.json.prepared.")
            and outcome == "moved"
        ):
            raise OSError("synthetic post-backup failure")
        return outcome

    def fail_before_final_link(
        source: object,
        target: object,
        *args: object,
        **kwargs: object,
    ) -> None:
        if (
            Path(target) == manifest_path
            and Path(source).name.startswith(".manifest.json.")
            and ".prepared." not in Path(source).name
        ):
            raise OSError("synthetic pre-link failure")
        original_link(source, target, *args, **kwargs)

    if failure_boundary == "after_backup":
        monkeypatch.setattr(
            freeze_module,
            "_rename_no_overwrite",
            fail_after_backup,
        )
    else:
        monkeypatch.setattr(freeze_module.os, "link", fail_before_final_link)

    with pytest.raises(RuntimeError, match="freeze approval failed"):
        _migrate(case)

    assert not manifest_path.exists()
    assert report_path.read_bytes() == expected_report
    recovery = [
        path
        for path in _manifest_recoveries_with_bytes(root, case.legacy_manifest_bytes)
        if path.name.startswith(".manifest.json.prepared.")
    ]
    assert recovery

    monkeypatch.setattr(
        freeze_module,
        "_rename_no_overwrite",
        original_rename_no_overwrite,
    )
    monkeypatch.setattr(freeze_module.os, "link", original_link)
    verified_recovery = recovery[0]
    recovery_state = freeze_module._probe_file_state(verified_recovery)
    assert recovery_state is not None
    assert recovery_state.identity == legacy_state.identity
    assert verified_recovery.read_bytes() == case.legacy_manifest_bytes
    assert _sha256(verified_recovery.read_bytes()) == _sha256(
        case.legacy_manifest_bytes
    )
    recovery_relative = verified_recovery.relative_to(root).as_posix()
    assert (
        freeze_module.recover_freeze_manifest(
            data_root=root,
            recovery_path=recovery_relative,
            expected_sha256=_sha256(case.legacy_manifest_bytes),
        )
        == "created"
    )
    assert (
        freeze_module.recover_freeze_manifest(
            data_root=root,
            recovery_path=recovery_relative,
            expected_sha256=_sha256(case.legacy_manifest_bytes),
        )
        == "matched"
    )
    assert isinstance(_migrate(case), FreezeManifestV2)


def test_migration_post_action_link_error_commits_verified_v2(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _migration_fixture(tmp_path)
    root = case.prepared.data_root
    manifest_path = root / "manifest.json"
    original_link = freeze_module.os.link
    linked_then_raised = False

    def link_then_raise(
        source: object,
        target: object,
        *args: object,
        **kwargs: object,
    ) -> None:
        nonlocal linked_then_raised
        original_link(source, target, *args, **kwargs)
        source_path = Path(source)
        if (
            Path(target) == manifest_path
            and source_path.name.startswith(".manifest.json.")
            and ".prepared." not in source_path.name
        ):
            linked_then_raised = True
            raise OSError("synthetic post-link error")

    def quarantine_must_not_run(_: Path) -> None:
        raise AssertionError("verified post-action publication must not quarantine")

    monkeypatch.setattr(freeze_module.os, "link", link_then_raise)
    monkeypatch.setattr(
        freeze_module,
        "_move_manifest_to_quarantine_no_overwrite",
        quarantine_must_not_run,
    )

    migrated = _migrate(case)

    assert linked_then_raised
    assert isinstance(migrated, FreezeManifestV2)
    assert load_freeze_manifest(manifest_path, data_root=root) == migrated


def test_migration_stable_postcheck_failure_happens_before_final_link(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _migration_fixture(tmp_path)
    root = case.prepared.data_root
    manifest_path = root / "manifest.json"
    original_stable = freeze_module._stable_evidence_files
    original_link = freeze_module.os.link
    final_link_calls = 0

    @contextmanager
    def fail_after_stable_postcheck(evidence: object) -> Iterator[None]:
        with original_stable(evidence):
            yield
        raise RuntimeError("freeze approval failed")

    def record_final_link(
        source: object,
        target: object,
        *args: object,
        **kwargs: object,
    ) -> None:
        nonlocal final_link_calls
        source_path = Path(source)
        if (
            Path(target) == manifest_path
            and source_path.name.startswith(".manifest.json.")
            and ".prepared." not in source_path.name
        ):
            final_link_calls += 1
        original_link(source, target, *args, **kwargs)

    monkeypatch.setattr(
        freeze_module,
        "_stable_evidence_files",
        fail_after_stable_postcheck,
    )
    monkeypatch.setattr(freeze_module.os, "link", record_final_link)

    with pytest.raises(RuntimeError, match="freeze approval failed"):
        _migrate(case)

    assert final_link_calls == 0
    assert not manifest_path.exists()
    with pytest.raises((FileNotFoundError, ValueError)):
        load_freeze_manifest(manifest_path, data_root=root)
    assert _manifest_recoveries_with_bytes(root, case.legacy_manifest_bytes)


@pytest.mark.parametrize(
    "invalid_recovery",
    ["wrong_hash", "noncanonical_v1", "non_v1"],
)
def test_recover_freeze_manifest_rejects_invalid_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid_recovery: str,
) -> None:
    case = _migration_fixture(tmp_path)
    root = case.prepared.data_root
    manifest_path = root / "manifest.json"
    recovery = _leave_exact_v1_recovery(case, monkeypatch)
    expected_sha256 = _sha256(case.legacy_manifest_bytes)
    if invalid_recovery == "wrong_hash":
        expected_sha256 = _sha256(b"wrong-operator-hash")
    elif invalid_recovery == "noncanonical_v1":
        payload = json.loads(recovery.read_bytes())
        recovery.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        expected_sha256 = _sha256(recovery.read_bytes())
    else:
        recovery.write_bytes(b'{"not":"a-frozen-v1-manifest"}\n')
        expected_sha256 = _sha256(recovery.read_bytes())

    with pytest.raises(ValueError, match="freeze recovery failed"):
        freeze_module.recover_freeze_manifest(
            data_root=root,
            recovery_path=recovery.relative_to(root).as_posix(),
            expected_sha256=expected_sha256,
        )

    assert not manifest_path.exists()


def test_recover_freeze_manifest_validates_legacy_evidence_bindings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _migration_fixture(tmp_path)
    root = case.prepared.data_root
    manifest_path = root / "manifest.json"
    recovery = _leave_exact_v1_recovery(case, monkeypatch)
    (root / "dev" / "gold.jsonl").write_bytes(b"mutated-gold-evidence\n")

    with pytest.raises(ValueError, match="freeze recovery failed"):
        freeze_module.recover_freeze_manifest(
            data_root=root,
            recovery_path=recovery.relative_to(root).as_posix(),
            expected_sha256=_sha256(case.legacy_manifest_bytes),
        )

    assert not manifest_path.exists()
    assert recovery.read_bytes() == case.legacy_manifest_bytes


def test_recover_freeze_manifest_stable_postcheck_failure_prevents_final_link(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _migration_fixture(tmp_path)
    root = case.prepared.data_root
    manifest_path = root / "manifest.json"
    recovery = _leave_exact_v1_recovery(case, monkeypatch)
    original_stable = freeze_module._stable_evidence_files
    original_link = freeze_module.os.link
    final_link_calls = 0

    @contextmanager
    def fail_after_stable_postcheck(evidence: object) -> Iterator[None]:
        with original_stable(evidence):
            yield
        raise RuntimeError("freeze recovery failed")

    def record_recovery_link(
        source: object,
        target: object,
        *args: object,
        **kwargs: object,
    ) -> None:
        nonlocal final_link_calls
        if Path(source) == recovery and Path(target) == manifest_path:
            final_link_calls += 1
        original_link(source, target, *args, **kwargs)

    monkeypatch.setattr(
        freeze_module,
        "_stable_evidence_files",
        fail_after_stable_postcheck,
    )
    monkeypatch.setattr(freeze_module.os, "link", record_recovery_link)

    with pytest.raises(RuntimeError, match="freeze recovery failed"):
        freeze_module.recover_freeze_manifest(
            data_root=root,
            recovery_path=recovery.relative_to(root).as_posix(),
            expected_sha256=_sha256(case.legacy_manifest_bytes),
        )

    assert final_link_calls == 0
    assert not manifest_path.exists()
    assert recovery.read_bytes() == case.legacy_manifest_bytes


@pytest.mark.skipif(os.name == "nt", reason="POSIX permits held-path replacement")
def test_recover_freeze_manifest_rejects_recovery_path_replaced_after_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _migration_fixture(tmp_path)
    root = case.prepared.data_root
    manifest_path = root / "manifest.json"
    recovery = _leave_exact_v1_recovery(case, monkeypatch)
    replacement = root / "replacement-recovery.tmp"
    replacement.write_bytes(case.legacy_manifest_bytes)
    original_link = freeze_module.os.link
    replaced = False

    def replace_recovery_then_link(
        source: object,
        target: object,
        *args: object,
        **kwargs: object,
    ) -> None:
        nonlocal replaced
        if Path(target) == manifest_path and Path(source) == recovery:
            os.replace(replacement, recovery)
            replaced = True
        original_link(source, target, *args, **kwargs)

    monkeypatch.setattr(freeze_module.os, "link", replace_recovery_then_link)

    with pytest.raises(RuntimeError, match="freeze recovery failed"):
        freeze_module.recover_freeze_manifest(
            data_root=root,
            recovery_path=recovery.relative_to(root).as_posix(),
            expected_sha256=_sha256(case.legacy_manifest_bytes),
        )

    assert replaced
    assert not manifest_path.exists()
    assert recovery.read_bytes() == case.legacy_manifest_bytes


def test_recover_freeze_manifest_never_overwrites_concurrent_manifest_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _migration_fixture(tmp_path)
    root = case.prepared.data_root
    manifest_path = root / "manifest.json"
    recovery = _leave_exact_v1_recovery(case, monkeypatch)
    owner_bytes = b"concurrent-recovery-owner"
    original_link = freeze_module.os.link
    owner_inserted = False

    def owner_wins_before_recovery_link(
        source: object,
        target: object,
        *args: object,
        **kwargs: object,
    ) -> None:
        nonlocal owner_inserted
        if Path(source) == recovery and Path(target) == manifest_path:
            manifest_path.write_bytes(owner_bytes)
            owner_inserted = True
        original_link(source, target, *args, **kwargs)

    monkeypatch.setattr(
        freeze_module.os,
        "link",
        owner_wins_before_recovery_link,
    )

    with pytest.raises(FileExistsError, match="freeze recovery failed"):
        freeze_module.recover_freeze_manifest(
            data_root=root,
            recovery_path=recovery.relative_to(root).as_posix(),
            expected_sha256=_sha256(case.legacy_manifest_bytes),
        )

    assert owner_inserted
    assert manifest_path.read_bytes() == owner_bytes
    assert recovery.read_bytes() == case.legacy_manifest_bytes


def test_recover_freeze_manifest_accepts_post_action_link_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _migration_fixture(tmp_path)
    root = case.prepared.data_root
    manifest_path = root / "manifest.json"
    recovery = _leave_exact_v1_recovery(case, monkeypatch)
    original_link = freeze_module.os.link
    linked_then_raised = False

    def link_recovery_then_raise(
        source: object,
        target: object,
        *args: object,
        **kwargs: object,
    ) -> None:
        nonlocal linked_then_raised
        original_link(source, target, *args, **kwargs)
        if Path(source) == recovery and Path(target) == manifest_path:
            linked_then_raised = True
            raise OSError("synthetic recovery post-link error")

    monkeypatch.setattr(freeze_module.os, "link", link_recovery_then_raise)

    status = freeze_module.recover_freeze_manifest(
        data_root=root,
        recovery_path=recovery.relative_to(root).as_posix(),
        expected_sha256=_sha256(case.legacy_manifest_bytes),
    )

    assert linked_then_raised
    assert status == "created"
    assert recovery.exists()
    assert manifest_path.samefile(recovery)
    assert isinstance(
        load_freeze_manifest(manifest_path, data_root=root),
        FreezeManifestV1,
    )


def test_recover_freeze_manifest_matches_exact_existing_different_inode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _migration_fixture(tmp_path)
    root = case.prepared.data_root
    manifest_path = root / "manifest.json"
    recovery = _leave_exact_v1_recovery(case, monkeypatch)
    manifest_path.write_bytes(recovery.read_bytes())
    assert not manifest_path.samefile(recovery)

    status = freeze_module.recover_freeze_manifest(
        data_root=root,
        recovery_path=recovery.relative_to(root).as_posix(),
        expected_sha256=_sha256(case.legacy_manifest_bytes),
    )

    assert status == "matched"
    assert manifest_path.read_bytes() == case.legacy_manifest_bytes
    assert recovery.read_bytes() == case.legacy_manifest_bytes


def test_migration_bound_manifest_hash_only_postcheck_rejects_descriptor_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _migration_fixture(tmp_path)
    root = case.prepared.data_root
    manifest_path = root / "manifest.json"
    original_link = freeze_module.os.link
    original_stable_bound = freeze_module._stable_bound_artifact_hash
    original_sha256_descriptor = freeze_module._sha256_descriptor
    descriptor_changed = False
    final_link_calls = 0
    legacy_sha256 = _sha256(case.legacy_manifest_bytes)

    @contextmanager
    def change_descriptor_before_bound_postcheck(
        artifact: object,
    ) -> Iterator[None]:
        nonlocal descriptor_changed
        with original_stable_bound(artifact):
            yield
            descriptor_changed = True

    def record_final_link(
        source: object,
        target: object,
        *args: object,
        **kwargs: object,
    ) -> None:
        nonlocal final_link_calls
        source_path = Path(source)
        if (
            Path(target) == manifest_path
            and source_path.name.startswith(".manifest.json.")
            and ".prepared." not in source_path.name
        ):
            final_link_calls += 1
        original_link(source, target, *args, **kwargs)

    def changed_bound_descriptor_hash(descriptor: int) -> str:
        digest = original_sha256_descriptor(descriptor)
        if descriptor_changed and digest == legacy_sha256:
            return _sha256(b"changed-bound-manifest-descriptor")
        return digest

    monkeypatch.setattr(
        freeze_module.os,
        "link",
        record_final_link,
    )
    monkeypatch.setattr(
        freeze_module,
        "_stable_bound_artifact_hash",
        change_descriptor_before_bound_postcheck,
    )
    monkeypatch.setattr(
        freeze_module,
        "_sha256_descriptor",
        changed_bound_descriptor_hash,
    )

    with pytest.raises(RuntimeError, match="freeze approval failed"):
        _migrate(case)

    assert descriptor_changed
    assert final_link_calls == 0
    assert not manifest_path.exists()
    assert _manifest_recoveries_with_bytes(root, case.legacy_manifest_bytes)


def test_migration_changed_bound_descriptor_keeps_recovery_non_authoritative(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _migration_fixture(tmp_path)
    root = case.prepared.data_root
    manifest_path = root / "manifest.json"
    original_link = freeze_module.os.link
    original_sha256_descriptor = freeze_module._sha256_descriptor
    publish_failed = False
    legacy_sha256 = _sha256(case.legacy_manifest_bytes)

    def fail_manifest_publish(
        source: object,
        target: object,
        *args: object,
        **kwargs: object,
    ) -> None:
        nonlocal publish_failed
        source_path = Path(source)
        if (
            Path(target) == manifest_path
            and source_path.name.startswith(".manifest.json.")
            and ".prepared." not in source_path.name
        ):
            publish_failed = True
            raise OSError("synthetic manifest publish failure")
        original_link(source, target, *args, **kwargs)

    def changed_bound_descriptor_hash(descriptor: int) -> str:
        digest = original_sha256_descriptor(descriptor)
        if publish_failed and digest == legacy_sha256:
            return _sha256(b"changed-bound-manifest-descriptor")
        return digest

    monkeypatch.setattr(freeze_module.os, "link", fail_manifest_publish)
    monkeypatch.setattr(
        freeze_module,
        "_sha256_descriptor",
        changed_bound_descriptor_hash,
    )

    with pytest.raises(RuntimeError, match="freeze approval failed"):
        _migrate(case)

    assert publish_failed
    assert not manifest_path.exists()
    assert _manifest_recoveries_with_bytes(root, case.legacy_manifest_bytes)


@pytest.mark.skipif(os.name == "nt", reason="POSIX permits held-inode mutation")
def test_migration_never_restores_backup_mutated_before_publish_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _migration_fixture(tmp_path)
    root = case.prepared.data_root
    manifest_path = root / "manifest.json"
    attacker_bytes = b"attacker-mutated-retained-v1-inode"
    original_link = freeze_module.os.link
    mutated_backup: Path | None = None
    quarantine_attempted = False

    def mutate_backup_then_fail_publish(
        source: object,
        target: object,
        *args: object,
        **kwargs: object,
    ) -> None:
        nonlocal mutated_backup
        source_path = Path(source)
        if (
            Path(target) == manifest_path
            and source_path.name.startswith(".manifest.json.")
            and ".prepared." not in source_path.name
        ):
            candidates = list(root.glob(".manifest.json.prepared.*.tmp"))
            assert len(candidates) == 1
            mutated_backup = candidates[0]
            mutated_backup.write_bytes(attacker_bytes)
            raise OSError("synthetic manifest publish failure")
        original_link(source, target, *args, **kwargs)

    monkeypatch.setattr(
        freeze_module.os,
        "link",
        mutate_backup_then_fail_publish,
    )

    def fail_quarantine(_: Path) -> None:
        nonlocal quarantine_attempted
        quarantine_attempted = True
        return None

    monkeypatch.setattr(
        freeze_module,
        "_move_manifest_to_quarantine_no_overwrite",
        fail_quarantine,
    )

    with pytest.raises(RuntimeError, match="freeze approval failed"):
        _migrate(case)

    assert mutated_backup is not None
    assert quarantine_attempted
    assert not manifest_path.exists()
    assert mutated_backup.read_bytes() == attacker_bytes


def test_migration_publish_failure_never_hardlinks_v1_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _migration_fixture(tmp_path)
    root = case.prepared.data_root
    manifest_path = root / "manifest.json"
    attacker_bytes = b"attacker-mutated-v1-recovery"
    original_link = freeze_module.os.link
    recovery_link_attempted = False
    quarantine_attempted = False

    def fail_publish_and_record_recovery_link(
        source: object,
        target: object,
        *args: object,
        **kwargs: object,
    ) -> None:
        nonlocal recovery_link_attempted
        source_path = Path(source)
        if Path(target) == manifest_path and source_path.name.startswith(
            ".manifest.json.prepared."
        ):
            recovery_link_attempted = True
            original_link(source, target, *args, **kwargs)
            try:
                source_path.write_bytes(attacker_bytes)
            except PermissionError:
                pass
            return
        if (
            Path(target) == manifest_path
            and source_path.name.startswith(".manifest.json.")
            and ".prepared." not in source_path.name
        ):
            raise OSError("synthetic manifest publish failure")
        original_link(source, target, *args, **kwargs)

    monkeypatch.setattr(
        freeze_module.os,
        "link",
        fail_publish_and_record_recovery_link,
    )

    def fail_quarantine(_: Path) -> None:
        nonlocal quarantine_attempted
        quarantine_attempted = True
        return None

    monkeypatch.setattr(
        freeze_module,
        "_move_manifest_to_quarantine_no_overwrite",
        fail_quarantine,
    )

    with pytest.raises(RuntimeError, match="freeze approval failed"):
        _migrate(case)

    assert not quarantine_attempted
    assert not recovery_link_attempted
    assert not manifest_path.exists()
    assert _manifest_recoveries_with_bytes(root, case.legacy_manifest_bytes)
    assert not _manifest_recoveries_with_bytes(root, attacker_bytes)


def test_migration_unmodified_bound_manifest_remains_recovery_after_link_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _migration_fixture(tmp_path)
    root = case.prepared.data_root
    manifest_path = root / "manifest.json"
    original_link = freeze_module.os.link
    publish_failed = False

    def fail_manifest_publish(
        source: object,
        target: object,
        *args: object,
        **kwargs: object,
    ) -> None:
        nonlocal publish_failed
        source_path = Path(source)
        if (
            Path(target) == manifest_path
            and source_path.name.startswith(".manifest.json.")
            and ".prepared." not in source_path.name
        ):
            publish_failed = True
            raise OSError("synthetic manifest publish failure")
        original_link(source, target, *args, **kwargs)

    monkeypatch.setattr(freeze_module.os, "link", fail_manifest_publish)

    with pytest.raises(RuntimeError, match="freeze approval failed"):
        _migrate(case)

    assert publish_failed
    assert not manifest_path.exists()
    assert _manifest_recoveries_with_bytes(root, case.legacy_manifest_bytes)


def test_migration_guarded_publish_preserves_concurrent_manifest_owner_and_backup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _migration_fixture(tmp_path)
    root = case.prepared.data_root
    manifest_path = root / "manifest.json"
    concurrent_bytes = b"concurrent-manifest-owner"
    original_link = freeze_module.os.link
    injected = False

    def insert_concurrent_owner(
        source: object,
        target: object,
        *args: object,
        **kwargs: object,
    ) -> None:
        nonlocal injected
        if (
            Path(target) == manifest_path
            and Path(source).name.startswith(".manifest.json.")
            and ".prepared." not in Path(source).name
            and not injected
        ):
            manifest_path.write_bytes(concurrent_bytes)
            injected = True
        original_link(source, target, *args, **kwargs)

    monkeypatch.setattr(
        freeze_module.os,
        "link",
        insert_concurrent_owner,
    )

    with pytest.raises(RuntimeError, match="freeze approval failed"):
        _migrate(case)

    assert injected
    assert manifest_path.read_bytes() == concurrent_bytes
    recovery = _manifest_recoveries_with_bytes(root, case.legacy_manifest_bytes)
    assert recovery
    assert (root / case.approval_report_path).is_file()

    monkeypatch.setattr(freeze_module.os, "link", original_link)
    manifest_path.unlink()
    os.replace(recovery[0], manifest_path)
    assert isinstance(_migrate(case), FreezeManifestV2)


def test_migration_post_backup_failure_never_attempts_recovery_link(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _migration_fixture(tmp_path)
    root = case.prepared.data_root
    manifest_path = root / "manifest.json"
    original_rename_no_overwrite = freeze_module._rename_no_overwrite
    original_link = freeze_module.os.link
    recovery_link_attempted = False

    def fail_after_backup(source: Path, target: Path) -> str:
        outcome = original_rename_no_overwrite(source, target)
        if (
            source == manifest_path
            and target.name.startswith(".manifest.json.prepared.")
            and outcome == "moved"
        ):
            raise OSError("synthetic post-backup failure")
        return outcome

    def record_recovery_link(
        source: object,
        target: object,
        *args: object,
        **kwargs: object,
    ) -> None:
        nonlocal recovery_link_attempted
        if (
            Path(target) == manifest_path
            and Path(source).name.startswith(".manifest.json.prepared.")
        ):
            recovery_link_attempted = True
        original_link(source, target, *args, **kwargs)

    monkeypatch.setattr(
        freeze_module,
        "_rename_no_overwrite",
        fail_after_backup,
    )
    monkeypatch.setattr(freeze_module.os, "link", record_recovery_link)

    with pytest.raises(RuntimeError, match="freeze approval failed"):
        _migrate(case)

    assert not recovery_link_attempted
    assert not manifest_path.exists()
    recovery = _manifest_recoveries_with_bytes(root, case.legacy_manifest_bytes)
    assert recovery


def test_migration_post_replace_probe_failure_preserves_v1_backup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _migration_fixture(tmp_path)
    root = case.prepared.data_root
    manifest_path = root / "manifest.json"
    original_rename_no_overwrite = freeze_module._rename_no_overwrite
    original_read_bytes = Path.read_bytes
    original_stat = Path.stat
    fail_backup_probes = False

    def replace_then_fail(source: Path, target: Path) -> str:
        nonlocal fail_backup_probes
        outcome = original_rename_no_overwrite(source, target)
        if (
            source == manifest_path
            and target.name.startswith(".manifest.json.prepared.")
            and outcome == "moved"
        ):
            fail_backup_probes = True
            raise OSError("synthetic post-replace failure")
        return outcome

    def fail_recovery_read(path: Path) -> bytes:
        if fail_backup_probes and path.name.startswith(".manifest.json.prepared."):
            raise OSError("synthetic backup read failure")
        return original_read_bytes(path)

    def fail_recovery_stat(
        path: Path,
        *,
        follow_symlinks: bool = True,
    ) -> os.stat_result:
        if fail_backup_probes and path.name.startswith(".manifest.json.prepared."):
            raise OSError("synthetic backup identity failure")
        return original_stat(path, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(
        freeze_module,
        "_rename_no_overwrite",
        replace_then_fail,
    )
    monkeypatch.setattr(Path, "read_bytes", fail_recovery_read)
    monkeypatch.setattr(Path, "stat", fail_recovery_stat)

    with pytest.raises(RuntimeError, match="freeze approval failed"):
        _migrate(case)

    fail_backup_probes = False
    assert not manifest_path.exists()
    recovery = _manifest_recoveries_with_bytes(root, case.legacy_manifest_bytes)
    assert recovery


def test_migration_commit_cleanup_preserves_replaced_backup_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _migration_fixture(tmp_path)
    root = case.prepared.data_root
    owner = root / "commit-cleanup-owner.tmp"
    owner_bytes = b"commit-cleanup-backup-owner"
    owner.write_bytes(owner_bytes)
    original_rename_no_overwrite = freeze_module._rename_no_overwrite
    owner_inserted = False

    def owner_replaces_before_cleanup(source: Path, target: Path) -> str:
        nonlocal owner_inserted
        if (
            source.name.startswith(".manifest.json.prepared.")
            and not owner_inserted
        ):
            if os.name == "nt":
                _native_replace_windows(owner, source)
            else:
                os.replace(owner, source)
            owner_inserted = True
        return original_rename_no_overwrite(source, target)

    monkeypatch.setattr(
        freeze_module,
        "_rename_no_overwrite",
        owner_replaces_before_cleanup,
    )

    assert isinstance(_migrate(case), FreezeManifestV2)

    assert owner_inserted
    retained = _manifest_recoveries_with_bytes(root, owner_bytes)
    assert retained


def test_backup_candidate_collision_preserves_owner_and_uses_fresh_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _prepared_tree(tmp_path)
    plan = _approval_plan(fixture)
    root = fixture.data_root
    manifest_path = root / "manifest.json"
    owner_bytes = b"backup-candidate-owner"
    original_rename_no_overwrite = freeze_module._rename_no_overwrite
    collided_candidate: Path | None = None
    backup_attempts = 0

    def occupy_first_candidate_then_rename(source: Path, target: Path) -> str:
        nonlocal backup_attempts, collided_candidate
        if source == manifest_path and target.name.startswith(
            ".manifest.json.prepared."
        ):
            backup_attempts += 1
            if backup_attempts == 1:
                target.write_bytes(owner_bytes)
                collided_candidate = target
        return original_rename_no_overwrite(source, target)

    monkeypatch.setattr(
        freeze_module,
        "_rename_no_overwrite",
        occupy_first_candidate_then_rename,
    )

    freeze_module._replace_manifest_guarded(
        manifest_path,
        plan.prepared_manifest_bytes,
        plan.frozen_manifest_bytes,
    )

    assert backup_attempts >= 2
    assert collided_candidate is not None
    assert collided_candidate.read_bytes() == owner_bytes
    assert manifest_path.read_bytes() == plan.frozen_manifest_bytes
    recoveries = list(root.glob(".manifest.json.prepared.*"))
    assert any(
        path != collided_candidate
        and path.read_bytes() == plan.prepared_manifest_bytes
        for path in recoveries
    )


def test_backup_quarantine_owner_replacing_after_second_probe_is_never_unlinked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _prepared_tree(tmp_path)
    plan = _approval_plan(fixture)
    root = fixture.data_root
    manifest_path = root / "manifest.json"
    owner = root / "post-probe-cleanup-owner.tmp"
    owner_bytes = b"owner-after-cleanup-identity-probe"
    owner.write_bytes(owner_bytes)
    original_probe = freeze_module._probe_file_state
    owner_inserted = False
    owner_quarantine: Path | None = None

    def replace_after_quarantine_probe(
        path: Path,
    ) -> object:
        nonlocal owner_inserted, owner_quarantine
        state = original_probe(path)
        if (
            not owner_inserted
            and path.name.startswith(".manifest.json.prepared.")
            and ".cleanup." in path.name
            and state is not None
        ):
            if os.name == "nt":
                _native_replace_windows(owner, path)
            else:
                os.replace(owner, path)
            owner_inserted = True
            owner_quarantine = path
        return state

    monkeypatch.setattr(
        freeze_module,
        "_probe_file_state",
        replace_after_quarantine_probe,
    )

    freeze_module._replace_manifest_guarded(
        manifest_path,
        plan.prepared_manifest_bytes,
        plan.frozen_manifest_bytes,
    )

    assert owner_inserted
    assert owner_quarantine is not None
    assert owner_quarantine.read_bytes() == owner_bytes
    assert manifest_path.read_bytes() == plan.frozen_manifest_bytes


def test_manifest_publisher_initial_fstat_failure_closes_descriptor_and_retains_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _prepared_tree(tmp_path)
    plan = _approval_plan(fixture)
    root = fixture.data_root
    manifest_path = root / "manifest.json"
    original_mkstemp = freeze_module.tempfile.mkstemp
    original_fstat = freeze_module.os.fstat
    first_descriptor: int | None = None
    first_path: Path | None = None
    triggered = False

    def capture_first_mkstemp(*args: object, **kwargs: object) -> tuple[int, str]:
        nonlocal first_descriptor, first_path
        descriptor, name = original_mkstemp(*args, **kwargs)
        if first_descriptor is None:
            first_descriptor = descriptor
            first_path = Path(name)
        return descriptor, name

    def fail_first_fstat(descriptor: int) -> os.stat_result:
        nonlocal triggered
        if descriptor == first_descriptor and not triggered:
            triggered = True
            raise OSError("synthetic initial manifest temp fstat failure")
        return original_fstat(descriptor)

    monkeypatch.setattr(freeze_module.tempfile, "mkstemp", capture_first_mkstemp)
    monkeypatch.setattr(freeze_module.os, "fstat", fail_first_fstat)

    with pytest.raises(OSError, match="synthetic initial manifest temp fstat failure"):
        freeze_module._replace_manifest_guarded(
            manifest_path,
            plan.prepared_manifest_bytes,
            plan.frozen_manifest_bytes,
        )

    assert triggered
    assert first_descriptor is not None
    with pytest.raises(OSError):
        original_fstat(first_descriptor)
    assert first_path is not None
    assert first_path.read_bytes() == b""
    assert manifest_path.read_bytes() == plan.prepared_manifest_bytes
    assert not list(root.glob(".manifest.json.prepared.*"))


def test_manifest_publisher_backup_name_failure_quarantines_v2_temporary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _prepared_tree(tmp_path)
    plan = _approval_plan(fixture)
    root = fixture.data_root
    manifest_path = root / "manifest.json"
    original_token_hex = freeze_module.secrets.token_hex
    calls = 0

    def fail_first_name_generation(length: int) -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("synthetic backup name generation failure")
        return original_token_hex(length)

    monkeypatch.setattr(freeze_module.secrets, "token_hex", fail_first_name_generation)

    with pytest.raises(RuntimeError, match="freeze approval failed"):
        freeze_module._replace_manifest_guarded(
            manifest_path,
            plan.prepared_manifest_bytes,
            plan.frozen_manifest_bytes,
        )

    assert manifest_path.read_bytes() == plan.prepared_manifest_bytes
    recoveries = list(root.glob(".manifest.json.*.tmp"))
    assert any(path.read_bytes() == plan.frozen_manifest_bytes for path in recoveries)
    assert not list(root.glob(".manifest.json.prepared.*"))


def test_manifest_publisher_close_failure_quarantines_v2_temporary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _prepared_tree(tmp_path)
    plan = _approval_plan(fixture)
    root = fixture.data_root
    manifest_path = root / "manifest.json"
    original_fdopen = freeze_module.os.fdopen

    class CloseThenFail:
        def __init__(self, descriptor: int, mode: str) -> None:
            self._file = original_fdopen(descriptor, mode)

        def __enter__(self) -> CloseThenFail:
            return self

        def __exit__(self, *args: object) -> None:
            del args
            self.close()

        def close(self) -> None:
            self._file.close()
            raise OSError("synthetic manifest temp close failure")

        def write(self, content: bytes) -> int:
            return self._file.write(content)

        def flush(self) -> None:
            self._file.flush()

        def fileno(self) -> int:
            return self._file.fileno()

    monkeypatch.setattr(freeze_module.os, "fdopen", CloseThenFail)

    with pytest.raises(OSError, match="synthetic manifest temp close failure"):
        freeze_module._replace_manifest_guarded(
            manifest_path,
            plan.prepared_manifest_bytes,
            plan.frozen_manifest_bytes,
        )

    assert manifest_path.read_bytes() == plan.prepared_manifest_bytes
    recoveries = list(root.glob(".manifest.json.*.tmp"))
    assert any(path.read_bytes() == plan.frozen_manifest_bytes for path in recoveries)
    assert not list(root.glob(".manifest.json.prepared.*"))


def test_migration_precommit_backup_proof_failure_never_links_or_quarantines(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _migration_fixture(tmp_path)
    root = case.prepared.data_root
    manifest_path = root / "manifest.json"
    original_link = freeze_module.os.link
    original_recovery_matches = freeze_module._manifest_recovery_matches
    quarantine_attempted = False
    final_link_calls = 0
    recovery_proof_calls = 0

    def record_final_link(
        source: object,
        target: object,
        *args: object,
        **kwargs: object,
    ) -> None:
        nonlocal final_link_calls
        source_path = Path(source)
        if (
            Path(target) == manifest_path
            and source_path.name.startswith(".manifest.json.")
            and ".prepared." not in source_path.name
        ):
            final_link_calls += 1
        original_link(source, target, *args, **kwargs)

    def fail_second_backup_proof(*args: object, **kwargs: object) -> bool:
        nonlocal recovery_proof_calls
        recovery_proof_calls += 1
        if recovery_proof_calls == 2:
            return False
        return original_recovery_matches(*args, **kwargs)

    def fail_manifest_quarantine(path: Path) -> None:
        nonlocal quarantine_attempted
        if path == manifest_path:
            quarantine_attempted = True
        return None

    monkeypatch.setattr(
        freeze_module.os,
        "link",
        record_final_link,
    )
    monkeypatch.setattr(
        freeze_module,
        "_move_manifest_to_quarantine_no_overwrite",
        fail_manifest_quarantine,
    )
    monkeypatch.setattr(
        freeze_module,
        "_manifest_recovery_matches",
        fail_second_backup_proof,
    )

    with pytest.raises(RuntimeError, match="freeze approval failed"):
        _migrate(case)

    assert recovery_proof_calls == 2
    assert final_link_calls == 0
    assert not quarantine_attempted
    assert not manifest_path.exists()
    recovery = _manifest_recoveries_with_bytes(root, case.legacy_manifest_bytes)
    assert recovery


def test_migration_different_owner_at_final_link_is_preserved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _migration_fixture(tmp_path)
    root = case.prepared.data_root
    manifest_path = root / "manifest.json"
    owner = root / "replacement-owner.tmp"
    owner_bytes = b"different-final-link-owner"
    owner.write_bytes(owner_bytes)
    original_link = freeze_module.os.link
    owner_inserted = False

    def insert_owner_and_raise(
        source: object,
        target: object,
        *args: object,
        **kwargs: object,
    ) -> None:
        nonlocal owner_inserted
        if (
            Path(target) == manifest_path
            and Path(source).name.startswith(".manifest.json.")
            and ".prepared." not in Path(source).name
        ):
            if os.name == "nt":
                _native_replace_windows(owner, manifest_path)
            else:
                os.replace(owner, manifest_path)
            owner_inserted = True
            raise OSError("synthetic different final owner")
        original_link(source, target, *args, **kwargs)

    monkeypatch.setattr(freeze_module.os, "link", insert_owner_and_raise)

    with pytest.raises(RuntimeError, match="freeze approval failed"):
        _migrate(case)

    assert owner_inserted
    assert manifest_path.read_bytes() == owner_bytes
    recovery = _manifest_recoveries_with_bytes(root, case.legacy_manifest_bytes)
    assert recovery


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


@pytest.mark.skipif(os.name != "nt", reason="Windows raw-handle cleanup contract")
@pytest.mark.parametrize("failure_stage", ["open_osfhandle", "initial_file_id"])
@pytest.mark.parametrize("concurrent_owner", [False, True])
def test_windows_migration_cleans_report_temp_when_raw_handle_setup_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
    concurrent_owner: bool,
) -> None:
    import ctypes
    import msvcrt

    case = _migration_fixture(tmp_path)
    root = case.prepared.data_root
    report_parent = root / "freeze_reports"
    original_open_osfhandle = msvcrt.open_osfhandle
    original_file_id = freeze_schema_module._windows_file_id
    triggered = False
    replaced_temp_path: Path | None = None
    attacker = root / "concurrent-temp-owner.tmp"
    attacker_bytes = b"concurrent-temp-owner"
    if concurrent_owner:
        attacker.write_bytes(attacker_bytes)

    def report_temp_path(handle: int) -> Path | None:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        buffer = ctypes.create_unicode_buffer(32768)
        if not kernel32.GetFinalPathNameByHandleW(
            ctypes.c_void_p(handle), buffer, len(buffer), 0
        ):
            return None
        path = Path(
            freeze_schema_module._normalized_windows_path(buffer.value)
        )
        return path if path.name.startswith(".v2-approval.json.") else None

    def inject_concurrent_owner(path: Path) -> None:
        nonlocal replaced_temp_path
        if concurrent_owner:
            _native_replace_windows(attacker, path)
            replaced_temp_path = path

    def fail_open_osfhandle(handle: int, flags: int) -> int:
        nonlocal triggered
        path = report_temp_path(handle)
        if (
            failure_stage == "open_osfhandle"
            and not triggered
            and path is not None
        ):
            inject_concurrent_owner(path)
            triggered = True
            raise OSError("synthetic open_osfhandle failure")
        return original_open_osfhandle(handle, flags)

    def fail_initial_file_id(handle: int) -> tuple[int, bytes]:
        nonlocal triggered
        path = report_temp_path(handle)
        if (
            failure_stage == "initial_file_id"
            and not triggered
            and path is not None
        ):
            inject_concurrent_owner(path)
            triggered = True
            raise OSError("synthetic initial FILE_ID failure")
        return original_file_id(handle)

    monkeypatch.setattr(msvcrt, "open_osfhandle", fail_open_osfhandle)
    monkeypatch.setattr(
        freeze_schema_module,
        "_windows_file_id",
        fail_initial_file_id,
    )

    with pytest.raises(RuntimeError, match="freeze migration failed"):
        _migrate(case)

    assert triggered
    assert (root / "manifest.json").read_bytes() == case.legacy_manifest_bytes
    assert not (root / case.approval_report_path).exists()
    temporaries = list(report_parent.glob(".v2-approval.json.*.tmp"))
    if concurrent_owner:
        assert temporaries == [replaced_temp_path]
        assert replaced_temp_path is not None
        assert replaced_temp_path.read_bytes() == attacker_bytes
    else:
        assert not temporaries


@pytest.mark.skipif(os.name != "nt", reason="Windows delete-handle fallback contract")
@pytest.mark.parametrize("failure_stage", ["open_osfhandle", "initial_file_id"])
def test_windows_report_temp_cleanup_falls_back_only_with_proven_path_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    import ctypes
    import msvcrt

    case = _migration_fixture(tmp_path)
    root = case.prepared.data_root
    report_parent = root / "freeze_reports"
    original_open_osfhandle = msvcrt.open_osfhandle
    original_file_id = freeze_schema_module._windows_file_id
    triggered = False

    def is_report_temp(handle: int) -> bool:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        buffer = ctypes.create_unicode_buffer(32768)
        return bool(
            kernel32.GetFinalPathNameByHandleW(
                ctypes.c_void_p(handle), buffer, len(buffer), 0
            )
            and Path(
                freeze_schema_module._normalized_windows_path(buffer.value)
            ).name.startswith(".v2-approval.json.")
        )

    def fail_open_osfhandle(handle: int, flags: int) -> int:
        nonlocal triggered
        if (
            failure_stage == "open_osfhandle"
            and not triggered
            and is_report_temp(handle)
        ):
            triggered = True
            raise OSError("synthetic open_osfhandle failure")
        return original_open_osfhandle(handle, flags)

    def fail_initial_file_id(handle: int) -> tuple[int, bytes]:
        nonlocal triggered
        if (
            failure_stage == "initial_file_id"
            and not triggered
            and is_report_temp(handle)
        ):
            triggered = True
            raise OSError("synthetic initial FILE_ID failure")
        return original_file_id(handle)

    monkeypatch.setattr(msvcrt, "open_osfhandle", fail_open_osfhandle)
    monkeypatch.setattr(
        freeze_schema_module,
        "_windows_file_id",
        fail_initial_file_id,
    )
    monkeypatch.setattr(
        freeze_schema_module,
        "_delete_windows_handle",
        lambda handle: False,
    )

    with pytest.raises(RuntimeError, match="freeze migration failed"):
        _migrate(case)

    assert triggered
    assert (root / "manifest.json").read_bytes() == case.legacy_manifest_bytes
    assert not (root / case.approval_report_path).exists()
    temporaries = list(report_parent.glob(".v2-approval.json.*.tmp"))
    if failure_stage == "open_osfhandle":
        assert not temporaries
    else:
        assert len(temporaries) == 1
        assert temporaries[0].read_bytes() == b""


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
    manifest_path = fixture.data_root / "manifest.json"
    assert manifest_path.read_bytes() == plan.frozen_manifest_bytes
    prepared_recovery = _manifest_recoveries_with_bytes(
        fixture.data_root,
        plan.prepared_manifest_bytes,
    )
    frozen_recovery = _manifest_recoveries_with_bytes(
        fixture.data_root,
        plan.frozen_manifest_bytes,
    )
    assert prepared_recovery
    assert frozen_recovery
    assert any(path.samefile(manifest_path) for path in frozen_recovery)
    assert all(not path.samefile(manifest_path) for path in prepared_recovery)
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
    recovery = list(fixture.data_root.glob(".manifest.json.prepared.*.tmp"))
    assert recovery == []


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
    recovery = _manifest_recoveries_with_bytes(
        fixture.data_root,
        plan.frozen_manifest_bytes,
    )
    assert recovery
    assert all(
        not path.samefile(fixture.data_root / "manifest.json") for path in recovery
    )


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


def test_approve_stable_postcheck_failure_prevents_manifest_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _prepared_tree(tmp_path)
    plan = _approval_plan(fixture)
    manifest_path = fixture.data_root / "manifest.json"
    original_stable = freeze_module._stable_evidence_files
    original_link = freeze_module.os.link
    final_link_calls = 0

    @contextmanager
    def fail_after_stable_postcheck(evidence: object) -> Iterator[None]:
        with original_stable(evidence):
            yield
        raise RuntimeError("freeze approval failed")

    def record_final_link(source: object, target: object) -> None:
        nonlocal final_link_calls
        if Path(target) == manifest_path:
            final_link_calls += 1
        original_link(source, target)

    monkeypatch.setattr(
        freeze_module,
        "_stable_evidence_files",
        fail_after_stable_postcheck,
    )
    monkeypatch.setattr(freeze_module.os, "link", record_final_link)
    with pytest.raises(RuntimeError, match="freeze approval failed"):
        approve_freeze(data_root=fixture.data_root, plan=plan)

    assert final_link_calls == 0
    assert not manifest_path.exists()
    recovery = _manifest_recoveries_with_bytes(
        fixture.data_root,
        plan.prepared_manifest_bytes,
    )
    assert recovery


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


def test_committed_freeze_retains_non_authoritative_v1_and_v2_recovery(
    tmp_path: Path,
) -> None:
    fixture = _prepared_tree(tmp_path)
    plan = _approval_plan(fixture)
    manifest_path = fixture.data_root / "manifest.json"
    assert approve_freeze(data_root=fixture.data_root, plan=plan) == "created"
    assert manifest_path.read_bytes() == plan.frozen_manifest_bytes
    prepared_recovery = _manifest_recoveries_with_bytes(
        fixture.data_root,
        plan.prepared_manifest_bytes,
    )
    frozen_recovery = _manifest_recoveries_with_bytes(
        fixture.data_root,
        plan.frozen_manifest_bytes,
    )
    assert prepared_recovery
    assert frozen_recovery
    assert all(not path.samefile(manifest_path) for path in prepared_recovery)
    assert any(path.samefile(manifest_path) for path in frozen_recovery)

    assert approve_freeze(data_root=fixture.data_root, plan=plan) == "matched"
    if os.name == "nt":
        assert not _manifest_recoveries_with_bytes(
            fixture.data_root,
            plan.prepared_manifest_bytes,
        )
        assert not _manifest_recoveries_with_bytes(
            fixture.data_root,
            plan.frozen_manifest_bytes,
        )
    else:
        assert prepared_recovery
        assert frozen_recovery


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
    assert approve_freeze(data_root=fixture.data_root, plan=plan) == "matched"
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
    assert approve_freeze(data_root=fixture.data_root, plan=plan) == "matched"
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


def test_failed_publish_preserves_prepared_recovery_backup(
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
