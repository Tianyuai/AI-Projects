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

from paper_search.evaluation.freeze import (
    audit_freeze_candidate,
    main,
    parse_zero_answer_policies,
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
    for path in (
        fixture.type_domain_labels,
        fixture.constraint_labels,
        fixture.overlap_labels,
    ):
        rows = _label_rows(path)
        for row in rows:
            row["annotator"] = sentinel
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

    exit_code = main(_cli_args(fixture))

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

    exit_code = main([*_cli_args(fixture), *extra_args])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert captured.err.rstrip() == "freeze approval failed"


def test_cli_requires_all_explicit_inputs() -> None:
    with pytest.raises(SystemExit) as error:
        main([])

    assert error.value.code == 2


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

    exit_code = main(args)

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert (
        captured.err.rstrip()
        == "freeze audit failed: private annotations are invalid"
    )
    assert sentinel not in captured.err

def test_cli_rejects_report_path_outside_freeze_reports(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = _prepared_tree(tmp_path)
    outside_report = tmp_path / "outside-report.json"

    exit_code = main(
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