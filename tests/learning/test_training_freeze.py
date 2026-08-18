from __future__ import annotations

import json
from pathlib import Path

from paper_search.learning.training_freeze import freeze_pasa_training_data


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _row(qid: str, question: str, paper_id: str) -> dict[str, object]:
    return {
        "qid": qid,
        "question": question,
        "answer": [f"Title {paper_id}"],
        "answer_arxiv_id": [paper_id],
        "source_meta": {"published_time": "2024-01-01"},
    }


def test_freeze_excludes_only_training_side_of_cross_role_components(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _write_jsonl(
        raw / "AutoScholarQuery/train.jsonl",
        [
            _row("train-exact", "Graph retrieval methods", "1111.00001"),
            _row("train-gold", "Different query", "2222.00002"),
            _row("train-safe", "Completely safe query", "3333.00003"),
        ],
    )
    _write_jsonl(
        raw / "AutoScholarQuery/dev.jsonl",
        [
            _row("dev-exact", "graph retrieval methods", "4444.00004"),
            _row("dev-gold", "Another development query", "2222.00002"),
        ],
    )
    _write_jsonl(
        raw / "AutoScholarQuery/test.jsonl",
        [_row("test-1", "Held out query", "5555.00005")],
    )
    _write_jsonl(
        raw / "RealScholarQuery/test.jsonl",
        [_row("real-1", "Real held out query", "6666.00006")],
    )

    manifest = freeze_pasa_training_data(
        raw_root=raw,
        private_output_root=tmp_path / "private",
        manifest_path=tmp_path / "manifest.json",
        revision="fixed-revision",
        asta_access_status="unauthorized",
        expected_counts={
            "AutoScholarQuery/train.jsonl": 3,
            "AutoScholarQuery/dev.jsonl": 2,
            "AutoScholarQuery/test.jsonl": 1,
            "RealScholarQuery/test.jsonl": 1,
        },
    )

    assert manifest.source_counts == {
        "pasa/auto_dev": 2,
        "pasa/auto_test": 1,
        "pasa/auto_train": 3,
        "pasa/real_test": 1,
    }
    assert manifest.frozen_counts["pasa/auto_train"] == 1
    assert manifest.excluded_training_count == 2
    assert manifest.training_isolation_verified is True
    assert manifest.isolation_issue_counts == {
        "exact_query_overlap": 1,
        "gold_paper_overlap": 1,
    }
    assert manifest.asta_access_status == "unauthorized"

    training_rows = [
        json.loads(line)
        for line in (tmp_path / "private/partitions/pasa_auto_train.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["query_id"] for row in training_rows] == ["train-safe"]


def test_freeze_does_not_transitively_exclude_training_only_neighbors(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    _write_jsonl(
        raw / "AutoScholarQuery/train.jsonl",
        [
            _row("train-direct", "Shared query", "1111.00001"),
            _row("train-indirect", "Unique training query", "1111.00001"),
        ],
    )
    _write_jsonl(
        raw / "AutoScholarQuery/dev.jsonl",
        [_row("dev", "shared query", "2222.00002")],
    )
    _write_jsonl(
        raw / "AutoScholarQuery/test.jsonl",
        [_row("test", "Held out query", "3333.00003")],
    )
    _write_jsonl(
        raw / "RealScholarQuery/test.jsonl",
        [_row("real", "Real held out query", "4444.00004")],
    )

    manifest = freeze_pasa_training_data(
        raw_root=raw,
        private_output_root=tmp_path / "private",
        manifest_path=tmp_path / "manifest.json",
        revision="fixed-revision",
        asta_access_status="unauthorized",
        expected_counts={
            "AutoScholarQuery/train.jsonl": 2,
            "AutoScholarQuery/dev.jsonl": 1,
            "AutoScholarQuery/test.jsonl": 1,
            "RealScholarQuery/test.jsonl": 1,
        },
    )

    assert manifest.excluded_training_count == 1
    assert manifest.transitive_component_warning_count == 1
    assert manifest.training_isolation_verified is True
    training_rows = [
        json.loads(line)
        for line in (tmp_path / "private/partitions/pasa_auto_train.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["query_id"] for row in training_rows] == ["train-indirect"]


def test_freeze_preserves_final_test_and_removes_conflicting_development(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    _write_jsonl(
        raw / "AutoScholarQuery/train.jsonl",
        [_row("train", "Training query", "1111.00001")],
    )
    _write_jsonl(
        raw / "AutoScholarQuery/dev.jsonl",
        [
            _row("dev-conflict", "Development conflict", "2222.00002"),
            _row("dev-safe", "Development safe", "3333.00003"),
        ],
    )
    _write_jsonl(
        raw / "AutoScholarQuery/test.jsonl",
        [_row("test", "Final query", "2222.00002")],
    )
    _write_jsonl(
        raw / "RealScholarQuery/test.jsonl",
        [_row("real", "Real final query", "4444.00004")],
    )

    manifest = freeze_pasa_training_data(
        raw_root=raw,
        private_output_root=tmp_path / "private",
        manifest_path=tmp_path / "manifest.json",
        revision="fixed-revision",
        asta_access_status="excluded_by_scope",
        expected_counts={
            "AutoScholarQuery/train.jsonl": 1,
            "AutoScholarQuery/dev.jsonl": 2,
            "AutoScholarQuery/test.jsonl": 1,
            "RealScholarQuery/test.jsonl": 1,
        },
    )

    assert manifest.datasets_in_scope == ["pasa"]
    assert manifest.excluded_development_count == 1
    assert manifest.frozen_counts["pasa/auto_dev"] == 1
    assert manifest.frozen_counts["pasa/auto_test"] == 1
    assert manifest.final_cross_role_issue_count == 0


def test_freeze_is_byte_deterministic_for_the_same_sources(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    for relative, row in (
        ("AutoScholarQuery/train.jsonl", _row("train", "Safe", "1111.00001")),
        ("AutoScholarQuery/dev.jsonl", _row("dev", "Development", "2222.00002")),
        ("AutoScholarQuery/test.jsonl", _row("test", "Test", "3333.00003")),
        ("RealScholarQuery/test.jsonl", _row("real", "Real", "4444.00004")),
    ):
        _write_jsonl(raw / relative, [row])
    expected = {
        "AutoScholarQuery/train.jsonl": 1,
        "AutoScholarQuery/dev.jsonl": 1,
        "AutoScholarQuery/test.jsonl": 1,
        "RealScholarQuery/test.jsonl": 1,
    }

    freeze_pasa_training_data(
        raw_root=raw,
        private_output_root=tmp_path / "private-a",
        manifest_path=tmp_path / "manifest-a.json",
        revision="fixed-revision",
        asta_access_status="unauthorized",
        expected_counts=expected,
    )
    freeze_pasa_training_data(
        raw_root=raw,
        private_output_root=tmp_path / "private-b",
        manifest_path=tmp_path / "manifest-b.json",
        revision="fixed-revision",
        asta_access_status="unauthorized",
        expected_counts=expected,
    )

    assert (tmp_path / "manifest-a.json").read_bytes() == (
        tmp_path / "manifest-b.json"
    ).read_bytes()
    assert (tmp_path / "private-a/partitions/pasa_auto_train.jsonl").read_bytes() == (
        tmp_path / "private-b/partitions/pasa_auto_train.jsonl"
    ).read_bytes()
