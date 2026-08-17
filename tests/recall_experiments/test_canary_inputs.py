from __future__ import annotations

import json
from pathlib import Path

import pytest

from paper_search.recall_experiments.canary_inputs import (
    LoadedCanaryInput,
    RecallCase,
    load_canary_input,
    load_jsonl_cases,
    load_single_case,
)


def test_single_query_becomes_one_unscored_recall_case() -> None:
    cases = load_single_case("Which paper introduced dataset distillation?", "user-001")

    assert cases == (
        RecallCase(
            query_id="user-001",
            query="Which paper introduced dataset distillation?",
            gold_paper_ids=None,
        ),
    )


def test_jsonl_accepts_scored_rows_with_one_shape(tmp_path: Path) -> None:
    path = tmp_path / "queries.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "query_id": "q-1",
                        "query": "first",
                        "gold_paper_ids": ["arxiv:1811.10959"],
                    }
                ),
                json.dumps(
                    {
                        "query_id": "q-2",
                        "query": "second",
                        "gold_paper_ids": ["arxiv:1811.10959"],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    cases = load_jsonl_cases(path)

    assert [case.query_id for case in cases] == ["q-1", "q-2"]
    assert cases[0].gold_paper_ids == ("arxiv:1811.10959",)
    assert cases[1].gold_paper_ids == ("arxiv:1811.10959",)


def test_jsonl_rejects_mixed_gold_coverage(tmp_path: Path) -> None:
    path = tmp_path / "queries.jsonl"
    path.write_text(
        '{"query_id":"q-1","query":"first"}\n'
        '{"query_id":"q-2","query":"second","gold_paper_ids":["arxiv:1"]}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="all cases must consistently include Gold"):
        load_jsonl_cases(path)


def test_cases_reject_duplicate_query_ids(tmp_path: Path) -> None:
    path = tmp_path / "queries.jsonl"
    path.write_text(
        '{"query_id":"q-1","query":"first"}\n'
        '{"query_id":"q-1","query":"second"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="query IDs must be unique"):
        load_jsonl_cases(path)


def test_unified_input_loader_binds_single_query_bytes() -> None:
    loaded = load_canary_input(
        input_kind="single",
        query="graph retrieval",
        query_id="user-1",
        source_path=None,
        identifier_map_path=None,
        workspace_root=Path.cwd(),
    )

    assert isinstance(loaded, LoadedCanaryInput)
    assert loaded.input_kind == "single"
    assert loaded.evaluation_status == "not_available"
    assert loaded.input_sha256.startswith("sha256:")
    assert loaded.identifier_map_bytes is None


def test_scored_jsonl_requires_identifier_map(tmp_path: Path) -> None:
    source = tmp_path / "queries.jsonl"
    source.write_text(
        '{"query_id":"q-1","query":"first","gold_paper_ids":["arxiv:1811.10959"]}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="identifier map is required"):
        load_canary_input(
            input_kind="jsonl",
            query=None,
            query_id=None,
            source_path=source,
            identifier_map_path=None,
            workspace_root=Path.cwd(),
        )
