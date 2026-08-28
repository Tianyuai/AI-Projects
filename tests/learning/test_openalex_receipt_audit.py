from __future__ import annotations

import json
from pathlib import Path

from paper_search.learning.openalex_receipt_audit import audit_openalex_receipts


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_pair(
    root: Path,
    query_id: str,
    *,
    action_id: str,
    paper_id: str,
    arxiv_id: str | None = None,
) -> None:
    base = root / "key-01" / "openalex" / f"batch-{query_id}"
    _write_json(
        base / "generation" / "attempt-01" / f"{query_id}.json",
        {
            "query_id": query_id,
            "attempt_status": "succeeded",
            "actions": [
                {
                    "action_id": action_id,
                    "action_type": "text_search",
                    "payload": {"query_text": f"query {query_id}"},
                }
            ],
        },
    )
    _write_json(
        base / "retrieval" / "attempt-01" / f"{query_id}.json",
        {
            "query_id": query_id,
            "attempt_status": "succeeded",
            "results": [
                {
                    "action_id": action_id,
                    "action_type": "text_search",
                    "errors": [],
                    "hits": [
                        {
                            "canonical_id": paper_id,
                            "title": "A paper",
                            **({"arxiv_id": arxiv_id} if arxiv_id is not None else {}),
                        }
                    ],
                    "infrastructure_failure": False,
                }
            ],
        },
    )


def _partition(path: Path, *query_ids: str) -> None:
    path.write_text(
        "".join(
            json.dumps(
                {
                    "query_id": query_id,
                    "role": "training",
                    "split": "auto_train",
                    "query": f"query {query_id}",
                    "gold_paper_ids": ["gold-1"],
                }
            )
            + "\n"
            for query_id in query_ids
        ),
        encoding="utf-8",
    )


def test_audit_is_summary_only_and_incremental(tmp_path: Path) -> None:
    root = tmp_path / "window"
    partition = tmp_path / "partition.jsonl"
    state = tmp_path / "state.json"
    _partition(partition, "q1", "q2")
    _write_pair(root, "q1", action_id="a1", paper_id="gold-1")

    first = audit_openalex_receipts(
        receipt_roots=[root], partition_path=partition, state_path=state
    )
    assert first["files_new"] == 1
    assert first["files_reused"] == 0
    assert first["candidate_ready_query_count"] == 1
    assert first["trainable_query_count"] == 1
    assert first["network_calls"] == 0
    assert first["llm_calls"] == 0
    assert "records" not in first

    second = audit_openalex_receipts(
        receipt_roots=[root], partition_path=partition, state_path=state
    )
    assert second["files_new"] == 0
    assert second["files_reused"] == 1
    assert second["unique_candidate_count"] == 1

    _write_pair(root, "q2", action_id="a2", paper_id="p2")
    third = audit_openalex_receipts(
        receipt_roots=[root], partition_path=partition, state_path=state
    )
    assert third["files_new"] == 1
    assert third["candidate_ready_query_count"] == 2
    assert third["unique_candidate_count"] == 2


def test_audit_counts_invalid_receipts_without_emitting_paths(tmp_path: Path) -> None:
    root = tmp_path / "window"
    partition = tmp_path / "partition.jsonl"
    state = tmp_path / "state.json"
    _partition(partition, "q1")
    base = root / "key-01" / "openalex" / "batch-q1"
    _write_json(
        base / "retrieval" / "attempt-01" / "q1.json",
        {"query_id": "q1", "attempt_status": "succeeded", "results": []},
    )
    (base / "retrieval" / "attempt-01" / "broken.json").write_text(
        "not-json", encoding="utf-8"
    )

    result = audit_openalex_receipts(
        receipt_roots=[root],
        partition_path=partition,
        state_path=state,
        include_query_rows=True,
    )
    assert result["parse_error_count"] == 1
    assert result["candidate_ready_query_count"] == 0
    assert "broken.json" not in json.dumps(result)


def test_audit_preserves_previous_daily_root_when_a_new_root_is_supplied(
    tmp_path: Path,
) -> None:
    root1 = tmp_path / "window-01"
    root2 = tmp_path / "window-02"
    partition = tmp_path / "partition.jsonl"
    state = tmp_path / "state.json"
    _partition(partition, "q1", "q2")
    _write_pair(root1, "q1", action_id="a1", paper_id="gold-1")
    _write_pair(root2, "q2", action_id="a2", paper_id="gold-1")
    first = audit_openalex_receipts(
        receipt_roots=[root1], partition_path=partition, state_path=state
    )
    assert first["unique_candidate_count"] == 1
    second = audit_openalex_receipts(
        receipt_roots=[root2], partition_path=partition, state_path=state
    )
    assert second["files_new"] == 1
    assert second["files_removed"] == 0
    assert second["unique_candidate_count"] == 2
    assert second["receipt_roots"] == sorted(
        [str(root1.resolve()), str(root2.resolve())]
    )


def test_audit_matches_arxiv_gold_through_candidate_alias(tmp_path: Path) -> None:
    root = tmp_path / "window"
    partition = tmp_path / "partition.jsonl"
    state = tmp_path / "state.json"
    partition.write_text(
        json.dumps(
            {
                "query_id": "q1",
                "role": "training",
                "split": "auto_train",
                "query": "query q1",
                "gold_paper_ids": ["arxiv:2301.01234"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_pair(
        root,
        "q1",
        action_id="a1",
        paper_id="doi:10.1145/example",
        arxiv_id="2301.01234v2",
    )

    result = audit_openalex_receipts(
        receipt_roots=[root],
        partition_path=partition,
        state_path=state,
        include_query_rows=True,
    )

    assert result["candidate_ready_query_count"] == 1
    assert result["gold_hit_query_count"] == 1
    assert result["gold_hit_count"] == 1
    assert result["query_rows"] == [
        {
            "query_id": "q1",
            "candidate_count": 1,
            "gold_paper_count": 1,
            "gold_hit_count": 1,
            "hard_negative_candidate_count": 0,
            "positive_and_hard_negative": False,
        }
    ]
