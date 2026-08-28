from __future__ import annotations

import json

from paper_search.learning.openalex_daily_schedule import SearchActionIdentity
from paper_search.learning.openalex_dev_backfill import (
    assign_work_to_keys,
    build_work,
    inventory_core4_receipts,
    select_query_ids,
)


def _identity(text: str) -> SearchActionIdentity:
    return SearchActionIdentity(
        action_type="text_search",
        search_mode="lexical",
        normalized_text=text,
    )


def test_select_query_ids_excludes_conflicts_and_preserves_existing_complete() -> None:
    selected = select_query_ids(
        required_query_ids=["q1", "q2", "q3", "q4"],
        complete_query_ids=["q1"],
        conflict_query_ids=["q2"],
        target_count=3,
    )
    assert selected == ("q1", "q3", "q4")


def test_assign_work_respects_per_key_budget() -> None:
    required = {query_id: (_identity(query_id),) for query_id in ("q1", "q2", "q3")}
    work = build_work(tuple(required), required)
    assigned = assign_work_to_keys(work, key_count=2, max_calls=2)
    assert sum(len(items) for items in assigned.values()) == 3
    assert all(
        sum(len(item.missing_actions) for item in items) <= 2
        for items in assigned.values()
    )


def test_inventory_finds_receipts_below_independent_batch_directory(tmp_path) -> None:
    identity = _identity("alpha")
    batch = tmp_path / "key-01" / "openalex" / "batch-0001"
    generation = batch / "generation" / "attempt-01"
    retrieval = batch / "retrieval" / "attempt-01"
    generation.mkdir(parents=True)
    retrieval.mkdir(parents=True)
    (generation / "q1.json").write_text(
        json.dumps(
            {
                "query_id": "q1",
                "attempt_status": "succeeded",
                "generation_provenance": {
                    "candidate_policy": "core4-semantic-boolean-missing-actions-v1",
                    "collection_mode": "scheduled_missing_actions",
                },
                "actions": [
                    {
                        "action_id": "a1",
                        "action_type": "text_search",
                        "payload": {"query_text": "alpha", "search_mode": "lexical"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (retrieval / "q1.json").write_text(
        json.dumps(
            {
                "attempt_status": "succeeded",
                "results": [
                    {
                        "action_id": "a1",
                        "errors": [],
                        "infrastructure_failure": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = inventory_core4_receipts(
        receipt_roots=[tmp_path], required={"q1": (identity,)}
    )

    assert result["complete_query_ids"] == ["q1"], result
    assert result["files_scanned"] == 2
