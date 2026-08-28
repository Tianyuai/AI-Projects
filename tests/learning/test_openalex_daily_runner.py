from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import pytest

from paper_search.learning.openalex_daily_runner import (
    batch_has_provider_quota_exhaustion,
    is_recoverable_canary_failure,
    is_scheduled_work_complete,
    load_scheduled_training_shard,
    select_smoke_work,
)
from paper_search.learning.openalex_daily_schedule import (
    OpenAlexDailyTrainingPlan,
    ScheduledQueryActions,
    SearchActionIdentity,
    build_daily_openalex_schedule,
)


def _work(query_id: str, text: str) -> ScheduledQueryActions:
    return ScheduledQueryActions(
        query_id=query_id,
        missing_actions=(
            SearchActionIdentity(
                action_type="text_search",
                search_mode="lexical",
                normalized_text=text,
            ),
        ),
    )


def _write_plan(path: Path, partition_bytes: bytes) -> str:
    schedule = build_daily_openalex_schedule(
        [_work("q-1", "first query"), _work("q-2", "second query")],
        first_window=date(2026, 8, 19),
        last_training_window=date(2026, 8, 19),
        final_test_window=date(2026, 8, 31),
        key_count=2,
        max_search_calls_per_key=1,
    )
    plan = OpenAlexDailyTrainingPlan(
        schema_version="openalex-daily-training-plan-v1",
        partition_sha256="sha256:" + hashlib.sha256(partition_bytes).hexdigest(),
        receipt_inventory_sha256="sha256:" + "2" * 64,
        required_query_count=2,
        required_search_actions=2,
        reused_search_actions=0,
        missing_search_actions=2,
        schedule=schedule,
    )
    content = json.dumps(
        plan.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    path.write_bytes(content)
    return "sha256:" + hashlib.sha256(content).hexdigest()


def test_load_scheduled_shard_binds_plan_partition_and_key_slot(
    tmp_path: Path,
) -> None:
    partition = tmp_path / "auto_train.jsonl"
    partition_bytes = (
        json.dumps(
            {
                "dataset": "pasa",
                "split": "auto_train",
                "role": "training",
                "query_id": "q-1",
                "query": "first query",
                "gold_paper_ids": ["arxiv:1"],
            }
        )
        + "\n"
        + json.dumps(
            {
                "dataset": "pasa",
                "split": "auto_train",
                "role": "training",
                "query_id": "q-2",
                "query": "second query",
                "gold_paper_ids": ["arxiv:2"],
            }
        )
        + "\n"
    ).encode("utf-8")
    partition.write_bytes(partition_bytes)
    plan_path = tmp_path / "plan.json"
    plan_sha256 = _write_plan(plan_path, partition_bytes)

    prepared = load_scheduled_training_shard(
        plan_path=plan_path,
        expected_plan_sha256=plan_sha256,
        partition_path=partition,
        window=date(2026, 8, 19),
        key_slot=2,
    )

    assert prepared.key_slot == 2
    assert prepared.planned_search_calls == 1
    assert prepared.remaining_search_calls == 1
    assert [row["query_id"] for row in prepared.rows] == ["q-2"]
    assert [item.query_id for item in prepared.work] == ["q-2"]


def test_load_scheduled_shard_rejects_wrong_plan_hash(tmp_path: Path) -> None:
    partition = tmp_path / "auto_train.jsonl"
    partition_bytes = b'{"dataset":"pasa"}\n'
    partition.write_bytes(partition_bytes)
    plan_path = tmp_path / "plan.json"
    _write_plan(plan_path, partition_bytes)

    with pytest.raises(ValueError, match="plan hash"):
        load_scheduled_training_shard(
            plan_path=plan_path,
            expected_plan_sha256="sha256:" + "0" * 64,
            partition_path=partition,
            window=date(2026, 8, 19),
            key_slot=1,
        )


def test_smoke_selects_one_productive_action_for_each_key(tmp_path: Path) -> None:
    partition_bytes = b"partition"
    plan_path = tmp_path / "plan.json"
    _write_plan(plan_path, partition_bytes)
    plan = OpenAlexDailyTrainingPlan.model_validate_json(plan_path.read_bytes())

    first = select_smoke_work(plan, key_slot=1)
    second = select_smoke_work(plan, key_slot=2)

    assert len(first.missing_actions) == 1
    assert len(second.missing_actions) == 1
    assert first.query_id != second.query_id


def test_only_no_valid_repeat_is_a_recoverable_batch_failure() -> None:
    assert is_recoverable_canary_failure(
        RuntimeError("canary produced no valid repeat")
    )
    assert not is_recoverable_canary_failure(RuntimeError("programming defect"))


def test_batch_receipt_reports_provider_quota_exhaustion(tmp_path: Path) -> None:
    receipt = tmp_path / "retrieval" / "attempt-01" / "q-1.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text(
        json.dumps(
            {
                "query_id": "q-1",
                "attempt_status": "failed",
                "errors": [
                    {
                        "code": "quota_exhausted",
                        "provider": "openalex",
                        "retryable": False,
                    }
                ],
                "results": [],
            }
        ),
        encoding="utf-8",
    )

    assert batch_has_provider_quota_exhaustion(tmp_path)


def test_load_scheduled_shard_removes_successful_resume_receipts(
    tmp_path: Path,
) -> None:
    partition = tmp_path / "auto_train.jsonl"
    partition_bytes = (
        json.dumps(
            {
                "dataset": "pasa",
                "split": "auto_train",
                "role": "training",
                "query_id": "q-1",
                "query": "first query",
                "gold_paper_ids": ["arxiv:1"],
            }
        )
        + "\n"
        + json.dumps(
            {
                "dataset": "pasa",
                "split": "auto_train",
                "role": "training",
                "query_id": "q-2",
                "query": "second query",
                "gold_paper_ids": ["arxiv:2"],
            }
        )
        + "\n"
    ).encode("utf-8")
    partition.write_bytes(partition_bytes)
    plan_path = tmp_path / "plan.json"
    plan_sha256 = _write_plan(plan_path, partition_bytes)
    receipts = tmp_path / "receipts"
    generation = receipts / "generation" / "attempt-01" / "q-2.json"
    retrieval = receipts / "retrieval" / "attempt-01" / "q-2.json"
    generation.parent.mkdir(parents=True)
    retrieval.parent.mkdir(parents=True)
    generation.write_text(
        json.dumps(
            {
                "query_id": "q-2",
                "attempt_status": "succeeded",
                "actions": [
                    {
                        "action_id": "old-id",
                        "action_type": "text_search",
                        "payload": {
                            "query_text": "second query",
                            "search_mode": "lexical",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    retrieval.write_text(
        json.dumps(
            {
                "query_id": "q-2",
                "attempt_status": "succeeded",
                "results": [
                    {
                        "action_id": "old-id",
                        "errors": [],
                        "infrastructure_failure": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    prepared = load_scheduled_training_shard(
        plan_path=plan_path,
        expected_plan_sha256=plan_sha256,
        partition_path=partition,
        window=date(2026, 8, 19),
        key_slot=2,
        completed_receipt_roots=[receipts],
    )

    assert prepared.remaining_search_calls == 0
    assert prepared.rows == ()
    assert prepared.work == ()
    assert is_scheduled_work_complete([_work("q-2", "second query")], [receipts])
