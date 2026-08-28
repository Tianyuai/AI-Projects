"""Load one frozen OpenAlex daily shard without exposing secret material."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from datetime import date
from pathlib import Path

from pydantic import Field

from paper_search.domain.models import DomainModel, Sha256
from paper_search.learning.openalex_daily_schedule import (
    OpenAlexDailyTrainingPlan,
    ScheduledQueryActions,
    build_missing_action_work,
    load_settled_search_action_identities,
)


class PreparedOpenAlexTrainingShard(DomainModel):
    plan_sha256: Sha256
    window: date
    key_slot: int = Field(strict=True, gt=0)
    max_search_calls: int = Field(strict=True, gt=0)
    planned_search_calls: int = Field(strict=True, gt=0)
    remaining_search_calls: int = Field(strict=True, ge=0)
    rows: tuple[dict[str, object], ...]
    work: tuple[ScheduledQueryActions, ...]


def _sha256(content: bytes) -> Sha256:
    return "sha256:" + hashlib.sha256(content).hexdigest()


def _load_training_rows(path: Path, *, expected_sha256: Sha256) -> list[dict[str, object]]:
    content = path.read_bytes()
    if _sha256(content) != expected_sha256:
        raise ValueError("training partition does not match the frozen plan")
    rows = [
        json.loads(line)
        for line in content.decode("utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("training partition is empty")
    for row in rows:
        if (
            not isinstance(row, dict)
            or row.get("dataset") != "pasa"
            or row.get("split") != "auto_train"
            or row.get("role") != "training"
        ):
            raise ValueError("daily OpenAlex runner requires isolated auto_train rows")
    return rows


def load_scheduled_training_shard(
    *,
    plan_path: Path,
    expected_plan_sha256: Sha256,
    partition_path: Path,
    window: date,
    key_slot: int,
    completed_receipt_roots: Sequence[Path] = (),
) -> PreparedOpenAlexTrainingShard:
    """Bind one scheduled shard to its exact plan and auto_train partition."""

    plan_bytes = plan_path.read_bytes()
    observed_plan_sha256 = _sha256(plan_bytes)
    if observed_plan_sha256 != expected_plan_sha256:
        raise ValueError("OpenAlex training plan hash does not match")
    plan = OpenAlexDailyTrainingPlan.model_validate_json(plan_bytes)
    rows = _load_training_rows(
        partition_path,
        expected_sha256=plan.partition_sha256,
    )
    shard = next(
        (
            item
            for item in plan.schedule.shards
            if item.window == window and item.key_slot == key_slot
        ),
        None,
    )
    if shard is None:
        raise ValueError("OpenAlex training shard is not present in the frozen plan")
    row_by_query = {str(row.get("query_id")): row for row in rows}
    if len(row_by_query) != len(rows):
        raise ValueError("training partition query IDs must be unique")
    required_by_query = {
        item.query_id: item.missing_actions for item in shard.queries
    }
    completed = (
        load_settled_search_action_identities(completed_receipt_roots)
        if completed_receipt_roots
        else {}
    )
    remaining_work = build_missing_action_work(required_by_query, completed)
    selected_rows: list[dict[str, object]] = []
    for item in remaining_work:
        row = row_by_query.get(item.query_id)
        if row is None:
            raise ValueError(f"scheduled query is missing from partition: {item.query_id}")
        selected_rows.append(row)
    return PreparedOpenAlexTrainingShard(
        plan_sha256=observed_plan_sha256,
        window=window,
        key_slot=key_slot,
        max_search_calls=shard.max_search_calls,
        planned_search_calls=shard.planned_search_calls,
        remaining_search_calls=sum(
            len(item.missing_actions) for item in remaining_work
        ),
        rows=tuple(selected_rows),
        work=tuple(remaining_work),
    )


def select_smoke_work(
    plan: OpenAlexDailyTrainingPlan,
    *,
    key_slot: int,
) -> ScheduledQueryActions:
    """Select one productive, deterministic action for a key-slot smoke."""

    for shard in plan.schedule.shards:
        if shard.key_slot == key_slot:
            first = shard.queries[0]
            return ScheduledQueryActions(
                query_id=first.query_id,
                missing_actions=(first.missing_actions[0],),
            )
    raise ValueError("OpenAlex key slot has no scheduled smoke work")


def is_scheduled_work_complete(
    work: Sequence[ScheduledQueryActions],
    receipt_roots: Sequence[Path],
) -> bool:
    """Return true only when every scheduled identity has a successful receipt."""

    required = {item.query_id: item.missing_actions for item in work}
    completed = load_settled_search_action_identities(receipt_roots)
    return not build_missing_action_work(required, completed)


def is_recoverable_canary_failure(error: RuntimeError) -> bool:
    """Recognize the service outcome produced by a failed provider batch."""

    return str(error) == "canary produced no valid repeat"


def batch_has_provider_quota_exhaustion(run_path: Path) -> bool:
    """Return true when a saved retrieval receipt records exhausted OpenAlex quota."""

    retrieval_root = run_path / "retrieval"
    for receipt_path in retrieval_root.glob("attempt-*/*.json"):
        try:
            payload = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        errors = payload.get("errors") if isinstance(payload, dict) else None
        if not isinstance(errors, list):
            continue
        if any(
            isinstance(error, dict)
            and error.get("provider") == "openalex"
            and error.get("code") == "quota_exhausted"
            for error in errors
        ):
            return True
    return False


__all__ = [
    "PreparedOpenAlexTrainingShard",
    "batch_has_provider_quota_exhaustion",
    "is_scheduled_work_complete",
    "is_recoverable_canary_failure",
    "load_scheduled_training_shard",
    "select_smoke_work",
]
