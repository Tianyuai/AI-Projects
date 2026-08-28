"""Fail-closed live/replay delivery equivalence checks."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

from paper_search.evaluation.official_adapter import InternalPredictionRecord


def _canonical_bytes(records: Sequence[InternalPredictionRecord]) -> bytes:
    rows = [
        InternalPredictionRecord.model_validate(record).model_dump(mode="json")
        for record in records
    ]
    return (
        json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def compare_delivery_predictions(
    live_predictions: Sequence[InternalPredictionRecord],
    replay_predictions: Sequence[InternalPredictionRecord],
) -> dict[str, object]:
    """Require identical query coverage and final ranked IDs across both modes."""

    live = [InternalPredictionRecord.model_validate(row) for row in live_predictions]
    replay = [InternalPredictionRecord.model_validate(row) for row in replay_predictions]
    live_ids = [row.query_id for row in live]
    replay_ids = [row.query_id for row in replay]
    if live_ids != replay_ids:
        raise ValueError("live/replay query coverage or order mismatch")
    mismatched = [
        left.query_id
        for left, right in zip(live, replay, strict=True)
        if left.selected_paper_ids != right.selected_paper_ids
    ]
    if mismatched:
        raise ValueError(
            "live/replay ranking mismatch for " + ", ".join(mismatched[:10])
        )
    live_bytes = _canonical_bytes(live)
    replay_bytes = _canonical_bytes(replay)
    return {
        "schema_version": "delivery-live-replay-rehearsal-v1",
        "passed": True,
        "query_count": len(live),
        "identical_query_count": len(live),
        "mismatched_query_ids": [],
        "live_predictions_sha256": _sha256(live_bytes),
        "replay_predictions_sha256": _sha256(replay_bytes),
        "test_partition_touched": False,
    }


__all__ = ["compare_delivery_predictions"]
