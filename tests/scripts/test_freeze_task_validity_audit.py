from __future__ import annotations

from scripts.freeze_task_validity_audit import _build_evidence_rows


def test_evidence_rows_assign_blind_hit_and_miss_cohorts() -> None:
    rows = [
        {"query_id": "q-1", "fold": 1},
        {"query_id": "q-2", "fold": 2},
    ]

    evidence = _build_evidence_rows(rows, hit_query_ids={"q-2"}, role="training")

    assert evidence == [
        {"query_id": "q-1", "role": "training", "cohort": "miss", "fold": 1},
        {"query_id": "q-2", "role": "training", "cohort": "hit", "fold": 2},
    ]
