from __future__ import annotations

import json
import sqlite3
from datetime import date
from pathlib import Path

from paper_search.domain.models import QuerySpec
from scripts import prepare_query_adaptive_recall_pilot as pilot_module
from scripts.prepare_query_adaptive_recall_pilot import (
    _load_excluded_query_ids,
    _sampling_bucket,
)


def test_validation_sampling_bucket_does_not_depend_on_gold_labels() -> None:
    spec = QuerySpec(
        original_query="find benchmark studies",
        research_goal="find benchmark studies",
        datasets=["CO3D"],
    )
    one_gold = {
        "query": "find benchmark studies",
        "gold_paper_ids": ["arxiv:1"],
    }
    many_gold = {
        "query": "find benchmark studies",
        "gold_paper_ids": ["arxiv:1", "arxiv:2", "arxiv:3"],
    }

    assert _sampling_bucket(one_gold, spec=spec, length_cuts=[3, 6, 9]) == (
        "dataset",
        "1",
    )
    assert _sampling_bucket(many_gold, spec=spec, length_cuts=[3, 6, 9]) == (
        "dataset",
        "1",
    )


def test_exclusion_inventory_unions_manifests_and_saved_receipt_queries(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"sample": [{"query_id": "q-manifest"}]}),
        encoding="utf-8",
    )
    receipt_root = tmp_path / "receipts"
    generation = receipt_root / "openalex" / "batch-0001" / "generation" / "attempt-01"
    generation.mkdir(parents=True)
    (generation / "q-receipt.json").write_text(
        json.dumps({"query_id": "q-receipt"}),
        encoding="utf-8",
    )

    assert _load_excluded_query_ids(
        manifest_paths=[manifest],
        receipt_roots=[receipt_root],
    ) == {"q-manifest", "q-receipt"}


def test_quota_snapshot_can_use_current_ledger_window(tmp_path: Path) -> None:
    ledger = tmp_path / "quota.sqlite3"
    with sqlite3.connect(ledger) as connection:
        connection.execute(
            """
            CREATE TABLE quota_usage (
                window TEXT NOT NULL,
                key_slot INTEGER NOT NULL,
                max_search_calls INTEGER NOT NULL,
                used_search_calls INTEGER NOT NULL,
                PRIMARY KEY (window, key_slot)
            )
            """
        )
        connection.executemany(
            "INSERT INTO quota_usage VALUES (?, ?, ?, ?)",
            [
                ("2026-08-24", 1, 880, 700),
                ("2026-08-25", 1, 880, 91),
                ("2026-08-25", 2, 880, 98),
            ],
        )

    assert hasattr(pilot_module, "_load_quota_usage_from_ledger")
    snapshot = pilot_module._load_quota_usage_from_ledger(
        ledger,
        window=date(2026, 8, 25),
    )

    assert snapshot == {
        "window": "2026-08-25",
        "capacity": 1760,
        "used": 189,
        "remaining": 1571,
        "rows": [
            {
                "window": "2026-08-25",
                "key_slot": 1,
                "max_search_calls": 880,
                "used_search_calls": 91,
            },
            {
                "window": "2026-08-25",
                "key_slot": 2,
                "max_search_calls": 880,
                "used_search_calls": 98,
            },
        ],
    }


def test_sampling_digest_is_bound_to_candidate_policy() -> None:
    assert hasattr(pilot_module, "_sampling_digest")
    assert pilot_module._sampling_digest(
        "q-1", candidate_policy="query-adaptive-high-recall-v1"
    ) != pilot_module._sampling_digest(
        "q-1", candidate_policy="query-adaptive-high-recall-v2"
    )
