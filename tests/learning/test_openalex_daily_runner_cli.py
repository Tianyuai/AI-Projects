from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

from paper_search.learning.openalex_daily_schedule import (
    ScheduledQueryActions,
    SearchActionIdentity,
    current_openalex_quota_window,
)
from scripts import run_openalex_daily_training as daily_cli
from scripts.run_openalex_daily_training import build_parser, quota_is_exhausted


def test_daily_runner_cli_requires_plan_hash_and_explicit_mode() -> None:
    args = build_parser().parse_args(
        [
            "inspect",
            "--plan",
            "plan.json",
            "--plan-sha256",
            "sha256:" + "1" * 64,
            "--partition",
            "auto_train.jsonl",
            "--window",
            "2026-08-19",
            "--key-slot",
            "3",
        ]
    )

    assert args.command == "inspect"
    assert args.plan_sha256 == "sha256:" + "1" * 64
    assert args.key_slot == 3


def test_daily_runner_stops_when_the_key_hard_cap_is_reached() -> None:
    assert not quota_is_exhausted(899, 900)
    assert quota_is_exhausted(900, 900)


def test_daily_runner_reports_hard_cap_reached_on_the_last_batch() -> None:
    assert daily_cli._reported_hard_cap_status(
        stopped_before_batch=False,
        used_search_calls=900,
        max_search_calls=900,
    )


def test_daily_runner_stops_key_after_provider_quota_exhaustion(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    calls = 0

    class FakeBundle:
        async def aclose(self) -> None:
            return None

    class FakeService:
        def __init__(self, *, workspace_root: Path) -> None:
            del workspace_root

        async def run(self, **kwargs: object) -> None:
            nonlocal calls
            calls += 1
            output_path = kwargs["output_path"]
            assert isinstance(output_path, Path)
            receipt = output_path / "retrieval" / "attempt-01" / "q-1.json"
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
            raise RuntimeError("canary produced no valid repeat")

    async def fake_build_bundle(**kwargs: object) -> FakeBundle:
        del kwargs
        return FakeBundle()

    monkeypatch.setattr(daily_cli, "load_runtime_profile", lambda path: object())
    monkeypatch.setattr(
        daily_cli,
        "resolve_runtime_secrets",
        lambda profile, openalex_key_slot: object(),
    )
    monkeypatch.setattr(daily_cli, "load_recall_recipe", lambda path: object())
    monkeypatch.setattr(daily_cli, "build_live_runtime_bundle", fake_build_bundle)
    monkeypatch.setattr(daily_cli, "RecallCanaryService", FakeService)
    rows = tuple(
        {
            "query_id": query_id,
            "query": query_id,
            "gold_paper_ids": [f"arxiv:{query_id}"],
        }
        for query_id in ("q-1", "q-2")
    )
    work = tuple(
        ScheduledQueryActions(
            query_id=query_id,
            missing_actions=(
                SearchActionIdentity(
                    action_type="text_search",
                    search_mode="lexical",
                    normalized_text=query_id,
                ),
            ),
        )
        for query_id in ("q-1", "q-2")
    )

    result = asyncio.run(
        daily_cli._execute_work(
            workspace_root=tmp_path,
            rows=rows,
            work=work,
            output_root=tmp_path / "output",
            ledger_path=tmp_path / "quota.sqlite3",
            window=current_openalex_quota_window(datetime.now(UTC)),
            key_slot=1,
            max_search_calls=900,
            chunk_size=1,
            profile_path=tmp_path / "profile.yaml",
            recipe_path=tmp_path / "recipe.yaml",
        )
    )

    assert calls == 1
    assert result["external_quota_exhausted"] is True
