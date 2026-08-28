from __future__ import annotations

import asyncio
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from paper_search.domain.models import QuerySpec
from paper_search.learning.candidate_ceiling import QueryAdaptiveHighRecallGenerator
from paper_search.learning.openalex_daily_schedule import (
    OpenAlexDailyQuotaExceededError,
    OpenAlexDailyTrainingPlan,
    SQLiteOpenAlexDailyQuotaLedger,
    ScheduledMissingActionGenerator,
    ScheduledQueryActions,
    SearchActionIdentity,
    build_daily_openalex_schedule,
    build_missing_action_work,
    build_openalex_daily_training_plan,
    load_completed_search_action_identities,
    estimate_max_openalex_search_api_calls,
    load_settled_search_action_identities,
    required_a_prime_action_identities,
)
from paper_search.recall_experiments.contracts import RecallGenerationContext


def _actions(query_id: str, count: int) -> ScheduledQueryActions:
    return ScheduledQueryActions(
        query_id=query_id,
        missing_actions=tuple(
            SearchActionIdentity(
                action_type="text_search",
                search_mode="lexical",
                normalized_text=f"{query_id} action {index}",
            )
            for index in range(count)
        ),
    )


def test_raw_openalex_request_estimate_accounts_for_pagination_and_semantic_cap() -> None:
    actions = (
        SearchActionIdentity(
            action_type="text_search",
            search_mode="lexical",
            normalized_text="lexical query",
        ),
        SearchActionIdentity(
            action_type="text_search",
            search_mode="semantic",
            normalized_text="semantic query",
        ),
        SearchActionIdentity(
            action_type="title_search",
            search_mode="lexical",
            normalized_text="exact title",
        ),
    )

    assert estimate_max_openalex_search_api_calls(
        actions, max_results_per_action=100
    ) == 5
    assert estimate_max_openalex_search_api_calls(
        actions, max_results_per_action=50
    ) == 3


def test_raw_openalex_request_estimate_rejects_invalid_depth() -> None:
    with pytest.raises(ValueError, match="between 1 and 300"):
        estimate_max_openalex_search_api_calls(
            _actions("q", 1).missing_actions,
            max_results_per_action=0,
        )


def test_daily_schedule_assigns_every_action_once_with_per_key_hard_caps() -> None:
    work = [_actions(f"q-{index}", count) for index, count in enumerate((6, 5, 4, 3, 2))]

    schedule = build_daily_openalex_schedule(
        work,
        first_window=date(2026, 8, 18),
        last_training_window=date(2026, 8, 19),
        final_test_window=date(2026, 8, 20),
        key_count=2,
        max_search_calls_per_key=6,
    )

    assert schedule.schema_version == "openalex-daily-training-schedule-v1"
    assert schedule.timezone == "Asia/Shanghai"
    assert schedule.reset_hour_local == 8
    assert schedule.final_test_window == date(2026, 8, 20)
    assert sum(shard.planned_search_calls for shard in schedule.shards) == 20
    assert all(shard.planned_search_calls <= 6 for shard in schedule.shards)
    assert {(shard.window, shard.key_slot) for shard in schedule.shards} == {
        (date(2026, 8, 18), 1),
        (date(2026, 8, 18), 2),
        (date(2026, 8, 19), 1),
        (date(2026, 8, 19), 2),
    }
    scheduled = [
        (query.query_id, action)
        for shard in schedule.shards
        for query in shard.queries
        for action in query.missing_actions
    ]
    expected = [
        (query.query_id, action)
        for query in work
        for action in query.missing_actions
    ]
    assert sorted(scheduled, key=str) == sorted(expected, key=str)
    assert len(scheduled) == len(set(scheduled))


def test_daily_schedule_is_deterministic_independent_of_input_order() -> None:
    work = [_actions("q-b", 2), _actions("q-a", 3), _actions("q-c", 1)]

    first = build_daily_openalex_schedule(
        work,
        first_window=date(2026, 8, 18),
        last_training_window=date(2026, 8, 18),
        final_test_window=date(2026, 8, 19),
        key_count=2,
        max_search_calls_per_key=4,
    )
    second = build_daily_openalex_schedule(
        list(reversed(work)),
        first_window=date(2026, 8, 18),
        last_training_window=date(2026, 8, 18),
        final_test_window=date(2026, 8, 19),
        key_count=2,
        max_search_calls_per_key=4,
    )

    assert first == second


def test_training_plan_freezes_source_hashes_and_reuse_counts() -> None:
    rows = [
        {
            "query_id": "q-1",
            "query": "Which methods improve academic paper retrieval quality?",
            "dataset": "pasa",
            "split": "auto_train",
            "role": "training",
        }
    ]
    required = asyncio.run(required_a_prime_action_identities(rows))
    reused = required["q-1"][:1]

    plan = asyncio.run(
        build_openalex_daily_training_plan(
            rows,
            partition_sha256="sha256:" + "1" * 64,
            completed_by_query={"q-1": frozenset(reused)},
            first_window=date(2026, 8, 18),
            last_training_window=date(2026, 8, 30),
            final_test_window=date(2026, 8, 31),
            key_count=11,
            max_search_calls_per_key=900,
        )
    )

    assert isinstance(plan, OpenAlexDailyTrainingPlan)
    assert plan.schema_version == "openalex-daily-training-plan-v1"
    assert plan.partition_sha256 == "sha256:" + "1" * 64
    assert plan.required_query_count == 1
    assert plan.required_search_actions == len(required["q-1"])
    assert plan.reused_search_actions == 1
    assert plan.missing_search_actions == len(required["q-1"]) - 1
    assert plan.receipt_inventory_sha256.startswith("sha256:")
    assert plan.schedule.planned_search_calls == plan.missing_search_actions


def test_scheduled_generator_emits_only_the_frozen_missing_action() -> None:
    query = "Which methods improve academic paper retrieval quality?"
    context = RecallGenerationContext(
        query_id="q-1",
        original_query=query,
        query_spec=QuerySpec(original_query=query, research_goal=query),
    )
    required = asyncio.run(
        required_a_prime_action_identities(
            [
                {
                    "query_id": "q-1",
                    "query": query,
                    "dataset": "pasa",
                    "split": "auto_train",
                    "role": "training",
                }
            ]
        )
    )
    expected = required["q-1"][-1]
    generator = ScheduledMissingActionGenerator(
        [ScheduledQueryActions(query_id="q-1", missing_actions=(expected,))]
    )

    result = asyncio.run(generator.generate(context))

    assert len(result.action_batch.actions) == 1
    assert result.provenance["scheduled_missing_action_count"] == "1"
    emitted = result.action_batch.actions[0].model_dump(mode="json")
    assert SearchActionIdentity.model_validate(
        {
            "action_type": emitted["action_type"],
            "search_mode": emitted["payload"].get("search_mode", "lexical"),
            "normalized_text": emitted["payload"].get(
                "query_text", emitted["payload"].get("title_text")
            ),
        }
    ) == expected


def test_scheduled_generator_can_filter_a_query_adaptive_source() -> None:
    query = "Find image classification papers using ViT on ImageNet-C"
    spec = QuerySpec(
        original_query=query,
        research_goal=query,
        methods=["ViT"],
        tasks=["image classification"],
        datasets=["ImageNet-C"],
    )
    context = RecallGenerationContext(
        query_id="q-adaptive",
        original_query=query,
        query_spec=QuerySpec(original_query=query, research_goal=query),
    )
    source = QueryAdaptiveHighRecallGenerator(
        frozen_query_specs={"q-adaptive": spec}
    )
    source_result = asyncio.run(source.generate(context))
    expected_action = source_result.action_batch.actions[-1]
    expected = SearchActionIdentity.model_validate(
        {
            "action_type": expected_action.action_type,
            "search_mode": expected_action.payload.search_mode,
            "normalized_text": expected_action.payload.query_text,
        }
    )
    generator = ScheduledMissingActionGenerator(
        [ScheduledQueryActions(query_id="q-adaptive", missing_actions=(expected,))],
        source=source,
    )

    result = asyncio.run(generator.generate(context))

    assert len(result.action_batch.actions) == 1
    assert result.action_batch.actions[0] == expected_action
    assert result.provenance["source_candidate_policy"] == (
        "query-adaptive-high-recall-v2"
    )


def test_daily_schedule_fails_closed_when_training_capacity_is_insufficient() -> None:
    with pytest.raises(ValueError, match="insufficient OpenAlex training capacity"):
        build_daily_openalex_schedule(
            [_actions("q-1", 6), _actions("q-2", 6)],
            first_window=date(2026, 8, 18),
            last_training_window=date(2026, 8, 18),
            final_test_window=date(2026, 8, 19),
            key_count=1,
            max_search_calls_per_key=10,
        )


def test_scheduled_query_rejects_duplicate_action_identity() -> None:
    action = SearchActionIdentity(
        action_type="text_search",
        search_mode="semantic",
        normalized_text="query",
    )

    with pytest.raises(ValueError, match="duplicate missing action identity"):
        ScheduledQueryActions(query_id="q-1", missing_actions=(action, action))


def _write_action_receipts(
    root: Path,
    *,
    query_id: str,
    actions: list[dict[str, object]],
    results: list[dict[str, object]],
) -> None:
    generation = root / "generation" / "attempt-01" / f"{query_id}.json"
    retrieval = root / "retrieval" / "attempt-01" / f"{query_id}.json"
    generation.parent.mkdir(parents=True)
    retrieval.parent.mkdir(parents=True)
    generation.write_text(
        json.dumps(
            {
                "query_id": query_id,
                "attempt_status": "succeeded",
                "actions": actions,
            }
        ),
        encoding="utf-8",
    )
    retrieval.write_text(
        json.dumps(
            {
                "query_id": query_id,
                "attempt_status": "succeeded",
                "results": results,
            }
        ),
        encoding="utf-8",
    )


def test_receipt_inventory_reuses_only_successful_normalized_action_identity(
    tmp_path: Path,
) -> None:
    root = tmp_path / "receipts"
    _write_action_receipts(
        root,
        query_id="q-1",
        actions=[
            {
                "action_id": "old-lexical-id",
                "action_type": "text_search",
                "payload": {"query_text": "  Query  Text ", "search_mode": "lexical"},
            },
            {
                "action_id": "old-semantic-id",
                "action_type": "text_search",
                "payload": {"query_text": "Query Text", "search_mode": "semantic"},
            },
        ],
        results=[
            {
                "action_id": "old-lexical-id",
                "errors": [],
                "infrastructure_failure": False,
            },
            {
                "action_id": "old-semantic-id",
                "errors": [{"code": "rate_limited"}],
                "infrastructure_failure": False,
            },
        ],
    )

    completed = load_completed_search_action_identities([root])

    assert completed == {
        "q-1": frozenset(
            {
                SearchActionIdentity(
                    action_type="text_search",
                    search_mode="lexical",
                    normalized_text="query text",
                )
            }
        )
    }


def test_receipt_inventory_reuses_successful_action_from_failed_mixed_attempt(
    tmp_path: Path,
) -> None:
    root = tmp_path / "partial-receipts"
    generation = root / "generation" / "attempt-01" / "q-partial.json"
    retrieval = root / "retrieval" / "attempt-01" / "q-partial.json"
    generation.parent.mkdir(parents=True)
    retrieval.parent.mkdir(parents=True)
    actions = [
        {
            "action_id": "lexical-ok",
            "action_type": "text_search",
            "payload": {"query_text": "usable lexical action"},
        },
        {
            "action_id": "semantic-failed",
            "action_type": "text_search",
            "payload": {
                "query_text": "failed semantic action",
                "search_mode": "semantic",
            },
        },
    ]
    generation.write_text(
        json.dumps(
            {
                "query_id": "q-partial",
                "attempt_status": "succeeded",
                "actions": actions,
            }
        ),
        encoding="utf-8",
    )
    retrieval.write_text(
        json.dumps(
            {
                "query_id": "q-partial",
                "attempt_status": "failed",
                "results": [
                    {
                        "action_id": "lexical-ok",
                        "errors": [],
                        "infrastructure_failure": False,
                    },
                    {
                        "action_id": "semantic-failed",
                        "errors": [{"code": "provider_error"}],
                        "infrastructure_failure": True,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    completed = load_completed_search_action_identities([root])

    assert completed == {
        "q-partial": frozenset(
            {
                SearchActionIdentity(
                    action_type="text_search",
                    search_mode="lexical",
                    normalized_text="usable lexical action",
                )
            }
        )
    }


def test_receipt_inventory_treats_generation_without_retrieval_as_incomplete(
    tmp_path: Path,
) -> None:
    root = tmp_path / "receipts"
    generation = (
        root
        / "batch-incomplete"
        / "generation"
        / "attempt-01"
        / "q-incomplete.json"
    )
    generation.parent.mkdir(parents=True)
    generation.write_text(
        json.dumps(
            {
                "query_id": "q-incomplete",
                "attempt_status": "succeeded",
                "actions": [
                    {
                        "action_id": "lexical",
                        "action_type": "text_search",
                        "payload": {
                            "query_text": "unfinished query",
                            "search_mode": "lexical",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    _write_action_receipts(
        root / "batch-complete",
        query_id="q-complete",
        actions=[
            {
                "action_id": "semantic",
                "action_type": "text_search",
                "payload": {
                    "query_text": "completed query",
                    "search_mode": "semantic",
                },
            }
        ],
        results=[
            {
                "action_id": "semantic",
                "errors": [],
                "infrastructure_failure": False,
            }
        ],
    )

    completed = load_completed_search_action_identities([root])

    assert "q-incomplete" not in completed
    assert completed["q-complete"] == frozenset(
        {
            SearchActionIdentity(
                action_type="text_search",
                search_mode="semantic",
                normalized_text="completed query",
            )
        }
    )


def test_receipt_inventory_ignores_known_non_search_action_results(
    tmp_path: Path,
) -> None:
    root = tmp_path / "mixed-receipts"
    _write_action_receipts(
        root,
        query_id="q-mixed",
        actions=[
            {
                "action_id": "lexical",
                "action_type": "text_search",
                "payload": {
                    "query_text": "search query",
                    "search_mode": "lexical",
                },
            },
            {
                "action_id": "graph",
                "action_type": "citation_expand",
                "payload": {"paper_ids": ["openalex:W1"]},
            },
        ],
        results=[
            {
                "action_id": "lexical",
                "errors": [],
                "infrastructure_failure": False,
            },
            {
                "action_id": "graph",
                "errors": [],
                "infrastructure_failure": False,
            },
        ],
    )

    completed = load_completed_search_action_identities([root])

    assert completed["q-mixed"] == frozenset(
        {
            SearchActionIdentity(
                action_type="text_search",
                search_mode="lexical",
                normalized_text="search query",
            )
        }
    )


def test_missing_action_work_matches_identity_not_legacy_action_id(
    tmp_path: Path,
) -> None:
    root = tmp_path / "receipts"
    _write_action_receipts(
        root,
        query_id="q-1",
        actions=[
            {
                "action_id": "semantic-backfill-original",
                "action_type": "text_search",
                "payload": {"query_text": "Original Query", "search_mode": "semantic"},
            }
        ],
        results=[
            {
                "action_id": "semantic-backfill-original",
                "errors": [],
                "infrastructure_failure": False,
            }
        ],
    )
    semantic = SearchActionIdentity(
        action_type="text_search",
        search_mode="semantic",
        normalized_text="original query",
    )
    lexical = SearchActionIdentity(
        action_type="text_search",
        search_mode="lexical",
        normalized_text="original query",
    )

    work = build_missing_action_work(
        {"q-1": (lexical, semantic), "q-2": (lexical,)},
        load_completed_search_action_identities([root]),
    )

    assert work == [
        ScheduledQueryActions(query_id="q-1", missing_actions=(lexical,)),
        ScheduledQueryActions(query_id="q-2", missing_actions=(lexical,)),
    ]


def test_settled_inventory_keeps_terminal_decode_error_but_retries_network_error(
    tmp_path: Path,
) -> None:
    terminal_root = tmp_path / "terminal"
    transient_root = tmp_path / "transient"
    action = {
        "action_id": "search",
        "action_type": "text_search",
        "payload": {"query_text": "query", "search_mode": "lexical"},
    }
    _write_action_receipts(
        terminal_root,
        query_id="q-terminal",
        actions=[action],
        results=[
            {
                "action_id": "search",
                "errors": [{"code": "invalid_work", "retryable": False}],
                "infrastructure_failure": False,
            }
        ],
    )
    _write_action_receipts(
        transient_root,
        query_id="q-transient",
        actions=[action],
        results=[
            {
                "action_id": "search",
                "errors": [{"code": "network_error", "retryable": False}],
                "infrastructure_failure": True,
            }
        ],
    )

    settled = load_settled_search_action_identities(
        [terminal_root, transient_root]
    )

    assert "q-terminal" in settled
    assert "q-transient" not in settled


def test_required_a_prime_actions_are_generated_only_for_auto_train() -> None:
    required = asyncio.run(
        required_a_prime_action_identities(
            [
                {
                    "query_id": "q-1",
                    "query": "Which methods improve academic paper retrieval quality?",
                    "dataset": "pasa",
                    "split": "auto_train",
                    "role": "training",
                }
            ]
        )
    )

    identities = required["q-1"]
    assert 5 <= len(identities) <= 6
    assert len(identities) == len(set(identities))
    assert sum(identity.search_mode == "semantic" for identity in identities) == 1
    assert all(identity.action_type in {"text_search", "title_search"} for identity in identities)


def test_required_a_prime_actions_reject_final_test_rows() -> None:
    with pytest.raises(ValueError, match="auto_train"):
        asyncio.run(
            required_a_prime_action_identities(
                [
                    {
                        "query_id": "q-test",
                        "query": "hidden evaluation query",
                        "dataset": "pasa",
                        "split": "auto_test",
                        "role": "final_test",
                    }
                ]
            )
        )


def test_daily_quota_ledger_counts_attempts_atomically_and_survives_restart(
    tmp_path: Path,
) -> None:
    beijing = timezone(timedelta(hours=8), name="Asia/Shanghai")
    now = datetime(2026, 8, 18, 12, tzinfo=beijing)
    path = tmp_path / "quota.sqlite3"
    ledger = SQLiteOpenAlexDailyQuotaLedger(
        path,
        window=date(2026, 8, 18),
        key_slot=3,
        max_search_calls=2,
        clock=lambda: now,
    )

    assert ledger.claim_attempt() == 1
    assert ledger.claim_attempt() == 2
    with pytest.raises(OpenAlexDailyQuotaExceededError, match="hard cap"):
        ledger.claim_attempt()

    reopened = SQLiteOpenAlexDailyQuotaLedger(
        path,
        window=date(2026, 8, 18),
        key_slot=3,
        max_search_calls=2,
        clock=lambda: now,
    )
    assert reopened.used_search_calls == 2
    with pytest.raises(OpenAlexDailyQuotaExceededError, match="hard cap"):
        reopened.claim_attempt()


def test_daily_quota_ledger_rejects_a_stale_beijing_window(tmp_path: Path) -> None:
    ledger = SQLiteOpenAlexDailyQuotaLedger(
        tmp_path / "quota.sqlite3",
        window=date(2026, 8, 18),
        key_slot=1,
        max_search_calls=900,
        clock=lambda: datetime(
            2026,
            8,
            19,
            8,
            0,
            tzinfo=timezone(timedelta(hours=8), name="Asia/Shanghai"),
        ),
    )

    with pytest.raises(ValueError, match="current OpenAlex quota window"):
        ledger.claim_attempt()
