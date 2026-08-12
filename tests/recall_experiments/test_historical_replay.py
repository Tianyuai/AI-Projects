"""Exactness-bounded normalization tests for historical candidate-pool evidence."""

from __future__ import annotations

from pathlib import Path

import pytest


WORKSPACE_ROOT = Path(__file__).parents[2]
INVENTORY = WORKSPACE_ROOT / "runs" / "_recall_history_inventory" / "source-inventory.json"
CONFIG_ROOT = WORKSPACE_ROOT / "configs" / "recall_experiments" / "historical"


def test_historical_loader_normalizes_all_bound_methods_with_explicit_terminal_states() -> None:
    from paper_search.recall_experiments.inputs.historical import load_historical_replays

    replay = load_historical_replays(
        inventory_path=INVENTORY,
        config_root=CONFIG_ROOT,
        workspace_root=WORKSPACE_ROOT,
    )

    assert set(replay.methods) == {
        "query-rewrite",
        "llm-query-variants",
        "query-evolution",
        "title-candidates",
        "citation-expansion",
    }
    assert replay.methods["query-evolution"].fixed_actions is not None
    assert replay.methods["query-evolution"].frozen_dataset is not None
    assert replay.methods["query-evolution"].terminal_state == "not_comparable"
    assert replay.methods["query-evolution"].per_query_equality == "not_provable"
    assert replay.methods["query-evolution"].exact_actions_available is True
    assert replay.methods["query-evolution"].exact_provider_responses_available is True
    assert replay.methods["query-evolution"].semantic_mismatch is not None
    assert "per_query_gold_hits" in replay.methods["query-evolution"].unprovable_fields
    assert replay.methods["query-rewrite"].terminal_state == "aggregate_only"
    assert replay.methods["llm-query-variants"].terminal_state == "aggregate_only"
    assert replay.methods["title-candidates"].terminal_state == "aggregate_only"
    assert replay.methods["citation-expansion"].terminal_state == "insufficient_historical_evidence"
    assert replay.scheme_b_terminal_state == "insufficient_historical_evidence"


def test_query_evolution_replays_exact_actions_and_reconstructs_response_candidate_ids() -> None:
    from paper_search.recall_experiments.inputs.historical import load_historical_replays

    method = load_historical_replays(
        inventory_path=INVENTORY,
        config_root=CONFIG_ROOT,
        workspace_root=WORKSPACE_ROOT,
    ).methods["query-evolution"]

    assert method.fixed_actions is not None
    assert method.frozen_dataset is not None
    assert method.candidate_pool_ids_by_query
    assert set(method.fixed_actions) == {
        record.query_id for record in method.frozen_dataset.queries
    }
    assert all(isinstance(candidate_ids, tuple) for candidate_ids in method.candidate_pool_ids_by_query.values())
    assert method.historical_baseline is None
    assert method.gold_hit_ids_by_query is None
    assert method.frozen_dataset.evaluation_materials is None


def test_history_loader_rejects_inventory_hash_drift_before_parsing_sources(tmp_path: Path) -> None:
    import json

    from paper_search.recall_experiments.inputs.historical import HistoricalReplayError, load_historical_replays

    copied = json.loads(INVENTORY.read_text(encoding="utf-8"))
    copied["methods"][0]["source_sha256"][0] = "0" * 64
    inventory = tmp_path / "source-inventory.json"
    inventory.write_text(json.dumps(copied), encoding="utf-8")

    try:
        load_historical_replays(
            inventory_path=inventory,
            config_root=CONFIG_ROOT,
            workspace_root=WORKSPACE_ROOT,
        )
    except HistoricalReplayError as error:
        assert "inventory hash mismatch" in str(error)
    else:
        raise AssertionError("expected inventory hash mismatch")


def test_history_loader_rejects_valid_but_different_task_zero_gold_association_binding(
    tmp_path: Path,
) -> None:
    import json

    from paper_search.recall_experiments.inputs.historical import HistoricalReplayError, load_historical_replays

    copied = json.loads(INVENTORY.read_text(encoding="utf-8"))
    copied["gold_catalog"]["association_source"] = {
        "path": "docs/evidence/title-retention-offline-2026-08-09.json",
        "sha256": "5ef7ccbc7d56bae4ee82cbec52036b7f4e6b2f02aa68eeee6f3953b15d9d0737",
        "kind": "association_only",
    }
    inventory = tmp_path / "source-inventory.json"
    inventory.write_text(json.dumps(copied), encoding="utf-8")

    with pytest.raises(HistoricalReplayError, match="Task 0 Gold association binding"):
        load_historical_replays(
            inventory_path=inventory,
            config_root=CONFIG_ROOT,
            workspace_root=WORKSPACE_ROOT,
        )


def test_aggregate_metrics_come_from_task_zero_metric_evidence_path() -> None:
    from paper_search.recall_experiments.inputs.historical import load_historical_replays

    title = load_historical_replays(
        inventory_path=INVENTORY,
        config_root=CONFIG_ROOT,
        workspace_root=WORKSPACE_ROOT,
    ).methods["title-candidates"]

    assert title.aggregate_metrics is not None
    variants = title.aggregate_metrics["variants"]
    assert isinstance(variants, list)
    assert variants[0]["name"] == "historical_rrf"


def test_history_loader_rejects_metric_evidence_not_bound_to_its_method(tmp_path: Path) -> None:
    import json

    from paper_search.recall_experiments.inputs.historical import HistoricalReplayError, load_historical_replays

    copied = json.loads(INVENTORY.read_text(encoding="utf-8"))
    title = next(item for item in copied["methods"] if item["method_id"] == "title-candidates")
    title["metric_evidence"] = {"available": True, "path": "data/dev/gold.jsonl"}
    inventory = tmp_path / "source-inventory.json"
    inventory.write_text(json.dumps(copied), encoding="utf-8")

    with pytest.raises(HistoricalReplayError, match="metric evidence path"):
        load_historical_replays(
            inventory_path=inventory,
            config_root=CONFIG_ROOT,
            workspace_root=WORKSPACE_ROOT,
        )


def test_evaluator_rejects_historical_dataset_without_bound_identifier_material() -> None:
    from paper_search.recall_experiments.evaluator import CandidateRecallEvaluator
    from paper_search.recall_experiments.inputs.historical import load_historical_replays

    dataset = load_historical_replays(
        inventory_path=INVENTORY,
        config_root=CONFIG_ROOT,
        workspace_root=WORKSPACE_ROOT,
    ).methods["query-evolution"].frozen_dataset

    assert dataset is not None
    with pytest.raises(ValueError, match="evaluation materials are unavailable"):
        CandidateRecallEvaluator().preflight(dataset)


def test_historical_comparison_reports_every_method_without_a_false_scheme_b_pass(tmp_path: Path) -> None:
    from paper_search.recall_experiments.composition import compare_historical_replays

    payload = compare_historical_replays(
        inventory_path=INVENTORY,
        config_root=CONFIG_ROOT,
        output_path=tmp_path / "comparison",
        workspace_root=WORKSPACE_ROOT,
    )

    assert payload["scheme_b_terminal_state"] == "insufficient_historical_evidence"
    assert {item["method_id"] for item in payload["methods"]} == {
        "query-rewrite",
        "llm-query-variants",
        "query-evolution",
        "title-candidates",
        "citation-expansion",
    }
    query_evolution = next(item for item in payload["methods"] if item["method_id"] == "query-evolution")
    assert query_evolution["terminal_state"] == "not_comparable"
    assert query_evolution["semantic_mismatch"]
    assert query_evolution["gold_hit_ids_by_query"] is None
    assert (tmp_path / "comparison" / "historical-replay-comparison.json").is_file()


def test_historical_compare_cli_uses_the_task_zero_inventory(
    tmp_path: Path, capsys: object, monkeypatch: object
) -> None:
    from paper_search.cli import main

    monkeypatch.chdir(WORKSPACE_ROOT)  # type: ignore[attr-defined]
    assert main(
        [
            "recall",
            "compare",
            "--config-root",
            str(CONFIG_ROOT),
            "--out",
            str(tmp_path / "comparison"),
        ]
    ) == 0
    assert "insufficient_historical_evidence" in capsys.readouterr().out  # type: ignore[attr-defined]
