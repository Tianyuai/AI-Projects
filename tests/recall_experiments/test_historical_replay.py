"""Exactness-bounded normalization tests for historical candidate-pool evidence."""

from __future__ import annotations

from pathlib import Path


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
    assert replay.methods["query-evolution"].per_query_equality == "not_comparable"
    assert replay.methods["query-evolution"].semantic_mismatch is not None
    assert "per_query_gold_hits" in replay.methods["query-evolution"].unprovable_fields
    assert replay.methods["query-rewrite"].terminal_state == "aggregate_only"
    assert replay.methods["llm-query-variants"].terminal_state == "aggregate_only"
    assert replay.methods["title-candidates"].terminal_state == "aggregate_only"
    assert replay.methods["citation-expansion"].terminal_state == "insufficient_historical_evidence"
    assert replay.scheme_b_terminal_state == "insufficient_historical_evidence"


def test_query_evolution_replays_exact_action_and_candidate_ids_without_claiming_gold_hits() -> None:
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
