from __future__ import annotations

import json
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from paper_search.config import load_budget
from paper_search.control.budget import HardBudgetController
from paper_search.evaluation.dataset import IdentifierMap
from paper_search.learning.lexical_bridge import (
    LexicalBridgeExample,
    SupervisedLexicalBridge,
)
from paper_search.learning.lexical_bridge_deployment import (
    freeze_lexical_bridge_model,
)
from paper_search.recall_experiments.canary_runtime import (
    _search_reservation_calls,
    load_runtime_profile,
)
from scripts import run_provider_recall_comparison as provider_comparison
from scripts.run_provider_recall_comparison import (
    _build_generator_override,
    _checked_inter_batch_delay,
    _load_infrastructure_failure_marker,
    _load_fixed_actions,
    _load_selected_query_ids,
    _load_rows,
    _parse_providers,
    _provider_overlap,
    _runtime_profile_name,
    _write_infrastructure_failure_marker,
)


def _fixed_action(query_text: str, *, mode: str = "lexical") -> dict[str, object]:
    return {
        "actions": [
            {
                "action_id": "query-native-title-phrase-v3",
                "strategy": "candidate-family:query-native-title-phrase-v3",
                "action_type": "text_search",
                "payload": {"query_text": query_text, "search_mode": mode},
            }
        ]
    }


def test_provider_comparison_binds_production_combined_identifier_map() -> None:
    loader = getattr(
        provider_comparison,
        "_load_production_identifier_context",
        None,
    )

    assert callable(loader)
    context = loader(
        workspace_root=Path.cwd(),
        lock_path=Path("deliverables/evaluator/live-evaluator.lock.yaml"),
    )
    identifier_map = IdentifierMap.from_bytes(
        context.identifier_map_bytes,
        source="provider comparison production identity context",
    )

    assert identifier_map.resolve("doi:10.1609/aaai.v34i01.5421") == (
        identifier_map.resolve("arxiv:2010.01532")
    )
    assert context.evidence["binding"] == "production_combined_identifier_map"
    assert context.evidence["combined_identifier_map_sha256"] == (
        "sha256:" + hashlib.sha256(context.identifier_map_bytes).hexdigest()
    )


def test_provider_comparison_rejects_resume_from_another_identity_context() -> None:
    validator = getattr(
        provider_comparison,
        "_validate_report_identifier_context",
        None,
    )

    assert callable(validator)
    report = SimpleNamespace(
        execution_identity=SimpleNamespace(
            identifier_map_sha256="sha256:" + "1" * 64,
        )
    )
    with pytest.raises(ValueError, match="identifier context"):
        validator(report, expected_sha256="sha256:" + "2" * 64)


def test_provider_comparison_accepts_one_non_test_partition(tmp_path) -> None:
    path = tmp_path / "train.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(
                {
                    "dataset": "pasa",
                    "split": "auto_train",
                    "role": "training",
                    "query_id": f"q-{index}",
                    "query": f"query {index}",
                    "gold_paper_ids": [f"arxiv:2001.0000{index}"],
                }
            )
            for index in range(2)
        ),
        encoding="utf-8",
    )

    rows, identity = _load_rows(path, limit=1)

    assert len(rows) == 1
    assert identity == ("pasa", "auto_train", "training")


def test_provider_comparison_refuses_final_test_partition(tmp_path) -> None:
    path = tmp_path / "test.jsonl"
    path.write_text(
        json.dumps(
            {
                "dataset": "pasa",
                "split": "test",
                "role": "final_test",
                "query_id": "q-1",
                "query": "query",
                "gold_paper_ids": ["arxiv:2001.00001"],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="final_test"):
        _load_rows(path, limit=None)


def test_exploration_collection_is_restricted_to_training_role() -> None:
    generator = _build_generator_override(
        collection_mode="exploration",
        role="training",
        max_actions=3,
        workspace_root=None,
    )

    assert generator.exploration_policy == "anchor-compress-rotate-v1"
    with pytest.raises(ValueError, match="training role"):
        _build_generator_override(
            collection_mode="exploration",
            role="development",
            max_actions=3,
            workspace_root=None,
        )


def test_exploration_collection_requires_three_action_budget() -> None:
    with pytest.raises(ValueError, match="exactly three"):
        _build_generator_override(
            collection_mode="exploration",
            role="training",
            max_actions=2,
            workspace_root=None,
        )


def test_candidate_ceiling_collection_allows_development_but_not_final_test() -> None:
    generator = _build_generator_override(
        collection_mode="candidate_ceiling",
        role="training",
        max_actions=12,
        workspace_root=None,
    )

    assert generator.candidate_policy == "full-controlled-candidate-pool-v2"
    development = _build_generator_override(
        collection_mode="candidate_ceiling",
        role="development",
        max_actions=12,
        workspace_root=None,
    )
    assert development.candidate_policy == "full-controlled-candidate-pool-v2"
    with pytest.raises(ValueError, match="final_test"):
        _build_generator_override(
            collection_mode="candidate_ceiling",
            role="final_test",
            max_actions=12,
            workspace_root=None,
        )


def test_structured_graph_collection_uses_frozen_openalex_generator() -> None:
    generator = _build_generator_override(
        collection_mode="structured_graph",
        role="training",
        max_actions=12,
        workspace_root=None,
    )

    assert generator.candidate_policy == "structured-graph-candidate-pool-v1"
    assert _runtime_profile_name("structured_graph") == "candidate-ceiling-live.yaml"
    development = _build_generator_override(
        collection_mode="structured_graph",
        role="development",
        max_actions=12,
        workspace_root=None,
    )
    assert development.candidate_policy == "structured-graph-candidate-pool-v1"
    with pytest.raises(ValueError, match="final_test"):
        _build_generator_override(
            collection_mode="structured_graph",
            role="final_test",
            max_actions=12,
            workspace_root=None,
        )


def test_semantic_backfill_collection_allows_development_but_forbids_final_test() -> None:
    generator = _build_generator_override(
        collection_mode="semantic_backfill",
        role="training",
        max_actions=1,
        workspace_root=None,
    )

    assert generator.backfill_policy == "openalex-semantic-backfill-v1"
    assert _runtime_profile_name("semantic_backfill") == "semantic-backfill-live.yaml"
    development = _build_generator_override(
        collection_mode="semantic_backfill",
        role="development",
        max_actions=1,
        workspace_root=None,
    )
    assert development.backfill_policy == "openalex-semantic-backfill-v1"
    with pytest.raises(ValueError, match="forbids final_test"):
        _build_generator_override(
            collection_mode="semantic_backfill",
            role="final_test",
            max_actions=1,
            workspace_root=None,
        )


def test_production_lexical_collection_loads_current_pairwise_policy(
    tmp_path: Path,
) -> None:
    model_path = (
        tmp_path
        / "data/training_private/models/cpu-pairwise-action-ranker-openalex-v1.f64"
    )
    model_path.parent.mkdir(parents=True)
    weights = b"\0" * (256 * 8)
    model_path.write_bytes(weights)
    manifest_path = tmp_path / "data/training/cpu-pairwise-action-ranker-openalex-v1.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "cpu-pairwise-action-ranker-experiment-v1",
                "model_id": "cpu-pairwise-action-ranker-v1",
                "target_provider": "openalex",
                "dimension": 256,
                "epochs": 3,
                "learning_rate": 0.08,
                "l2": 1e-6,
                "seed": 17,
                "confidence_threshold": 0.4,
                "model_sha256": "sha256:" + hashlib.sha256(weights).hexdigest(),
            }
        ),
        encoding="utf-8",
    )

    generator = _build_generator_override(
        collection_mode="production_lexical",
        role="training",
        max_actions=3,
        workspace_root=tmp_path,
    )

    assert generator.model_id == "cpu-pairwise-action-ranker-v1"
    assert generator.source_sha256 == "sha256:" + hashlib.sha256(weights).hexdigest()
    assert (
        _runtime_profile_name("production_lexical")
        == "production-lexical-live.yaml"
    )
    with pytest.raises(ValueError, match="exactly twelve"):
        _build_generator_override(
            collection_mode="candidate_ceiling",
            role="training",
            max_actions=3,
            workspace_root=None,
        )


def test_lexical_bridge_collection_wraps_production_policy_explicitly(
    tmp_path: Path,
) -> None:
    pairwise_path = (
        tmp_path
        / "data/training_private/models/cpu-pairwise-action-ranker-openalex-v1.f64"
    )
    pairwise_path.parent.mkdir(parents=True)
    weights = b"\0" * (256 * 8)
    pairwise_path.write_bytes(weights)
    pairwise_manifest = (
        tmp_path / "data/training/cpu-pairwise-action-ranker-openalex-v1.json"
    )
    pairwise_manifest.parent.mkdir(parents=True)
    pairwise_manifest.write_text(
        json.dumps(
            {
                "schema_version": "cpu-pairwise-action-ranker-experiment-v1",
                "model_id": "cpu-pairwise-action-ranker-v1",
                "target_provider": "openalex",
                "dimension": 256,
                "epochs": 3,
                "learning_rate": 0.08,
                "l2": 1e-6,
                "seed": 17,
                "confidence_threshold": 0.4,
                "model_sha256": "sha256:" + hashlib.sha256(weights).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    bridge = SupervisedLexicalBridge.fit(
        [
            LexicalBridgeExample(
                query="multimodal representation learning",
                gold_titles=("Cross modal retrieval",),
            ),
            LexicalBridgeExample(
                query="multi modal representation alignment",
                gold_titles=("Cross-modal retrieval",),
            ),
        ],
        representation="word_char",
        learning_objective="neighbor_idf",
    )
    bridge_path = (
        tmp_path
        / "data/training_private/models/supervised-lexical-bridge-openalex-v2.joblib"
    )
    bridge_manifest = (
        tmp_path / "data/training/supervised-lexical-bridge-openalex-v2.json"
    )
    freeze_lexical_bridge_model(
        bridge,
        model_path=bridge_path,
        manifest_path=bridge_manifest,
        training_query_count=2,
        raw_train_sha256="sha256:" + "1" * 64,
        train_partition_sha256="sha256:" + "2" * 64,
        training_oof_sha256="sha256:" + "3" * 64,
        independent_dev_sha256="sha256:" + "4" * 64,
    )

    generator = _build_generator_override(
        collection_mode="lexical_bridge",
        role="development",
        max_actions=4,
        workspace_root=tmp_path,
    )

    assert generator.model_id == "supervised-lexical-bridge-openalex-v2"
    assert generator.generator_type == "local_cpu"
    assert generator.source_sha256.startswith("sha256:")
    assert _runtime_profile_name("lexical_bridge") == "lexical-bridge-live.yaml"


def test_fixed_budget_collection_uses_six_action_openalex_generator() -> None:
    generator = _build_generator_override(
        collection_mode="fixed_budget_openalex",
        role="development",
        max_actions=6,
        workspace_root=None,
    )

    assert generator.candidate_policy == "fixed-budget-openalex-v1"
    assert generator.max_openalex_actions == 6
    assert (
        _runtime_profile_name("fixed_budget_openalex")
        == "fixed-budget-openalex-live.yaml"
    )
    with pytest.raises(ValueError, match="exactly six"):
        _build_generator_override(
            collection_mode="fixed_budget_openalex",
            role="development",
            max_actions=5,
            workspace_root=None,
        )


def test_a_prime_collection_uses_frozen_core4_semantic_boolean_generator() -> None:
    generator = _build_generator_override(
        collection_mode="core4_semantic_boolean",
        role="development",
        max_actions=6,
        workspace_root=None,
    )

    assert generator.candidate_policy == "core4-semantic-boolean-v1"
    assert _runtime_profile_name("core4_semantic_boolean") == "fixed-budget-openalex-live.yaml"


def test_provider_comparison_loads_three_disjoint_ceiling_batches(tmp_path) -> None:
    path = tmp_path / "train.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(
                {
                    "dataset": "pasa",
                    "split": "auto_train",
                    "role": "training",
                    "query_id": f"q-{index:03d}",
                    "query": f"query {index}",
                    "gold_paper_ids": [f"arxiv:2001.{index:05d}"],
                }
            )
            for index in range(90)
        ),
        encoding="utf-8",
    )

    batches = [
        _load_rows(
            path,
            limit=10,
            sample_batch_index=index,
            sample_batch_count=3,
        )[0]
        for index in range(3)
    ]

    ids = [{str(row["query_id"]) for row in batch} for batch in batches]
    assert all(len(batch) == 10 for batch in batches)
    assert not ids[0].intersection(ids[1] | ids[2])
    assert not ids[1].intersection(ids[2])


def test_provider_comparison_selects_exact_frozen_query_ids(tmp_path) -> None:
    path = tmp_path / "train.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(
                {
                    "dataset": "pasa",
                    "split": "auto_train",
                    "role": "training",
                    "query_id": f"q-{index}",
                    "query": f"query {index}",
                    "gold_paper_ids": [f"arxiv:2001.0000{index}"],
                }
            )
            for index in range(5)
        ),
        encoding="utf-8",
    )

    rows, _ = _load_rows(
        path,
        limit=None,
        selected_query_ids=frozenset({"q-1", "q-4"}),
    )

    assert [row["query_id"] for row in rows] == ["q-1", "q-4"]
    with pytest.raises(ValueError, match="missing from partition"):
        _load_rows(
            path,
            limit=None,
            selected_query_ids=frozenset({"q-missing"}),
        )


def test_provider_comparison_loads_explicit_frozen_query_ids(tmp_path) -> None:
    path = tmp_path / "selection.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "provider-recall-query-selection-v1",
                "query_ids": ["q-3", "q-1"],
            }
        ),
        encoding="utf-8",
    )

    query_ids, digest = _load_selected_query_ids(path)

    assert query_ids == frozenset({"q-1", "q-3"})
    assert digest.startswith("sha256:")
    path.write_text(
        json.dumps(
            {
                "schema_version": "provider-recall-query-selection-v1",
                "query_ids": ["q-1", "q-1"],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate"):
        _load_selected_query_ids(path)


def test_provider_comparison_loads_hash_bound_single_lexical_actions(
    tmp_path: Path,
) -> None:
    path = tmp_path / "actions.json"
    path.write_text(
        json.dumps(
            {
                "q-1": _fixed_action("graph neural networks"),
                "q-2": _fixed_action("parameter efficient tuning"),
            }
        ),
        encoding="utf-8",
    )

    actions, digest = _load_fixed_actions(path, query_ids=("q-1", "q-2"))

    assert set(actions) == {"q-1", "q-2"}
    assert digest == "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="coverage"):
        _load_fixed_actions(path, query_ids=("q-1",))


def test_provider_comparison_rejects_nonlexical_fixed_actions(tmp_path: Path) -> None:
    path = tmp_path / "actions.json"
    path.write_text(
        json.dumps({"q-1": _fixed_action("graph neural networks", mode="semantic")}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="one lexical text action"):
        _load_fixed_actions(path, query_ids=("q-1",))


def test_provider_comparison_reports_cross_provider_unique_and_union_hits() -> None:
    def report(*rows: tuple[str, list[str]]) -> SimpleNamespace:
        return SimpleNamespace(
            result=SimpleNamespace(
                per_query=[
                    SimpleNamespace(query_id=query_id, gold_hit_ids=gold_hit_ids)
                    for query_id, gold_hit_ids in rows
                ]
            )
        )

    overlap = _provider_overlap(
        {
            "openalex": [
                report(("q-1", ["g-1"]), ("q-2", ["g-2"]), ("q-3", []))
            ],
            "semantic_scholar": [
                report(("q-1", []), ("q-2", ["g-2"]), ("q-3", ["g-3"]))
            ],
        }
    )

    assert overlap == {
        "intersection_gold_hits": 1,
        "openalex_only_gold_hits": 1,
        "semantic_scholar_only_gold_hits": 1,
        "union_gold_hits": 3,
        "intersection_gold_hit_queries": 1,
        "openalex_only_gold_hit_queries": 1,
        "semantic_scholar_only_gold_hit_queries": 1,
        "union_gold_hit_queries": 3,
    }


def test_provider_selection_allows_openalex_only_without_duplicates() -> None:
    assert _parse_providers(["openalex"]) == ("openalex",)
    assert _parse_providers(["openalex", "semantic_scholar"]) == (
        "openalex",
        "semantic_scholar",
    )
    with pytest.raises(ValueError, match="duplicate"):
        _parse_providers(["openalex", "openalex"])


def test_inter_batch_delay_is_explicitly_bounded() -> None:
    assert _checked_inter_batch_delay(12.0) == 12.0
    with pytest.raises(ValueError, match="between 0 and 60"):
        _checked_inter_batch_delay(-0.1)
    with pytest.raises(ValueError, match="between 0 and 60"):
        _checked_inter_batch_delay(60.1)


def test_candidate_ceiling_uses_batch_sized_runtime_budget() -> None:
    assert _runtime_profile_name("candidate_ceiling") == "candidate-ceiling-live.yaml"
    assert _runtime_profile_name("policy") == "default-live.yaml"
    assert _runtime_profile_name("exploration") == "default-live.yaml"


def test_candidate_ceiling_batch_budget_is_not_hard_stopped_at_zero_usage() -> None:
    budget = load_budget("configs/budget_candidate_ceiling_batch4.yaml")

    controller = HardBudgetController(budget, formal_live=True)

    assert controller.stop_status() == "continue"


def test_candidate_ceiling_runtime_enables_openalex_request_pacing() -> None:
    profile = load_runtime_profile(
        Path("configs/recall_experiments/runtime/candidate-ceiling-live.yaml")
    )

    assert profile.openalex_minimum_request_interval_seconds == 0.2


def test_openalex_reservation_covers_configured_key_rotation() -> None:
    assert _search_reservation_calls("openalex", configured_key_count=7) == 7
    assert _search_reservation_calls("openalex", configured_key_count=1) == 3
    assert _search_reservation_calls("semantic_scholar", configured_key_count=1) == 3


def test_infrastructure_failure_marker_is_resume_loadable(tmp_path: Path) -> None:
    run_path = tmp_path / "semantic_scholar" / "batch-0007-retry-002"

    written = _write_infrastructure_failure_marker(
        run_path=run_path,
        provider="semantic_scholar",
        batch_stem="batch-0007",
        query_ids=("q-7",),
    )
    loaded = _load_infrastructure_failure_marker(
        run_path / "infrastructure-failure.json"
    )

    assert loaded == written
    assert loaded["schema_version"] == "provider-recall-infrastructure-failure-v1"
    assert loaded["query_ids"] == ["q-7"]
    assert loaded["reason"] == "no_valid_repeat"
    assert loaded["retryable"] is True
    assert "error_message" not in loaded
