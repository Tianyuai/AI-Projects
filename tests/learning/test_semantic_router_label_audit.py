from __future__ import annotations

from paper_search.learning.method_route_labels import MethodRouteLabel
from paper_search.learning.contracts import PolicyActionCandidate
from paper_search.learning.provider_action_labels import ProviderActionLabel
from paper_search.learning.semantic_router_label_audit import (
    LabelCompressionCriteria,
    audit_binary_label_compression,
    baseline_gold_hit_counts,
)


def _label(
    query_id: str,
    *,
    gold_count: int,
    marginal_hits: int,
) -> MethodRouteLabel:
    return MethodRouteLabel(
        dataset="pasa",
        split="auto_train",
        role="training",
        method="semantic",
        query_id=query_id,
        query=f"query {query_id}",
        routing_label="beneficial" if marginal_hits else "not_beneficial",
        gold_association_count=gold_count,
        marginal_gold_hit_count=marginal_hits,
        marginal_recall=marginal_hits / gold_count,
        search_api_calls=1,
    )


def test_label_compression_requires_repeated_single_and_multi_hit_strata() -> None:
    rows = [
        _label("a-single", gold_count=3, marginal_hits=1),
        _label("a-multi", gold_count=3, marginal_hits=2),
        _label("b-single", gold_count=4, marginal_hits=1),
        _label("b-multi", gold_count=4, marginal_hits=3),
    ]
    metadata = {
        "a-single": {"intent_family": "request", "length_bucket": "q1", "gold_count_bucket": "3_plus"},
        "a-multi": {"intent_family": "request", "length_bucket": "q1", "gold_count_bucket": "3_plus"},
        "b-single": {"intent_family": "which_what", "length_bucket": "q2", "gold_count_bucket": "3_plus"},
        "b-multi": {"intent_family": "which_what", "length_bucket": "q2", "gold_count_bucket": "3_plus"},
    }
    baseline_hits = {query_id: 0 for query_id in metadata}
    criteria = LabelCompressionCriteria(
        minimum_overall_examples_per_strength=2,
        minimum_examples_per_strength_per_stratum=1,
        minimum_qualifying_strata_per_family=2,
        minimum_qualifying_families=2,
    )

    audit = audit_binary_label_compression(
        rows,
        metadata=metadata,
        baseline_gold_hit_counts=baseline_hits,
        criteria=criteria,
    )

    assert audit["evidence_crosses_strata"] is True
    assert audit["overall_strength_counts"] == {"multi_hit": 2, "single_hit": 2}
    assert audit["observed_api_call_distribution"] == {"1": 4}
    assert set(audit["qualifying_families"]) == {"intent_family", "length_bucket"}


def test_label_compression_does_not_infer_evidence_from_one_stratum() -> None:
    rows = [
        _label("single", gold_count=3, marginal_hits=1),
        _label("multi", gold_count=3, marginal_hits=2),
    ]
    metadata = {
        query_id: {
            "intent_family": "request",
            "length_bucket": "q1",
            "gold_count_bucket": "3_plus",
        }
        for query_id in ("single", "multi")
    }
    criteria = LabelCompressionCriteria(
        minimum_overall_examples_per_strength=1,
        minimum_examples_per_strength_per_stratum=1,
        minimum_qualifying_strata_per_family=2,
        minimum_qualifying_families=2,
    )

    audit = audit_binary_label_compression(
        rows,
        metadata=metadata,
        baseline_gold_hit_counts={"single": 0, "multi": 0},
        criteria=criteria,
    )

    assert audit["evidence_crosses_strata"] is False
    assert audit["qualifying_families"] == []


def test_label_compression_reports_lexical_coverage_without_gold_leak_features() -> None:
    rows = [
        _label("none", gold_count=3, marginal_hits=1),
        _label("partial", gold_count=3, marginal_hits=1),
    ]
    metadata = {
        query_id: {
            "intent_family": "request",
            "length_bucket": "q1",
            "gold_count_bucket": "3_plus",
        }
        for query_id in ("none", "partial")
    }

    audit = audit_binary_label_compression(
        rows,
        metadata=metadata,
        baseline_gold_hit_counts={"none": 0, "partial": 1},
        criteria=LabelCompressionCriteria(),
    )

    strata = audit["families"]["lexical_coverage"]
    assert {row["value"] for row in strata} == {"none", "partial"}


def test_baseline_gold_hit_counts_use_union_across_lexical_actions() -> None:
    action = PolicyActionCandidate(
        action_id="lexical-1",
        action_type="text_search",
        text="query",
        origin="original_query",
        provider_hint="openalex",
    )

    def row(action_id: str, gold_ids: tuple[str, ...]) -> ProviderActionLabel:
        return ProviderActionLabel(
            dataset="pasa",
            split="auto_train",
            role="training",
            query_id="q1",
            query="query",
            provider="openalex",
            action=action.model_copy(update={"action_id": action_id}),
            retrieval_status="available",
            gold_association_count=3,
            gold_hit_ids=gold_ids,
            gold_hit_count=len(gold_ids),
            action_recall=len(gold_ids) / 3,
            novel_over_anchor_hit_count=0,
        )

    counts = baseline_gold_hit_counts(
        [row("lexical-1", ("openalex:a",)), row("lexical-2", ("openalex:a", "openalex:b"))]
    )

    assert counts == {"q1": 2}
