from __future__ import annotations

from paper_search.learning.candidate_family_diagnostics import (
    summarize_candidate_family_batch,
)
from paper_search.learning.contracts import PolicyActionCandidate
from paper_search.learning.provider_action_labels import ProviderActionLabel


def _label(query_id: str, action_id: str, hits: tuple[str, ...]) -> ProviderActionLabel:
    return ProviderActionLabel(
        dataset="pasa",
        split="auto_train",
        role="training",
        query_id=query_id,
        query="graph retrieval",
        provider="openalex",
        action=PolicyActionCandidate(
            action_id=action_id,
            action_type="text_search",
            text=action_id,
            origin=(
                "original_query" if action_id.endswith("anchor") else "deterministic_rule"
            ),
            provider_hint="openalex",
            search_mode=("semantic" if "semantic" in action_id else "lexical"),
        ),
        retrieval_status="available",
        gold_association_count=3,
        gold_hit_ids=hits,
        gold_hit_count=len(hits),
        action_recall=len(hits) / 3,
        novel_over_anchor_hit_count=(0 if action_id.endswith("anchor") else len(hits)),
    )


def test_family_summary_reports_non_additive_incremental_oracle_lift() -> None:
    labels = []
    for query_id in ("q1", "q2"):
        labels.extend(
            [
                _label(query_id, "ceiling-candidate-anchor", ("g1",)),
                _label(query_id, "ceiling-candidate-semantic-original", ("g2",)),
                _label(query_id, "ceiling-candidate-boolean-relaxed", ("g3",)),
                _label(query_id, "ceiling-candidate-prf-1", ("g2",)),
            ]
        )

    summary = summarize_candidate_family_batch(labels, batch_id="fold-1")

    assert summary.query_count == 2
    assert summary.baseline_all_candidate_macro_recall == 1 / 3
    assert summary.v2_all_candidate_macro_recall == 1.0
    assert summary.baseline_oracle_at_3_macro_recall == 1 / 3
    assert summary.v2_oracle_at_3_macro_recall == 1.0
    assert summary.semantic_incremental_macro_recall == 1 / 3
    assert summary.boolean_phrase_incremental_macro_recall == 1 / 3
    assert summary.prf_incremental_macro_recall == 1 / 3
