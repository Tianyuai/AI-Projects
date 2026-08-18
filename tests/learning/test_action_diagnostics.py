from __future__ import annotations

import pytest

from paper_search.learning.action_diagnostics import diagnose_action_selection
from paper_search.learning.contracts import PolicyActionCandidate
from paper_search.learning.provider_action_labels import ProviderActionLabel


def _label(
    action_id: str,
    provider: str,
    hits: tuple[str, ...],
    *,
    origin: str = "deterministic_rule",
) -> ProviderActionLabel:
    action = PolicyActionCandidate(
        action_id=action_id,
        action_type="text_search",
        text=action_id.replace("-", " "),
        origin=origin,
        provider_hint="either",
    )
    return ProviderActionLabel(
        dataset="pasa",
        split="auto_dev",
        role="development",
        query_id="q-1",
        query="graph retrieval",
        provider=provider,
        action=action,
        retrieval_status="available",
        gold_association_count=3,
        gold_hit_ids=hits,
        gold_hit_count=len(hits),
        action_recall=len(hits) / 3,
        novel_over_anchor_hit_count=len(hits),
    )


def test_oracle_selects_complementary_action_provider_pairs() -> None:
    labels = [
        _label("anchor", "openalex", ("g1",), origin="original_query"),
        _label("anchor", "semantic_scholar", ("g1",), origin="original_query"),
        _label("method-query", "openalex", ("g1", "g2")),
        _label("method-query", "semantic_scholar", ("g1",)),
        _label("dataset-query", "openalex", ()),
        _label("dataset-query", "semantic_scholar", ("g3",)),
    ]

    result = diagnose_action_selection(
        labels,
        selected_action_provider_pairs=[
            ("anchor", "openalex"),
            ("method-query", "openalex"),
        ],
        max_actions=2,
    )

    assert result.selected_gold_hit_count == 2
    assert result.selected_recall == 2 / 3
    assert result.oracle_gold_hit_count == 3
    assert result.oracle_recall == 1.0
    assert result.selection_gap == pytest.approx(1 / 3)
    assert set(result.oracle_action_provider_pairs) == {
        ("method-query", "openalex"),
        ("dataset-query", "semantic_scholar"),
    }


def test_oracle_ignores_unavailable_provider_actions() -> None:
    available = _label("anchor", "openalex", ("g1",), origin="original_query")
    unavailable = _label("expanded", "semantic_scholar", ())
    unavailable = unavailable.model_copy(
        update={
            "retrieval_status": "unavailable",
            "gold_association_count": None,
            "gold_hit_count": None,
            "action_recall": None,
            "novel_over_anchor_hit_count": None,
        }
    )

    result = diagnose_action_selection(
        [available, unavailable],
        selected_action_provider_pairs=[("anchor", "openalex")],
        max_actions=2,
    )

    assert result.available_action_provider_count == 1
    assert result.oracle_action_provider_pairs == (("anchor", "openalex"),)
