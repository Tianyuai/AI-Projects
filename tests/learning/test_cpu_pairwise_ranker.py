from __future__ import annotations

from pathlib import Path

import pytest

from paper_search.learning.contracts import PolicyActionCandidate, QueryPolicyInput
from paper_search.learning.cpu_pairwise_ranker import CpuPairwiseActionRanker
from paper_search.learning.provider_action_labels import ProviderActionLabel
from paper_search.query.parser import rule_fallback


def _action(action_id: str, text: str) -> PolicyActionCandidate:
    return PolicyActionCandidate(
        action_id=action_id,
        action_type="text_search",
        text=text,
        origin="deterministic_rule",
        provider_hint="either",
    )


def _label(
    *,
    query_id: str,
    query: str,
    action: PolicyActionCandidate,
    hit: bool,
    status: str = "available",
    provider: str = "openalex",
) -> ProviderActionLabel:
    if status == "unavailable":
        return ProviderActionLabel(
            dataset="pasa",
            split="auto_train",
            role="training",
            query_id=query_id,
            query=query,
            provider=provider,
            action=action,
            retrieval_status="unavailable",
            error_codes=("rate_limited",),
        )
    return ProviderActionLabel(
        dataset="pasa",
        split="auto_train",
        role="training",
        query_id=query_id,
        query=query,
        provider=provider,
        action=action,
        retrieval_status="available",
        gold_association_count=1,
        gold_hit_ids=("doi:10.1/hit",) if hit else (),
        gold_hit_count=int(hit),
        action_recall=float(hit),
        novel_over_anchor_hit_count=int(hit),
    )


def _request(query_id: str, query: str) -> QueryPolicyInput:
    return QueryPolicyInput(
        query_id=query_id,
        original_query=query,
        query_kind="semantic",
        query_spec=rule_fallback(query),
        allowed_action_types=["text_search", "title_search"],
        max_actions=2,
    )


def test_pairwise_ranker_learns_query_action_compatibility() -> None:
    graph = _action("graph", "graph diffusion retrieval")
    protein = _action("protein", "protein structure folding")
    graph_query = "Find graph diffusion retrieval papers"
    protein_query = "Find protein structure folding papers"
    rows = [
        _label(query_id="g", query=graph_query, action=graph, hit=True),
        _label(query_id="g", query=graph_query, action=protein, hit=False),
        _label(query_id="p", query=protein_query, action=graph, hit=False),
        _label(query_id="p", query=protein_query, action=protein, hit=True),
    ]
    ranker = CpuPairwiseActionRanker(
        target_provider="openalex",
        dimension=2048,
        epochs=30,
        seed=7,
    )

    pair_count = ranker.fit(rows)

    assert pair_count == 2
    assert ranker.score(_request("g", graph_query), [graph, protein])[0] > ranker.score(
        _request("g", graph_query), [graph, protein]
    )[1]
    assert ranker.score(
        _request("p", protein_query), [graph, protein]
    )[1] > ranker.score(_request("p", protein_query), [graph, protein])[0]


def test_pairwise_ranker_masks_unavailable_receipts_and_requires_preference_pairs() -> None:
    query = "Find graph diffusion retrieval papers"
    graph = _action("graph", "graph diffusion retrieval")
    unavailable = _action("unavailable", "graph neural retrieval")
    ranker = CpuPairwiseActionRanker(target_provider="openalex", epochs=2)

    with pytest.raises(ValueError, match="preference pair"):
        ranker.fit(
            [
                _label(query_id="g", query=query, action=graph, hit=True),
                _label(
                    query_id="g",
                    query=query,
                    action=unavailable,
                    hit=False,
                    status="unavailable",
                ),
            ]
        )


def test_pairwise_ranker_uses_only_the_target_provider_for_fit() -> None:
    query = "Find graph diffusion retrieval papers"
    graph = _action("graph", "graph diffusion retrieval")
    protein = _action("protein", "protein structure folding")
    rows = [
        _label(query_id="g", query=query, action=graph, hit=True),
        _label(query_id="g", query=query, action=protein, hit=False),
        _label(
            query_id="g",
            query=query,
            action=graph,
            hit=False,
            provider="semantic_scholar",
        ),
        _label(
            query_id="g",
            query=query,
            action=protein,
            hit=True,
            provider="semantic_scholar",
        ),
    ]

    pair_count = CpuPairwiseActionRanker(epochs=2).fit(rows)

    assert pair_count == 1


def test_pairwise_ranker_rejects_development_labels_during_fit() -> None:
    query = "Find graph diffusion retrieval papers"
    graph = _action("graph", "graph diffusion retrieval")
    protein = _action("protein", "protein structure folding")
    rows = [
        _label(query_id="g", query=query, action=graph, hit=True),
        _label(query_id="g", query=query, action=protein, hit=False),
    ]
    rows = [row.model_copy(update={"split": "auto_dev", "role": "development"}) for row in rows]

    with pytest.raises(ValueError, match="development"):
        CpuPairwiseActionRanker(target_provider="openalex").fit(rows)


def test_pairwise_ranker_save_load_is_deterministic(tmp_path: Path) -> None:
    query = "Find graph diffusion retrieval papers"
    graph = _action("graph", "graph diffusion retrieval")
    protein = _action("protein", "protein structure folding")
    rows = [
        _label(query_id="g", query=query, action=graph, hit=True),
        _label(query_id="g", query=query, action=protein, hit=False),
    ]
    first = CpuPairwiseActionRanker(dimension=256, epochs=3, seed=7)
    second = CpuPairwiseActionRanker(dimension=256, epochs=3, seed=7)
    first.fit(rows)
    second.fit(rows)
    first_path = tmp_path / "first.f64"
    second_path = tmp_path / "second.f64"

    first.save(first_path)
    second.save(second_path)
    loaded = CpuPairwiseActionRanker.load(
        first_path,
        target_provider="openalex",
        dimension=256,
        epochs=3,
        seed=7,
    )

    assert first_path.read_bytes() == second_path.read_bytes()
    assert loaded.score(_request("g", query), [graph, protein]) == first.score(
        _request("g", query), [graph, protein]
    )


def test_pairwise_ranker_load_rejects_wrong_weight_size(tmp_path: Path) -> None:
    path = tmp_path / "broken.f64"
    path.write_bytes(b"broken")

    with pytest.raises(ValueError, match="weight size"):
        CpuPairwiseActionRanker.load(
            path,
            target_provider="openalex",
            dimension=256,
        )
