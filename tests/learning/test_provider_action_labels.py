from __future__ import annotations

from paper_search.domain.models import ErrorDetail, Paper, UsageActual
from paper_search.learning.contracts import PolicyActionCandidate
from paper_search.learning.provider_action_labels import (
    ProviderActionObservation,
    build_provider_action_labels,
)


def _action(action_id: str, text: str, *, origin: str = "deterministic_rule"):
    return PolicyActionCandidate(
        action_id=action_id,
        action_type="text_search",
        text=text,
        origin=origin,
        provider_hint="either",
    )


def _paper(arxiv_id: str) -> Paper:
    return Paper(
        canonical_id=f"arxiv:{arxiv_id}",
        arxiv_id=arxiv_id,
        title=f"Paper {arxiv_id}",
        sources=["openalex"],
    )


def test_real_receipt_labels_measure_hits_and_novel_gain_over_anchor() -> None:
    observations = [
        ProviderActionObservation(
            provider="openalex",
            action=_action("anchor", "graph retrieval", origin="original_query"),
            hits=[_paper("2001.00001")],
            usage=UsageActual(search_api_calls=1, cost_cny="0"),
        ),
        ProviderActionObservation(
            provider="openalex",
            action=_action("expanded", "graph diffusion retrieval"),
            hits=[_paper("2001.00001"), _paper("2002.00002")],
            usage=UsageActual(search_api_calls=1, cost_cny="0"),
        ),
    ]

    labels = build_provider_action_labels(
        dataset="pasa",
        split="auto_train",
        role="training",
        query_id="q-1",
        query="Which works study graph retrieval?",
        gold_paper_ids=["arxiv:2001.00001", "arxiv:2002.00002"],
        observations=observations,
    )

    assert labels[0].retrieval_status == "available"
    assert labels[0].gold_hit_count == 1
    assert labels[0].gold_hit_ids == ("doi:10.48550/arxiv.2001.00001",)
    assert labels[0].action_recall == 0.5
    assert labels[0].novel_over_anchor_hit_count == 0
    assert labels[1].gold_hit_count == 2
    assert labels[1].action_recall == 1.0
    assert labels[1].novel_over_anchor_hit_count == 1


def test_infrastructure_failure_is_unavailable_not_a_negative_label() -> None:
    observation = ProviderActionObservation(
        provider="semantic_scholar",
        action=_action("expanded", "graph diffusion retrieval"),
        hits=[],
        usage=UsageActual(search_api_calls=3, cost_cny="0"),
        errors=[
            ErrorDetail(
                code="rate_limited",
                message="provider rate limited",
                retryable=True,
                provider="semantic_scholar",
            )
        ],
        infrastructure_failure=True,
    )

    [label] = build_provider_action_labels(
        dataset="pasa",
        split="auto_train",
        role="training",
        query_id="q-1",
        query="Which works study graph retrieval?",
        gold_paper_ids=["arxiv:2001.00001"],
        observations=[observation],
    )

    assert label.retrieval_status == "unavailable"
    assert label.gold_hit_count is None
    assert label.action_recall is None
    assert label.novel_over_anchor_hit_count is None
    assert label.gold_hit_ids == ()
    assert label.error_codes == ("rate_limited",)


def test_zero_hit_success_is_an_available_zero_reward_label() -> None:
    observation = ProviderActionObservation(
        provider="openalex",
        action=_action("expanded", "unrelated terms"),
        hits=[],
        usage=UsageActual(search_api_calls=1, cost_cny="0"),
    )

    [label] = build_provider_action_labels(
        dataset="pasa",
        split="auto_train",
        role="training",
        query_id="q-1",
        query="Which works study graph retrieval?",
        gold_paper_ids=["arxiv:2001.00001"],
        observations=[observation],
    )

    assert label.retrieval_status == "available"
    assert label.gold_hit_count == 0
    assert label.action_recall == 0.0


def test_invalid_request_is_unavailable_even_when_transport_did_not_fail() -> None:
    observation = ProviderActionObservation(
        provider="openalex",
        action=_action("semantic", "semantic query"),
        hits=[],
        usage=UsageActual(search_api_calls=1, cost_cny="0"),
        errors=[
            ErrorDetail(
                code="invalid_request",
                message="request rejected",
                retryable=False,
                provider="openalex",
            )
        ],
        infrastructure_failure=False,
    )

    [label] = build_provider_action_labels(
        dataset="pasa",
        split="auto_train",
        role="training",
        query_id="q-1",
        query="Which works study graph retrieval?",
        gold_paper_ids=["arxiv:2001.00001"],
        observations=[observation],
    )

    assert label.retrieval_status == "unavailable"
    assert label.gold_hit_count is None
    assert label.error_codes == ("invalid_request",)
