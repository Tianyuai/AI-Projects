from datetime import UTC, datetime

import pytest

from paper_search.domain.models import ErrorDetail, Paper, ProviderResult, UsageActual
from paper_search.evaluation.dataset import IdentifierMap
from paper_search.ranking.fusion import fuse_provider_results


def _paper(
    canonical_id: str,
    *,
    doi: str | None = None,
    openalex_id: str | None = None,
    semantic_scholar_id: str | None = None,
    sources: list[str],
) -> Paper:
    return Paper(
        canonical_id=canonical_id,
        title=f"Paper {canonical_id}",
        doi=doi,
        openalex_id=openalex_id,
        semantic_scholar_id=semantic_scholar_id,
        sources=sources,
    )


def _result(provider: str, papers: list[Paper], *, failed: bool = False) -> ProviderResult[list[Paper]]:
    return ProviderResult[list[Paper]](
        data=papers,
        usage=UsageActual(search_api_calls=1),
        provenance={
            "provider": provider,
            "endpoint": "/search",
            "model_id": "fixture",
            "requested_at": datetime(2026, 7, 23, tzinfo=UTC).isoformat(),
            "response_hash": f"sha256:{provider}",
        },
        cache_hit=False,
        latency_ms=1,
        errors=(
            [
                ErrorDetail(
                    code="timeout",
                    message="synthetic failure",
                    retryable=True,
                    provider=provider,
                )
            ]
            if failed
            else []
        ),
    )


def test_rrf_is_deterministic_and_uses_stable_tie_breaks() -> None:
    results = {
        "openalex": _result(
            "openalex",
            [
                _paper("openalex:W2", openalex_id="W2", sources=["openalex"]),
                _paper("openalex:W1", openalex_id="W1", sources=["openalex"]),
            ],
        ),
        "semantic_scholar": _result(
            "semantic_scholar",
            [
                _paper("s2:S1", semantic_scholar_id="S1", sources=["semantic_scholar"]),
                _paper("s2:S2", semantic_scholar_id="S2", sources=["semantic_scholar"]),
            ],
        ),
    }

    first = fuse_provider_results(results, method="rrf", rrf_k=60)
    second = fuse_provider_results(results, method="rrf", rrf_k=60)

    assert first == second
    assert [item.paper.canonical_id for item in first] == [
        "openalex:W2",
        "s2:S1",
        "openalex:W1",
        "s2:S2",
    ]
    assert first[0].score == pytest.approx(1 / 61)


def test_weighted_fusion_changes_provider_contribution() -> None:
    results = {
        "openalex": _result(
            "openalex",
            [_paper("openalex:W1", openalex_id="W1", sources=["openalex"])],
        ),
        "semantic_scholar": _result(
            "semantic_scholar",
            [_paper("s2:S1", semantic_scholar_id="S1", sources=["semantic_scholar"])],
        ),
    }

    fused = fuse_provider_results(
        results,
        method="weighted",
        provider_weights={"openalex": 0.25, "semantic_scholar": 0.75},
    )

    assert [item.paper.canonical_id for item in fused] == ["s2:S1", "openalex:W1"]
    assert [item.score for item in fused] == pytest.approx([0.75, 0.25])


def test_cross_source_duplicate_retains_ids_sources_and_rank_evidence() -> None:
    results = {
        "openalex": _result(
            "openalex",
            [
                _paper(
                    "openalex:W1",
                    doi="10.9999/shared",
                    openalex_id="W1",
                    sources=["openalex"],
                )
            ],
        ),
        "semantic_scholar": _result(
            "semantic_scholar",
            [
                _paper(
                    "s2:S1",
                    doi="https://doi.org/10.9999/SHARED",
                    semantic_scholar_id="S1",
                    sources=["semantic_scholar"],
                )
            ],
        ),
    }

    fused = fuse_provider_results(results, method="rrf", rrf_k=10)

    assert len(fused) == 1
    assert fused[0].paper.openalex_id == "W1"
    assert fused[0].paper.semantic_scholar_id == "S1"
    assert set(fused[0].paper.sources) == {"openalex", "semantic_scholar"}
    assert fused[0].source_ranks == {"openalex": 1, "semantic_scholar": 1}
    assert fused[0].score == pytest.approx(2 / 11)


def test_fusion_uses_frozen_aliases_for_cross_provider_identity() -> None:
    results = {
        "openalex": _result(
            "openalex",
            [_paper("openalex:W1", openalex_id="W1", sources=["openalex"])],
        ),
        "semantic_scholar": _result(
            "semantic_scholar",
            [
                _paper(
                    "s2:S1",
                    semantic_scholar_id="S1",
                    sources=["semantic_scholar"],
                )
            ],
        ),
    }
    aliases = IdentifierMap.from_bytes(
        b'{"openalex:W1":"arxiv:2401.00001",'
        b'"s2:S1":"arxiv:2401.00001"}'
    )

    fused = fuse_provider_results(results, method="rrf", id_map=aliases)

    assert len(fused) == 1
    assert fused[0].paper.openalex_id == "W1"
    assert fused[0].paper.semantic_scholar_id == "S1"
    assert fused[0].source_ranks == {"openalex": 1, "semantic_scholar": 1}


def test_failed_provider_does_not_hide_valid_sibling_results() -> None:
    results = {
        "openalex": _result("openalex", [], failed=True),
        "semantic_scholar": _result(
            "semantic_scholar",
            [_paper("s2:S1", semantic_scholar_id="S1", sources=["semantic_scholar"])],
        ),
    }

    fused = fuse_provider_results(results, method="rrf")

    assert [item.paper.canonical_id for item in fused] == ["s2:S1"]


@pytest.mark.parametrize(
    ("method", "weights"),
    [
        ("unknown", None),
        ("weighted", None),
        ("weighted", {"openalex": -1.0}),
    ],
)
def test_invalid_fusion_configuration_is_rejected(
    method: str,
    weights: dict[str, float] | None,
) -> None:
    with pytest.raises(ValueError):
        fuse_provider_results({}, method=method, provider_weights=weights)
