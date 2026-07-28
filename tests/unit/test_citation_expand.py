from __future__ import annotations

import hashlib

import pytest

from paper_search.domain.models import CitationEdge, CitationExpansion, Paper, ProviderPaperId


def make_seed(*, canonical_id: str, semantic_scholar_id: str) -> Paper:
    return Paper(
        canonical_id=canonical_id,
        title=f"Seed {canonical_id}",
        semantic_scholar_id=semantic_scholar_id,
        sources=["semantic_scholar"],
    )


def make_expanded(
    *,
    canonical_id: str,
    semantic_scholar_id: str,
    title: str | None = None,
) -> Paper:
    return Paper(
        canonical_id=canonical_id,
        title=title or f"Expanded {canonical_id}",
        semantic_scholar_id=semantic_scholar_id,
        sources=["semantic_scholar"],
    )


def s2(identifier: str) -> ProviderPaperId:
    return ProviderPaperId(provider="semantic_scholar", value=identifier)


def edge(*, citing: str, cited: str) -> CitationEdge:
    return CitationEdge(
        provider="semantic_scholar",
        citing_provider_id=s2(citing),
        cited_provider_id=s2(cited),
    )


def test_expand_one_hop_maps_forward_and_backward_edges() -> None:
    from paper_search.graph.citation_expand import expand_one_hop

    seed = make_seed(canonical_id="doi:seed", semantic_scholar_id="S-SEED")
    cited = make_expanded(canonical_id="s2:ref", semantic_scholar_id="S-REF")
    citing = make_expanded(canonical_id="s2:cit", semantic_scholar_id="S-CIT")
    mapping = {
        s2("S-SEED"): "doi:seed",
        s2("S-REF"): "doi:ref",
        s2("S-CIT"): "doi:cit",
    }

    result = expand_one_hop(
        [seed],
        CitationExpansion(
            papers=[cited, citing],
            raw_edges=[
                edge(citing="S-SEED", cited="S-REF"),
                edge(citing="S-CIT", cited="S-SEED"),
            ],
        ),
        mapping,
    )

    assert [paper.canonical_id for paper in result.papers] == [
        "doi:seed",
        "doi:ref",
        "doi:cit",
    ]
    assert [(resolved.citing_canonical_id, resolved.cited_canonical_id) for resolved in result.edges] == [
        ("doi:seed", "doi:ref"),
        ("doi:cit", "doi:seed"),
    ]
    assert result.edges[0].source_edge_hash == "sha256:" + hashlib.sha256(
        "semantic_scholar|doi:seed|doi:ref".encode("utf-8")
    ).hexdigest()
    assert result.skipped_edge_count == 0
    assert result.truncated is False
    assert result.warnings == []


def test_expand_one_hop_counts_unresolved_and_malformed_edges_aggregately() -> None:
    from paper_search.graph.citation_expand import expand_one_hop

    seed = make_seed(canonical_id="doi:seed", semantic_scholar_id="S-SEED")
    expanded = make_expanded(canonical_id="s2:ref", semantic_scholar_id="S-REF")
    malformed = CitationEdge.model_construct(
        provider="semantic_scholar",
        citing_provider_id=ProviderPaperId(provider="openalex", value="W-BAD"),
        cited_provider_id=s2("S-SEED"),
        citing_canonical_id=None,
        cited_canonical_id=None,
    )

    result = expand_one_hop(
        [seed],
        CitationExpansion.model_construct(
            papers=[expanded],
            raw_edges=[
                edge(citing="S-SEED", cited="S-MISSING"),
                malformed,
            ],
        ),
        {
            s2("S-SEED"): "doi:seed",
        },
    )

    assert result.papers == [seed]
    assert result.edges == []
    assert result.skipped_edge_count == 2
    assert result.warnings == ["unresolved_citation_edge"]
    serialized = result.model_dump()["warnings"]
    assert serialized == ["unresolved_citation_edge"]
    assert "S-MISSING" not in serialized[0]
    assert "W-BAD" not in serialized[0]


def test_expand_one_hop_drops_self_edges_and_duplicate_edges_and_papers() -> None:
    from paper_search.graph.citation_expand import expand_one_hop

    seed = make_seed(canonical_id="doi:seed", semantic_scholar_id="S-SEED")
    first = make_expanded(
        canonical_id="s2:first",
        semantic_scholar_id="S-DUP-A",
        title="First duplicate survives",
    )
    second = make_expanded(
        canonical_id="s2:second",
        semantic_scholar_id="S-DUP-B",
        title="Second duplicate is dropped",
    )

    result = expand_one_hop(
        [seed],
        CitationExpansion(
            papers=[first, second],
            raw_edges=[
                edge(citing="S-SEED", cited="S-SEED"),
                edge(citing="S-SEED", cited="S-DUP-A"),
                edge(citing="S-SEED", cited="S-DUP-B"),
            ],
        ),
        {
            s2("S-SEED"): "doi:seed",
            s2("S-DUP-A"): "doi:dup",
            s2("S-DUP-B"): "doi:dup",
        },
    )

    assert [paper.canonical_id for paper in result.papers] == ["doi:seed", "doi:dup"]
    assert [paper.title for paper in result.papers] == ["Seed doi:seed", "First duplicate survives"]
    assert [(resolved.citing_canonical_id, resolved.cited_canonical_id) for resolved in result.edges] == [
        ("doi:seed", "doi:dup")
    ]
    assert result.skipped_edge_count == 0
    assert result.warnings == []


def test_expand_one_hop_applies_max_seed_and_max_expanded_truncation() -> None:
    from paper_search.graph.citation_expand import expand_one_hop

    seeds = [
        make_seed(canonical_id="doi:seed-1", semantic_scholar_id="S-SEED-1"),
        make_seed(canonical_id="doi:seed-2", semantic_scholar_id="S-SEED-2"),
    ]
    expansion = CitationExpansion(
        papers=[
            make_expanded(canonical_id="s2:a", semantic_scholar_id="S-A"),
            make_expanded(canonical_id="s2:b", semantic_scholar_id="S-B"),
            make_expanded(canonical_id="s2:c", semantic_scholar_id="S-C"),
        ],
        raw_edges=[
            edge(citing="S-SEED-1", cited="S-A"),
            edge(citing="S-SEED-2", cited="S-B"),
            edge(citing="S-SEED-1", cited="S-C"),
        ],
    )
    mapping = {
        s2("S-SEED-1"): "doi:seed-1",
        s2("S-SEED-2"): "doi:seed-2",
        s2("S-A"): "doi:a",
        s2("S-B"): "doi:b",
        s2("S-C"): "doi:c",
    }

    result = expand_one_hop(seeds, expansion, mapping, max_seeds=1, max_expanded=1)

    assert [paper.canonical_id for paper in result.papers] == ["doi:seed-1", "doi:a"]
    assert [(resolved.citing_canonical_id, resolved.cited_canonical_id) for resolved in result.edges] == [
        ("doi:seed-1", "doi:a")
    ]
    assert result.truncated is True


def test_expand_one_hop_handles_empty_expansion() -> None:
    from paper_search.graph.citation_expand import CitationExpansionResult, expand_one_hop

    seed = make_seed(canonical_id="doi:seed", semantic_scholar_id="S-SEED")

    result = expand_one_hop(
        [seed],
        CitationExpansion(papers=[], raw_edges=[]),
        {s2("S-SEED"): "doi:seed"},
    )

    assert result == CitationExpansionResult(
        papers=[seed],
        edges=[],
        skipped_edge_count=0,
        truncated=False,
        warnings=[],
    )


@pytest.mark.parametrize(
    ("max_seeds", "max_expanded"),
    [(0, 1), (1, 0), (-1, 1), (1, -1), (True, 1), (1, False)],
)
def test_expand_one_hop_rejects_invalid_limits(max_seeds: object, max_expanded: object) -> None:
    from paper_search.graph.citation_expand import expand_one_hop

    seed = make_seed(canonical_id="doi:seed", semantic_scholar_id="S-SEED")

    with pytest.raises(ValueError):
        expand_one_hop(
            [seed],
            CitationExpansion(papers=[], raw_edges=[]),
            {s2("S-SEED"): "doi:seed"},
            max_seeds=max_seeds,
            max_expanded=max_expanded,
        )


def test_expand_one_hop_rejects_empty_and_duplicate_seed_ids() -> None:
    from paper_search.graph.citation_expand import expand_one_hop

    seed = make_seed(canonical_id="doi:seed", semantic_scholar_id="S-SEED")

    with pytest.raises(ValueError, match="seeds must not be empty"):
        expand_one_hop([], CitationExpansion(papers=[], raw_edges=[]), {})
    with pytest.raises(ValueError, match="seed canonical IDs must be unique"):
        expand_one_hop(
            [seed, seed.model_copy()],
            CitationExpansion(papers=[], raw_edges=[]),
            {s2("S-SEED"): "doi:seed"},
        )


def test_expand_one_hop_is_deterministic_across_repeated_calls() -> None:
    from paper_search.graph.citation_expand import expand_one_hop

    seed = make_seed(canonical_id="doi:seed", semantic_scholar_id="S-SEED")
    expansion = CitationExpansion(
        papers=[make_expanded(canonical_id="s2:ref", semantic_scholar_id="S-REF")],
        raw_edges=[edge(citing="S-SEED", cited="S-REF")],
    )
    mapping = {
        s2("S-SEED"): "doi:seed",
        s2("S-REF"): "doi:ref",
    }

    first = expand_one_hop([seed], expansion, mapping)
    second = expand_one_hop([seed], expansion, mapping)

    assert first.model_dump() == second.model_dump()
    assert first.model_dump_json() == second.model_dump_json()
