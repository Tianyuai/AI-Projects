from __future__ import annotations

from paper_search.domain.models import Paper
from paper_search.evaluation.conservative_identity_aliases import (
    build_conservative_pasa_identifier_aliases,
)
from paper_search.evaluation.dataset import normalize_title


def _abstract(*, shared: int, private: str) -> str:
    common = " ".join(f"shared{index}" for index in range(shared))
    return f"{common} {private}"


def test_conservative_alias_requires_unique_title_abstract_and_year_evidence() -> None:
    title = "Image Super-Resolution via Iterative Refinement"
    pasa = Paper(
        canonical_id="arxiv:2104.07636",
        arxiv_id="2104.07636",
        title=title,
        abstract=_abstract(shared=80, private="pasa extension"),
        publication_year=2021,
        sources=["pasa_paper_database"],
    )
    candidate = Paper(
        canonical_id="doi:10.1109/tpami.2022.3204461",
        doi="10.1109/tpami.2022.3204461",
        openalex_id="W3155072588",
        title=title,
        abstract=_abstract(shared=80, private="publisher extension"),
        publication_year=2022,
        sources=["openalex"],
    )

    aliases, evidence, decisions = build_conservative_pasa_identifier_aliases(
        [candidate],
        {normalize_title(title): (pasa,)},
    )

    assert aliases == {
        "doi:10.1109/tpami.2022.3204461": "arxiv:2104.07636",
        "openalex:W3155072588": "arxiv:2104.07636",
    }
    assert len(evidence) == 1
    assert evidence[0]["target_id"] == "arxiv:2104.07636"
    assert evidence[0]["shared_abstract_token_count"] >= 80
    assert evidence[0]["publication_year_distance"] == 1
    assert decisions == {"accepted": 1}


def test_conservative_alias_rejects_title_only_ambiguous_and_stale_matches() -> None:
    title = "A Distinctive Scientific Paper Title"
    base = Paper(
        canonical_id="doi:10.1000/candidate",
        doi="10.1000/candidate",
        title=title,
        abstract=_abstract(shared=80, private="candidate"),
        publication_year=2022,
        sources=["openalex"],
    )
    reference = Paper(
        canonical_id="arxiv:2201.00001",
        arxiv_id="2201.00001",
        title=title,
        abstract=_abstract(shared=80, private="reference"),
        publication_year=2022,
        sources=["pasa_paper_database"],
    )
    duplicate = reference.model_copy(
        update={"canonical_id": "arxiv:2201.00002", "arxiv_id": "2201.00002"}
    )

    cases = [
        (base.model_copy(update={"abstract": None}), (reference,), "missing_abstract"),
        (
            base.model_copy(update={"abstract": "completely unrelated short text"}),
            (reference,),
            "insufficient_abstract_overlap",
        ),
        (base, (reference, duplicate), "ambiguous_pasa_title"),
        (
            base.model_copy(update={"publication_year": 2015}),
            (reference,),
            "publication_year_mismatch",
        ),
    ]

    for candidate, references, expected_reason in cases:
        aliases, evidence, decisions = build_conservative_pasa_identifier_aliases(
            [candidate],
            {normalize_title(title): references},
        )
        assert aliases == {}
        assert evidence == []
        assert decisions == {expected_reason: 1}
