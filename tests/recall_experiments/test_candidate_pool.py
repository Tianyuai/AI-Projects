from __future__ import annotations

import inspect

import pytest

from paper_search.domain.models import ErrorDetail, Paper
import paper_search.recall_experiments.candidate_pool as candidate_pool_module
from paper_search.recall_experiments.candidate_pool import CandidatePoolBuilder
from paper_search.recall_experiments.contracts import RetrievalActionResult
from paper_search.recall_experiments.stages import CandidateStagePipeline, StageResult


class GuardedActionResult(RetrievalActionResult):
    @property
    def gold_documents(self) -> object:
        raise AssertionError("candidate pools must not access Gold documents")

    @property
    def evaluation(self) -> object:
        raise AssertionError("candidate pools must not access evaluation data")

    @property
    def identifier_map(self) -> object:
        raise AssertionError("candidate pools must not access identifier maps")


def _paper(
    canonical_id: str,
    title: str,
    **fields: object,
) -> Paper:
    return Paper(canonical_id=canonical_id, title=title, **fields)


def _result(
    action_id: str,
    action_type: str,
    hits: list[Paper],
    *,
    provenance: dict[str, str] | None = None,
    errors: list[ErrorDetail] | None = None,
) -> RetrievalActionResult:
    return RetrievalActionResult(
        action_id=action_id,
        action_type=action_type,  # type: ignore[arg-type]
        hits=hits,
        provenance=provenance or {"provider": action_type},
        errors=errors or [],
    )


def test_production_pool_preserves_action_result_rank_order_and_all_source_evidence() -> None:
    first = _paper("first", "First result")
    duplicate = _paper("duplicate", "First result", authors=["Ada Lovelace"], publication_year=2024)
    last = _paper("last", "Last result")
    results = [
        _result("text-1", "text_search", [first, last], provenance={"provider": "openalex"}),
        _result("title-1", "title_search", [duplicate], provenance={"provider": "semantic_scholar"}),
    ]

    pool = CandidatePoolBuilder("production-dedup-v1").build("query-1", results)

    assert pool.query_id == "query-1"
    assert pool.policy_version == "production-dedup-v1"
    assert [entry.paper.canonical_id for entry in pool.entries] == ["duplicate", "last"]
    assert [evidence.model_dump() for evidence in pool.entries[0].source_evidence] == [
        {"action_id": "text-1", "action_type": "text_search", "provenance": {"provider": "openalex"}},
        {
            "action_id": "title-1",
            "action_type": "title_search",
            "provenance": {"provider": "semantic_scholar"},
        },
    ]
    assert [evidence.action_id for evidence in pool.entries[1].source_evidence] == ["text-1"]
    assert pool.model_dump(mode="json")["policy_version"] == "production-dedup-v1"


@pytest.mark.parametrize(
    "left,right",
    [
        (
            _paper("doi-first", "DOI first", doi="10.1000/example"),
            _paper("doi-second", "A different title", doi="https://doi.org/10.1000/example"),
        ),
        (
            _paper("external-first", "External first", openalex_id="W123"),
            _paper("external-second", "Different external", openalex_id="openalex:W123"),
        ),
        (
            _paper("title-first", "An Exact Title"),
            _paper("title-second", " an exact title "),
        ),
        (
            _paper("fuzzy-first", "A Reliable Study of Candidate Pools", authors=["Ada Lovelace"], publication_year=2024),
            _paper("fuzzy-second", "A Reliable Study of Candidate Pool", authors=["Ada Lovelace"], publication_year=2024),
        ),
    ],
)
def test_production_policy_uses_existing_duplicate_rules(left: Paper, right: Paper) -> None:
    pool = CandidatePoolBuilder("production-dedup-v1").build(
        "query-1",
        [_result("text-1", "text_search", [left]), _result("title-1", "title_search", [right])],
    )

    assert len(pool.entries) == 1
    assert [evidence.action_id for evidence in pool.entries[0].source_evidence] == ["text-1", "title-1"]


def test_legacy_policy_keeps_first_canonical_id_without_fuzzy_merging() -> None:
    first = _paper("shared", "Legacy first")
    repeated_id = _paper("shared", "Legacy later", abstract="new metadata")
    fuzzy_title = _paper(
        "fuzzy-other",
        "Legacy firs",
        authors=["Ada Lovelace"],
        publication_year=2024,
    )

    pool = CandidatePoolBuilder("canonical-id-first-v1").build(
        "query-1",
        [_result("text-1", "text_search", [first, fuzzy_title]), _result("title-1", "title_search", [repeated_id])],
    )

    assert pool.policy_version == "canonical-id-first-v1"
    assert [entry.paper.canonical_id for entry in pool.entries] == ["shared", "fuzzy-other"]
    assert pool.entries[0].paper == first
    assert [evidence.action_id for evidence in pool.entries[0].source_evidence] == ["text-1", "title-1"]


def test_pool_keeps_partial_result_hits_and_handles_empty_results_deterministically() -> None:
    partial = _result(
        "text-1",
        "text_search",
        [_paper("partial", "Partial success")],
        errors=[
            ErrorDetail(
                code="network_error",
                message="one page failed",
                retryable=True,
                provider="openalex",
            )
        ],
    )
    empty = _result("title-1", "title_search", [])
    builder = CandidatePoolBuilder("production-dedup-v1")

    assert builder.build("query-empty", []).entries == []
    assert [entry.paper.canonical_id for entry in builder.build("query-1", [partial, empty]).entries] == [
        "partial"
    ]
    assert builder.build("query-1", [partial, empty]) == builder.build("query-1", [partial, empty])


def test_builder_rejects_unknown_policy_and_cannot_receive_gold_or_identifier_map(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="unknown candidate pool policy"):
        CandidatePoolBuilder("unknown-v1")

    signature = inspect.signature(CandidatePoolBuilder.build)
    assert "gold" not in signature.parameters
    assert "identifier_map" not in signature.parameters

    original_deduplicator = candidate_pool_module.deduplicate_papers
    calls: list[tuple[list[Paper], object]] = []

    def deduplicator_spy(papers: list[Paper], *, id_map: object) -> object:
        assert id_map is None
        calls.append((papers, id_map))
        return original_deduplicator(papers, id_map=None)

    monkeypatch.setattr(candidate_pool_module, "deduplicate_papers", deduplicator_spy)
    guarded_result = GuardedActionResult(
        action_id="text-1",
        action_type="text_search",
        hits=[_paper("paper-1", "Safe candidate")],
    )
    pool = CandidatePoolBuilder("production-dedup-v1").build(
        "query-1",
        [guarded_result],
    )

    assert [entry.paper.canonical_id for entry in pool.entries] == ["paper-1"]
    assert [evidence.action_id for evidence in pool.entries[0].source_evidence] == ["text-1"]
    assert len(calls) == 1
    assert calls[0][0] == guarded_result.hits
    assert calls[0][1] is None


class ReplacingStage:
    def __init__(self) -> None:
        self.contexts: list[object] = []

    def apply(self, pool: object, context: object) -> StageResult:
        self.contexts.append(context)
        return StageResult(pool=pool)  # type: ignore[arg-type]


class ArtifactMutatingStage:
    def apply(self, pool: object, context: object) -> StageResult:
        artifacts = context  # type: ignore[assignment]
        artifacts["generation_artifact"]["actions"].append("forbidden")  # type: ignore[index]
        artifacts["retrieval_artifact"]["results"].append("forbidden")  # type: ignore[index]
        return StageResult(pool=pool)  # type: ignore[arg-type]


def test_empty_stage_pipeline_returns_the_exact_pool_and_non_empty_stages_are_explicit() -> None:
    pool = CandidatePoolBuilder("production-dedup-v1").build(
        "query-1", [_result("text-1", "text_search", [_paper("paper-1", "One")])]
    )
    generation_artifact = {"actions": ["text-1"]}
    retrieval_artifact = {"results": ["paper-1"]}
    context = {"generation_artifact": generation_artifact, "retrieval_artifact": retrieval_artifact}

    assert CandidateStagePipeline().apply(pool, context) is pool

    stage = ReplacingStage()
    assert CandidateStagePipeline(stages=(stage,)).apply(pool, context) is pool
    assert stage.contexts == [context]
    assert CandidateStagePipeline(stages=(ArtifactMutatingStage(),)).apply(pool, context) is pool
    assert generation_artifact == {"actions": ["text-1"]}
    assert retrieval_artifact == {"results": ["paper-1"]}
