from __future__ import annotations

import pytest

from paper_search.domain.models import Paper
from paper_search.ranking.rerank import ConstraintReranker


class RecordingEvaluator:
    def __init__(self, responses: dict[str, object]) -> None:
        self._responses = responses
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def __call__(self, paper: Paper, constraints: tuple[str, ...]) -> object:
        self.calls.append((paper.canonical_id, constraints))
        response = self._responses[paper.canonical_id]
        if isinstance(response, Exception):
            raise response
        return response


def _paper(identifier: str, title: str) -> Paper:
    return Paper(canonical_id=identifier, title=title, abstract=f"{title} abstract")


def test_reranker_normalizes_constraints_stably_sorts_and_preserves_full_papers() -> None:
    papers = [
        _paper("paper:1", "Alpha"),
        _paper("paper:2", "Beta"),
        _paper("paper:3", "Gamma"),
    ]
    evaluator = RecordingEvaluator(
        {
            "paper:1": {
                "matched_constraints": ["constraint a"],
                "unmatched_constraints": ["constraint b"],
                "relevance_score": 0.8,
                "raw_reasoning": "private detail one",
            },
            "paper:2": {
                "matched_constraints": ["constraint a"],
                "unmatched_constraints": ["constraint b"],
                "relevance_score": 0.8,
                "raw_reasoning": "private detail two",
            },
            "paper:3": {
                "matched_constraints": [],
                "unmatched_constraints": ["constraint a", "constraint b"],
                "relevance_score": 0.2,
                "raw_reasoning": "private detail three",
            },
        }
    )

    result = ConstraintReranker(evaluator, batch_size=3, max_batches=1).rerank(
        papers,
        ["  constraint a  ", "\nconstraint   b\t"],
    )

    assert evaluator.calls == [
        ("paper:1", ("constraint a", "constraint b")),
        ("paper:2", ("constraint a", "constraint b")),
        ("paper:3", ("constraint a", "constraint b")),
    ]
    assert [item.paper.canonical_id for item in result.ranked] == [
        "paper:1",
        "paper:2",
        "paper:3",
    ]
    assert result.ranked[0].paper.abstract == "Alpha abstract"
    assert result.ranked[0].assessment.matched_constraint_count == 1
    assert result.ranked[0].assessment.unmatched_constraint_count == 1
    assert result.ranked[0].assessment.constraint_coverage == pytest.approx(0.5)
    assert result.ranked[0].score == pytest.approx(0.71)
    assert result.status == "applied"
    assert result.processed_count == 3
    assert result.batch_count == 1
    assert result.truncated is False
    assert result.warnings == []


def test_reranker_combines_relevance_and_coverage_deterministically() -> None:
    papers = [
        _paper("paper:coverage", "Coverage winner"),
        _paper("paper:relevance", "Relevance winner"),
    ]
    evaluator = RecordingEvaluator(
        {
            "paper:coverage": {
                "matched_constraint_count": 2,
                "unmatched_constraint_count": 0,
                "relevance_score": 0.5,
            },
            "paper:relevance": {
                "matched_constraint_count": 0,
                "unmatched_constraint_count": 2,
                "relevance_score": 0.9,
            },
        }
    )

    result = ConstraintReranker(evaluator).rerank(
        papers,
        ["constraint a", "constraint b"],
    )

    assert [item.paper.canonical_id for item in result.ranked] == [
        "paper:coverage",
        "paper:relevance",
    ]
    assert result.ranked[0].score == pytest.approx(0.65)
    assert result.ranked[1].score == pytest.approx(0.63)
    assert result.ranked[0].assessment.constraint_coverage == pytest.approx(1.0)
    assert result.ranked[1].assessment.constraint_coverage == pytest.approx(0.0)


def test_reranker_enforces_candidate_and_batch_limits() -> None:
    papers = [_paper(f"paper:{index}", f"Paper {index}") for index in range(5)]
    evaluator = RecordingEvaluator(
        {
            paper.canonical_id: {
                "matched_constraint_count": 1,
                "unmatched_constraint_count": 0,
                "relevance_score": 0.5,
            }
            for paper in papers
        }
    )

    result = ConstraintReranker(
        evaluator,
        max_candidates=5,
        batch_size=2,
        max_batches=2,
    ).rerank(papers, ["constraint"])

    assert evaluator.calls == [
        ("paper:0", ("constraint",)),
        ("paper:1", ("constraint",)),
        ("paper:2", ("constraint",)),
        ("paper:3", ("constraint",)),
    ]
    assert [item.paper.canonical_id for item in result.ranked] == [
        "paper:0",
        "paper:1",
        "paper:2",
        "paper:3",
    ]
    assert result.processed_count == 4
    assert result.batch_count == 2
    assert result.truncated is True


def test_reranker_returns_applied_without_evaluator_for_empty_papers() -> None:
    evaluator = RecordingEvaluator({})

    result = ConstraintReranker(evaluator).rerank([], ["constraint"])

    assert result.ranked == []
    assert result.status == "applied"
    assert result.processed_count == 0
    assert result.batch_count == 0
    assert result.truncated is False
    assert evaluator.calls == []


def test_reranker_returns_zero_coverage_without_calling_evaluator_for_empty_constraints() -> None:
    papers = [_paper("paper:1", "Alpha"), _paper("paper:2", "Beta")]
    evaluator = RecordingEvaluator({})

    result = ConstraintReranker(evaluator).rerank(papers, ["   ", "\n\t"])

    assert [item.paper.canonical_id for item in result.ranked] == [
        "paper:1",
        "paper:2",
    ]
    assert [item.score for item in result.ranked] == [0.0, 0.0]
    assert all(item.assessment.constraint_coverage == 0.0 for item in result.ranked)
    assert all(item.assessment.relevance_score == 0.0 for item in result.ranked)
    assert result.status == "applied"
    assert result.processed_count == 0
    assert result.batch_count == 0
    assert result.truncated is False
    assert evaluator.calls == []


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_candidates": 0}, "max_candidates"),
        ({"max_candidates": -1}, "max_candidates"),
        ({"max_candidates": True}, "max_candidates"),
        ({"batch_size": 0}, "batch_size"),
        ({"batch_size": 16}, "batch_size"),
        ({"batch_size": True}, "batch_size"),
        ({"max_batches": 0}, "max_batches"),
        ({"max_batches": 3}, "max_batches"),
        ({"max_batches": True}, "max_batches"),
    ],
)
def test_reranker_rejects_invalid_limits(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        ConstraintReranker(RecordingEvaluator({}), **kwargs)


def test_reranker_degrades_on_evaluator_failure_without_leaking_exception_text() -> None:
    private_detail = "private evaluator failure with synthetic query"
    papers = [
        _paper("paper:2", "Second"),
        _paper("paper:1", "First"),
    ]
    evaluator = RecordingEvaluator(
        {
            "paper:2": RuntimeError(private_detail),
            "paper:1": {
                "matched_constraint_count": 1,
                "unmatched_constraint_count": 0,
                "relevance_score": 0.9,
            },
        }
    )

    result = ConstraintReranker(evaluator).rerank(papers, ["constraint"])

    assert [item.paper.canonical_id for item in result.ranked] == [
        "paper:2",
        "paper:1",
    ]
    assert [item.score for item in result.ranked] == [0.0, 0.0]
    assert result.status == "degraded"
    assert result.processed_count == 0
    assert result.batch_count == 0
    assert result.truncated is False
    assert result.warnings == ["rerank_unavailable"]
    assert private_detail not in result.model_dump_json()


def test_reranker_is_repeatable_and_serializes_only_aggregate_metadata() -> None:
    papers = [_paper("paper:1", "Alpha")]
    evaluator = RecordingEvaluator(
        {
            "paper:1": {
                "matched_constraints": ["query like constraint"],
                "unmatched_constraints": [],
                "relevance_score": 0.4,
                "raw_reasoning": "arbitrary evaluator string",
            }
        }
    )
    reranker = ConstraintReranker(evaluator)

    first = reranker.rerank(papers, [" query   like   constraint "])
    second = reranker.rerank(papers, [" query   like   constraint "])

    assert first.model_dump() == second.model_dump()
    serialized = first.model_dump_json()
    assert "query like constraint" not in serialized
    assert "arbitrary evaluator string" not in serialized
    assert "raw_reasoning" not in serialized
