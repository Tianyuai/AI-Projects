from __future__ import annotations

import pytest

from paper_search.domain.models import Paper
from paper_search.evaluation.dataset import EvaluationQuery
from paper_search.recall_experiments.candidate_pool import CandidatePoolBuilder
from paper_search.recall_experiments.contracts import RetrievalActionResult, SeedCandidate
from paper_search.recall_experiments.evaluator import (
    CandidateRecallEvaluator,
    HistoricalReplayEvidence,
    RecallAttempt,
    compare_exact_replay,
    compare_regenerated,
)
from paper_search.recall_experiments.inputs.gold_catalog import (
    SealedGoldDocumentCatalog,
    SealedGoldDocumentRecord,
)
from paper_search.recall_experiments.inputs.base import (
    FrozenRecallDataset,
    OpaqueEvaluationMaterials,
)


def _dataset(*, seed_ids: list[str] | None = None) -> FrozenRecallDataset:
    queries = [
        EvaluationQuery(
            query_id="q-one",
            query="first query",
            relevant_paper_ids=["doi:10.1000/one", "arxiv:2401.00002"],
        ),
        EvaluationQuery(
            query_id="q-two",
            query="second query",
            relevant_paper_ids=["doi:10.1000/two"],
        ),
    ]
    return FrozenRecallDataset(
        queries=queries,
        source_hashes={},
        evaluation_materials=OpaqueEvaluationMaterials(
            gold_records=queries,
            identifier_map_bytes=(
                b'{"arxiv:2401.00002":"doi:10.1000/other",'
                b'"openalex:W2":"doi:10.1000/two"}'
            ),
            identifier_map_sha256="sha256:" + "0" * 64,
        ),
        seed_candidates=[
            SeedCandidate(paper=Paper(canonical_id=identifier, title="Seed", sources=["openalex"]))
            for identifier in (seed_ids or ["doi:10.1000/seed"])
        ],
    )


def _pool(query_id: str, *identifiers: str, policy: str = "production-dedup-v1"):
    return CandidatePoolBuilder(policy).build(
        query_id,
        [
            RetrievalActionResult(
                action_id="action-1",
                action_type="text_search",
                hits=[Paper(canonical_id=value, title=value, sources=["openalex"]) for value in identifiers],
            )
        ],
    )


def test_evaluate_counts_unique_resolved_gold_associations_and_recall_only() -> None:
    result = CandidateRecallEvaluator().evaluate(
        _dataset(),
        [
            _pool("q-one", "arxiv:2401.00002", "doi:10.1000/one"),
            _pool("q-two", "openalex:W2", "doi:10.1000/miss"),
        ],
    )

    assert [(row.gold_association_count, row.gold_hit_count) for row in result.per_query] == [
        (2, 2),
        (1, 1),
    ]
    assert result.gold_association_count == 3
    assert result.gold_hit_count == 3
    assert result.macro_candidate_recall == 1.0
    public_fields = set(type(result).model_fields)
    assert not {"precision", "top_k", "ranking", "f1", "mrr", "ndcg"}.intersection(public_fields)


def test_preflight_rejects_resolved_duplicates_seed_gold_and_denominator_mismatch() -> None:
    duplicate = _dataset()
    duplicate.evaluation_materials.gold_records[1].relevant_paper_ids.append("openalex:W2")
    with pytest.raises(ValueError, match="duplicate resolved Gold association"):
        CandidateRecallEvaluator().preflight(duplicate)

    with pytest.raises(ValueError, match="seed candidate resolves to a Gold ID"):
        CandidateRecallEvaluator().preflight(_dataset(seed_ids=["openalex:W2"]))

    mismatched = _dataset()
    mismatched.evaluation_materials.gold_records.pop()
    with pytest.raises(ValueError, match="denominator"):
        CandidateRecallEvaluator().preflight(mismatched)

    mismatched_associations = _dataset().model_copy(
        update={
            "queries": [
                EvaluationQuery(
                    query_id="q-one",
                    query="first query",
                    relevant_paper_ids=["doi:10.1000/one"],
                ),
                _dataset().queries[1],
            ]
        }
    )
    with pytest.raises(ValueError, match="denominator"):
        CandidateRecallEvaluator().preflight(mismatched_associations)


def test_preflight_is_the_only_identifier_map_parser_and_returns_generation_safe_contexts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import paper_search.evaluation.dataset as dataset_module

    calls = 0
    original = dataset_module.IdentifierMap.from_bytes

    def spy(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        return original(*args[1:], **kwargs)

    monkeypatch.setattr(dataset_module.IdentifierMap, "from_bytes", classmethod(spy))
    prepared = CandidateRecallEvaluator().preflight(_dataset())

    assert calls == 1
    assert [context.query_id for context in prepared.generation_contexts] == ["q-one", "q-two"]
    assert all(not hasattr(context, "identifier_map") for context in prepared.generation_contexts)


def test_exact_replay_requires_policy_and_per_query_identity() -> None:
    evaluator = CandidateRecallEvaluator()
    current = evaluator.evaluate(
        _dataset(), [_pool("q-one", "doi:10.1000/one"), _pool("q-two", "doi:10.1000/two")]
    )
    historical = HistoricalReplayEvidence.from_repeat(current)

    assert compare_exact_replay(current, historical).conclusion == "passed"
    assert compare_exact_replay(
        current,
        historical.model_copy(update={"candidate_pool_policy_version": "canonical-id-first-v1"}),
    ).conclusion == "not_comparable"
    aggregate = historical.model_copy(update={"per_query": None})
    aggregate_comparison = compare_exact_replay(current, aggregate)
    assert aggregate_comparison.conclusion == "passed"
    assert aggregate_comparison.per_query_comparison == "not_provable"

    different_hits = historical.model_copy(
        update={
            "per_query": [
                historical.per_query[0].model_copy(update={"gold_hit_ids": ["doi:10.1000/other"]}),
                historical.per_query[1],
            ]
        }
    )
    assert compare_exact_replay(current, different_hits).conclusion == "failed"


def test_regenerated_comparison_keeps_attempts_and_requires_three_valid_repeats() -> None:
    evaluator = CandidateRecallEvaluator()
    passing = evaluator.evaluate(
        _dataset(), [_pool("q-one", "doi:10.1000/one"), _pool("q-two", "doi:10.1000/two")]
    )
    historical = HistoricalReplayEvidence.from_repeat(passing)
    attempts = [
        RecallAttempt(attempt_id="attempt-01", infrastructure_failure=True),
        RecallAttempt(attempt_id="attempt-02", valid_repeat_ordinal=1, result=passing),
        RecallAttempt(attempt_id="attempt-03", valid_repeat_ordinal=2, result=passing),
        RecallAttempt(attempt_id="attempt-04", valid_repeat_ordinal=3, result=passing),
    ]

    comparison = compare_regenerated(attempts, historical, "production-dedup-v1")
    assert comparison.conclusion == "passed"
    assert comparison.valid_repeat_count == 3
    assert comparison.passing_repeat_count == 3
    assert comparison.hit_count_summary == {"min": 2, "median": 2, "max": 2}
    assert comparison.macro_candidate_recall_summary == {"min": 0.75, "median": 0.75, "max": 0.75}

    insufficient = compare_regenerated(
        [RecallAttempt(attempt_id=f"attempt-{ordinal:02d}", infrastructure_failure=True) for ordinal in range(1, 6)],
        historical,
        "production-dedup-v1",
    )
    assert insufficient.conclusion == "insufficient_valid_repeats"
    assert len(insufficient.attempts) == 5


def test_regenerated_comparison_returns_not_comparable_before_tolerance_math() -> None:
    result = CandidateRecallEvaluator().evaluate(
        _dataset(), [_pool("q-one", "doi:10.1000/one"), _pool("q-two", "doi:10.1000/two")]
    )
    historical = HistoricalReplayEvidence.from_repeat(result).model_copy(
        update={"candidate_pool_policy_version": "canonical-id-first-v1"}
    )
    comparison = compare_regenerated(
        [RecallAttempt(attempt_id="attempt-01", valid_repeat_ordinal=1, result=result)],
        historical,
        "production-dedup-v1",
    )
    assert comparison.conclusion == "not_comparable"
    assert comparison.valid_repeat_count == 1


def test_evaluate_rejects_legacy_pool_when_recipe_locks_production_policy() -> None:
    from paper_search.recall_experiments.recipes import RecallMethodRecipe

    recipe = RecallMethodRecipe.model_validate(
        {
            "method_id": "production-policy",
            "generator": {
                "type": "manual_actions",
                "actions": "actions.json",
                "gold_visibility": "blind",
            },
            "retrieval": {
                "allowed_actions": ["text_search"],
                "backend": "snapshot_replay",
                "max_results_per_action": 1,
                "max_total_actions": 1,
            },
            "candidate_pool": {"policy_version": "production-dedup-v1"},
            "evaluation": {"repeat_count": 1, "max_repeat_attempts": 1},
        }
    )

    with pytest.raises(ValueError, match="recipe candidate pool policy"):
        CandidateRecallEvaluator(recipe).evaluate(
            _dataset(),
            [
                _pool("q-one", "doi:10.1000/one", policy="canonical-id-first-v1"),
                _pool("q-two", "doi:10.1000/two", policy="canonical-id-first-v1"),
            ],
        )


@pytest.mark.parametrize(
    "attempt_ids",
    [
        ["attempt-05"],
        ["attempt-05", "attempt-03", "attempt-01"],
        ["attempt-01", "attempt-03"],
        ["attempt-02", "attempt-01"],
        ["attempt-01", "attempt-01"],
    ],
)
def test_regenerated_attempts_must_be_an_ordered_contiguous_prefix(
    attempt_ids: list[str],
) -> None:
    attempts = [RecallAttempt(attempt_id=attempt_id, infrastructure_failure=True) for attempt_id in attempt_ids]

    with pytest.raises(ValueError, match="ordered contiguous prefix"):
        compare_regenerated(attempts, None, "production-dedup-v1")


def test_regenerated_retention_uses_gold_associations_not_candidate_overlap() -> None:
    evaluator = CandidateRecallEvaluator()
    historical_result = evaluator.evaluate(
        _dataset(),
        [
            _pool("q-one", "doi:10.1000/one", "doi:10.1000/ignore"),
            _pool("q-two", "doi:10.1000/two"),
        ],
    )
    current = evaluator.evaluate(
        _dataset(),
        [
            _pool("q-one", "doi:10.1000/other", "doi:10.1000/ignore"),
            _pool("q-two", "doi:10.1000/two"),
        ],
    )
    comparison = compare_regenerated(
        [
            RecallAttempt(attempt_id=f"attempt-{ordinal:02d}", valid_repeat_ordinal=ordinal, result=current)
            for ordinal in range(1, 4)
        ],
        HistoricalReplayEvidence.from_repeat(historical_result),
        "production-dedup-v1",
    )

    assert comparison.conclusion == "failed"
    assert [attempt.historical_gold_retention for attempt in comparison.attempts] == [0.5, 0.5, 0.5]


def test_regenerated_gold_tolerance_compares_hits_not_denominator() -> None:
    evaluator = CandidateRecallEvaluator()
    historical = evaluator.evaluate(
        _dataset(),
        [_pool("q-one", "doi:10.1000/one"), _pool("q-two", "doi:10.1000/two")],
    )
    current = historical.model_copy(update={"gold_hit_count": historical.gold_hit_count + 2})

    comparison = compare_regenerated(
        [
            RecallAttempt(
                attempt_id=f"attempt-{ordinal:02d}",
                valid_repeat_ordinal=ordinal,
                result=current,
            )
            for ordinal in range(1, 4)
        ],
        HistoricalReplayEvidence.from_repeat(historical),
        "production-dedup-v1",
    )

    assert comparison.conclusion == "failed"
    assert comparison.passing_repeat_count == 0


def test_oracle_preflight_rejects_blind_overlap_and_missing_catalog_titles() -> None:
    recipe = {
        "method_id": "oracle-method",
        "generator": {"type": "manual_actions", "actions": "actions.json", "gold_visibility": "oracle"},
        "retrieval": {
            "allowed_actions": ["text_search"],
            "backend": "snapshot_replay",
            "max_results_per_action": 1,
            "max_total_actions": 1,
        },
        "evaluation": {"repeat_count": 1, "max_repeat_attempts": 1},
    }
    from paper_search.recall_experiments.recipes import RecallMethodRecipe, SampleBinding

    oracle_sample = SampleBinding(sample_id="oracle", query_ids=["q-one", "q-two"])
    blind_sample = SampleBinding(sample_id="blind", query_ids=["q-two"])
    with pytest.raises(ValueError, match="Oracle and Blind"):
        CandidateRecallEvaluator(
            RecallMethodRecipe.model_validate(recipe), sample=oracle_sample, blind_sample=blind_sample
        ).preflight(_dataset())

    catalog = SealedGoldDocumentCatalog(
        records=[
            SealedGoldDocumentRecord(
                query_id="q-one", gold_paper_id="doi:10.1000/one", title="First title"
            )
        ],
        source_hashes={},
        source_manifest=[],
        source_manifest_sha256="sha256:" + "0" * 64,
        catalog_sha256="sha256:" + "0" * 64,
        status="complete",
    )
    with pytest.raises(ValueError, match="lacks a title"):
        CandidateRecallEvaluator(
            RecallMethodRecipe.model_validate(recipe), sample=oracle_sample, gold_catalog=catalog
        ).preflight(_dataset())
