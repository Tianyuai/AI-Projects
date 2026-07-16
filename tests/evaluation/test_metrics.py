from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from paper_search.evaluation.dataset import EvaluationQuery, IdentifierMap, PredictionRecord
from paper_search.evaluation.metrics import (
    EvaluationResult,
    MetricSummary,
    deduplicate_ranked,
    evaluate,
    score_query,
)


@pytest.mark.parametrize(
    ("gold", "predicted", "expected"),
    [
        ([], [], 1.0),
        (["doi:10.1000/a"], [], 0.0),
        ([], ["doi:10.1000/a"], 0.0),
    ],
)
def test_empty_set_contract(
    gold: list[str],
    predicted: list[str],
    expected: float,
) -> None:
    result = score_query(gold, predicted)

    assert result.precision == expected
    assert result.recall == expected
    assert result.f1 == expected
    assert result.recall_at_5 == expected
    assert result.recall_at_10 == expected
    assert result.recall_at_20 == expected


def test_deduplicate_ranked_keeps_first_occurrence() -> None:
    values = ["openalex:W2", "openalex:W1", "openalex:W2", "openalex:W3"]

    assert deduplicate_ranked(values) == [
        "openalex:W2",
        "openalex:W1",
        "openalex:W3",
    ]


def test_recall_at_k_uses_first_ranked_occurrence_and_hits_keep_rank_order() -> None:
    result = score_query(
        ["openalex:W1", "openalex:W2", "openalex:W7"],
        [
            "openalex:W1",
            "openalex:W1",
            "openalex:W3",
            "openalex:W4",
            "openalex:W5",
            "openalex:W6",
            "openalex:W2",
            "openalex:W7",
        ],
    )

    assert result.predicted_ids == [
        "openalex:W1",
        "openalex:W3",
        "openalex:W4",
        "openalex:W5",
        "openalex:W6",
        "openalex:W2",
        "openalex:W7",
    ]
    assert result.hit_ids == ["openalex:W1", "openalex:W2", "openalex:W7"]
    assert result.recall_at_5 == pytest.approx(1 / 3)
    assert result.recall_at_10 == 1.0
    assert result.recall_at_20 == 1.0


def test_score_query_reports_counts_and_set_metrics() -> None:
    result = score_query(
        ["openalex:W1", "openalex:W2", "openalex:W3"],
        ["openalex:W3", "openalex:W4", "openalex:W1"],
    )

    assert result.true_positive_count == 2
    assert result.false_positive_count == 1
    assert result.false_negative_count == 1
    assert result.precision == pytest.approx(2 / 3)
    assert result.recall == pytest.approx(2 / 3)
    assert result.f1 == pytest.approx(2 / 3)


def test_evaluate_reports_macro_micro_and_missing_predictions() -> None:
    gold = [
        EvaluationQuery(
            query_id="q1",
            query="one",
            relevant_paper_ids=["openalex:W1"],
        ),
        EvaluationQuery(query_id="q2", query="two", relevant_paper_ids=[]),
    ]
    predictions = [
        PredictionRecord(query_id="q1", predicted_paper_ids=["openalex:W1"]),
    ]

    result = evaluate(gold, predictions)

    assert list(result.per_query) == ["q1", "q2"]
    assert result.summary.query_count == 2
    assert result.summary.missing_prediction_count == 1
    assert result.summary.macro_precision == 1.0
    assert result.summary.macro_recall == 1.0
    assert result.summary.macro_f1 == 1.0
    assert result.summary.macro_recall_at_5 == 1.0
    assert result.summary.macro_recall_at_10 == 1.0
    assert result.summary.macro_recall_at_20 == 1.0
    assert result.summary.micro_precision == 1.0
    assert result.summary.micro_recall == 1.0
    assert result.summary.micro_f1 == 1.0


def test_evaluate_micro_metrics_sum_counts_before_scoring() -> None:
    gold = [
        EvaluationQuery(
            query_id="q1",
            query="one",
            relevant_paper_ids=["openalex:W1"],
        ),
        EvaluationQuery(
            query_id="q2",
            query="two",
            relevant_paper_ids=["openalex:W2"],
        ),
    ]
    predictions = [
        PredictionRecord(
            query_id="q1",
            predicted_paper_ids=["openalex:W1", "openalex:W3"],
        ),
        PredictionRecord(query_id="q2", predicted_paper_ids=[]),
    ]

    result = evaluate(gold, predictions)

    assert result.summary.macro_f1 == pytest.approx(1 / 3)
    assert result.summary.micro_precision == pytest.approx(1 / 2)
    assert result.summary.micro_recall == pytest.approx(1 / 2)
    assert result.summary.micro_f1 == pytest.approx(1 / 2)


def test_evaluate_empty_input_has_explicit_perfect_empty_contract() -> None:
    result = evaluate([], [])

    assert result.per_query == {}
    assert result.summary.query_count == 0
    assert result.summary.missing_prediction_count == 0
    assert result.summary.macro_precision == 1.0
    assert result.summary.macro_recall == 1.0
    assert result.summary.macro_f1 == 1.0
    assert result.summary.macro_recall_at_5 == 1.0
    assert result.summary.macro_recall_at_10 == 1.0
    assert result.summary.macro_recall_at_20 == 1.0
    assert result.summary.micro_precision == 1.0
    assert result.summary.micro_recall == 1.0
    assert result.summary.micro_f1 == 1.0


def test_evaluate_rejects_unknown_prediction_query() -> None:
    gold = [EvaluationQuery(query_id="q1", query="one")]
    predictions = [PredictionRecord(query_id="unknown")]

    with pytest.raises(ValueError, match="unknown prediction query_id: unknown"):
        evaluate(gold, predictions)


@pytest.mark.parametrize("duplicate_side", ["gold", "predictions"])
def test_evaluate_rejects_duplicate_query_ids(duplicate_side: str) -> None:
    gold = [EvaluationQuery(query_id="q1", query="one")]
    predictions = [PredictionRecord(query_id="q1")]
    if duplicate_side == "gold":
        gold.append(EvaluationQuery(query_id="q1", query="duplicate"))
    else:
        predictions.append(PredictionRecord(query_id="q1"))

    with pytest.raises(ValueError, match=rf"duplicate {duplicate_side} query_id: q1"):
        evaluate(gold, predictions)


def test_evaluate_applies_identifier_map_before_comparison_and_deduplication(
    tmp_path: Path,
) -> None:
    path = tmp_path / "id-map.json"
    path.write_text(
        '{"doi:10.1000/a":"openalex:W1",'
        '"arxiv:2501.10120":"openalex:W1"}',
        encoding="utf-8",
    )
    identifier_map = IdentifierMap.from_path(path)
    gold = [
        EvaluationQuery(
            query_id="q1",
            query="mapped",
            relevant_paper_ids=["doi:10.1000/a"],
        )
    ]
    predictions = [
        PredictionRecord(
            query_id="q1",
            predicted_paper_ids=["arxiv:2501.10120", "doi:10.1000/a"],
        )
    ]

    result = evaluate(gold, predictions, id_map=identifier_map)

    assert result.per_query["q1"].gold_ids == ["openalex:W1"]
    assert result.per_query["q1"].predicted_ids == ["openalex:W1"]
    assert result.per_query["q1"].hit_ids == ["openalex:W1"]
    assert result.per_query["q1"].precision == 1.0


def test_metric_models_are_frozen() -> None:
    query_metrics = score_query([], [])
    summary = MetricSummary(
        query_count=0,
        missing_prediction_count=0,
        macro_precision=1.0,
        macro_recall=1.0,
        macro_f1=1.0,
        macro_recall_at_5=1.0,
        macro_recall_at_10=1.0,
        macro_recall_at_20=1.0,
        micro_precision=1.0,
        micro_recall=1.0,
        micro_f1=1.0,
    )
    result = EvaluationResult(summary=summary, per_query={})

    with pytest.raises(ValidationError):
        query_metrics.f1 = 0.0
    with pytest.raises(ValidationError):
        summary.query_count = 1
    with pytest.raises(ValidationError):
        result.per_query = {"q1": query_metrics}
