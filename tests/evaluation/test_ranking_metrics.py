from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from paper_search.evaluation.dataset import EvaluationQuery, IdentifierMap, PredictionRecord
from paper_search.evaluation.ranking_metrics import (
    RankingEvaluationResult,
    RankingSummary,
    evaluate_ranking,
    main,
    score_ranking,
)


@pytest.mark.parametrize(
    ("gold", "predicted", "expected"),
    [
        ([], [], 1.0),
        (["doi:10.1000/a"], [], 0.0),
        ([], ["doi:10.1000/a"], 0.0),
    ],
)
def test_ranking_empty_set_contract(
    gold: list[str],
    predicted: list[str],
    expected: float,
) -> None:
    result = score_ranking(gold, predicted)

    assert result.mrr == expected
    assert result.ndcg == expected


def test_mrr_is_reciprocal_rank_of_first_hit() -> None:
    result = score_ranking(
        ["openalex:W2", "openalex:W7"],
        ["openalex:W1", "openalex:W2", "openalex:W7"],
    )

    assert result.mrr == pytest.approx(1 / 2)


def test_mrr_zero_when_no_hit() -> None:
    result = score_ranking(
        ["openalex:W1", "openalex:W2"],
        ["openalex:W3", "openalex:W4"],
    )

    assert result.mrr == 0.0


def test_ndcg_scores_ranked_binary_relevance() -> None:
    result = score_ranking(
        ["openalex:W1", "openalex:W2", "openalex:W3"],
        ["openalex:W1", "openalex:W3", "openalex:W4", "openalex:W2"],
    )

    assert result.ndcg == pytest.approx(0.96746, abs=1e-5)


def test_ndcg_is_one_for_perfect_ordering() -> None:
    result = score_ranking(
        ["openalex:W1", "openalex:W2"],
        ["openalex:W1", "openalex:W2"],
    )

    assert result.ndcg == pytest.approx(1.0)


def test_score_ranking_deduplicates_ranked_predictions() -> None:
    result = score_ranking(
        ["openalex:W1", "openalex:W2"],
        ["openalex:W1", "openalex:W1", "openalex:W2"],
    )

    assert result.predicted_ids == ["openalex:W1", "openalex:W2"]
    assert result.mrr == pytest.approx(1.0)
    assert result.ndcg == pytest.approx(1.0)


def test_evaluate_ranking_reports_macro_averages_and_missing_predictions() -> None:
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
        PredictionRecord(query_id="q1", predicted_paper_ids=["openalex:W1"]),
    ]

    result = evaluate_ranking(gold, predictions)

    assert list(result.per_query) == ["q1", "q2"]
    assert result.summary.query_count == 2
    assert result.summary.missing_prediction_count == 1
    assert result.per_query["q1"].mrr == 1.0
    assert result.per_query["q1"].ndcg == pytest.approx(1.0)
    assert result.per_query["q2"].mrr == 0.0
    assert result.per_query["q2"].ndcg == 0.0
    assert result.summary.macro_mrr == pytest.approx(0.5)
    assert result.summary.macro_ndcg == pytest.approx(0.5)


def test_evaluate_ranking_empty_input_has_explicit_perfect_empty_contract() -> None:
    result = evaluate_ranking([], [])

    assert result.per_query == {}
    assert result.summary.query_count == 0
    assert result.summary.missing_prediction_count == 0
    assert result.summary.macro_mrr == 1.0
    assert result.summary.macro_ndcg == 1.0


def test_evaluate_ranking_rejects_unknown_prediction_query() -> None:
    gold = [EvaluationQuery(query_id="q1", query="one")]
    predictions = [PredictionRecord(query_id="unknown")]

    with pytest.raises(ValueError, match="unknown prediction query_id: unknown"):
        evaluate_ranking(gold, predictions)


@pytest.mark.parametrize("duplicate_side", ["gold", "predictions"])
def test_evaluate_ranking_rejects_duplicate_query_ids(duplicate_side: str) -> None:
    gold = [EvaluationQuery(query_id="q1", query="one")]
    predictions = [PredictionRecord(query_id="q1")]
    if duplicate_side == "gold":
        gold.append(EvaluationQuery(query_id="q1", query="duplicate"))
    else:
        predictions.append(PredictionRecord(query_id="q1"))

    with pytest.raises(ValueError, match=rf"duplicate {duplicate_side} query_id: q1"):
        evaluate_ranking(gold, predictions)


def test_evaluate_ranking_applies_identifier_map_before_scoring(
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

    result = evaluate_ranking(gold, predictions, id_map=identifier_map)

    assert result.per_query["q1"].gold_ids == ["openalex:W1"]
    assert result.per_query["q1"].predicted_ids == ["openalex:W1"]
    assert result.per_query["q1"].mrr == 1.0


def test_ranking_metric_models_are_frozen() -> None:
    metrics = score_ranking([], [])
    summary = RankingSummary(
        query_count=0,
        missing_prediction_count=0,
        macro_mrr=1.0,
        macro_ndcg=1.0,
    )
    result = RankingEvaluationResult(summary=summary, per_query={})

    with pytest.raises(ValidationError):
        metrics.mrr = 0.0
    with pytest.raises(ValidationError):
        summary.query_count = 1
    with pytest.raises(ValidationError):
        result.per_query = {"q1": metrics}


def test_cli_writes_stable_standalone_ranking_json(tmp_path: Path) -> None:
    gold_path = tmp_path / "gold.jsonl"
    pred_path = tmp_path / "predictions.jsonl"
    out_path = tmp_path / "ranking-metrics.json"
    gold_path.write_text(
        json.dumps(
            {
                "query_id": "q1",
                "query": "one",
                "relevant_paper_ids": ["openalex:W1"],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    pred_path.write_text(
        json.dumps(
            {"query_id": "q1", "selected_paper_ids": ["openalex:W1"]}
        )
        + "\n",
        encoding="utf-8",
    )

    assert (
        main(
            [
                "--gold",
                str(gold_path),
                "--pred",
                str(pred_path),
                "--out",
                str(out_path),
            ]
        )
        == 0
    )

    payload = json.loads(out_path.read_bytes())
    assert payload["contract_version"] == "task2-ranking-evaluation-v1"
    assert payload["summary"]["macro_mrr"] == 1.0
    assert payload["summary"]["macro_ndcg"] == 1.0
    assert payload["per_query"]["q1"]["mrr"] == 1.0
    assert payload["per_query"]["q1"]["ndcg"] == pytest.approx(1.0)
