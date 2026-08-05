"""Standalone ranked-retrieval metrics (MRR / NDCG) outside the formal contract.

The formal ``EvaluationResult`` contract stays unchanged so historical frozen
runs remain verifiable; ranking metrics are computed offline into their own
artifact by ``paper-search ranking-metrics``.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Sequence
from pathlib import Path

from pydantic import Field

from paper_search.domain.models import DomainModel, NonEmptyStr, NonNegativeInt
from paper_search.evaluation.dataset import (
    EvaluationQuery,
    IdentifierMap,
    PredictionRecord,
    read_jsonl,
    sha256_file,
)
from paper_search.evaluation.metrics import (
    _IdentifierMapInputError,
    _load_identifier_map_snapshot,
    _write_json_atomic,
    deduplicate_ranked,
)
from paper_search.evaluation.official_adapter import (
    InternalPredictionRecord,
    adapt_prediction_record,
)


CONTRACT_VERSION = "task2-ranking-evaluation-v1"


class RankingMetrics(DomainModel):
    """Ranked-retrieval metrics for one evaluation query."""

    gold_ids: list[NonEmptyStr]
    predicted_ids: list[NonEmptyStr]
    mrr: float = Field(ge=0, le=1, allow_inf_nan=False)
    ndcg: float = Field(ge=0, le=1, allow_inf_nan=False)


class RankingSummary(DomainModel):
    """Macro ranked-retrieval metrics across an evaluation collection."""

    query_count: NonNegativeInt
    missing_prediction_count: NonNegativeInt
    macro_mrr: float = Field(ge=0, le=1, allow_inf_nan=False)
    macro_ndcg: float = Field(ge=0, le=1, allow_inf_nan=False)


class RankingEvaluationResult(DomainModel):
    """Aggregate summary plus ranking metrics keyed by query identifier."""

    summary: RankingSummary
    per_query: dict[str, RankingMetrics]


def score_ranking(gold: Sequence[str], predicted: Sequence[str]) -> RankingMetrics:
    """Score one gold set against a ranked prediction sequence with MRR/NDCG."""
    gold_ids = deduplicate_ranked(gold)
    predicted_ids = deduplicate_ranked(predicted)
    gold_set = set(gold_ids)

    if not gold_ids and not predicted_ids:
        mrr = ndcg = 1.0
    elif not gold_ids or not predicted_ids:
        mrr = ndcg = 0.0
    else:
        first_hit_rank = next(
            (
                index
                for index, identifier in enumerate(predicted_ids, start=1)
                if identifier in gold_set
            ),
            None,
        )
        mrr = 1.0 / first_hit_rank if first_hit_rank is not None else 0.0
        dcg = sum(
            1.0 / math.log2(index + 1)
            for index, identifier in enumerate(predicted_ids, start=1)
            if identifier in gold_set
        )
        idcg = sum(
            1.0 / math.log2(index + 1) for index in range(1, len(gold_ids) + 1)
        )
        ndcg = dcg / idcg if idcg else 0.0

    return RankingMetrics(
        gold_ids=gold_ids,
        predicted_ids=predicted_ids,
        mrr=mrr,
        ndcg=ndcg,
    )


def evaluate_ranking(
    gold: Sequence[EvaluationQuery],
    predictions: Sequence[PredictionRecord],
    *,
    id_map: IdentifierMap | None = None,
) -> RankingEvaluationResult:
    """Score every gold query and aggregate macro MRR/NDCG."""
    gold_by_id: dict[str, EvaluationQuery] = {}
    for gold_record_input in gold:
        if gold_record_input.query_id in gold_by_id:
            raise ValueError(f"duplicate gold query_id: {gold_record_input.query_id}")
        gold_by_id[gold_record_input.query_id] = gold_record_input

    predictions_by_id: dict[str, PredictionRecord] = {}
    for prediction_record_input in predictions:
        if prediction_record_input.query_id in predictions_by_id:
            raise ValueError(
                f"duplicate predictions query_id: {prediction_record_input.query_id}"
            )
        predictions_by_id[prediction_record_input.query_id] = prediction_record_input

    for query_id in predictions_by_id:
        if query_id not in gold_by_id:
            raise ValueError(f"unknown prediction query_id: {query_id}")

    resolve = id_map.resolve if id_map is not None else lambda value: value
    per_query: dict[str, RankingMetrics] = {}
    missing_prediction_count = 0
    for query_id, gold_record in gold_by_id.items():
        prediction_record = predictions_by_id.get(query_id)
        if prediction_record is None:
            missing_prediction_count += 1
            predicted_ids: Sequence[str] = ()
        else:
            predicted_ids = prediction_record.predicted_paper_ids

        per_query[query_id] = score_ranking(
            [resolve(identifier) for identifier in gold_record.relevant_paper_ids],
            [resolve(identifier) for identifier in predicted_ids],
        )

    query_count = len(per_query)

    def macro(values: Sequence[float]) -> float:
        if not values:
            return 1.0
        return sum(values) / len(values)

    summary = RankingSummary(
        query_count=query_count,
        missing_prediction_count=missing_prediction_count,
        macro_mrr=macro([metrics.mrr for metrics in per_query.values()]),
        macro_ndcg=macro([metrics.ndcg for metrics in per_query.values()]),
    )
    return RankingEvaluationResult(summary=summary, per_query=per_query)


def run_cli(
    gold_path: Path,
    pred_path: Path,
    out_path: Path,
    id_map_path: Path | None,
) -> int:
    try:
        gold = read_jsonl(gold_path, EvaluationQuery)
        external_predictions = read_jsonl(pred_path, InternalPredictionRecord)
        predictions = [adapt_prediction_record(record) for record in external_predictions]
        identifier_map: IdentifierMap | None = None
        identifier_map_hash: str | None = None
        if id_map_path is not None:
            identifier_map, identifier_map_hash = _load_identifier_map_snapshot(
                id_map_path
            )
        result = evaluate_ranking(gold, predictions, id_map=identifier_map)

        input_hashes = {
            "gold": sha256_file(gold_path),
            "predictions": sha256_file(pred_path),
        }
        if identifier_map_hash is not None:
            input_hashes["id_map"] = identifier_map_hash

        payload = {
            "contract_version": CONTRACT_VERSION,
            "input_hashes": input_hashes,
            "summary": result.summary.model_dump(mode="json"),
            "per_query": {
                query_id: result.per_query[query_id].model_dump(mode="json")
                for query_id in sorted(result.per_query)
            },
        }
        _write_json_atomic(out_path, payload)
    except _IdentifierMapInputError as error:
        print(f"ranking evaluation failed: {error}", file=sys.stderr)
        return 2
    except (OSError, ValueError) as error:
        print(f"ranking evaluation failed: {error}", file=sys.stderr)
        return 2
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute standalone ranked-retrieval metrics (MRR/NDCG)"
    )
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--id-map", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return run_cli(args.gold, args.pred, args.out, args.id_map)


if __name__ == "__main__":
    raise SystemExit(main())
