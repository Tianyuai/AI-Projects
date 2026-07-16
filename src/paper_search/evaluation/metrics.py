from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import Field

from paper_search.domain.models import DomainModel, NonEmptyStr, NonNegativeInt
from paper_search.evaluation.dataset import (
    EvaluationQuery,
    IdentifierMap,
    PredictionRecord,
    read_jsonl,
    sha256_file,
)
from paper_search.evaluation.official_adapter import (
    InternalPredictionRecord,
    adapt_prediction_record,
)


CONTRACT_VERSION = "task2-evaluation-v1"


class QueryMetrics(DomainModel):
    """Set and ranked-retrieval metrics for one evaluation query."""

    gold_ids: list[NonEmptyStr]
    predicted_ids: list[NonEmptyStr]
    hit_ids: list[NonEmptyStr]
    true_positive_count: NonNegativeInt
    false_positive_count: NonNegativeInt
    false_negative_count: NonNegativeInt
    precision: float = Field(ge=0, le=1, allow_inf_nan=False)
    recall: float = Field(ge=0, le=1, allow_inf_nan=False)
    f1: float = Field(ge=0, le=1, allow_inf_nan=False)
    recall_at_5: float = Field(ge=0, le=1, allow_inf_nan=False)
    recall_at_10: float = Field(ge=0, le=1, allow_inf_nan=False)
    recall_at_20: float = Field(ge=0, le=1, allow_inf_nan=False)


class MetricSummary(DomainModel):
    """Macro and micro metrics across an evaluation collection."""

    query_count: NonNegativeInt
    missing_prediction_count: NonNegativeInt
    macro_precision: float = Field(ge=0, le=1, allow_inf_nan=False)
    macro_recall: float = Field(ge=0, le=1, allow_inf_nan=False)
    macro_f1: float = Field(ge=0, le=1, allow_inf_nan=False)
    macro_recall_at_5: float = Field(ge=0, le=1, allow_inf_nan=False)
    macro_recall_at_10: float = Field(ge=0, le=1, allow_inf_nan=False)
    macro_recall_at_20: float = Field(ge=0, le=1, allow_inf_nan=False)
    micro_precision: float = Field(ge=0, le=1, allow_inf_nan=False)
    micro_recall: float = Field(ge=0, le=1, allow_inf_nan=False)
    micro_f1: float = Field(ge=0, le=1, allow_inf_nan=False)


class EvaluationResult(DomainModel):
    """Aggregate summary plus metrics keyed by gold query identifier."""

    summary: MetricSummary
    per_query: dict[str, QueryMetrics]


def deduplicate_ranked(values: Sequence[str]) -> list[str]:
    """Keep each identifier's first ranked occurrence."""
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def score_query(gold: Sequence[str], predicted: Sequence[str]) -> QueryMetrics:
    """Score one normalized gold set against a ranked prediction sequence."""
    gold_ids = deduplicate_ranked(gold)
    predicted_ids = deduplicate_ranked(predicted)
    gold_set = set(gold_ids)
    predicted_set = set(predicted_ids)
    hit_ids = [identifier for identifier in predicted_ids if identifier in gold_set]

    true_positive_count = len(hit_ids)
    false_positive_count = len(predicted_set - gold_set)
    false_negative_count = len(gold_set - predicted_set)

    if not gold_ids and not predicted_ids:
        precision = recall = f1 = 1.0
    elif not gold_ids or not predicted_ids:
        precision = recall = f1 = 0.0
    else:
        precision = true_positive_count / len(predicted_ids)
        recall = true_positive_count / len(gold_ids)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    def recall_at(cutoff: int) -> float:
        if not gold_ids:
            return 1.0 if not predicted_ids else 0.0
        if not predicted_ids:
            return 0.0
        hits = sum(identifier in gold_set for identifier in predicted_ids[:cutoff])
        return hits / len(gold_ids)

    return QueryMetrics(
        gold_ids=gold_ids,
        predicted_ids=predicted_ids,
        hit_ids=hit_ids,
        true_positive_count=true_positive_count,
        false_positive_count=false_positive_count,
        false_negative_count=false_negative_count,
        precision=precision,
        recall=recall,
        f1=f1,
        recall_at_5=recall_at(5),
        recall_at_10=recall_at(10),
        recall_at_20=recall_at(20),
    )


def _micro_metrics(
    true_positive_count: int,
    false_positive_count: int,
    false_negative_count: int,
) -> tuple[float, float, float]:
    if true_positive_count == false_positive_count == false_negative_count == 0:
        return 1.0, 1.0, 1.0

    precision_denominator = true_positive_count + false_positive_count
    recall_denominator = true_positive_count + false_negative_count
    precision = (
        true_positive_count / precision_denominator if precision_denominator else 0.0
    )
    recall = true_positive_count / recall_denominator if recall_denominator else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def evaluate(
    gold: Sequence[EvaluationQuery],
    predictions: Sequence[PredictionRecord],
    *,
    id_map: IdentifierMap | None = None,
) -> EvaluationResult:
    """Evaluate normalized records, rejecting ambiguous query alignment."""
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
    per_query: dict[str, QueryMetrics] = {}
    missing_prediction_count = 0
    for query_id, gold_record in gold_by_id.items():
        prediction_record = predictions_by_id.get(query_id)
        if prediction_record is None:
            missing_prediction_count += 1
            predicted_ids: Sequence[str] = ()
        else:
            predicted_ids = prediction_record.predicted_paper_ids

        per_query[query_id] = score_query(
            [resolve(identifier) for identifier in gold_record.relevant_paper_ids],
            [resolve(identifier) for identifier in predicted_ids],
        )

    query_count = len(per_query)

    def macro(values: Sequence[float]) -> float:
        if not values:
            return 1.0
        return sum(values) / len(values)

    true_positive_count = sum(
        metrics.true_positive_count for metrics in per_query.values()
    )
    false_positive_count = sum(
        metrics.false_positive_count for metrics in per_query.values()
    )
    false_negative_count = sum(
        metrics.false_negative_count for metrics in per_query.values()
    )
    micro_precision, micro_recall, micro_f1 = _micro_metrics(
        true_positive_count,
        false_positive_count,
        false_negative_count,
    )

    summary = MetricSummary(
        query_count=query_count,
        missing_prediction_count=missing_prediction_count,
        macro_precision=macro([metrics.precision for metrics in per_query.values()]),
        macro_recall=macro([metrics.recall for metrics in per_query.values()]),
        macro_f1=macro([metrics.f1 for metrics in per_query.values()]),
        macro_recall_at_5=macro(
            [metrics.recall_at_5 for metrics in per_query.values()]
        ),
        macro_recall_at_10=macro(
            [metrics.recall_at_10 for metrics in per_query.values()]
        ),
        macro_recall_at_20=macro(
            [metrics.recall_at_20 for metrics in per_query.values()]
        ),
        micro_precision=micro_precision,
        micro_recall=micro_recall,
        micro_f1=micro_f1,
    )
    return EvaluationResult(summary=summary, per_query=per_query)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(
                payload,
                temporary,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        temporary_path.replace(path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate ranked paper predictions")
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--id-map", type=Path)
    return parser


def _run_cli(args: argparse.Namespace) -> None:
    gold = read_jsonl(args.gold, EvaluationQuery)
    external_predictions = read_jsonl(args.pred, InternalPredictionRecord)
    predictions = [adapt_prediction_record(record) for record in external_predictions]
    identifier_map = IdentifierMap.from_path(args.id_map) if args.id_map else None
    result = evaluate(gold, predictions, id_map=identifier_map)

    input_hashes = {
        "gold": sha256_file(args.gold),
        "predictions": sha256_file(args.pred),
    }
    if args.id_map:
        input_hashes["id_map"] = sha256_file(args.id_map)

    payload: dict[str, Any] = {
        "contract_version": CONTRACT_VERSION,
        "input_hashes": input_hashes,
        "summary": result.summary.model_dump(mode="json"),
        "per_query": {
            query_id: result.per_query[query_id].model_dump(mode="json")
            for query_id in sorted(result.per_query)
        },
    }
    _write_json_atomic(args.out, payload)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        _run_cli(args)
    except (OSError, ValueError) as error:
        print(f"evaluation failed: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
