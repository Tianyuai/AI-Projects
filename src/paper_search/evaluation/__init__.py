"""Offline dataset adaptation and evaluation utilities."""

from typing import TYPE_CHECKING

from paper_search.evaluation.annotation import (
    AgreementReport,
    AnnotationRecord,
    FieldAgreement,
    cohen_kappa,
    compare_annotations,
)
from paper_search.evaluation.dataset import (
    EvaluationQuery,
    IdentifierMap,
    PredictionRecord,
    answer_count_bucket,
    normalize_paper_id,
    normalize_title,
    read_jsonl,
    sha256_file,
    stratified_sample,
    write_frozen_bytes,
    write_jsonl_atomic,
)
from paper_search.evaluation.official_adapter import (
    InternalPredictionRecord,
    PaSaRecord,
    adapt_pasa_record,
    adapt_prediction_record,
)


if TYPE_CHECKING:
    from paper_search.evaluation.metrics import (
        EvaluationResult,
        MetricSummary,
        QueryMetrics,
        deduplicate_ranked,
        evaluate,
        score_query,
    )


_METRIC_EXPORTS = frozenset(
    {
        "EvaluationResult",
        "MetricSummary",
        "QueryMetrics",
        "deduplicate_ranked",
        "evaluate",
        "score_query",
    }
)


def __getattr__(name: str) -> object:
    """Load metric exports lazily so ``python -m ...metrics`` stays warning-free."""
    if name not in _METRIC_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from paper_search.evaluation.metrics import (
        EvaluationResult,
        MetricSummary,
        QueryMetrics,
        deduplicate_ranked,
        evaluate,
        score_query,
    )

    exports: dict[str, object] = {
        "EvaluationResult": EvaluationResult,
        "MetricSummary": MetricSummary,
        "QueryMetrics": QueryMetrics,
        "deduplicate_ranked": deduplicate_ranked,
        "evaluate": evaluate,
        "score_query": score_query,
    }
    value = exports[name]
    globals()[name] = value
    return value


__all__ = [
    "AgreementReport",
    "AnnotationRecord",
    "EvaluationQuery",
    "EvaluationResult",
    "FieldAgreement",
    "IdentifierMap",
    "InternalPredictionRecord",
    "MetricSummary",
    "PaSaRecord",
    "PredictionRecord",
    "QueryMetrics",
    "adapt_pasa_record",
    "adapt_prediction_record",
    "answer_count_bucket",
    "cohen_kappa",
    "compare_annotations",
    "deduplicate_ranked",
    "evaluate",
    "normalize_paper_id",
    "normalize_title",
    "read_jsonl",
    "score_query",
    "sha256_file",
    "stratified_sample",
    "write_frozen_bytes",
    "write_jsonl_atomic",
]
