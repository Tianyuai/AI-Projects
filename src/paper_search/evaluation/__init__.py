"""Offline dataset adaptation and evaluation utilities."""

from typing import TYPE_CHECKING

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
from paper_search.evaluation.predictions import (
    prediction_from_response,
    write_response_predictions,
)


if TYPE_CHECKING:
    from paper_search.evaluation.annotation import (
        AgreementReport,
        AnnotationRecord,
        AnnotationValidationSummary,
        FieldAgreement,
        TypeDomainAnnotationRecord,
        cohen_kappa,
        compare_annotations,
        validate_annotation_file,
    )
    from paper_search.evaluation.metrics import (
        EvaluationResult,
        MetricSummary,
        QueryMetrics,
        deduplicate_ranked,
        evaluate,
        score_query,
    )


_ANNOTATION_EXPORTS = frozenset(
    {
        "AgreementReport",
        "AnnotationRecord",
        "AnnotationValidationSummary",
        "FieldAgreement",
        "TypeDomainAnnotationRecord",
        "cohen_kappa",
        "compare_annotations",
        "validate_annotation_file",
    }
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
    """Load executable-submodule exports lazily so ``python -m`` stays warning-free."""
    if name in _ANNOTATION_EXPORTS:
        from paper_search.evaluation.annotation import (
            AgreementReport,
            AnnotationRecord,
            AnnotationValidationSummary,
            FieldAgreement,
            TypeDomainAnnotationRecord,
            cohen_kappa,
            compare_annotations,
            validate_annotation_file,
        )

        exports: dict[str, object] = {
            "AgreementReport": AgreementReport,
            "AnnotationRecord": AnnotationRecord,
            "AnnotationValidationSummary": AnnotationValidationSummary,
            "FieldAgreement": FieldAgreement,
            "TypeDomainAnnotationRecord": TypeDomainAnnotationRecord,
            "cohen_kappa": cohen_kappa,
            "compare_annotations": compare_annotations,
            "validate_annotation_file": validate_annotation_file,
        }
    elif name in _METRIC_EXPORTS:
        from paper_search.evaluation.metrics import (
            EvaluationResult,
            MetricSummary,
            QueryMetrics,
            deduplicate_ranked,
            evaluate,
            score_query,
        )

        exports = {
            "EvaluationResult": EvaluationResult,
            "MetricSummary": MetricSummary,
            "QueryMetrics": QueryMetrics,
            "deduplicate_ranked": deduplicate_ranked,
            "evaluate": evaluate,
            "score_query": score_query,
        }
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    value = exports[name]
    globals()[name] = value
    return value


__all__ = [
    "AgreementReport",
    "AnnotationRecord",
    "AnnotationValidationSummary",
    "EvaluationQuery",
    "EvaluationResult",
    "FieldAgreement",
    "IdentifierMap",
    "InternalPredictionRecord",
    "MetricSummary",
    "PaSaRecord",
    "PredictionRecord",
    "QueryMetrics",
    "TypeDomainAnnotationRecord",
    "adapt_pasa_record",
    "adapt_prediction_record",
    "answer_count_bucket",
    "cohen_kappa",
    "compare_annotations",
    "deduplicate_ranked",
    "evaluate",
    "normalize_paper_id",
    "normalize_title",
    "prediction_from_response",
    "read_jsonl",
    "score_query",
    "sha256_file",
    "stratified_sample",
    "validate_annotation_file",
    "write_frozen_bytes",
    "write_jsonl_atomic",
    "write_response_predictions",
]
