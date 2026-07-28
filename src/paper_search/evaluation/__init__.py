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
    write_prediction_records,
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
    from paper_search.evaluation.experiments import (
        ExperimentAggregate,
        ExperimentRecord,
        build_experiment_record,
        write_experiment_record,
    )
    from paper_search.evaluation.statistics import (
        BootstrapInterval,
        MacroF1Comparison,
        bootstrap_mean_interval,
        compare_macro_f1,
    )
    from paper_search.evaluation.synthetic_baseline import (
        SYNTHETIC_QUERIES,
        run_synthetic_baseline,
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
_EXPERIMENT_EXPORTS = frozenset(
    {
        "ExperimentAggregate",
        "ExperimentRecord",
        "build_experiment_record",
        "write_experiment_record",
    }
)
_STATISTICS_EXPORTS = frozenset(
    {
        "BootstrapInterval",
        "MacroF1Comparison",
        "bootstrap_mean_interval",
        "compare_macro_f1",
    }
)
_SYNTHETIC_BASELINE_EXPORTS = frozenset(
    {
        "SYNTHETIC_QUERIES",
        "run_synthetic_baseline",
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
    elif name in _EXPERIMENT_EXPORTS:
        from paper_search.evaluation.experiments import (
            ExperimentAggregate,
            ExperimentRecord,
            build_experiment_record,
            write_experiment_record,
        )

        exports = {
            "ExperimentAggregate": ExperimentAggregate,
            "ExperimentRecord": ExperimentRecord,
            "build_experiment_record": build_experiment_record,
            "write_experiment_record": write_experiment_record,
        }
    elif name in _STATISTICS_EXPORTS:
        from paper_search.evaluation.statistics import (
            BootstrapInterval,
            MacroF1Comparison,
            bootstrap_mean_interval,
            compare_macro_f1,
        )

        exports = {
            "BootstrapInterval": BootstrapInterval,
            "MacroF1Comparison": MacroF1Comparison,
            "bootstrap_mean_interval": bootstrap_mean_interval,
            "compare_macro_f1": compare_macro_f1,
        }
    elif name in _SYNTHETIC_BASELINE_EXPORTS:
        from paper_search.evaluation.synthetic_baseline import (
            SYNTHETIC_QUERIES,
            run_synthetic_baseline,
        )

        exports = {
            "SYNTHETIC_QUERIES": SYNTHETIC_QUERIES,
            "run_synthetic_baseline": run_synthetic_baseline,
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
    "BootstrapInterval",
    "EvaluationQuery",
    "EvaluationResult",
    "ExperimentAggregate",
    "ExperimentRecord",
    "FieldAgreement",
    "IdentifierMap",
    "InternalPredictionRecord",
    "MacroF1Comparison",
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
    "compare_macro_f1",
    "deduplicate_ranked",
    "evaluate",
    "bootstrap_mean_interval",
    "build_experiment_record",
    "normalize_paper_id",
    "normalize_title",
    "prediction_from_response",
    "read_jsonl",
    "score_query",
    "sha256_file",
    "stratified_sample",
    "SYNTHETIC_QUERIES",
    "run_synthetic_baseline",
    "validate_annotation_file",
    "write_frozen_bytes",
    "write_experiment_record",
    "write_jsonl_atomic",
    "write_prediction_records",
    "write_response_predictions",
]
