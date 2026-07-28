from pathlib import Path

import pytest

from paper_search.storage.experiment import ExperimentRecordStore


def test_record_is_aggregate_only_and_owner_only() -> None:
    from paper_search.evaluation import ExperimentAggregate, build_experiment_record

    record = build_experiment_record(
        run_id="synthetic-dev-001",
        config_hash="sha256:" + "a" * 64,
        git_sha="b" * 40,
        split="dev",
        phase="tuning",
        modules={"embedding": False},
        prompt_versions={"analysis": "query-analyze-v1"},
        model_metadata={"embedding": "fixture-embedding-v1"},
        aggregate=ExperimentAggregate(
            query_count=3,
            macro_f1=0.5,
            macro_recall=0.75,
            search_api_calls=4,
            llm_calls=1,
            input_tokens=10,
            output_tokens=2,
            cost_cny=0.12,
            latency_ms=45,
            failure_count=1,
        ),
        artifact_hashes={"predictions": "sha256:" + "c" * 64},
    )

    assert record.annotation_policy == "owner_only_provisional"
    assert record.aggregate.failure_rate == pytest.approx(1 / 3)
    assert "synthetic query" not in record.model_dump_json()


@pytest.mark.parametrize(
    ("split", "phase", "message"),
    [
        ("validation", "tuning", "selection_only"),
        ("validation", "selection_only", "selection_only"),
    ],
)
def test_validation_records_require_selection_only_phase(
    split: str,
    phase: str,
    message: str,
) -> None:
    from paper_search.evaluation import ExperimentAggregate, build_experiment_record

    aggregate = ExperimentAggregate(
        query_count=2,
        macro_f1=0.25,
        macro_recall=0.5,
        search_api_calls=1,
        llm_calls=0,
        input_tokens=0,
        output_tokens=0,
        cost_cny=0.0,
        latency_ms=10,
        failure_count=0,
    )

    if phase == "selection_only":
        build_experiment_record(
            run_id="validation-selection-001",
            config_hash="sha256:" + "a" * 64,
            git_sha="b" * 40,
            split=split,
            phase=phase,
            modules={"embedding": False},
            prompt_versions={},
            model_metadata={},
            aggregate=aggregate,
            artifact_hashes={"predictions": "sha256:" + "c" * 64},
        )
        return

    with pytest.raises(ValueError, match=message):
        build_experiment_record(
            run_id="validation-tuning-001",
            config_hash="sha256:" + "a" * 64,
            git_sha="b" * 40,
            split=split,
            phase=phase,
            modules={"embedding": False},
            prompt_versions={},
            model_metadata={},
            aggregate=aggregate,
            artifact_hashes={"predictions": "sha256:" + "c" * 64},
        )


def test_failure_count_cannot_exceed_query_count() -> None:
    from paper_search.evaluation import ExperimentAggregate

    with pytest.raises(ValueError, match="failure_count"):
        ExperimentAggregate(
            query_count=1,
            macro_f1=0.5,
            macro_recall=0.5,
            search_api_calls=1,
            llm_calls=0,
            input_tokens=0,
            output_tokens=0,
            cost_cny=0.0,
            latency_ms=1,
            failure_count=2,
        )


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("query_count", 0, "query_count"),
        ("macro_f1", 1.1, "macro_f1"),
        ("macro_recall", -0.1, "macro_recall"),
    ],
)
def test_aggregate_validates_positive_counts_and_metric_ranges(
    field_name: str,
    value: float,
    message: str,
) -> None:
    from paper_search.evaluation import ExperimentAggregate

    payload: dict[str, float | int] = {
        "query_count": 1,
        "macro_f1": 0.5,
        "macro_recall": 0.5,
        "search_api_calls": 1,
        "llm_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cost_cny": 0.0,
        "latency_ms": 1,
        "failure_count": 0,
    }
    payload[field_name] = value

    with pytest.raises(ValueError, match=message):
        ExperimentAggregate(**payload)


def test_write_experiment_record_uses_immutable_canonical_store(tmp_path: Path) -> None:
    from paper_search.evaluation import (
        ExperimentAggregate,
        build_experiment_record,
        write_experiment_record,
    )

    record = build_experiment_record(
        run_id="synthetic-dev-002",
        config_hash="sha256:" + "d" * 64,
        git_sha="e" * 40,
        split="dev",
        phase="selection_only",
        modules={"embedding": False, "rerank": False},
        prompt_versions={"analysis": "query-analyze-v1"},
        model_metadata={"embedding": "fixture-embedding-v1"},
        aggregate=ExperimentAggregate(
            query_count=4,
            macro_f1=0.6,
            macro_recall=0.8,
            search_api_calls=5,
            llm_calls=0,
            input_tokens=0,
            output_tokens=0,
            cost_cny=0.0,
            latency_ms=20,
            failure_count=1,
        ),
        artifact_hashes={"predictions": "sha256:" + "f" * 64},
    )
    store = ExperimentRecordStore(tmp_path)

    path = write_experiment_record(store, record)

    assert path.name == "synthetic-dev-002.json"
    assert store.read(record.run_id) == record.model_dump(mode="json")
    assert path.read_bytes().endswith(b"\n")
    second_path = write_experiment_record(store, record)
    assert second_path == path
    with pytest.raises(FileExistsError):
        write_experiment_record(
            store,
            record.model_copy(
                update={
                    "aggregate": record.aggregate.model_copy(update={"search_api_calls": 6}),
                }
            ),
        )
