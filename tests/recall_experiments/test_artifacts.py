from __future__ import annotations

import json

import pytest

from paper_search.recall_experiments.artifacts import RecallArtifactWriter
from paper_search.recall_experiments.contracts import RecallActionBatch
from paper_search.recall_experiments.generation.base import GenerationResult


def test_writer_creates_an_immutable_canonical_run_layout(tmp_path) -> None:
    writer = RecallArtifactWriter(tmp_path)
    run_path = writer.start_run(
        "run-01",
        recipe_lock={"recipe_hash": "sha256:" + "1" * 64},
        sample_manifest={"sample_hash": "sha256:" + "2" * 64, "input_hashes": {}},
    )

    generation = writer.write_generation(
        "attempt-01",
        "query-1",
        {"actions": []},
        attempt_status="running",
        valid_repeat_ordinal=None,
    )
    retrieval = writer.write_retrieval(
        "attempt-01",
        "query-1",
        {"results": []},
        attempt_status="failed",
        valid_repeat_ordinal=None,
        errors=[{"message": "Authorization: Bearer very-secret", "token": "value"}],
    )
    pool = writer.write_candidate_pool(
        "attempt-01",
        "query-1",
        {"entries": []},
        attempt_status="failed",
        valid_repeat_ordinal=None,
    )
    report = writer.write_report(
        {"attempts": [{"attempt_id": "attempt-01", "attempt_status": "failed", "valid_repeat_ordinal": None}]}
    )

    assert run_path / "recipe.lock.yaml" == tmp_path / "run-01" / "recipe.lock.yaml"
    assert generation == run_path / "generation" / "attempt-01" / "query-1.json"
    assert retrieval == run_path / "retrieval" / "attempt-01" / "query-1.json"
    assert pool == run_path / "candidate-pools" / "attempt-01" / "query-1.json"
    assert report == run_path / "recall-report.json"
    assert (run_path / "sample-manifest.json").is_file()
    generation_bytes = generation.read_bytes()
    assert generation_bytes == b'{"actions":[],"attempt_id":"attempt-01","attempt_status":"running","query_id":"query-1","valid_repeat_ordinal":null}\n'
    retrieval_payload = json.loads(retrieval.read_text(encoding="utf-8"))
    assert "very-secret" not in retrieval.read_text(encoding="utf-8")
    assert retrieval_payload["errors"][0]["message"] == "[REDACTED]"
    assert retrieval_payload["errors"][0]["token"] == "[REDACTED]"

    with pytest.raises(FileExistsError):
        writer.write_generation(
            "attempt-01",
            "query-1",
            {"actions": []},
            attempt_status="running",
            valid_repeat_ordinal=None,
        )
    with pytest.raises(FileExistsError):
        writer.start_run("run-01", recipe_lock={}, sample_manifest={})


def test_writer_preserves_fixed_source_bytes_and_rejects_unsafe_paths_and_secrets(tmp_path) -> None:
    writer = RecallArtifactWriter(tmp_path)
    writer.start_run("run-01", recipe_lock={}, sample_manifest={})
    raw_batch = b'{\n  "actions": []\n}'
    generation = GenerationResult(
        query_id="query-1",
        action_batch=RecallActionBatch(actions=[]),
        artifact_bytes=raw_batch,
    )

    generation_path = writer.write_generation(
        "attempt-01",
        "query-1",
        generation,
        attempt_status="succeeded",
        valid_repeat_ordinal=1,
    )
    payload = json.loads(generation_path.read_text(encoding="utf-8"))
    assert payload["immutable_action_batch_utf8"].encode("utf-8") == raw_batch

    with pytest.raises(ValueError, match="path component"):
        writer.write_generation(
            "../../attempt-01",
            "query-2",
            {"actions": []},
            attempt_status="succeeded",
            valid_repeat_ordinal=1,
        )
    retrieval = writer.write_retrieval(
        "attempt-01",
        "query-2",
        {"results": []},
        attempt_status="failed",
        valid_repeat_ordinal=None,
        errors=[{"message": "request failed: x-api-key: top-secret; token=abc123"}],
    )
    assert "top-secret" not in retrieval.read_text(encoding="utf-8")
    assert "abc123" not in retrieval.read_text(encoding="utf-8")


@pytest.mark.parametrize("unsafe", ["C:", "C:stream", "query:alternate", "\\\\server", "/root", ".", ".."])
def test_writer_rejects_portable_unsafe_path_components(tmp_path, unsafe: str) -> None:
    writer = RecallArtifactWriter(tmp_path)
    writer.start_run("run-01", recipe_lock={}, sample_manifest={})

    with pytest.raises(ValueError, match="path component"):
        writer.write_generation(
            unsafe,
            "query-1",
            {"actions": []},
            attempt_status="succeeded",
            valid_repeat_ordinal=1,
        )


@pytest.mark.parametrize("ordinal", [0, 4])
def test_writer_rejects_repeat_ordinals_outside_one_through_three(tmp_path, ordinal: int) -> None:
    writer = RecallArtifactWriter(tmp_path)
    writer.start_run("run-01", recipe_lock={}, sample_manifest={})

    with pytest.raises(ValueError, match="valid_repeat_ordinal"):
        writer.write_generation(
            "attempt-01",
            "query-1",
            {"actions": []},
            attempt_status="succeeded",
            valid_repeat_ordinal=ordinal,
        )


@pytest.mark.parametrize("invalid_number", [float("nan"), float("inf"), float("-inf")])
def test_writer_rejects_non_finite_json_numbers(tmp_path, invalid_number: float) -> None:
    writer = RecallArtifactWriter(tmp_path)
    writer.start_run("run-01", recipe_lock={}, sample_manifest={})

    with pytest.raises(ValueError):
        writer.write_retrieval(
            "attempt-01",
            "query-1",
            {"value": invalid_number},
            attempt_status="failed",
            valid_repeat_ordinal=None,
        )
