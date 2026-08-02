from __future__ import annotations

import json
import hashlib
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest
import yaml

from paper_search.application.artifacts import FormalRunWorkspace, RunManifest
from paper_search.application.locks import CandidateLock, ReplayLock, lock_sha256
from paper_search.control.ledger import LedgerReport
from paper_search.domain.models import DependencyStatus, UsageActual, UsageEstimate
from paper_search.evaluation.business_results import (
    BusinessResultRecord,
    business_result_sha256,
)
from paper_search.evaluation.execution_adapter import EvaluationExecutionRecord
from paper_search.evaluation.gates import GateEvaluation
from paper_search.evaluation.dataset import EvaluationQuery, PredictionRecord
from paper_search.evaluation.metrics import evaluate
from paper_search.evaluation.official_adapter import InternalPredictionRecord
from paper_search.storage.dependency_snapshot import DependencySnapshotManifestV2


NOW = datetime(2026, 8, 2, tzinfo=UTC)
SHA_A = "sha256:" + "a" * 64
SHA_B = "sha256:" + "b" * 64


def _manifest(run_id: str = "formal-1") -> RunManifest:
    candidate = _candidate_lock()
    return RunManifest(
        run_id=run_id,
        status="incomplete",
        gate_result="not_applicable",
        execution_mode="live",
        split="dev",
        frozen_manifest_sha256=candidate.frozen_data.manifest.sha256,
        partition_sha256=candidate.frozen_data.partition_sha256,
        identifier_map_sha256=candidate.frozen_data.identifier_map.sha256,
        source_git_sha=candidate.source_git_sha,
        tracked_source_dirty=False,
        config_hash=lock_sha256(candidate),
        input_lock_sha256=_sha256(_input_lock_bytes()),
        prompt_version=candidate.baseline.prompt_version,
        snapshot_set_id=_snapshot_set_id(),
        snapshot_manifest_sha256=_snapshot_sha256(),
        experiment_name="main-baseline",
        optional_modules={"embedding": False},
        started_at=NOW,
        ended_at=None,
        readiness_summary=[
            DependencyStatus(dependency="llm", state="ready", cache_hit=False, error_codes=[]),
            DependencyStatus(dependency="openalex", state="ready", cache_hit=False, error_codes=[]),
            DependencyStatus(dependency="semantic_scholar", state="ready", cache_hit=False, error_codes=[]),
        ],
        failure_count=0,
    )


def _workspace(
    tmp_path: Path,
    *,
    seal_snapshots: bool = True,
    **kwargs: object,
) -> FormalRunWorkspace:
    workspace = FormalRunWorkspace(
        runs_root=tmp_path / "runs",
        manifest=_manifest(),
        input_lock_bytes=_input_lock_bytes(),
        nonce_factory=lambda: "nonce",
        clock=lambda: NOW,
        **kwargs,
    )
    if seal_snapshots:
        assert workspace.seal_snapshots() == _snapshot()
    return workspace


def _execution(query_id: str) -> EvaluationExecutionRecord:
    return EvaluationExecutionRecord(
        query_id=query_id,
        run_id="formal-1",
        outcome_kind="success",
        business_result_sha256=business_result_sha256(_business(query_id)),
        usage=UsageActual(),
        diagnostics=[],
        is_partial=False,
        planner_status="primary",
        planner_fallback=False,
        stop_reason="completed",
    )


def _business(query_id: str) -> BusinessResultRecord:
    return BusinessResultRecord(
        query_id=query_id,
        query_analysis=None,
        selected_paper_ids=[],
        high_relevance=[],
        partial_relevance=[],
        citation_edges=[],
        is_partial=False,
        planner_status="primary",
        planner_fallback=False,
        warnings=[],
        stop_reason="completed",
        hard_failure_code=None,
    )


def _gate(result: str = "passed") -> GateEvaluation:
    return GateEvaluation(
        split="dev",
        formal_valid=True,
        quality_passed=result == "passed",
        gate_result=result,
        checks=[],
    )


def _ledger() -> LedgerReport:
    return LedgerReport(
        run_id="formal-1",
        reserved=UsageEstimate(cost_cny=Decimal("0")),
        actual=UsageActual(cost_cny=Decimal("0")),
        run_cap_cny=Decimal("18"),
        project_actual_cny=Decimal("0"),
        project_soft_stop_cny=Decimal("160"),
        project_hard_cap_cny=Decimal("200"),
        within_caps=True,
    )


def _snapshot() -> DependencySnapshotManifestV2:
    return DependencySnapshotManifestV2(
        snapshot_set_id=_snapshot_set_id(),
        sealed_at=NOW,
        entries=[],
    )


def _snapshot_bytes() -> bytes:
    return (
        json.dumps(
            _snapshot().model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _sha256(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _snapshot_sha256() -> str:
    return _sha256(_snapshot_bytes())


def _snapshot_set_id() -> str:
    return _sha256(b"[]")


def _input_lock_bytes() -> bytes:
    return Path("tests/fixtures/application/candidate.lock.yaml").read_bytes()


def _candidate_lock() -> CandidateLock:
    return CandidateLock.model_validate(yaml.safe_load(_input_lock_bytes()))


def _replay_lock() -> ReplayLock:
    candidate = _candidate_lock()
    return ReplayLock(
        schema_version="integrated-lock-v1",
        lock_kind="replay",
        created_at=NOW,
        source_capture_run_id="formal-1",
        source_git_sha=candidate.source_git_sha,
        runtime_allow_live=candidate.runtime_allow_live,
        frozen_data=candidate.frozen_data,
        baseline=candidate.baseline,
        budget_config=candidate.budget_config,
        pricing_policy=candidate.pricing_policy,
        quality_gates=candidate.quality_gates,
        capture_policy=candidate.capture_policy,
        snapshot_set_id=_snapshot_set_id(),
        snapshot_manifest_sha256=_snapshot_sha256(),
    )


def _lock_bytes(lock: ReplayLock) -> bytes:
    return yaml.safe_dump(
        lock.model_dump(mode="python"),
        sort_keys=False,
        allow_unicode=True,
    ).encode("utf-8")


def _populate(workspace: FormalRunWorkspace) -> None:
    for query_id in ("q1", "q2"):
        workspace.write_prediction(InternalPredictionRecord(query_id=query_id))
        workspace.write_execution(_execution(query_id))
        workspace.write_business_result(_business(query_id))
    gold = [EvaluationQuery(query_id=query_id, query=query_id) for query_id in ("q1", "q2")]
    predictions = [PredictionRecord(query_id=query_id) for query_id in ("q1", "q2")]
    workspace.write_metrics(evaluate(gold, predictions))
    workspace.write_usage(_ledger())


def test_workspace_stages_exact_lock_and_ordered_jsonl_tree(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    _populate(workspace)

    assert workspace.work_dir.name == ".incomplete-formal-1-nonce"
    assert (workspace.work_dir / "config.lock.yaml").read_bytes() == _input_lock_bytes()
    assert [json.loads(line)["query_id"] for line in (workspace.work_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()] == ["q1", "q2"]
    assert {path.name for path in workspace.work_dir.iterdir()} == {
        "run.json",
        "config.lock.yaml",
        "predictions.jsonl",
        "executions.jsonl",
        "business-results.jsonl",
        "metrics.json",
        "usage.json",
        "failures.jsonl", "snapshots",
    }


def test_constructor_rejects_input_lock_bytes_not_bound_by_manifest(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="input lock sha256"):
        FormalRunWorkspace(
            runs_root=tmp_path / "runs",
            manifest=_manifest().model_copy(update={"input_lock_sha256": SHA_B}),
            input_lock_bytes=_input_lock_bytes(),
            nonce_factory=lambda: "nonce",
            clock=lambda: NOW,
        )


@pytest.mark.parametrize("writer", ["write_prediction", "write_execution", "write_business_result"])
def test_jsonl_writers_reject_duplicate_query_ids(tmp_path: Path, writer: str) -> None:
    workspace = _workspace(tmp_path)
    record = {
        "write_prediction": InternalPredictionRecord(query_id="q1"),
        "write_execution": _execution("q1"),
        "write_business_result": _business("q1"),
    }[writer]
    method = getattr(workspace, writer)
    method(record)

    with pytest.raises(ValueError, match="duplicate query_id"):
        method(record)


def test_finalize_requires_sealed_snapshot_and_rejects_destination_collision(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path, seal_snapshots=False)
    _populate(workspace)
    with pytest.raises(RuntimeError, match="sealed snapshot"):
        workspace.finalize(gate_evaluation=_gate(), replay_lock=_replay_lock(), snapshot_manifest=None)  # type: ignore[arg-type]

    workspace.seal_snapshots()

    destination = tmp_path / "runs" / "formal-1"
    destination.mkdir()
    with pytest.raises(FileExistsError, match="destination"):
        workspace.finalize(gate_evaluation=_gate(), replay_lock=_replay_lock(), snapshot_manifest=_snapshot())


def test_complete_status_is_independent_from_failed_gate(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    _populate(workspace)

    destination = workspace.finalize(
        gate_evaluation=_gate("failed"),
        replay_lock=_replay_lock(),
        snapshot_manifest=_snapshot(),
    )

    run = json.loads((destination / "run.json").read_text(encoding="utf-8"))
    assert run["status"] == "complete"
    assert run["gate_result"] == "failed"
    assert set(path.name for path in destination.iterdir()) == {
        "run.json", "config.lock.yaml", "replay.lock.yaml", "snapshot-manifest.json",
        "predictions.jsonl", "executions.jsonl", "business-results.jsonl",
        "metrics.json", "usage.json", "failures.jsonl", "snapshots",
    }


def test_finalize_rejects_snapshot_bytes_not_matching_declared_digest(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    _populate(workspace)

    with pytest.raises(ValueError, match="snapshot manifest"):
        workspace.finalize(
            gate_evaluation=_gate(),
            replay_lock=_replay_lock(),
            snapshot_manifest=_snapshot().model_copy(
                update={"sealed_at": NOW + timedelta(seconds=1)}
            ),
        )


def test_replay_copies_exact_verified_lock_and_manifest_bytes(tmp_path: Path) -> None:
    snapshot_bytes = _snapshot().model_dump_json().encode("utf-8") + b"\n"
    replay_lock = _replay_lock().model_copy(
        update={"snapshot_manifest_sha256": _sha256(snapshot_bytes)}
    )
    input_lock_bytes = _lock_bytes(replay_lock)
    manifest = _manifest().model_copy(
        update={
            "execution_mode": "replay",
            "source_git_sha": replay_lock.source_git_sha,
            "config_hash": lock_sha256(replay_lock),
            "input_lock_sha256": _sha256(input_lock_bytes),
            "snapshot_set_id": replay_lock.snapshot_set_id,
            "snapshot_manifest_sha256": replay_lock.snapshot_manifest_sha256,
        }
    )
    replay_root = tmp_path / "replay-source"
    replay_root.mkdir()
    (replay_root / "snapshot-manifest.json").write_bytes(snapshot_bytes)
    workspace = FormalRunWorkspace(
        runs_root=tmp_path / "runs",
        manifest=manifest,
        input_lock_bytes=input_lock_bytes,
        nonce_factory=lambda: "nonce",
        clock=lambda: NOW,
        replay_snapshot_root=replay_root,
    )
    _populate(workspace)

    destination = workspace.finalize(
        gate_evaluation=_gate(),
        replay_lock=replay_lock,
        snapshot_manifest=_snapshot(),
    )

    assert (destination / "replay.lock.yaml").read_bytes() == input_lock_bytes
    assert (destination / "snapshot-manifest.json").read_bytes() == snapshot_bytes


def test_finalize_rejects_cross_file_query_order_and_coverage(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    for query_id in ("q1", "q2"):
        workspace.write_prediction(InternalPredictionRecord(query_id=query_id))
    for query_id in ("q2", "q1"):
        workspace.write_execution(_execution(query_id))
    for query_id in ("q1", "q2"):
        workspace.write_business_result(_business(query_id))
    workspace.write_metrics(evaluate([], []))
    workspace.write_usage(_ledger())

    with pytest.raises(ValueError, match="ordered query coverage"):
        workspace.finalize(
            gate_evaluation=_gate(),
            replay_lock=_replay_lock(),
            snapshot_manifest=_snapshot(),
        )


@pytest.mark.parametrize(("terminal", "expected"), [("fail", "failed"), ("interrupt", "interrupted")])
def test_failure_and_interrupt_publish_only_beneath_failed_root(tmp_path: Path, terminal: str, expected: str) -> None:
    workspace = _workspace(tmp_path)
    destination = workspace.fail("internal_error") if terminal == "fail" else workspace.interrupt()

    assert destination == tmp_path / "runs" / "_failed" / "formal-1"
    assert json.loads((destination / "run.json").read_text(encoding="utf-8"))["status"] == expected
    assert not (tmp_path / "runs" / "formal-1").exists()


def test_validator_failure_never_publishes_complete_destination(tmp_path: Path) -> None:
    def reject(path: Path) -> None:
        assert path.name == ".incomplete-formal-1-nonce"
        raise ValueError("injected validator failure")

    workspace = _workspace(tmp_path, validator=reject)
    _populate(workspace)

    with pytest.raises(ValueError, match="injected validator failure"):
        workspace.finalize(gate_evaluation=_gate(), replay_lock=_replay_lock(), snapshot_manifest=_snapshot())
    assert workspace.work_dir.exists()
    assert not (tmp_path / "runs" / "formal-1").exists()


def test_publication_failure_after_validator_keeps_only_staging(tmp_path: Path) -> None:
    validated: list[Path] = []

    def fail_publish(source: Path, destination: Path) -> None:
        assert validated == [source]
        assert destination == tmp_path / "runs" / "formal-1"
        raise OSError("injected publication failure")

    workspace = _workspace(
        tmp_path,
        validator=validated.append,
        publisher=fail_publish,
    )
    _populate(workspace)

    with pytest.raises(OSError, match="injected publication failure"):
        workspace.finalize(
            gate_evaluation=_gate(),
            replay_lock=_replay_lock(),
            snapshot_manifest=_snapshot(),
        )
    assert workspace.work_dir.exists()
    assert not (tmp_path / "runs" / "formal-1").exists()


def test_write_failure_stays_in_staging_directory(tmp_path: Path) -> None:
    def fail_write(path: Path, payload: bytes) -> None:
        if path.name == "metrics.json":
            raise OSError("injected write failure")
        path.write_bytes(payload)

    workspace = _workspace(tmp_path, writer=fail_write)
    with pytest.raises(OSError, match="injected write failure"):
        workspace.write_metrics(evaluate([], []))
    assert workspace.work_dir.exists()
    assert not (tmp_path / "runs" / "formal-1").exists()


def test_constructor_rejects_cross_device_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    real_stat = __import__("os").stat
    runs_root = (tmp_path / "runs").resolve()

    def different_device(path: object, **kwargs: object):
        result = real_stat(path, **kwargs)
        if Path(path) == runs_root:
            values = list(result)
            values[2] = result.st_dev + 1
            return __import__("os").stat_result(values)
        return result

    monkeypatch.setattr("paper_search.application.artifacts.os.stat", different_device)
    with pytest.raises(ValueError, match="same filesystem"):
        _workspace(tmp_path)
