from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import UTC, datetime

import numpy as np
import pytest

from paper_search.domain.models import BudgetReservation, UsageEstimate
from paper_search.learning.deployment import (
    build_cpu_action_analyzer_decorator,
    build_cpu_pairwise_action_analyzer_decorator,
    load_cpu_action_policy,
    load_cpu_pairwise_action_policy,
)


def _sha256(content: bytes) -> str:
    return "sha256:" + hashlib.sha256(content).hexdigest()


def _bundle(tmp_path):
    weights = np.zeros(256, dtype="<f8").tobytes()
    model_path = tmp_path / "weights.f64"
    model_path.write_bytes(weights)
    result_path = tmp_path / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "model_id": "cpu-action-ranker-v1",
                "dimension": 256,
                "learned": {"threshold": 0.4},
                "model_sha256": _sha256(weights),
            }
        ),
        encoding="utf-8",
    )
    return model_path, result_path


def _pairwise_bundle(tmp_path):
    weights = np.zeros(256, dtype="<f8").tobytes()
    model_path = tmp_path / "pairwise.f64"
    model_path.write_bytes(weights)
    manifest_path = tmp_path / "pairwise.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "cpu-pairwise-action-ranker-experiment-v1",
                "model_id": "cpu-pairwise-action-ranker-v1",
                "target_provider": "openalex",
                "dimension": 256,
                "epochs": 3,
                "learning_rate": 0.08,
                "l2": 1e-6,
                "seed": 7,
                "confidence_threshold": 0.4,
                "model_sha256": _sha256(weights),
            }
        ),
        encoding="utf-8",
    )
    return model_path, manifest_path


def test_loader_rejects_tampered_cpu_action_weights(tmp_path) -> None:
    model_path, result_path = _bundle(tmp_path)
    model_path.write_bytes(b"tampered")

    with pytest.raises(ValueError, match="hash mismatch"):
        load_cpu_action_policy(model_path=model_path, result_path=result_path)


def test_analyzer_decorator_runs_loaded_cpu_policy_through_original_boundary(
    tmp_path,
) -> None:
    model_path, result_path = _bundle(tmp_path)
    fallback_calls = 0

    async def fallback(query, reservation):
        nonlocal fallback_calls
        fallback_calls += 1
        raise AssertionError("zero-weight model at threshold 0.4 must not fallback")

    decorator = build_cpu_action_analyzer_decorator(
        model_path=model_path,
        result_path=result_path,
        max_actions=5,
    )
    analyzer = decorator(fallback)
    reservation = BudgetReservation(
        reservation_id="r1",
        action="query.analyze",
        reserved=UsageEstimate(llm_calls=1, input_tokens=100, output_tokens=100),
        expires_at=datetime(2030, 1, 1, tzinfo=UTC),
    )

    result = asyncio.run(
        analyzer(
            "Which paper proposed graph diffusion networks for retrieval?",
            reservation,
        )
    )

    assert fallback_calls == 0
    assert result.provenance["model_id"] == "cpu-action-ranker-v1"
    assert len(result.data["search_plan"]["subqueries"]) >= 2


def test_pairwise_loader_rejects_tampered_weights(tmp_path) -> None:
    model_path, manifest_path = _pairwise_bundle(tmp_path)
    model_path.write_bytes(b"tampered")

    with pytest.raises(ValueError, match="hash mismatch"):
        load_cpu_pairwise_action_policy(
            model_path=model_path,
            manifest_path=manifest_path,
        )


def test_pairwise_decorator_runs_loaded_policy_through_original_boundary(
    tmp_path,
) -> None:
    model_path, manifest_path = _pairwise_bundle(tmp_path)
    fallback_calls = 0

    async def fallback(query, reservation):
        nonlocal fallback_calls
        fallback_calls += 1
        raise AssertionError("zero-weight model at threshold 0.4 must not fallback")

    decorator = build_cpu_pairwise_action_analyzer_decorator(
        model_path=model_path,
        manifest_path=manifest_path,
        max_actions=5,
    )
    analyzer = decorator(fallback)
    reservation = BudgetReservation(
        reservation_id="r2",
        action="query.analyze",
        reserved=UsageEstimate(llm_calls=1, input_tokens=100, output_tokens=100),
        expires_at=datetime(2030, 1, 1, tzinfo=UTC),
    )

    result = asyncio.run(
        analyzer(
            "Which paper proposed graph diffusion networks for retrieval?",
            reservation,
        )
    )

    assert fallback_calls == 0
    assert result.provenance["model_id"] == "cpu-pairwise-action-ranker-v1"
    assert len(result.data["search_plan"]["subqueries"]) >= 2
