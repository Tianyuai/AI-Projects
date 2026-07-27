from __future__ import annotations

import asyncio
import socket
from pathlib import Path

import pytest

from paper_search.evaluation.official_adapter import InternalPredictionRecord
from paper_search.evaluation.synthetic_baseline import (
    SYNTHETIC_QUERIES,
    run_synthetic_baseline,
)
from paper_search.evaluation.synthetic_mocks import (
    SyntheticOrchestratorFactory,
    build_synthetic_search_service,
)
from paper_search.evaluation.synthetic_baseline import _run_synthetic_batch


def test_real_mock_stack_writes_only_ordered_synthetic_predictions(
    tmp_path: Path,
) -> None:
    output = tmp_path / "artifacts" / "predictions.jsonl"
    records = asyncio.run(run_synthetic_baseline(output=output))

    assert [record.query_id for record in records] == [
        request.query_id for request in SYNTHETIC_QUERIES
    ]
    assert records[0].selected_paper_ids == ["openalex:W100"]
    assert records[1] == InternalPredictionRecord(
        query_id="synthetic-empty-result",
        selected_paper_ids=[],
    )
    assert records[2].selected_paper_ids == ["openalex:W100"]
    assert [path.name for path in output.parent.iterdir()] == ["predictions.jsonl"]


def test_real_mock_stack_repeated_runs_are_byte_identical(
    tmp_path: Path,
) -> None:
    output = tmp_path / "predictions.jsonl"

    asyncio.run(
        run_synthetic_baseline(output=output)
    )
    first = output.read_bytes()
    asyncio.run(
        run_synthetic_baseline(output=output)
    )

    assert output.read_bytes() == first
    assert b"recall" not in first.lower()
    assert b"f1" not in first.lower()


def test_factory_creates_fresh_budget_controller_per_request(
    tmp_path: Path,
) -> None:
    factory = SyntheticOrchestratorFactory()
    service = build_synthetic_search_service(factory=factory)

    asyncio.run(
        _run_synthetic_batch(
            SYNTHETIC_QUERIES,
            search_service=service,
            output=tmp_path / "predictions.jsonl",
        )
    )

    assert len(factory.controllers) == len(SYNTHETIC_QUERIES)
    assert len({id(controller) for controller in factory.controllers}) == len(
        SYNTHETIC_QUERIES
    )


def test_public_runner_is_offline_when_network_is_blocked(
    tmp_path: Path,
) -> None:
    attempts: list[str] = []

    def reject_network(*args: object, **kwargs: object) -> None:
        attempts.append("network")
        raise AssertionError("synthetic baseline attempted network access")

    async def run_with_network_blocked() -> list[InternalPredictionRecord]:
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(socket.socket, "connect", reject_network)
            monkeypatch.setattr(socket.socket, "connect_ex", reject_network)
            monkeypatch.setattr(socket, "create_connection", reject_network)
            monkeypatch.setattr(socket, "getaddrinfo", reject_network)
            return await run_synthetic_baseline(
                output=tmp_path / "predictions.jsonl"
            )

    records = asyncio.run(run_with_network_blocked())

    assert attempts == []
    assert [record.selected_paper_ids for record in records] == [
        ["openalex:W100"],
        [],
        ["openalex:W100"],
    ]
