from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path

import pytest

from paper_search.api.contracts import SearchRequest
from paper_search.domain.models import StructuredSearchResponse
from paper_search.evaluation.synthetic_baseline import _run_synthetic_batch
from paper_search.evaluation.synthetic_mocks import build_synthetic_search_service

SUBPROCESS_ENV = {**os.environ, "PYTHONPATH": str(Path("src").resolve())}


def _run_cli(output: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "paper_search.evaluation.synthetic_baseline",
            "--output",
            str(output),
            *extra,
        ],
        check=False,
        capture_output=True,
        env=SUBPROCESS_ENV,
        text=True,
    )


def test_cli_writes_only_byte_stable_synthetic_predictions(
    tmp_path: Path,
) -> None:
    output = tmp_path / "artifacts" / "predictions.jsonl"

    first = _run_cli(output)
    assert first.returncode == 0, first.stderr
    assert first.stderr == ""
    first_bytes = output.read_bytes()

    second = _run_cli(output)
    assert second.returncode == 0, second.stderr
    assert second.stderr == ""
    assert output.read_bytes() == first_bytes
    assert [path.name for path in output.parent.iterdir()] == ["predictions.jsonl"]
    assert b"recall" not in first_bytes.lower()
    assert b"f1" not in first_bytes.lower()


def test_cli_rejects_formal_evaluation_arguments(tmp_path: Path) -> None:
    for argument in (
        "--gold",
        "--split",
        "--metrics",
        "--manifest",
        "--api-key",
        "--endpoint",
        "--out",
    ):
        output = tmp_path / f"{argument[2:]}.jsonl"
        result = _run_cli(output, argument, "forbidden")
        assert result.returncode == 2
        assert not output.exists()


def test_package_batch_exports_are_lazy_and_warning_free() -> None:
    script = """
import sys
import paper_search.evaluation as evaluation

assert "paper_search.evaluation.synthetic_baseline" not in sys.modules
from paper_search.evaluation import synthetic_baseline

for name in (
    "SYNTHETIC_QUERIES",
    "run_synthetic_baseline",
):
    assert getattr(evaluation, name) is getattr(synthetic_baseline, name)

for name in ("SyntheticSearchService", "validate_synthetic_requests"):
    assert not hasattr(evaluation, name)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        env=SUBPROCESS_ENV,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""


@pytest.mark.parametrize("failure_index", range(30))
def test_uncached_batch_isolates_each_failure(
    tmp_path: Path,
    failure_index: int,
) -> None:
    requests = tuple(
        SearchRequest(
            query_id=f"failure-isolation-{index}",
            query=f"Synthetic failure isolation query {index}",
            budget_profile="low",
            include_trace=False,
        )
        for index in range(31)
    )
    service = build_synthetic_search_service()
    calls: list[str] = []

    async def flaky_service(request: SearchRequest) -> StructuredSearchResponse:
        calls.append(request.query_id)
        if request.query_id == requests[failure_index].query_id:
            raise TimeoutError("synthetic uncached failure")
        return await service(request)

    output = tmp_path / "predictions.jsonl"
    records = asyncio.run(
        _run_synthetic_batch(requests, search_service=flaky_service, output=output)
    )

    assert len(records) == len(requests)
    assert calls == [request.query_id for request in requests]
    assert [record.query_id for record in records] == [
        request.query_id for request in requests
    ]
    assert records[failure_index].selected_paper_ids == []
    assert output.exists()
