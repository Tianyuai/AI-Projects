from __future__ import annotations

import json
from importlib.util import find_spec

import pytest
import torch

from paper_search import health


def _available_dependencies() -> dict[str, dict[str, object]]:
    return {
        name: {"available": True, "version": "test", "error": None}
        for name in health.RETRIEVAL_DEPENDENCIES
    }


def test_health_module_is_available() -> None:
    assert find_spec("paper_search.health") is not None


def test_cpu_only_runtime_is_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(health, "_dependency_report", _available_dependencies)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    report = health.collect_local_health(matrix_size=4)

    assert report["status"] == "ready"
    assert report["core"]["matrix_smoke"]["shape"] == [4, 4]
    assert report["core"]["matrix_smoke"]["finite"] is True
    assert report["accelerator"] == {
        "backend": "cuda",
        "status": "unavailable",
        "build": torch.version.cuda,
        "device": None,
        "matrix_smoke": None,
    }
    assert report["errors"] == []


def test_required_cuda_is_blocking_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(health, "_dependency_report", _available_dependencies)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    report = health.collect_local_health(matrix_size=4, require_accelerator="cuda")

    assert report["status"] == "degraded"
    assert report["errors"] == ["accelerator_required:cuda"]


def test_available_cuda_reports_deterministic_smoke(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    smoke = {"shape": [4, 4], "finite": True, "checksum": 64.0}
    monkeypatch.setattr(health, "_dependency_report", _available_dependencies)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        health,
        "_cuda_smoke",
        lambda _torch, _size: {"device": "test-cuda", "matrix_smoke": smoke},
    )

    report = health.collect_local_health(matrix_size=4)

    assert report["status"] == "ready"
    assert report["accelerator"]["status"] == "available"
    assert report["accelerator"]["device"] == "test-cuda"
    assert report["accelerator"]["matrix_smoke"] == smoke


def test_optional_cuda_smoke_failure_is_non_blocking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(health, "_dependency_report", _available_dependencies)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def fail_cuda(_torch: object, _size: int) -> dict[str, object]:
        raise RuntimeError("private device detail")

    monkeypatch.setattr(health, "_cuda_smoke", fail_cuda)

    report = health.collect_local_health(matrix_size=4)

    assert report["status"] == "ready"
    assert report["accelerator"]["status"] == "error"
    assert report["errors"] == []
    assert "private device detail" not in json.dumps(report)


def test_cpu_smoke_failure_is_blocking(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(health, "_dependency_report", _available_dependencies)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def fail_cpu(_torch: object, _size: int) -> dict[str, object]:
        raise RuntimeError("private cpu detail")

    monkeypatch.setattr(health, "_cpu_smoke", fail_cpu)

    report = health.collect_local_health(matrix_size=4)

    assert report["status"] == "degraded"
    assert report["core"]["matrix_smoke"] is None
    assert report["errors"] == ["cpu_smoke:RuntimeError"]
    assert "private cpu detail" not in json.dumps(report)


@pytest.mark.parametrize(
    "smoke",
    [
        {"shape": [4, 4], "finite": False, "checksum": 64.0},
        {"shape": [4, 4], "finite": True, "checksum": 63.0},
    ],
)
def test_incorrect_cpu_smoke_result_is_blocking(
    monkeypatch: pytest.MonkeyPatch,
    smoke: dict[str, object],
) -> None:
    monkeypatch.setattr(health, "_dependency_report", _available_dependencies)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(health, "_cpu_smoke", lambda _torch, _size: smoke)

    report = health.collect_local_health(matrix_size=4)

    assert report["status"] == "degraded"
    assert report["errors"] == ["cpu_smoke:invalid_result"]


@pytest.mark.parametrize(
    "smoke",
    [
        {"shape": [4, 4], "finite": False, "checksum": 64.0},
        {"shape": [4, 4], "finite": True, "checksum": 63.0},
    ],
)
def test_incorrect_required_cuda_smoke_result_is_blocking(
    monkeypatch: pytest.MonkeyPatch,
    smoke: dict[str, object],
) -> None:
    monkeypatch.setattr(health, "_dependency_report", _available_dependencies)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        health,
        "_cuda_smoke",
        lambda _torch, _size: {"device": "test-cuda", "matrix_smoke": smoke},
    )

    report = health.collect_local_health(matrix_size=4, require_accelerator="cuda")

    assert report["status"] == "degraded"
    assert report["accelerator"]["status"] == "error"
    assert report["errors"] == ["accelerator_required:cuda"]

def test_missing_dependency_is_blocking(monkeypatch: pytest.MonkeyPatch) -> None:
    dependencies = _available_dependencies()
    dependencies["faiss"] = {
        "available": False,
        "version": None,
        "error": "ImportError",
    }
    monkeypatch.setattr(health, "_dependency_report", lambda: dependencies)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    report = health.collect_local_health(matrix_size=4)

    assert report["status"] == "degraded"
    assert report["errors"] == ["dependency_missing:faiss"]


def test_health_cli_supports_optional_accelerator_requirement(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    captured: list[str | None] = []

    def collect(*, require_accelerator: str | None = None) -> dict[str, object]:
        captured.append(require_accelerator)
        status = "degraded" if require_accelerator else "ready"
        return {"status": status, "sentinel": "safe"}

    monkeypatch.setattr(health, "collect_local_health", collect)

    assert health.main([]) == 0
    assert health.main(["--require-accelerator", "cuda"]) == 1
    outputs = capsys.readouterr().out.strip().splitlines()
    assert captured == [None, "cuda"]
    assert [json.loads(item)["status"] for item in outputs] == ["ready", "degraded"]


def test_health_cli_rejects_unknown_accelerator() -> None:
    with pytest.raises(SystemExit) as exc_info:
        health.main(["--require-accelerator", "rocm"])

    assert exc_info.value.code == 2


def test_health_cli_outputs_json_without_environment_secrets(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    sentinel = "must-not-appear-in-health-output"
    monkeypatch.setenv("LLM_API_KEY", sentinel)
    monkeypatch.setattr(
        health,
        "collect_local_health",
        lambda *, require_accelerator=None: {"status": "ready"},
    )

    assert health.main([]) == 0
    output = capsys.readouterr().out
    assert json.loads(output)["status"] == "ready"
    assert sentinel not in output