from __future__ import annotations

import json

from paper_search import health


def test_supported_runtime_health_is_ready() -> None:
    report = health.collect_local_health()

    assert report["status"] == "ready"
    assert report["errors"] == []
    assert set(report["core"]["dependencies"]) == {"rank_bm25"}


def test_missing_dependency_is_reported(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(
        health,
        "_dependency_report",
        lambda: {"rank_bm25": {"available": False, "version": None, "error": "ImportError"}},
    )

    report = health.collect_local_health()

    assert report["status"] == "degraded"
    assert report["errors"] == ["dependency_missing:rank_bm25"]


def test_health_cli_outputs_secret_free_json(monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("LLM_API_KEY", "must-not-leak")

    assert health.main([]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ready"
    assert "must-not-leak" not in json.dumps(payload)
