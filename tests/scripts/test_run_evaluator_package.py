from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import scripts.run_evaluator_package as evaluator_package
from scripts.run_evaluator_package import main
from tests.integration.test_serve_process import _sealed_replay_fixture


def test_evaluator_package_script_is_directly_executable() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/run_evaluator_package.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_published_safe_replay_sample_runs_without_api_credentials(
    tmp_path: Path,
) -> None:
    environment = {
        "PYTHONPATH": os.pathsep.join((str(Path("src").resolve()), str(Path.cwd()))),
        "PYTHONUTF8": "1",
    }
    for name in ("SYSTEMROOT", "WINDIR", "TEMP", "TMP"):
        if value := os.environ.get(name):
            environment[name] = value
    sample = Path("examples/safe-replay")
    output = tmp_path / "predictions.jsonl"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_evaluator_package.py",
            "--queries",
            str(sample / "queries.jsonl"),
            "--output",
            str(output),
            "--lock",
            str(sample / "replay.lock.yaml"),
            "--snapshot-manifest",
            str(sample / "snapshots/smoke/snapshot-manifest.json"),
            "--artifact-root",
            str(sample),
            "--capture-output-root",
            str(tmp_path / "captures"),
            "--mode",
            "replay",
        ],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["mode"] == "replay"
    assert json.loads(output.read_text(encoding="utf-8"))["query_id"] == "safe-replay-001"


def test_one_command_evaluator_package_runs_replay_and_validates_output(
    tmp_path,
) -> None:
    fixture = _sealed_replay_fixture(tmp_path)
    queries = tmp_path / "queries.jsonl"
    output = tmp_path / "predictions.jsonl"
    queries.write_text(
        json.dumps(
            {
                "query_id": "judge-q1",
                "query": "resource-aware scholarly paper search",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = main(
        [
            "--queries",
            str(queries),
            "--output",
            str(output),
            "--lock",
            str(fixture["replay_lock"]),
            "--snapshot-manifest",
            str(fixture["manifest"]),
            "--artifact-root",
            str(fixture["root"]),
            "--capture-output-root",
            str(tmp_path / "captures"),
            "--mode",
            "replay",
        ]
    )

    assert result == 0
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row["query_id"] for row in rows] == ["judge-q1"]
    assert len(rows[0]["selected_paper_ids"]) > 0


def test_one_command_live_uses_candidate_lock_without_snapshot(
    tmp_path,
    monkeypatch,
) -> None:
    queries = tmp_path / "queries.jsonl"
    output = tmp_path / "predictions.jsonl"
    lock = tmp_path / "candidate.lock.yaml"
    queries.write_text(
        json.dumps({"query_id": "hidden-q1", "query": "hidden evaluator query"})
        + "\n",
        encoding="utf-8",
    )
    lock.write_text("placeholder", encoding="utf-8")
    events: list[str] = []

    class FakeSession:
        def record_execution(self, execution):
            events.append("record")

        def seal(self):
            events.append("seal")
            return SimpleNamespace(snapshot_set_id="sha256:" + "1" * 64), object()

        def publish(self):
            events.append("publish")
            return tmp_path / "published"

    class FakeService:
        async def execute(self, request, *, run_id):
            assert request.mode in {"live", "replay"}
            assert run_id.startswith(("evaluator-", "replay-check-"))
            return SimpleNamespace(
                outcome=SimpleNamespace(
                    kind="success",
                    response=SimpleNamespace(selected_paper_ids=["W-live-1"]),
                )
            )

    class FakeBundle:
        service = FakeService()
        config_hash = "sha256:" + "2" * 64
        artifact_factory = SimpleNamespace(
            start_capture=lambda **kwargs: FakeSession()
        )

        async def aclose(self):
            return None

    def fake_compose(**kwargs):
        if kwargs["mode"] == "replay":
            assert kwargs["lock_path"] == tmp_path / "published" / "replay.lock.yaml"
            assert kwargs["snapshot_manifest_path"] == (
                tmp_path / "published" / "snapshot-manifest.json"
            )
            return FakeBundle()
        assert kwargs["lock_path"] == lock
        assert kwargs["network_authorized"] is True
        assert kwargs["snapshot_manifest_path"] is None
        return FakeBundle()

    monkeypatch.setattr(evaluator_package.CompositionRoot, "compose", fake_compose)

    result = main(
        [
            "--queries",
            str(queries),
            "--output",
            str(output),
            "--lock",
            str(lock),
            "--artifact-root",
            str(tmp_path),
            "--capture-output-root",
            str(tmp_path / "captures"),
            "--mode",
            "live",
            "--verify-replay",
        ]
    )

    assert result == 0
    assert json.loads(output.read_text(encoding="utf-8"))["selected_paper_ids"] == [
        "W-live-1"
    ]
    assert events == ["record", "seal", "publish"]
