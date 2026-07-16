from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


FIXTURE_ROOT = Path("tests/fixtures/evaluation")
SUBPROCESS_ENV = {**os.environ, "PYTHONPATH": str(Path("src").resolve())}


def _run_cli(output: Path, *extra_args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "paper_search.evaluation.metrics",
            "--gold",
            str(FIXTURE_ROOT / "gold.jsonl"),
            "--pred",
            str(FIXTURE_ROOT / "predictions.jsonl"),
            "--out",
            str(output),
            *extra_args,
        ],
        check=False,
        capture_output=True,
        env=SUBPROCESS_ENV,
        text=True,
    )


def test_cli_writes_stable_metrics(tmp_path: Path) -> None:
    output = tmp_path / "metrics.json"

    first = _run_cli(output)
    assert first.returncode == 0, first.stderr
    assert first.stderr == ""
    first_bytes = output.read_bytes()

    second = _run_cli(output)
    assert second.returncode == 0, second.stderr
    assert output.read_bytes() == first_bytes

    payload = json.loads(first_bytes)
    assert payload["contract_version"] == "task2-evaluation-v1"
    assert set(payload["input_hashes"]) == {"gold", "predictions"}
    assert "macro_f1" in payload["summary"]
    assert "micro_f1" in payload["summary"]
    assert list(payload["per_query"]) == ["q1", "q2"]


def test_cli_hashes_optional_identifier_map(tmp_path: Path) -> None:
    output = tmp_path / "metrics.json"

    result = _run_cli(
        output,
        "--id-map",
        str(FIXTURE_ROOT / "id_map.json"),
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_bytes())
    assert set(payload["input_hashes"]) == {"gold", "predictions", "id_map"}
    assert payload["per_query"]["q1"]["gold_ids"][0] == "openalex:W1"


def test_package_metric_exports_are_lazy_and_public() -> None:
    script = """
import sys
import paper_search.evaluation as evaluation

assert "paper_search.evaluation.metrics" not in sys.modules
from paper_search.evaluation import metrics

for name in (
    "EvaluationResult",
    "MetricSummary",
    "QueryMetrics",
    "deduplicate_ranked",
    "evaluate",
    "score_query",
):
    assert getattr(evaluation, name) is getattr(metrics, name)
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        env=SUBPROCESS_ENV,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_cli_fails_without_replacing_existing_output(tmp_path: Path) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text("not-json\n", encoding="utf-8")
    output = tmp_path / "metrics.json"
    output.write_text("preserve-me\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "paper_search.evaluation.metrics",
            "--gold",
            str(bad),
            "--pred",
            str(FIXTURE_ROOT / "predictions.jsonl"),
            "--out",
            str(output),
        ],
        check=False,
        capture_output=True,
        env=SUBPROCESS_ENV,
        text=True,
    )

    assert result.returncode == 2
    assert output.read_text(encoding="utf-8") == "preserve-me\n"
