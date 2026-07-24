from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


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
    "validate_synthetic_requests",
):
    assert getattr(evaluation, name) is getattr(synthetic_baseline, name)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        env=SUBPROCESS_ENV,
        text=True,
    )
    assert result.returncode == 0, result.stderr
