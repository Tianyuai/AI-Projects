from __future__ import annotations

from pathlib import Path
import subprocess
import sys


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "train_large_scale_fusion.py"


def test_cli_exposes_prepare_only_mode() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--prepare-only" in completed.stdout
    assert "--reliability-pair-budget" in completed.stdout
    assert "--task-provenance-pair-budget" in completed.stdout
    assert "--entity-pair-budget" in completed.stdout
    assert "--hard-constraint-pair-budget" in completed.stdout
    assert "--production-replay-shard-dir" in completed.stdout
