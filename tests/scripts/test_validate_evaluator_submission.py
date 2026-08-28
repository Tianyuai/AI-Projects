from __future__ import annotations

from pathlib import Path

from scripts.validate_evaluator_submission import main


def test_repository_evaluator_examples_pass_submission_validation() -> None:
    root = Path(__file__).resolve().parents[2]

    assert main(
        [
            "--queries",
            str(root / "examples/evaluator/queries.jsonl"),
            "--predictions",
            str(root / "examples/evaluator/predictions.jsonl"),
        ]
    ) == 0
