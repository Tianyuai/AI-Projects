"""Recall experiment CLI boundaries stay offline until explicitly authorized."""

from __future__ import annotations

import json
import socket
from hashlib import sha256
from pathlib import Path

import pytest

from paper_search.cli import build_parser, main


WORKSPACE_ROOT = Path(__file__).parents[2]


def test_recall_parser_exposes_split_workflow_commands(tmp_path: Path) -> None:
    parser = build_parser()
    common = [
        "recall",
        "prepare-context",
        "--recipe",
        "recipe.yaml",
        "--sample",
        "sample.yaml",
        "--out",
        str(tmp_path / "prepared"),
    ]

    args = parser.parse_args(common)

    assert args.command == "recall"
    assert args.recall_command == "prepare-context"
    assert args.out == tmp_path / "prepared"


def test_prepare_context_reports_incomplete_oracle_catalog_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    attempts: list[str] = []

    def tripwire(*_args: object, **_kwargs: object) -> object:
        attempts.append("network")
        raise AssertionError("offline command attempted network access")

    monkeypatch.setattr(socket, "create_connection", tripwire)
    monkeypatch.setattr(socket, "getaddrinfo", tripwire)

    exit_code = main(
        [
            "recall",
            "prepare-context",
            "--recipe",
            str(WORKSPACE_ROOT / "configs/recall_experiments/methods/manual-oracle-smoke.yaml"),
            "--sample",
            str(WORKSPACE_ROOT / "configs/recall_experiments/samples/dev-smoke-3.yaml"),
            "--out",
            str(tmp_path / "prepared"),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert payload == {
        "error_code": "oracle_catalog_incomplete",
        "path": str(tmp_path / "prepared"),
        "status": "failed",
    }
    assert attempts == []


def test_run_rejects_live_recipe_without_explicit_authorization(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(
        [
            "recall",
            "run",
            "--recipe",
            str(WORKSPACE_ROOT / "configs/recall_experiments/methods/manual-oracle-smoke.yaml"),
            "--sample",
            str(WORKSPACE_ROOT / "configs/recall_experiments/samples/dev-smoke-3.yaml"),
            "--actions",
            str(tmp_path / "actions.json"),
            "--out",
            str(tmp_path / "run"),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert payload["error_code"] == "live_not_authorized"
    assert payload["status"] == "failed"


def test_manual_actions_prepare_and_validate_with_complete_synthetic_frozen_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    gold = b'{"query_id":"q-one","query":"synthetic query","relevant_paper_ids":["arxiv:2401.00001"]}\n'
    identifier_map = b'{"arxiv:2401.00001":"doi:10.1000/synthetic"}\n'
    (tmp_path / "gold.jsonl").write_bytes(gold)
    (tmp_path / "identifier-map.json").write_bytes(identifier_map)
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(
        """method_id: synthetic-manual
generator:
  type: manual_actions
  actions: actions.json
  gold_visibility: blind
retrieval:
  allowed_actions: [text_search]
  backend: snapshot_replay
  max_results_per_action: 5
  max_total_actions: 1
evaluation:
  repeat_count: 1
  max_repeat_attempts: 1
""",
        encoding="utf-8",
    )
    sample = tmp_path / "sample.yaml"
    sample.write_text(
        """sample_id: synthetic
query_ids: [q-one]
frozen_inputs:
  gold_associations:
    path: gold.jsonl
    sha256: sha256:%s
  identifier_map:
    path: identifier-map.json
    sha256: sha256:%s
"""
        % (sha256(gold).hexdigest(), sha256(identifier_map).hexdigest()),
        encoding="utf-8",
    )
    actions = tmp_path / "actions.json"
    actions.write_text(
        json.dumps(
            {
                "q-one": {
                    "actions": [
                        {
                            "action_id": "a-1",
                            "action_type": "text_search",
                            "strategy": "synthetic",
                            "payload": {"query_text": "synthetic query"},
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    prepared = tmp_path / "prepared"
    validated = tmp_path / "validated"

    assert main(["recall", "prepare-context", "--recipe", str(recipe), "--sample", str(sample), "--out", str(prepared)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "complete"
    assert main(["recall", "validate-actions", "--recipe", str(recipe), "--contexts", str(prepared), "--actions", str(actions), "--out", str(validated)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "complete"
    assert (validated / "generation" / "validated" / "q-one.json").is_file()

    assert main(["recall", "run", "--recipe", str(recipe), "--sample", str(sample), "--actions", str(actions), "--out", str(tmp_path / "replay")]) == 2
    assert json.loads(capsys.readouterr().out)["error_code"] == "snapshot_unavailable"
