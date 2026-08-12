"""Recall experiment CLI boundaries stay offline until explicitly authorized."""

from __future__ import annotations

import json
import socket
from asyncio import run
from hashlib import sha256
from pathlib import Path

import pytest

from paper_search.cli import build_parser, main
from paper_search.domain.models import Paper
from paper_search.recall_experiments.composition import (
    RecallRuntime,
    RecallTerminalError,
    compare_recall_artifacts,
    run_recall_experiment,
)
from paper_search.recall_experiments.generation.backends import LLMBackendResult
from paper_search.recall_experiments.retrieval.backends import (
    BackendCitationResult,
    BackendSearchResult,
)


WORKSPACE_ROOT = Path(__file__).parents[2]


def _write_synthetic_inputs(tmp_path: Path, *, backend: str = "snapshot_replay") -> tuple[Path, Path, Path]:
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
  backend: %s
  max_results_per_action: 5
  max_total_actions: 1
evaluation:
  repeat_count: 1
  max_repeat_attempts: 1
"""
        % backend,
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
            {"q-one": {"actions": [{"action_id": "a-1", "action_type": "text_search", "strategy": "synthetic", "payload": {"query_text": "synthetic query"}}]}}
        ),
        encoding="utf-8",
    )
    return recipe, sample, actions


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


def test_compare_writes_truthful_result_from_explicit_run_artifacts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    result = {
        "candidate_pool_policy_version": "production-dedup-v1",
        "gold_association_count": 1,
        "gold_hit_count": 1,
        "macro_candidate_recall": 1.0,
        "per_query": [{"query_id": "q-one", "candidate_pool_ids": ["doi:10.1000/one"], "gold_hit_ids": ["doi:10.1000/one"], "gold_association_count": 1, "gold_hit_count": 1, "candidate_recall": 1.0}],
    }
    for name in ("current", "historical"):
        path = tmp_path / name
        path.mkdir()
        (path / "recall-report.json").write_text(
            json.dumps({"attempts": [{"attempt_status": "succeeded", "result": result}]}), encoding="utf-8"
        )

    assert main(["recall", "compare", "--current", str(tmp_path / "current"), "--historical", str(tmp_path / "historical"), "--out", str(tmp_path / "comparison")]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["conclusion"] == "passed"
    assert json.loads((tmp_path / "comparison" / "recall-comparison.json").read_text()) == payload


def test_compare_rejects_execution_identity_mismatch(tmp_path: Path) -> None:
    result = {
        "candidate_pool_policy_version": "production-dedup-v1",
        "gold_association_count": 1,
        "gold_hit_count": 1,
        "macro_candidate_recall": 1.0,
        "per_query": [],
    }
    for name, recipe_sha in (("current", "a"), ("historical", "b")):
        path = tmp_path / name
        path.mkdir()
        (path / "recall-report.json").write_text(
            json.dumps(
                {
                    "schema_version": "candidate-recall-report-v1",
                    "execution_identity": {
                        "recipe_sha256": "sha256:" + recipe_sha * 64,
                        "sample_sha256": "sha256:" + "c" * 64,
                    },
                    "attempts": [{"attempt_status": "succeeded", "result": result}],
                }
            ),
            encoding="utf-8",
        )

    with pytest.raises(RecallTerminalError, match="config_mismatch"):
        compare_recall_artifacts(
            current_run=tmp_path / "current",
            historical_run=tmp_path / "historical",
            output_path=tmp_path / "comparison",
        )


def test_authorized_live_run_uses_injected_runtime_without_constructing_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    recipe, sample, actions = _write_synthetic_inputs(tmp_path, backend="live_provider")

    class FakeRuntime:
        async def search(self, *_args: object, **_kwargs: object) -> BackendSearchResult:
            return BackendSearchResult(hits=[Paper(canonical_id="doi:10.1000/synthetic", title="Synthetic", sources=["openalex"])])

        async def expand(self, *_args: object, **_kwargs: object) -> BackendCitationResult:
            return BackendCitationResult(direction="references")

        async def generate(self, *_args: object, **_kwargs: object) -> LLMBackendResult:
            return LLMBackendResult()

    runtime = FakeRuntime()
    injected = RecallRuntime(
        runtime,
        runtime,
        runtime,
        identity={
            "backend_identity": "fake-live-v1",
            "budget_policy_sha256": "sha256:" + "1" * 64,
            "pricing_policy_sha256": "sha256:" + "2" * 64,
        },
    )

    assert main(["recall", "run", "--recipe", str(recipe), "--sample", str(sample), "--actions", str(actions), "--allow-live", "--out", str(tmp_path / "live")], recall_runtime_factory=lambda _recipe: injected) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "complete"


def test_live_runtime_rejects_unbound_budget_or_pricing_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    recipe, sample, actions = _write_synthetic_inputs(tmp_path, backend="live_provider")

    class FakeRuntime:
        async def search(self, *_args: object, **_kwargs: object) -> BackendSearchResult:
            return BackendSearchResult()

        async def expand(self, *_args: object, **_kwargs: object) -> BackendCitationResult:
            return BackendCitationResult(direction="references")

        async def generate(self, *_args: object, **_kwargs: object) -> LLMBackendResult:
            return LLMBackendResult()

    runtime = FakeRuntime()
    injected = RecallRuntime(runtime, runtime, runtime, identity={"backend_identity": "fake"})

    with pytest.raises(RecallTerminalError, match="config_mismatch"):
        run(
            run_recall_experiment(
                recipe_path=recipe,
                sample_path=sample,
                output_path=tmp_path / "live",
                workspace_root=tmp_path,
                actions_path=actions,
                allow_live=True,
                live_runtime_factory=lambda _recipe: injected,
            )
        )


def test_fixed_actions_reject_command_line_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    recipe, sample, actions = _write_synthetic_inputs(tmp_path)
    recipe.write_text(recipe.read_text(encoding="utf-8").replace("manual_actions", "fixed_actions"), encoding="utf-8")

    with pytest.raises(RecallTerminalError, match="invalid_actions"):
        run(
            run_recall_experiment(
                recipe_path=recipe,
                sample_path=sample,
                output_path=tmp_path / "run",
                workspace_root=tmp_path,
                actions_path=actions,
                allow_live=False,
            )
        )


def test_fixed_actions_use_recipe_bound_bytes_for_exact_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    recipe, sample, _actions = _write_synthetic_inputs(tmp_path)
    recipe.write_text(
        recipe.read_text(encoding="utf-8").replace("manual_actions", "fixed_actions"),
        encoding="utf-8",
    )

    with pytest.raises(RecallTerminalError, match="snapshot_unavailable"):
        run(
            run_recall_experiment(
                recipe_path=recipe,
                sample_path=sample,
                output_path=tmp_path / "replay",
                workspace_root=tmp_path,
                actions_path=None,
                allow_live=False,
                snapshot_manifest_path=WORKSPACE_ROOT / "tests/fixtures/formal_run/replay/snapshot-manifest.json",
            )
    )

    generation = json.loads((tmp_path / "replay" / "generation" / "attempt-01" / "q-one.json").read_text())
    assert json.loads(generation["immutable_action_batch_utf8"]) == json.loads(
        (tmp_path / "actions.json").read_text()
    )["q-one"]


def test_manifest_backed_novel_replay_has_no_live_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import paper_search.retrieval.snapshot_adapters as snapshot_adapters

    monkeypatch.chdir(tmp_path)
    recipe, sample, actions = _write_synthetic_inputs(tmp_path)

    def no_client(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("novel replay constructed a live client")

    monkeypatch.setattr(snapshot_adapters.httpx, "AsyncClient", no_client)
    manifest = WORKSPACE_ROOT / "tests/fixtures/formal_run/replay/snapshot-manifest.json"

    assert main(["recall", "run", "--recipe", str(recipe), "--sample", str(sample), "--actions", str(actions), "--snapshot-manifest", str(manifest), "--out", str(tmp_path / "replay")]) == 2
    assert json.loads(capsys.readouterr().out)["error_code"] == "snapshot_unavailable"


def test_replay_llm_budget_estimate_uses_only_manifest_bound_usage() -> None:
    from paper_search.recall_experiments import composition

    manifest = {
        "schema_version": "dependency-snapshot-v2",
        "snapshot_set_id": "sha256:" + "1" * 64,
        "sealed_at": "2026-08-12T00:00:00Z",
        "entries": [
            {
                "request": {"dependency": "llm"},
                "usage": {
                    "llm_calls": 1,
                    "input_tokens": 123,
                    "output_tokens": 45,
                    "cost_cny": "0.12",
                    "elapsed_ms": 900,
                },
            },
            {"request": {"dependency": "openalex"}, "usage": {"cost_cny": "9.99"}},
        ],
    }

    initial, repair = composition._llm_replay_estimates_from_manifest(
        json.dumps(manifest).encode("utf-8")
    )

    assert initial == repair
    assert initial.llm_calls == 1
    assert initial.input_tokens == 123
    assert initial.output_tokens == 45
    assert str(initial.cost_cny) == "0.12"
