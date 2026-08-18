"""Recall experiment CLI boundaries stay offline until explicitly authorized."""

from __future__ import annotations

import json
import socket
from asyncio import run
from hashlib import sha256
from pathlib import Path

import pytest

from paper_search.cli import build_parser, main
from paper_search.control.budget import HardBudgetController
from paper_search.domain.models import Paper, SearchBudget
from paper_search.recall_experiments.composition import (
    RecallRuntime,
    RecallTerminalError,
    compare_recall_artifacts,
    run_recall_experiment,
)
from paper_search.recall_experiments.generation.backends import LLMBackendResult
from paper_search.recall_experiments.identity import ExecutionIdentity
from paper_search.recall_experiments.retrieval.backends import (
    BackendCitationResult,
    BackendSearchResult,
)


WORKSPACE_ROOT = Path(__file__).parents[2]


def _valid_execution_identity() -> dict[str, object]:
    snapshot_sha = "sha256:" + "d" * 64
    return {
        "identity_schema_version": "candidate-recall-execution-identity-v1",
        "method_id": "synthetic-manual",
        "recipe_sha256": "sha256:" + "a" * 64,
        "sample_sha256": "sha256:" + "b" * 64,
        "prompt_sha256": None,
        "generator_type": "manual_actions",
        "generator_model": None,
        "retrieval_backend": "snapshot_replay",
        "snapshot_manifest_sha256": snapshot_sha,
        "actions_sha256": "sha256:" + "c" * 64,
        "max_total_actions": 1,
        "max_results_per_action": 5,
        "candidate_pool_policy_version": "production-dedup-v1",
        "repeat_count": 1,
        "max_repeat_attempts": 1,
        "live_authorized": False,
        "runtime": {
            "backend_identity": "sealed_dependency_snapshot",
            "budget_policy": "recall-replay-v1",
            "pricing_provenance": "snapshot_bound_usage",
            "snapshot_manifest_sha256": snapshot_sha,
        },
    }


def test_canary_command_requires_live_authorization_before_output(tmp_path: Path) -> None:
    output = tmp_path / "canary"

    assert main(["recall", "canary", "--query", "test query", "--out", str(output)]) == 2
    assert not output.exists()


def _live_runtime_identity(*, model: str = "deepseek-v4-flash") -> dict[str, object]:
    pricing_hash = "sha256:" + "e" * 64
    controller_hash = "sha256:" + "f" * 64
    return {
        "identity_schema_version": "candidate-recall-live-runtime-v1",
        "controller_policy_sha256": controller_hash,
        "pricing_policy_sha256": pricing_hash,
        "dependencies": {
            "search": {
                "identity_schema_version": "live-dependency-runtime-identity-v1",
                "provider": "openalex",
                "dependency": "openalex",
                "adapter": "openalex-works-v1",
                "model": None,
                "version": "live-capture-search-v1",
                "endpoints": ["https://api.openalex.org/works"],
                "operations": ["search"],
                "pricing_policy_sha256": pricing_hash,
                "controller_policy_sha256": controller_hash,
            },
            "citation": {
                "identity_schema_version": "live-dependency-runtime-identity-v1",
                "provider": "semantic_scholar",
                "dependency": "semantic_scholar",
                "adapter": "semantic-graph-v1",
                "model": None,
                "version": "live-capture-search-v1",
                "endpoints": [
                    "https://api.semanticscholar.org/graph/v1/paper/search",
                    "https://api.semanticscholar.org/graph/v1/paper/batch",
                    "https://api.semanticscholar.org/graph/v1/paper/{paper_id}/references",
                    "https://api.semanticscholar.org/graph/v1/paper/{paper_id}/citations",
                ],
                "operations": ["search", "batch", "references", "citations"],
                "pricing_policy_sha256": pricing_hash,
                "controller_policy_sha256": controller_hash,
            },
            "llm": {
                "identity_schema_version": "live-dependency-runtime-identity-v1",
                "provider": "deepseek",
                "dependency": "llm",
                "adapter": "openai-compatible-json",
                "model": model,
                "version": "openai-compatible-client-v1",
                "endpoints": ["https://api.deepseek.com/v1/chat/completions"],
                "operations": ["generate_json"],
                "pricing_policy_sha256": pricing_hash,
                "controller_policy_sha256": controller_hash,
            },
        },
    }


def test_execution_identity_requires_generator_and_live_runtime_llm_models_to_match() -> None:
    payload = _valid_execution_identity()
    payload.update(
        {
            "prompt_sha256": "sha256:" + "9" * 64,
            "generator_type": "deepseek_prompt",
            "generator_model": "unexpected-model",
            "retrieval_backend": "live_provider",
            "snapshot_manifest_sha256": None,
            "actions_sha256": None,
            "live_authorized": True,
            "runtime": _live_runtime_identity(),
        }
    )

    with pytest.raises(ValueError, match="generator model"):
        ExecutionIdentity.model_validate(payload)


def test_execution_identity_allows_manual_live_retrieval_without_generator_model() -> None:
    payload = _valid_execution_identity()
    payload.update(
        {
            "retrieval_backend": "live_provider",
            "snapshot_manifest_sha256": None,
            "live_authorized": True,
            "runtime": _live_runtime_identity(),
        }
    )

    identity = ExecutionIdentity.model_validate(payload)

    assert identity.generator_type == "manual_actions"
    assert identity.generator_model is None
    assert identity.runtime.dependencies["llm"].model == "deepseek-v4-flash"


def _write_synthetic_inputs(tmp_path: Path, *, backend: str = "snapshot_replay") -> tuple[Path, Path, Path]:
    gold = b'{"query_id":"q-one","query":"synthetic query","relevant_paper_ids":["arxiv:2401.00001"]}\n'
    identifier_map = (
        b'{"arxiv:2401.00001":"doi:10.48550/arxiv.2401.00001",'
        b'"doi:10.1000/synthetic":"doi:10.48550/arxiv.2401.00001"}\n'
    )
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


def test_authorized_live_recipe_without_runtime_factory_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    recipe, sample, actions = _write_synthetic_inputs(
        tmp_path, backend="live_provider"
    )

    with pytest.raises(RecallTerminalError, match="live_runtime_unavailable"):
        run(
            run_recall_experiment(
                recipe_path=recipe,
                sample_path=sample,
                output_path=tmp_path / "live",
                workspace_root=tmp_path,
                actions_path=actions,
                allow_live=True,
            )
        )


def test_live_runtime_factory_surface_error_is_classified_as_config_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    recipe, sample, actions = _write_synthetic_inputs(
        tmp_path, backend="live_provider"
    )

    def invalid_factory(_recipe: object) -> RecallRuntime:
        raise ValueError("unadmitted provider surface")

    with pytest.raises(RecallTerminalError, match="config_mismatch"):
        run(
            run_recall_experiment(
                recipe_path=recipe,
                sample_path=sample,
                output_path=tmp_path / "live",
                workspace_root=tmp_path,
                actions_path=actions,
                allow_live=True,
                live_runtime_factory=invalid_factory,
            )
        )


def test_live_recipe_model_mismatch_fails_before_generator_or_provider_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import paper_search.recall_experiments.composition as composition

    monkeypatch.chdir(tmp_path)
    _manual_recipe, sample, _actions = _write_synthetic_inputs(
        tmp_path, backend="live_provider"
    )
    prompt = tmp_path / "mismatched-prompt.yaml"
    prompt.write_text(
        "name: mismatch\nversion: v1\nmodel: unexpected-model\n"
        "temperature: 0\ninstructions: [Never reached.]\n",
        encoding="utf-8",
    )
    recipe = tmp_path / "mismatched-recipe.yaml"
    recipe.write_text(
        """method_id: synthetic-mismatch
generator:
  type: deepseek_prompt
  prompt: mismatched-prompt.yaml
  model: unexpected-model
  temperature: 0
  gold_visibility: blind
  max_generated_actions: 1
  repair_attempts: 1
retrieval:
  allowed_actions: [text_search]
  backend: live_provider
  max_results_per_action: 5
  max_total_actions: 1
evaluation:
  repeat_count: 1
  max_repeat_attempts: 1
""",
        encoding="utf-8",
    )
    generator_calls: list[str] = []
    provider_calls: list[str] = []

    class FakeRuntime:
        async def search(self, *_args: object, **_kwargs: object) -> BackendSearchResult:
            provider_calls.append("search")
            return BackendSearchResult()

        async def expand(
            self, *_args: object, **_kwargs: object
        ) -> BackendCitationResult:
            provider_calls.append("citation")
            return BackendCitationResult(direction="references")

        async def generate(self, *_args: object, **_kwargs: object) -> LLMBackendResult:
            provider_calls.append("llm")
            return LLMBackendResult()

    fake = FakeRuntime()
    runtime = RecallRuntime(fake, fake, fake, identity=_live_runtime_identity())
    monkeypatch.setattr(composition, "_valid_live_runtime_identity", lambda _: True)

    def generator_tripwire(*_args: object, **_kwargs: object) -> object:
        generator_calls.append("constructed")
        raise AssertionError("generator construction must not be reached")

    monkeypatch.setattr(composition, "_build_offline_generator", generator_tripwire)

    with pytest.raises(RecallTerminalError, match="config_mismatch"):
        run(
            run_recall_experiment(
                recipe_path=recipe,
                sample_path=sample,
                output_path=tmp_path / "mismatch",
                workspace_root=tmp_path,
                actions_path=None,
                allow_live=True,
                live_runtime_factory=lambda _recipe: runtime,
            )
        )
    assert generator_calls == []
    assert provider_calls == []


def test_authorized_manual_live_run_does_not_bind_an_unused_llm_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import paper_search.recall_experiments.composition as composition

    monkeypatch.chdir(tmp_path)
    recipe, sample, actions = _write_synthetic_inputs(
        tmp_path, backend="live_provider"
    )
    provider_calls: list[str] = []

    class FakeRuntime:
        async def search(
            self, *_args: object, **_kwargs: object
        ) -> BackendSearchResult:
            provider_calls.append("search")
            return BackendSearchResult(
                hits=[
                    Paper(
                        canonical_id="doi:10.1000/synthetic",
                        title="Synthetic",
                        sources=["openalex"],
                    )
                ]
            )

        async def expand(
            self, *_args: object, **_kwargs: object
        ) -> BackendCitationResult:
            provider_calls.append("citation")
            return BackendCitationResult(direction="references")

        async def generate(
            self, *_args: object, **_kwargs: object
        ) -> LLMBackendResult:
            provider_calls.append("llm")
            return LLMBackendResult()

    fake = FakeRuntime()
    runtime = RecallRuntime(fake, fake, fake, identity=_live_runtime_identity())
    monkeypatch.setattr(composition, "_valid_live_runtime_identity", lambda _: True)

    output = tmp_path / "manual-live"
    run(
        run_recall_experiment(
            recipe_path=recipe,
            sample_path=sample,
            output_path=output,
            workspace_root=tmp_path,
            actions_path=actions,
            allow_live=True,
            live_runtime_factory=lambda _recipe: runtime,
        )
    )

    assert provider_calls == ["search"]
    report = json.loads((output / "recall-report.json").read_bytes())
    execution_identity = report["execution_identity"]
    assert execution_identity["generator_type"] == "manual_actions"
    assert execution_identity["generator_model"] is None
    assert execution_identity["runtime"]["dependencies"]["llm"]["model"] == "deepseek-v4-flash"


def test_manual_actions_prepare_and_validate_with_complete_synthetic_frozen_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    gold = b'{"query_id":"q-one","query":"synthetic query","relevant_paper_ids":["arxiv:2401.00001"]}\n'
    identifier_map = (
        b'{"arxiv:2401.00001":"doi:10.48550/arxiv.2401.00001",'
        b'"doi:10.1000/synthetic":"doi:10.48550/arxiv.2401.00001"}\n'
    )
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
            json.dumps(
                {
                    "schema_version": "candidate-recall-report-legacy-v0",
                    "attempts": [{"attempt_status": "succeeded", "result": result}],
                }
            ),
            encoding="utf-8",
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


def test_v1_compare_rejects_missing_execution_identity(tmp_path: Path) -> None:
    result = {
        "candidate_pool_policy_version": "production-dedup-v1",
        "gold_association_count": 1,
        "gold_hit_count": 1,
        "macro_candidate_recall": 1.0,
        "per_query": [],
    }
    for name in ("current", "historical"):
        path = tmp_path / name
        path.mkdir()
        (path / "recall-report.json").write_text(
            json.dumps(
                {
                    "schema_version": "candidate-recall-report-v1",
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


def test_compare_without_historical_validates_current_then_reports_insufficient_evidence(
    tmp_path: Path,
) -> None:
    current = tmp_path / "current"
    current.mkdir()
    result = {
        "candidate_pool_policy_version": "production-dedup-v1",
        "gold_association_count": 1,
        "gold_hit_count": 1,
        "macro_candidate_recall": 1.0,
        "per_query": [],
    }
    (current / "recall-report.json").write_text(
        json.dumps(
                {
                    "schema_version": "candidate-recall-report-v1",
                    "execution_identity": _valid_execution_identity(),
                    "attempts": [{"attempt_status": "succeeded", "result": result}],
                }
        ),
        encoding="utf-8",
    )

    comparison = compare_recall_artifacts(
        current_run=current,
        historical_run=None,
        output_path=tmp_path / "comparison",
    )

    assert comparison["conclusion"] == "insufficient_historical_evidence"
    assert comparison["per_query_comparison"] == "not_provable"


def test_compare_rejects_execution_identity_version_shell(tmp_path: Path) -> None:
    current = tmp_path / "current"
    current.mkdir()
    result = {
        "candidate_pool_policy_version": "production-dedup-v1",
        "gold_association_count": 1,
        "gold_hit_count": 1,
        "macro_candidate_recall": 1.0,
        "per_query": [],
    }
    (current / "recall-report.json").write_text(
        json.dumps(
            {
                "schema_version": "candidate-recall-report-v1",
                "execution_identity": {
                    "identity_schema_version": "candidate-recall-execution-identity-v1"
                },
                "attempts": [{"attempt_status": "succeeded", "result": result}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RecallTerminalError, match="config_mismatch"):
        compare_recall_artifacts(
            current_run=current,
            historical_run=None,
            output_path=tmp_path / "comparison",
        )


@pytest.mark.parametrize("missing_field", tuple(_valid_execution_identity()))
def test_compare_rejects_execution_identity_with_any_required_field_deleted(
    tmp_path: Path, missing_field: str
) -> None:
    current = tmp_path / "current"
    current.mkdir()
    identity = _valid_execution_identity()
    del identity[missing_field]
    result = {
        "candidate_pool_policy_version": "production-dedup-v1",
        "gold_association_count": 1,
        "gold_hit_count": 1,
        "macro_candidate_recall": 1.0,
        "per_query": [],
    }
    (current / "recall-report.json").write_text(
        json.dumps(
            {
                "schema_version": "candidate-recall-report-v1",
                "execution_identity": identity,
                "attempts": [{"attempt_status": "succeeded", "result": result}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RecallTerminalError, match="config_mismatch"):
        compare_recall_artifacts(
            current_run=current,
            historical_run=None,
            output_path=tmp_path / f"comparison-{missing_field}",
        )


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("recipe_sha256", "not-a-sha"),
        ("sample_sha256", "sha256:" + "0" * 64),
        ("max_total_actions", True),
        ("repeat_count", 0),
        ("generator_model", "unexpected-model"),
        ("live_authorized", True),
        ("live_authorized", 0),
    ],
)
def test_compare_rejects_malformed_or_conditionally_invalid_execution_identity(
    tmp_path: Path, field: str, bad_value: object
) -> None:
    current = tmp_path / "current"
    current.mkdir()
    identity = _valid_execution_identity()
    identity[field] = bad_value
    result = {
        "candidate_pool_policy_version": "production-dedup-v1",
        "gold_association_count": 1,
        "gold_hit_count": 1,
        "macro_candidate_recall": 1.0,
        "per_query": [],
    }
    (current / "recall-report.json").write_text(
        json.dumps(
            {
                "schema_version": "candidate-recall-report-v1",
                "execution_identity": identity,
                "attempts": [{"attempt_status": "succeeded", "result": result}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RecallTerminalError, match="config_mismatch"):
        compare_recall_artifacts(
            current_run=current,
            historical_run=None,
            output_path=tmp_path / f"comparison-{field}",
        )


def test_authorized_live_run_rejects_caller_declared_fake_runtime_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    recipe, sample, actions = _write_synthetic_inputs(tmp_path, backend="live_provider")

    class FakeRuntime:
        pricing_policy_sha256: str | None = None

        async def search(self, *_args: object, **_kwargs: object) -> BackendSearchResult:
            return BackendSearchResult(hits=[Paper(canonical_id="doi:10.1000/synthetic", title="Synthetic", sources=["openalex"])])

        async def expand(self, *_args: object, **_kwargs: object) -> BackendCitationResult:
            return BackendCitationResult(direction="references")

        async def generate(self, *_args: object, **_kwargs: object) -> LLMBackendResult:
            return LLMBackendResult()

    runtime = FakeRuntime()
    controller = HardBudgetController(SearchBudget(max_total_tokens=10_000, max_cost_cny=1))
    pricing_bytes = (
        WORKSPACE_ROOT / "tests/fixtures/pricing/pricing-policy-test-v1.yaml"
    ).read_bytes()
    from paper_search.recall_experiments.composition import (
        _budget_policy_sha256,
        _pricing_policy_sha256,
    )
    runtime.pricing_policy_sha256 = _pricing_policy_sha256(pricing_bytes)
    injected = RecallRuntime(
        runtime,
        runtime,
        runtime,
        identity={
            "backend_identity": "fake-live-v1",
            "budget_policy_sha256": _budget_policy_sha256(controller),
            "pricing_policy_sha256": _pricing_policy_sha256(pricing_bytes),
        },
        controller=controller,
        pricing_policy_bytes=pricing_bytes,
    )

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


def test_budget_policy_fingerprint_covers_formal_live_and_reservation_ttl() -> None:
    from paper_search.recall_experiments.composition import _budget_policy_sha256

    budget = SearchBudget(max_total_tokens=10_000, max_cost_cny=1)
    ordinary = HardBudgetController(budget, formal_live=False, reservation_ttl_seconds=120)
    formal = HardBudgetController(budget, formal_live=True, reservation_ttl_seconds=120)
    different_ttl = HardBudgetController(budget, formal_live=True, reservation_ttl_seconds=121)

    assert _budget_policy_sha256(formal) == formal.policy_fingerprint
    assert _budget_policy_sha256(ordinary) != _budget_policy_sha256(formal)
    assert _budget_policy_sha256(formal) != _budget_policy_sha256(different_ttl)


def test_build_live_runtime_accepts_exact_self_identifying_adapters(
    tmp_path: Path,
) -> None:
    import httpx

    from paper_search.control.pricing import (
        ActualCostPricer,
        canonical_pricing_policy_bytes,
        parse_pricing_policy_bytes,
    )
    from paper_search.llm.client import OpenAICompatibleLLMClient
    from paper_search.llm.snapshot_adapters import LiveCaptureLLMAnalyzer
    from paper_search.recall_experiments.composition import (
        build_live_runtime,
    )
    from paper_search.recall_experiments.generation.backends import BudgetedLLMBackend
    from paper_search.recall_experiments.identity import LiveRuntimeIdentity
    from paper_search.recall_experiments.retrieval.backends import (
        BudgetedCitationBackend,
        BudgetedSearchBackend,
    )
    from paper_search.retrieval.snapshot_adapters import LiveCaptureSearchProvider
    from paper_search.storage.dependency_snapshot import DependencyCaptureStore
    from paper_search.domain.models import UsageEstimate

    pricing_bytes = (
        WORKSPACE_ROOT / "tests/fixtures/pricing/pricing-policy-test-v1.yaml"
    ).read_bytes()
    pricer = ActualCostPricer(parse_pricing_policy_bytes(pricing_bytes))
    controller = HardBudgetController(
        SearchBudget(max_total_tokens=10_000, max_cost_cny=1),
        formal_live=True,
        reservation_ttl_seconds=91,
    )
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: pytest.fail(str(request)))
    )
    capture = DependencyCaptureStore(tmp_path / "capture")
    try:
        search = LiveCaptureSearchProvider(
            dependency="openalex",
            client=client,
            capture_store=capture,
            pricer=pricer,
            controller=controller,
        )
        citation = LiveCaptureSearchProvider(
            dependency="semantic_scholar",
            client=client,
            capture_store=capture,
            pricer=pricer,
            controller=controller,
        )
        llm_client = OpenAICompatibleLLMClient(
            client=client,
            base_url="https://api.deepseek.com/v1",
            model="deepseek-v4-flash",
            api_key="test-only",
        )
        analyzer = LiveCaptureLLMAnalyzer(
            client=llm_client,
            capture_store=capture,
            pricer=pricer,
            controller=controller,
            prompt_artifact_sha256="sha256:" + "a" * 64,
        )
        estimate = UsageEstimate(llm_calls=1, cost_cny=0)
        llm = BudgetedLLMBackend(
            analyzer=analyzer,
            controller=controller,
            initial_estimate=estimate,
            repair_estimate=estimate,
        )
        search_backend = BudgetedSearchBackend(
            provider=search,
            controller=controller,
            call_estimate=UsageEstimate(search_api_calls=1, cost_cny=0),
        )
        citation_backend = BudgetedCitationBackend(
            provider=citation,
            controller=controller,
            call_estimate=UsageEstimate(search_api_calls=1, cost_cny=0),
        )

        runtime = build_live_runtime(
            search_backend=search_backend,
            citation_backend=citation_backend,
            llm_backend=llm,
        )
        identity = LiveRuntimeIdentity.model_validate(runtime.identity)
        assert set(identity.dependencies) == {"search", "citation", "llm"}
        assert runtime.search_backend is search_backend
        assert runtime.citation_backend is citation_backend
        assert runtime.llm_backend is llm
        assert runtime.controller is controller
        assert runtime.pricing_policy_bytes == canonical_pricing_policy_bytes(
            parse_pricing_policy_bytes(pricing_bytes)
        )
        tampering = (
            (search_backend, {"provider": "openalex.evil"}),
            (
                search_backend,
                {"endpoints": ("https://api.openalex.org/evil",)},
            ),
            (search_backend, {"operations": ("search", "exfiltrate")}),
            (llm, {"model": "unexpected-model"}),
        )
        for backend, update in tampering:
            original = backend.dependency_identity
            backend._dependency_identity = original.model_copy(  # type: ignore[attr-defined]
                update=update
            )
            try:
                with pytest.raises(RecallTerminalError, match="config_mismatch"):
                    build_live_runtime(
                        search_backend=search_backend,
                        citation_backend=citation_backend,
                        llm_backend=llm,
                    )
            finally:
                backend._dependency_identity = original  # type: ignore[attr-defined]
    finally:
        run(client.aclose())


def test_build_live_runtime_rejects_distinct_pricer_object_before_dispatch(
    tmp_path: Path,
) -> None:
    import httpx

    from paper_search.control.pricing import ActualCostPricer, parse_pricing_policy_bytes
    from paper_search.domain.models import UsageEstimate
    from paper_search.llm.client import OpenAICompatibleLLMClient
    from paper_search.llm.snapshot_adapters import LiveCaptureLLMAnalyzer
    from paper_search.recall_experiments.composition import build_live_runtime
    from paper_search.recall_experiments.generation.backends import BudgetedLLMBackend
    from paper_search.recall_experiments.retrieval.backends import (
        BudgetedCitationBackend,
        BudgetedSearchBackend,
    )
    from paper_search.retrieval.snapshot_adapters import LiveCaptureSearchProvider
    from paper_search.storage.dependency_snapshot import DependencyCaptureStore

    pricing_bytes = (
        WORKSPACE_ROOT / "tests/fixtures/pricing/pricing-policy-test-v1.yaml"
    ).read_bytes()
    policy = parse_pricing_policy_bytes(pricing_bytes)
    first_pricer = ActualCostPricer(policy)
    second_pricer = ActualCostPricer(policy)
    controller = HardBudgetController(
        SearchBudget(max_total_tokens=10_000, max_cost_cny=1), formal_live=True
    )
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: pytest.fail(str(request)))
    )
    store = DependencyCaptureStore(tmp_path / "capture")
    try:
        search_provider = LiveCaptureSearchProvider(
            dependency="openalex",
            client=client,
            capture_store=store,
            pricer=first_pricer,
            controller=controller,
        )
        citation_provider = LiveCaptureSearchProvider(
            dependency="semantic_scholar",
            client=client,
            capture_store=store,
            pricer=first_pricer,
            controller=controller,
        )
        analyzer = LiveCaptureLLMAnalyzer(
            client=OpenAICompatibleLLMClient(
                client=client,
                base_url="https://api.deepseek.com/v1",
                model="deepseek-v4-flash",
                api_key="test-only",
            ),
            capture_store=store,
            pricer=second_pricer,
            controller=controller,
            prompt_artifact_sha256="sha256:" + "a" * 64,
        )
        search = BudgetedSearchBackend(
            provider=search_provider,
            controller=controller,
            call_estimate=UsageEstimate(search_api_calls=1),
        )
        citation = BudgetedCitationBackend(
            provider=citation_provider,
            controller=controller,
            call_estimate=UsageEstimate(search_api_calls=1),
        )
        llm = BudgetedLLMBackend(
            analyzer=analyzer,
            controller=controller,
            initial_estimate=UsageEstimate(llm_calls=1),
            repair_estimate=UsageEstimate(llm_calls=1),
        )
        with pytest.raises(RecallTerminalError, match="config_mismatch"):
            build_live_runtime(
                search_backend=search,
                citation_backend=citation,
                llm_backend=llm,
            )
    finally:
        run(client.aclose())


def test_valid_injected_live_runtime_reaches_mock_llm_and_search_only_when_authorized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import httpx

    from paper_search.control.pricing import ActualCostPricer, parse_pricing_policy_bytes
    from paper_search.domain.models import UsageEstimate
    from paper_search.llm.client import OpenAICompatibleLLMClient
    from paper_search.llm.snapshot_adapters import LiveCaptureLLMAnalyzer
    from paper_search.recall_experiments.composition import build_live_runtime
    from paper_search.recall_experiments.generation.backends import BudgetedLLMBackend
    from paper_search.recall_experiments.retrieval.backends import (
        BudgetedCitationBackend,
        BudgetedSearchBackend,
    )
    from paper_search.retrieval.snapshot_adapters import LiveCaptureSearchProvider
    from paper_search.storage.dependency_snapshot import DependencyCaptureStore

    monkeypatch.chdir(tmp_path)
    _manual_recipe, sample, _actions = _write_synthetic_inputs(
        tmp_path, backend="live_provider"
    )
    prompt = tmp_path / "prompt.yaml"
    prompt.write_text(
        "name: synthetic\nversion: v1\nmodel: deepseek-v4-flash\n"
        "temperature: 0\ninstructions: [Generate one text search.]\n",
        encoding="utf-8",
    )
    recipe = tmp_path / "live-recipe.yaml"
    recipe.write_text(
        """method_id: synthetic-live
generator:
  type: deepseek_prompt
  prompt: prompt.yaml
  model: deepseek-v4-flash
  temperature: 0
  gold_visibility: blind
  max_generated_actions: 1
  repair_attempts: 1
retrieval:
  allowed_actions: [text_search]
  backend: live_provider
  max_results_per_action: 5
  max_total_actions: 1
evaluation:
  repeat_count: 1
  max_repeat_attempts: 1
""",
        encoding="utf-8",
    )
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            calls.append("llm")
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "actions": [
                                            {
                                                "action_id": "a-1",
                                                "action_type": "text_search",
                                                "strategy": "synthetic",
                                                "payload": {
                                                    "query_text": "synthetic query"
                                                },
                                            }
                                        ]
                                    }
                                )
                            }
                        }
                    ],
                    "usage": {"prompt_tokens": 4, "completion_tokens": 3},
                },
                request=request,
            )
        calls.append("openalex")
        return httpx.Response(
            200,
            json={
                "meta": {"count": 1, "per_page": 5, "next_cursor": None},
                "results": [
                    {
                        "id": "https://openalex.org/W1",
                        "doi": "https://doi.org/10.1000/synthetic",
                        "title": "Synthetic",
                        "display_name": "Synthetic",
                        "authorships": [],
                        "publication_year": 2026,
                        "primary_location": None,
                        "cited_by_count": 0,
                        "is_retracted": False,
                    }
                ],
            },
            request=request,
        )

    pricing_bytes = (
        WORKSPACE_ROOT / "tests/fixtures/pricing/pricing-policy-test-v1.yaml"
    ).read_bytes().replace(b"deepseek-test-v1", b"deepseek-v4-flash")
    pricer = ActualCostPricer(parse_pricing_policy_bytes(pricing_bytes))
    controller = HardBudgetController(
        SearchBudget(max_total_tokens=10_000, max_cost_cny=1), formal_live=True
    )
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    store = DependencyCaptureStore(tmp_path / "capture")
    try:
        search_provider = LiveCaptureSearchProvider(
            dependency="openalex",
            client=client,
            capture_store=store,
            pricer=pricer,
            controller=controller,
        )
        citation_provider = LiveCaptureSearchProvider(
            dependency="semantic_scholar",
            client=client,
            capture_store=store,
            pricer=pricer,
            controller=controller,
        )
        analyzer = LiveCaptureLLMAnalyzer(
            client=OpenAICompatibleLLMClient(
                client=client,
                base_url="https://api.deepseek.com/v1",
                model="deepseek-v4-flash",
                api_key="test-only",
            ),
            capture_store=store,
            pricer=pricer,
            controller=controller,
            prompt_artifact_sha256="sha256:" + sha256(prompt.read_bytes()).hexdigest(),
        )
        runtime = build_live_runtime(
            search_backend=BudgetedSearchBackend(
                provider=search_provider,
                controller=controller,
                call_estimate=UsageEstimate(search_api_calls=1, cost_cny=0.01),
            ),
            citation_backend=BudgetedCitationBackend(
                provider=citation_provider,
                controller=controller,
                call_estimate=UsageEstimate(search_api_calls=1, cost_cny=0.01),
            ),
            llm_backend=BudgetedLLMBackend(
                analyzer=analyzer,
                controller=controller,
                initial_estimate=UsageEstimate(
                    llm_calls=1,
                    input_tokens=100,
                    output_tokens=100,
                    cost_cny=0.01,
                ),
                repair_estimate=UsageEstimate(
                    llm_calls=1,
                    input_tokens=100,
                    output_tokens=100,
                    cost_cny=0.01,
                ),
            ),
        )
        output = tmp_path / "authorized"
        run(
            run_recall_experiment(
                recipe_path=recipe,
                sample_path=sample,
                output_path=output,
                workspace_root=tmp_path,
                actions_path=None,
                allow_live=True,
                live_runtime_factory=lambda _recipe: runtime,
            )
        )
        assert calls == ["llm", "openalex"]
        report = json.loads((output / "recall-report.json").read_bytes())
        assert report["execution_identity"]["runtime"] == dict(runtime.identity)

        calls.clear()
        with pytest.raises(RecallTerminalError, match="live_not_authorized"):
            run(
                run_recall_experiment(
                    recipe_path=recipe,
                    sample_path=sample,
                    output_path=tmp_path / "unauthorized",
                    workspace_root=tmp_path,
                    actions_path=None,
                    allow_live=False,
                    live_runtime_factory=lambda _recipe: runtime,
                )
            )
        assert calls == []
    finally:
        run(client.aclose())


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


@pytest.mark.parametrize(
    ("budget_hash", "pricing_hash"),
    [("not-a-hash", "also-not-a-hash"), ("sha256:" + "0" * 64, "sha256:" + "0" * 64)],
)
def test_live_runtime_identity_rejects_forged_policy_hashes(
    budget_hash: str, pricing_hash: str
) -> None:
    from paper_search.recall_experiments.composition import _valid_live_runtime_identity

    class FakeRuntime:
        pricing_policy_sha256: str | None = None

    runtime = FakeRuntime()
    controller = HardBudgetController(SearchBudget(max_total_tokens=10_000, max_cost_cny=1))
    pricing_bytes = (
        WORKSPACE_ROOT / "tests/fixtures/pricing/pricing-policy-test-v1.yaml"
    ).read_bytes()
    runtime.pricing_policy_sha256 = pricing_hash
    injected = RecallRuntime(
        runtime,  # type: ignore[arg-type]
        runtime,  # type: ignore[arg-type]
        runtime,  # type: ignore[arg-type]
        identity={
            "backend_identity": "fake",
            "budget_policy_sha256": budget_hash,
            "pricing_policy_sha256": pricing_hash,
        },
        controller=controller,
        pricing_policy_bytes=pricing_bytes,
    )

    assert not _valid_live_runtime_identity(injected)


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
