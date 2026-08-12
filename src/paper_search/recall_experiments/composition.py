"""Composition boundary for offline-first candidate-recall commands.

This module is deliberately the only place that selects a generator or a
retrieval backend.  ``RecallExperimentRunner`` receives completed interfaces
and therefore has no method- or mode-specific branches.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from paper_search.control.budget import HardBudgetController
from paper_search.domain.models import ErrorDetail, Paper, SearchBudget, UsageEstimate
from paper_search.recall_experiments.artifacts import RecallArtifactWriter
from paper_search.recall_experiments.candidate_pool import CandidatePoolBuilder
from paper_search.recall_experiments.contracts import RecallGenerationContext
from paper_search.recall_experiments.evaluator import (
    CandidateRecallEvaluator,
    PreparedEvaluationContext,
    RecallRepeatResult,
    compare_exact_replay,
)
from paper_search.recall_experiments.generation.backends import (
    BudgetedLLMBackend,
    LLMBackend,
    LLMBackendResult,
)
from paper_search.recall_experiments.generation.base import QueryGenerator
from paper_search.recall_experiments.generation.deepseek import DeepSeekPromptGenerator, RecallPromptArtifact
from paper_search.recall_experiments.generation.fixed import FixedActionGenerator
from paper_search.recall_experiments.generation.manual import ManualActionGenerator
from paper_search.recall_experiments.inputs.formal_run import FormalRunInputSource
from paper_search.recall_experiments.inputs.historical import load_historical_replays
from paper_search.recall_experiments.inputs.gold_catalog import (
    GoldDocumentCatalogSource,
    SealedGoldDocumentCatalog,
)
from paper_search.recall_experiments.recipes import (
    DeepSeekPromptGeneratorRecipe,
    FixedActionsGeneratorRecipe,
    LoadedRecallRecipe,
    LoadedSampleBinding,
    ManualActionsGeneratorRecipe,
    RecallMethodRecipe,
    authorize_live_backend,
    load_recall_recipe,
    load_sample_binding,
    validate_recipe_sample_preflight,
)
from paper_search.recall_experiments.retrieval.backends import (
    BackendCitationResult,
    BackendSearchResult,
    BudgetedCitationBackend,
    BudgetedSearchBackend,
    CitationBackend,
    SearchBackend,
)
from paper_search.recall_experiments.retrieval.citation_expand import CitationExpandHandler
from paper_search.recall_experiments.retrieval.registry import RetrievalActionRegistry
from paper_search.recall_experiments.retrieval.text_search import TextSearchHandler
from paper_search.recall_experiments.retrieval.title_search import TitleSearchHandler
from paper_search.recall_experiments.runner import RecallExperimentRequest, RecallExperimentRunner
from paper_search.recall_experiments.stages import CandidateStagePipeline
from paper_search.retrieval.snapshot_adapters import LiveCaptureSearchProvider, ReplaySearchProvider
from paper_search.storage.dependency_snapshot import DependencySnapshotReader


TerminalCode = Literal[
    "config_mismatch",
    "invalid_actions",
    "live_not_authorized",
    "live_runtime_unavailable",
    "oracle_catalog_incomplete",
    "snapshot_unavailable",
]


class RecallTerminalError(RuntimeError):
    """A safe, provider-detail-free terminal command outcome."""

    def __init__(self, code: TerminalCode) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class PreparedRecallRun:
    loaded_recipe: LoadedRecallRecipe
    loaded_sample: LoadedSampleBinding
    prepared: PreparedEvaluationContext
    evaluator: CandidateRecallEvaluator

    @property
    def contexts(self) -> tuple[RecallGenerationContext, ...]:
        return self.prepared.generation_contexts


@dataclass(frozen=True)
class RecallRuntime:
    """Completed backend interfaces selected by composition, never by the runner."""

    search_backend: SearchBackend
    citation_backend: CitationBackend
    llm_backend: LLMBackend


RecallRuntimeFactory = Callable[[LoadedRecallRecipe], RecallRuntime]


def prepare_recall_run(recipe_path: Path, sample_path: Path, *, workspace_root: Path) -> PreparedRecallRun:
    """Verify frozen inputs and construct generator-safe contexts offline."""
    try:
        loaded_recipe = load_recall_recipe(recipe_path)
        loaded_sample = load_sample_binding(sample_path)
        validate_recipe_sample_preflight(loaded_recipe.recipe, loaded_sample.binding)
        dataset = FormalRunInputSource(workspace_root).load_queries(loaded_sample.binding)
        catalog = _load_catalog(loaded_recipe.recipe, loaded_sample, dataset, workspace_root)
        evaluator = CandidateRecallEvaluator(
            loaded_recipe.recipe,
            sample=loaded_sample.binding,
            gold_catalog=catalog,
        )
        return PreparedRecallRun(loaded_recipe, loaded_sample, evaluator.preflight(dataset), evaluator)
    except RecallTerminalError:
        raise
    except ValueError as error:
        message = str(error)
        if "complete sealed Gold catalog" in message or "oracle_catalog_" in message:
            raise RecallTerminalError("oracle_catalog_incomplete") from error
        raise RecallTerminalError("config_mismatch") from error
    except OSError as error:
        raise RecallTerminalError("config_mismatch") from error


def write_prepared_contexts(prepared: PreparedRecallRun, output_path: Path) -> Path:
    """Publish only sanitized, action-generation-visible contexts."""
    _start_output(output_path, prepared)
    for context in prepared.contexts:
        _write_new_json(output_path / "contexts" / f"{context.query_id}.json", context.model_dump(mode="json"))
    return output_path


def validate_pasted_actions(
    *, recipe_path: Path, contexts_path: Path, actions_path: Path, output_path: Path
) -> Path:
    """Freeze manually supplied actions without constructing an LLM or backend."""
    try:
        loaded_recipe = load_recall_recipe(recipe_path)
        contexts = _load_contexts(contexts_path)
        _validate_context_recipe_binding(contexts_path, loaded_recipe)
        generator = build_manual_generator(
            actions_path,
            contexts=contexts,
            recipe=loaded_recipe.recipe,
        )
        writer = RecallArtifactWriter(output_path.parent)
        writer.start_run(
            output_path.name,
            recipe_lock=loaded_recipe.recipe_bytes,
            sample_manifest={"query_ids": [context.query_id for context in contexts]},
        )
        for context in contexts:
            result = _run_generator(generator, context)
            writer.write_generation(
                "validated",
                context.query_id,
                result,
                attempt_status="succeeded",
                valid_repeat_ordinal=1,
            )
        return output_path
    except RecallTerminalError:
        raise
    except (TypeError, ValueError, OSError) as error:
        raise RecallTerminalError("invalid_actions") from error


async def run_recall_experiment(
    *,
    recipe_path: Path,
    sample_path: Path,
    output_path: Path,
    workspace_root: Path,
    actions_path: Path | None,
    allow_live: bool,
    snapshot_manifest_path: Path | None = None,
    live_runtime_factory: RecallRuntimeFactory | None = None,
) -> Path:
    """Run injected generator and handlers; live selection is explicit and fail-closed."""
    loaded_recipe = load_recall_recipe(recipe_path)
    recipe = loaded_recipe.recipe
    if actions_path is not None and not isinstance(recipe.generator, ManualActionsGeneratorRecipe):
        raise RecallTerminalError("invalid_actions")
    if recipe.retrieval.backend == "live_provider":
        try:
            authorize_live_backend(recipe, allow_live=allow_live)
        except PermissionError as error:
            raise RecallTerminalError("live_not_authorized") from error
    prepared = prepare_recall_run(recipe_path, sample_path, workspace_root=workspace_root)
    contexts = prepared.contexts
    if recipe.retrieval.backend == "live_provider":
        if live_runtime_factory is None:
            raise RecallTerminalError("live_runtime_unavailable")
        runtime = live_runtime_factory(loaded_recipe)
    else:
        runtime = (
            build_replay_runtime(snapshot_manifest_path, loaded_recipe)
            if snapshot_manifest_path is not None
            else unavailable_runtime()
        )
    generator = _build_offline_generator(loaded_recipe, contexts, actions_path, runtime.llm_backend)
    registry = build_handler_registry(runtime)
    writer = RecallArtifactWriter(output_path.parent)
    runner = RecallExperimentRunner(
        input_source=_PreparedInputSource(prepared),
        generator=generator,
        registry=cast(Any, registry),
        pool_builder=CandidatePoolBuilder(recipe.candidate_pool.policy_version),
        stages=CandidateStagePipeline(),
        evaluator=_PreparedEvaluator(prepared),
        writer=cast(Any, writer),
    )
    request = RecallExperimentRequest(
        run_id=output_path.name,
        sample=prepared.loaded_sample.binding,
        recipe_lock=_recipe_lock(loaded_recipe),
        sample_manifest={"query_ids": [context.query_id for context in contexts]},
        allowed_actions=recipe.retrieval.allowed_actions,
        max_actions=recipe.retrieval.max_total_actions,
        max_results_per_action=recipe.retrieval.max_results_per_action,
        repeat_count=recipe.evaluation.repeat_count,
        max_repeat_attempts=recipe.evaluation.max_repeat_attempts,
    )
    result = await runner.run(request)
    if any(attempt.attempt_status == "failed" for attempt in result.attempts):
        raise RecallTerminalError("snapshot_unavailable")
    return output_path


def compare_recall_artifacts(
    *, current_run: Path, historical_run: Path | None, output_path: Path
) -> dict[str, object]:
    """Compare explicit recall-report artifacts without inventing historical evidence."""
    try:
        current = _result_from_report(current_run)
        historical = _result_from_report(historical_run) if historical_run is not None else None
        comparison = compare_exact_replay(current, historical)
        payload = {
            **comparison.model_dump(mode="json"),
            "path": str(output_path),
            "status": "complete",
        }
        output_path.mkdir(parents=True, exist_ok=False)
        _write_new_json(output_path / "recall-comparison.json", payload)
        return payload
    except (OSError, TypeError, ValueError, KeyError) as error:
        raise RecallTerminalError("config_mismatch") from error


def compare_historical_replays(
    *,
    inventory_path: Path,
    config_root: Path,
    output_path: Path,
    workspace_root: Path,
) -> dict[str, object]:
    """Write all historical replay terminal states without manufacturing evidence."""
    try:
        replay = load_historical_replays(
            inventory_path=inventory_path,
            config_root=config_root,
            workspace_root=workspace_root,
        )
        methods = [
            {
                "method_id": method.method_id,
                "source_run_id": method.source_run_id,
                "source_hashes": method.source_hashes,
                "query_ids_available": method.query_ids_available,
                "evidence_level": method.evidence_level,
                "candidate_pool_policy_version": method.candidate_pool_policy_version,
                "fixed_actions_replayed": method.fixed_actions is not None,
                "candidate_pool_ids_by_query": method.candidate_pool_ids_by_query,
                "gold_hit_ids_by_query": method.gold_hit_ids_by_query,
                "aggregate_metrics": method.aggregate_metrics,
                "terminal_state": method.terminal_state,
                "per_query_equality": method.per_query_equality,
                "semantic_mismatch": method.semantic_mismatch,
                "unprovable_fields": method.unprovable_fields,
            }
            for method in replay.methods.values()
        ]
        payload: dict[str, object] = {
            "schema_version": "candidate-recall-historical-replay-v1",
            "methods": methods,
            "scheme_b_terminal_state": replay.scheme_b_terminal_state,
            "status": "complete",
            "path": str(output_path),
        }
        output_path.mkdir(parents=True, exist_ok=False)
        _write_new_json(output_path / "historical-replay-comparison.json", payload)
        return payload
    except (OSError, TypeError, ValueError, KeyError) as error:
        raise RecallTerminalError("config_mismatch") from error


def build_manual_generator(
    actions_path: Path, *, contexts: Sequence[RecallGenerationContext], recipe: RecallMethodRecipe
) -> ManualActionGenerator:
    return ManualActionGenerator(
        actions_path,
        expected_query_ids=[context.query_id for context in contexts],
        allowed_actions=recipe.retrieval.allowed_actions,
        max_actions=recipe.retrieval.max_total_actions,
    )


def build_fixed_generator(
    actions_path: Path, *, contexts: Sequence[RecallGenerationContext], recipe: RecallMethodRecipe
) -> FixedActionGenerator:
    decoded = json.loads(actions_path.read_bytes().decode("utf-8"))
    if not isinstance(decoded, Mapping):
        raise ValueError("fixed actions must map query IDs to action batches")
    return FixedActionGenerator(
        decoded,
        expected_query_ids=[context.query_id for context in contexts],
        allowed_actions=recipe.retrieval.allowed_actions,
        max_actions=recipe.retrieval.max_total_actions,
    )


def build_deepseek_generator(
    *,
    contexts: Sequence[RecallGenerationContext],
    recipe: DeepSeekPromptGeneratorRecipe,
    prompt_bytes: bytes,
    allowed_actions: Sequence[str],
    backend: LLMBackend | None = None,
) -> DeepSeekPromptGenerator:
    del contexts
    return DeepSeekPromptGenerator(
        backend=backend or _SnapshotUnavailableLLMBackend(),
        prompt=RecallPromptArtifact.from_yaml_bytes(prompt_bytes),
        visibility=recipe.gold_visibility,
        allowed_actions=allowed_actions,
        max_actions=recipe.max_generated_actions,
    )


generator_factories = {
    "manual_actions": build_manual_generator,
    "fixed_actions": build_fixed_generator,
    "deepseek_prompt": build_deepseek_generator,
}


def build_handler_registry(runtime: RecallRuntime) -> RetrievalActionRegistry:
    """Register completed handlers; no handler is selected by the runner."""
    handler_registry = RetrievalActionRegistry()
    handler_registry.register("text_search", build_text_handler(runtime))
    handler_registry.register("title_search", build_title_handler(runtime))
    handler_registry.register("citation_expand", build_citation_handler(runtime))
    return handler_registry


def build_text_handler(runtime: RecallRuntime) -> TextSearchHandler:
    return TextSearchHandler(backend=runtime.search_backend)


def build_title_handler(runtime: RecallRuntime) -> TitleSearchHandler:
    return TitleSearchHandler(backend=runtime.search_backend)


def build_citation_handler(runtime: RecallRuntime) -> CitationExpandHandler:
    return CitationExpandHandler(backend=runtime.citation_backend)


def unavailable_runtime() -> RecallRuntime:
    unavailable = _SnapshotUnavailableRuntime()
    return RecallRuntime(unavailable, unavailable, _SnapshotUnavailableLLMBackend())


def build_replay_runtime(manifest_path: Path, loaded_recipe: LoadedRecallRecipe) -> RecallRuntime:
    """Wrap sealed replay providers and analyzer without constructing a client."""
    content = manifest_path.read_bytes()
    reader = DependencySnapshotReader(
        manifest_path,
        snapshot_manifest_sha256="sha256:" + hashlib.sha256(content).hexdigest(),
    )
    controller = _recall_controller()
    search = BudgetedSearchBackend(
        provider=ReplaySearchProvider(dependency="openalex", reader=reader),
        controller=controller,
        call_estimate=_search_estimate(),
    )
    citation = BudgetedCitationBackend(
        provider=ReplaySearchProvider(dependency="semantic_scholar", reader=reader),
        controller=controller,
        call_estimate=_search_estimate(),
    )
    if not isinstance(loaded_recipe.recipe.generator, DeepSeekPromptGeneratorRecipe):
        return RecallRuntime(search, citation, _SnapshotUnavailableLLMBackend())
    assert loaded_recipe.prompt_sha256 is not None
    from paper_search.llm.snapshot_adapters import ReplayLLMAnalyzer

    analyzer = ReplayLLMAnalyzer(
        reader=reader,
        model_id=loaded_recipe.recipe.generator.model,
        prompt_artifact_sha256=loaded_recipe.prompt_sha256,
    )
    llm = BudgetedLLMBackend(
        analyzer=analyzer,
        controller=controller,
        initial_estimate=_llm_estimate(),
        repair_estimate=_llm_estimate(),
    )
    return RecallRuntime(search, citation, llm)


def build_live_runtime(
    *,
    search_provider: LiveCaptureSearchProvider,
    citation_provider: LiveCaptureSearchProvider,
    llm_backend: LLMBackend,
    controller: HardBudgetController,
) -> RecallRuntime:
    """Wrap explicitly supplied live providers; client creation remains outside runners."""
    return RecallRuntime(
        search_backend=BudgetedSearchBackend(
            provider=search_provider, controller=controller, call_estimate=_search_estimate()
        ),
        citation_backend=BudgetedCitationBackend(
            provider=citation_provider, controller=controller, call_estimate=_search_estimate()
        ),
        llm_backend=llm_backend,
    )


class _SnapshotUnavailableLLMBackend(LLMBackend):
    async def generate(self, request: object, call_kind: object) -> LLMBackendResult:
        del request, call_kind
        return LLMBackendResult(
            errors=[_snapshot_error("llm")], infrastructure_failure=True
        )


class _SnapshotUnavailableRuntime(SearchBackend, CitationBackend):
    async def search(
        self, action_id: str, query: str, filters: dict[str, object], limit: int
    ) -> BackendSearchResult:
        del action_id, query, filters, limit
        return BackendSearchResult(errors=[_snapshot_error("search")], infrastructure_failure=True)

    async def expand(
        self, action_id: str, seed: Paper, direction: Literal["references", "citations", "both"], limit: int
    ) -> BackendCitationResult:
        del action_id, seed, limit
        return BackendCitationResult(
            direction=direction, errors=[_snapshot_error("citation")], infrastructure_failure=True
        )


class _PreparedInputSource:
    def __init__(self, prepared: PreparedRecallRun) -> None:
        self._prepared = prepared

    def load_queries(self, sample: object) -> object:
        del sample
        return self._prepared


class _PreparedEvaluator:
    def __init__(self, prepared: PreparedRecallRun) -> None:
        self._prepared = prepared

    def preflight(self, dataset: object) -> PreparedEvaluationContext:
        del dataset
        return self._prepared.prepared

    def evaluate(self, prepared: object, pools: object) -> object:
        del prepared
        return self._prepared.evaluator.evaluate(self._prepared.prepared, cast(Any, pools))


def _build_offline_generator(
    loaded: LoadedRecallRecipe,
    contexts: Sequence[RecallGenerationContext],
    actions_path: Path | None,
    llm_backend: LLMBackend,
) -> QueryGenerator:
    recipe = loaded.recipe
    if isinstance(recipe.generator, ManualActionsGeneratorRecipe):
        return build_manual_generator(actions_path or Path(recipe.generator.actions), contexts=contexts, recipe=recipe)
    if isinstance(recipe.generator, FixedActionsGeneratorRecipe):
        return build_fixed_generator(actions_path or Path(recipe.generator.actions), contexts=contexts, recipe=recipe)
    assert isinstance(recipe.generator, DeepSeekPromptGeneratorRecipe)
    assert loaded.prompt_bytes is not None
    return build_deepseek_generator(
        contexts=contexts,
        recipe=recipe.generator,
        prompt_bytes=loaded.prompt_bytes,
        allowed_actions=recipe.retrieval.allowed_actions,
        backend=llm_backend,
    )


def _load_catalog(
    recipe: RecallMethodRecipe, loaded_sample: LoadedSampleBinding, dataset: object, workspace_root: Path
) -> SealedGoldDocumentCatalog | None:
    if recipe.generator.gold_visibility != "oracle":
        return None
    sample = loaded_sample.binding
    if sample.gold_document_catalog is None or sample.gold_document_catalog_manifest is None:
        raise RecallTerminalError("oracle_catalog_incomplete")
    frozen = sample.frozen_inputs
    if frozen is None:
        raise RecallTerminalError("config_mismatch")
    gold_records = getattr(getattr(dataset, "evaluation_materials"), "gold_records")
    catalog = GoldDocumentCatalogSource(workspace_root).load(
        sample.gold_document_catalog,
        manifest_binding=sample.gold_document_catalog_manifest,
        bound_paper_sources=frozen.bound_paper_sources,
        gold_associations=gold_records,
    )
    if catalog.status != "complete":
        raise RecallTerminalError("oracle_catalog_incomplete")
    return catalog


def _load_contexts(path: Path) -> tuple[RecallGenerationContext, ...]:
    context_dir = path / "contexts"
    contexts = tuple(
        RecallGenerationContext.model_validate_json(item.read_bytes())
        for item in sorted(context_dir.glob("*.json"))
    )
    if not contexts or len({context.query_id for context in contexts}) != len(contexts):
        raise ValueError("prepared contexts must contain unique query IDs")
    return contexts


def _validate_context_recipe_binding(path: Path, recipe: LoadedRecallRecipe) -> None:
    manifest = json.loads((path / "sample-manifest.json").read_bytes())
    if manifest.get("recipe_sha256") != recipe.recipe_sha256:
        raise ValueError("prepared contexts are bound to a different recipe")


def _start_output(path: Path, prepared: PreparedRecallRun) -> None:
    _start_output_from_loaded(path, prepared.loaded_recipe, prepared.contexts)


def _start_output_from_loaded(
    path: Path, recipe: LoadedRecallRecipe, contexts: Sequence[RecallGenerationContext]
) -> None:
    path.mkdir(parents=True, exist_ok=False)
    _write_new_json(
        path / "sample-manifest.json",
        {
            "query_ids": [context.query_id for context in contexts],
            "recipe_sha256": recipe.recipe_sha256,
        },
    )
    (path / "recipe.lock.yaml").write_bytes(recipe.recipe_bytes)


def _write_new_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        handle.write("\n")


def _run_generator(generator: QueryGenerator, context: RecallGenerationContext) -> Any:
    import asyncio

    return asyncio.run(generator.generate(context))


def _recipe_lock(loaded: LoadedRecallRecipe) -> dict[str, object]:
    return {"recipe_sha256": loaded.recipe_sha256, "recipe": loaded.recipe.model_dump(mode="json")}


def _result_from_report(run_path: Path) -> RecallRepeatResult:
    report = json.loads((run_path / "recall-report.json").read_bytes())
    attempts = report.get("attempts")
    if not isinstance(attempts, list):
        raise ValueError("recall report lacks attempts")
    results = [attempt.get("result") for attempt in attempts if attempt.get("attempt_status") == "succeeded"]
    if len(results) != 1 or not isinstance(results[0], Mapping):
        raise ValueError("recall report lacks one successful repeat")
    return RecallRepeatResult.model_validate(results[0])


def _snapshot_error(provider: str) -> ErrorDetail:
    return ErrorDetail(
        code="snapshot_unavailable",
        message="sealed snapshot is unavailable",
        retryable=False,
        provider=provider,
    )


def _recall_controller() -> HardBudgetController:
    return HardBudgetController(
        SearchBudget(max_total_tokens=10_000, max_cost_cny=1.0), formal_live=False
    )


def _search_estimate() -> UsageEstimate:
    return UsageEstimate(search_api_calls=1, elapsed_ms=30_000)


def _llm_estimate() -> UsageEstimate:
    return UsageEstimate(llm_calls=1, input_tokens=2_000, output_tokens=1_000, elapsed_ms=30_000)


__all__ = [
    "PreparedRecallRun",
    "RecallTerminalError",
    "RecallRuntime",
    "RecallRuntimeFactory",
    "build_citation_handler",
    "build_deepseek_generator",
    "build_fixed_generator",
    "build_manual_generator",
    "build_handler_registry",
    "build_live_runtime",
    "build_replay_runtime",
    "build_text_handler",
    "compare_recall_artifacts",
    "compare_historical_replays",
    "build_title_handler",
    "generator_factories",
    "prepare_recall_run",
    "run_recall_experiment",
    "unavailable_runtime",
    "validate_pasted_actions",
    "write_prepared_contexts",
]
