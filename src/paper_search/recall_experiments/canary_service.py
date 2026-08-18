"""One reusable execution service for scored and unscored recall canaries."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence
from typing import Any, cast

from paper_search.domain.models import QuerySpec, UsageActual
from paper_search.evaluation.dataset import IdentifierMap
from paper_search.recall_experiments.artifacts import RecallArtifactWriter
from paper_search.recall_experiments.candidate_pool import CandidatePoolBuilder
from paper_search.recall_experiments.canary_inputs import LoadedCanaryInput
from paper_search.recall_experiments.canary_reporting import (
    CanaryExecutionIdentity,
    CanaryInputIdentity,
    CanaryPerQueryResult,
    CanaryRecallResult,
    CanaryReport,
    compare_canary_results,
)
from paper_search.recall_experiments.composition import (
    build_deepseek_generator,
    build_fixed_generator,
    build_handler_registry,
    build_manual_generator,
    validate_live_runtime,
)
from paper_search.recall_experiments.contracts import CandidatePool, RecallGenerationContext
from paper_search.recall_experiments.generation.base import QueryGenerator
from paper_search.recall_experiments.paper_identity import EvidenceDrivenIdentifierResolver
from paper_search.recall_experiments.recipes import (
    DeepSeekPromptGeneratorRecipe,
    FixedActionsGeneratorRecipe,
    ManualActionsGeneratorRecipe,
    LoadedRecallRecipe,
)
from paper_search.recall_experiments.canary_runtime import RecallLiveRuntimeBundle
from paper_search.recall_experiments.runner import (
    RecallExperimentRequest,
    RecallExperimentRunner,
)
from paper_search.recall_experiments.stages import CandidateStagePipeline


@dataclass(frozen=True)
class _PreparedCanary:
    generation_contexts: tuple[RecallGenerationContext, ...]
    loaded_input: LoadedCanaryInput


class _InputSource:
    def load_queries(self, sample: object) -> object:
        if not isinstance(sample, LoadedCanaryInput):
            raise TypeError("canary sample must be LoadedCanaryInput")
        return sample


class _Evaluator:
    def __init__(self, policy_version: str) -> None:
        self._policy_version = policy_version

    def preflight(self, dataset: object) -> _PreparedCanary:
        if not isinstance(dataset, LoadedCanaryInput):
            raise TypeError("canary dataset must be LoadedCanaryInput")
        contexts = tuple(
            RecallGenerationContext(
                query_id=case.query_id,
                original_query=case.query,
                query_spec=QuerySpec(original_query=case.query, research_goal=case.query),
            )
            for case in dataset.cases
        )
        return _PreparedCanary(generation_contexts=contexts, loaded_input=dataset)

    def evaluate(
        self, prepared: object, pools: Sequence[CandidatePool]
    ) -> CanaryRecallResult:
        if not isinstance(prepared, _PreparedCanary):
            raise TypeError("canary evaluator received an invalid prepared input")
        pool_by_query = {pool.query_id: pool for pool in pools}
        if set(pool_by_query) != {case.query_id for case in prepared.loaded_input.cases}:
            raise ValueError("candidate pools do not cover the canary query set")
        resolver = self._identifier_resolver(prepared.loaded_input)
        rows: list[CanaryPerQueryResult] = []
        for case in prepared.loaded_input.cases:
            pool = pool_by_query[case.query_id]
            candidate_ids = [entry.paper.canonical_id for entry in pool.entries]
            if case.gold_paper_ids is None:
                rows.append(
                    CanaryPerQueryResult(
                        query_id=case.query_id,
                        candidate_pool_ids=candidate_ids,
                        candidate_count=len(candidate_ids),
                        evaluation_status="not_available",
                        gold_hit_ids=[],
                    )
                )
                continue
            assert resolver is not None
            resolved_candidates = frozenset().union(
                *(resolver.paper_identities(entry.paper) for entry in pool.entries)
            )
            resolved_gold = [resolver.resolve(item) for item in case.gold_paper_ids]
            hits = sorted(set(resolved_gold).intersection(resolved_candidates))
            rows.append(
                CanaryPerQueryResult(
                    query_id=case.query_id,
                    candidate_pool_ids=candidate_ids,
                    candidate_count=len(candidate_ids),
                    evaluation_status="available",
                    gold_hit_ids=hits,
                    gold_association_count=len(resolved_gold),
                    gold_hit_count=len(hits),
                    candidate_recall=len(hits) / len(resolved_gold),
                )
            )
        if prepared.loaded_input.evaluation_status == "not_available":
            return CanaryRecallResult(
                candidate_pool_policy_version=self._policy_version,
                evaluation_status="not_available",
                per_query=rows,
            )
        association_count = sum(row.gold_association_count or 0 for row in rows)
        hit_count = sum(row.gold_hit_count or 0 for row in rows)
        return CanaryRecallResult(
            candidate_pool_policy_version=self._policy_version,
            evaluation_status="available",
            per_query=rows,
            gold_association_count=association_count,
            gold_hit_count=hit_count,
            macro_candidate_recall=sum(row.candidate_recall or 0 for row in rows) / len(rows),
        )

    @staticmethod
    def _identifier_resolver(
        loaded: LoadedCanaryInput,
    ) -> EvidenceDrivenIdentifierResolver | None:
        if loaded.evaluation_status == "not_available":
            return None
        if loaded.identifier_map_bytes is None:
            raise ValueError("scored canary input lacks verified identifier-map bytes")
        identifier_map = IdentifierMap.from_bytes(
            loaded.identifier_map_bytes, source="canary identifier map"
        )
        return EvidenceDrivenIdentifierResolver(identifier_map)


class RecallCanaryService:
    """Connect inputs and a recipe to the existing generic Scheme-B runner."""

    def __init__(self, *, workspace_root: Path) -> None:
        self._workspace_root = workspace_root.resolve()

    async def run(
        self,
        *,
        loaded_recipe: LoadedRecallRecipe,
        loaded_input: LoadedCanaryInput,
        runtime_bundle: RecallLiveRuntimeBundle,
        output_path: Path,
        baseline_report_path: Path | None = None,
        generator_override: QueryGenerator | None = None,
    ) -> CanaryReport:
        recipe = loaded_recipe.recipe
        runtime_bundle.validate_capability()
        runtime = runtime_bundle.runtime
        if recipe.retrieval.backend != "live_provider":
            raise ValueError("canary service requires a verified live runtime")
        validate_live_runtime(runtime, recipe)
        if recipe.generator.gold_visibility != "blind":
            raise ValueError("the unified canary path requires blind generation")
        if recipe.evaluation.repeat_count != 1 or recipe.evaluation.max_repeat_attempts != 1:
            raise ValueError("quick canary reports require exactly one attempt")
        contexts = tuple(
            RecallGenerationContext(
                query_id=case.query_id,
                original_query=case.query,
                query_spec=QuerySpec(original_query=case.query, research_goal=case.query),
            )
            for case in loaded_input.cases
        )
        generator_recipe = recipe.generator
        generator: QueryGenerator
        if generator_override is not None:
            generator = generator_override
        elif isinstance(generator_recipe, FixedActionsGeneratorRecipe):
            generator = build_fixed_generator(
                self._workspace_root / generator_recipe.actions,
                contexts=contexts,
                recipe=recipe,
            )
        elif isinstance(generator_recipe, ManualActionsGeneratorRecipe):
            generator = build_manual_generator(
                self._workspace_root / generator_recipe.actions,
                contexts=contexts,
                recipe=recipe,
            )
        elif isinstance(generator_recipe, DeepSeekPromptGeneratorRecipe):
            if loaded_recipe.prompt_bytes is None:
                raise ValueError("DeepSeek recipe lacks bound prompt bytes")
            generator = build_deepseek_generator(
                contexts=contexts,
                recipe=generator_recipe,
                prompt_bytes=loaded_recipe.prompt_bytes,
                allowed_actions=recipe.retrieval.allowed_actions,
                backend=runtime.llm_backend,
            )
        else:  # pragma: no cover - closed recipe union
            raise TypeError("unsupported generator recipe")
        evaluator = _Evaluator(recipe.candidate_pool.policy_version)
        writer = RecallArtifactWriter(output_path.parent)
        runner = RecallExperimentRunner(
            input_source=_InputSource(),
            generator=generator,
            registry=cast(Any, build_handler_registry(runtime)),
            pool_builder=CandidatePoolBuilder(recipe.candidate_pool.policy_version),
            stages=CandidateStagePipeline(),
            evaluator=evaluator,
            writer=cast(Any, writer),
        )
        actions_sha256 = getattr(generator, "source_sha256", None)
        provisional_identity = {
            "identity_schema_version": "recall-canary-provisional-identity-v1",
            "method_id": recipe.method_id,
            "recipe_sha256": loaded_recipe.recipe_sha256,
            "input_sha256": loaded_input.input_sha256,
        }
        completed = await runner.run(
            RecallExperimentRequest(
                run_id=output_path.name,
                sample=loaded_input,
                recipe_lock={
                    "method_id": recipe.method_id,
                    "recipe_sha256": loaded_recipe.recipe_sha256,
                },
                sample_manifest={
                    "input_kind": loaded_input.input_kind,
                    "input_sha256": loaded_input.input_sha256,
                    "query_ids": [case.query_id for case in loaded_input.cases],
                },
                allowed_actions=recipe.retrieval.allowed_actions,
                max_actions=recipe.retrieval.max_total_actions,
                max_results_per_action=recipe.retrieval.max_results_per_action,
                repeat_count=recipe.evaluation.repeat_count,
                max_repeat_attempts=recipe.evaluation.max_repeat_attempts,
                execution_identity=provisional_identity,
            )
        )
        succeeded = [attempt for attempt in completed.attempts if attempt.result is not None]
        if not succeeded or not isinstance(succeeded[-1].result, CanaryRecallResult):
            raise RuntimeError("canary produced no valid repeat")
        result = succeeded[-1].result
        actions_by_query = _load_actions(output_path, succeeded[-1].attempt_id)
        snapshot_manifest_sha256, snapshot_set_id = await runtime_bundle.seal()
        override_type = getattr(generator, "generator_type", None)
        generator_type = override_type or recipe.generator.type
        execution_identity = CanaryExecutionIdentity(
            identity_schema_version="recall-canary-execution-identity-v1",
            method_id=recipe.method_id,
            recipe_sha256=loaded_recipe.recipe_sha256,
            input_sha256=loaded_input.input_sha256,
            identifier_map_sha256=loaded_input.identifier_map_sha256,
            generator_type=generator_type,
            generator_model=(
                getattr(generator, "model_id", None)
                if override_type is not None
                else getattr(recipe.generator, "model", None)
            ),
            prompt_sha256=(
                loaded_recipe.prompt_sha256
                if generator_type in {"deepseek_prompt", "local_cpu_fallback"}
                else None
            ),
            actions_sha256=actions_sha256,
            allowed_actions=tuple(recipe.retrieval.allowed_actions),
            max_total_actions=recipe.retrieval.max_total_actions,
            max_results_per_action=recipe.retrieval.max_results_per_action,
            candidate_pool_policy_version=recipe.candidate_pool.policy_version,
            runtime=runtime.identity,
            snapshot_manifest_sha256=snapshot_manifest_sha256,
            snapshot_set_id=snapshot_set_id,
        )
        comparison = None
        if baseline_report_path is not None:
            baseline = CanaryReport.model_validate_json(baseline_report_path.read_bytes())
            comparison = compare_canary_results(
                result,
                baseline.result,
                identities_match=_comparison_identity(baseline.execution_identity)
                == _comparison_identity(execution_identity),
            )
        report = CanaryReport(
            run_id=completed.run_id,
            input=CanaryInputIdentity(
                input_kind=loaded_input.input_kind,
                input_sha256=loaded_input.input_sha256,
                evaluation_status=loaded_input.evaluation_status,
                query_ids=tuple(case.query_id for case in loaded_input.cases),
            ),
            execution_identity=execution_identity,
            actions_by_query=actions_by_query,
            usage=runtime.controller.committed_usage if runtime.controller is not None else UsageActual(),
            result=result,
            comparison=comparison,
        )
        writer.write_canary_report(report.model_dump(mode="json"))
        return report


def _load_actions(output_path: Path, attempt_id: str) -> dict[str, list[dict[str, object]]]:
    directory = output_path / "generation" / attempt_id
    result: dict[str, list[dict[str, object]]] = {}
    for path in sorted(directory.glob("*.json")):
        payload = json.loads(path.read_bytes())
        actions = payload.get("actions") if isinstance(payload, dict) else None
        query_id = payload.get("query_id") if isinstance(payload, dict) else None
        if not isinstance(query_id, str) or not isinstance(actions, list) or not all(
            isinstance(action, dict) for action in actions
        ):
            raise ValueError("generation artifact does not contain a valid action list")
        result[query_id] = actions
    return result


def _comparison_identity(identity: CanaryExecutionIdentity) -> dict[str, object]:
    return identity.model_dump(
        mode="json", exclude={"snapshot_manifest_sha256", "snapshot_set_id"}
    )


__all__ = ["RecallCanaryService"]
