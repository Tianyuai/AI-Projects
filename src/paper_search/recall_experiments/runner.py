"""Generic orchestration for isolated candidate-recall experiment components."""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from paper_search.recall_experiments.contracts import (
    CandidatePool,
    RecallGenerationContext,
    RetrievalActionResult,
    RetrievalExecutionContext,
)
from paper_search.recall_experiments.generation.base import GenerationResult, QueryGenerator
from paper_search.recall_experiments.validation import validate_action_batch


class _InputSource(Protocol):
    def load_queries(self, sample: object) -> object: ...


class _PreparedEvaluator(Protocol):
    def preflight(self, dataset: object) -> object: ...

    def evaluate(self, prepared: object, pools: Sequence[CandidatePool]) -> object: ...


class _Registry(Protocol):
    def resolve(self, action_type: object) -> object: ...


class _ActionHandler(Protocol):
    async def execute(
        self, action: object, context: RetrievalExecutionContext
    ) -> RetrievalActionResult: ...


class _PoolBuilder(Protocol):
    def build(self, query_id: str, action_results: Sequence[RetrievalActionResult]) -> CandidatePool: ...


class _Stages(Protocol):
    def apply(self, pool: CandidatePool, context: object) -> CandidatePool: ...


class _ArtifactWriter(Protocol):
    def start_run(
        self, run_id: str, *, recipe_lock: Mapping[str, object], sample_manifest: Mapping[str, object]
    ) -> object: ...

    def write_generation(self, *args: object, **kwargs: object) -> object: ...

    def write_retrieval(self, *args: object, **kwargs: object) -> object: ...

    def write_candidate_pool(self, *args: object, **kwargs: object) -> object: ...

    def write_report(self, report: Mapping[str, object]) -> object: ...


@dataclass(frozen=True)
class RecallExperimentRequest:
    """Composition-owned settings; the runner never selects a method or backend."""

    run_id: str
    sample: object
    recipe_lock: Mapping[str, object]
    sample_manifest: Mapping[str, object]
    allowed_actions: Collection[str]
    max_actions: int
    max_results_per_action: int
    repeat_count: int
    max_repeat_attempts: int | None = None
    provider_filters: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.repeat_count < 1:
            raise ValueError("repeat_count must be positive")
        if self.max_actions < 1 or self.max_results_per_action < 1:
            raise ValueError("retrieval limits must be positive")
        if self.max_attempts < self.repeat_count or self.max_attempts > 5:
            raise ValueError("max_repeat_attempts must be between repeat_count and 5")

    @property
    def max_attempts(self) -> int:
        return self.max_repeat_attempts if self.max_repeat_attempts is not None else self.repeat_count


@dataclass(frozen=True)
class RecallExperimentAttempt:
    attempt_id: str
    attempt_status: str
    valid_repeat_ordinal: int | None
    result: object | None = None


@dataclass(frozen=True)
class RecallExperimentResult:
    run_id: str
    attempts: tuple[RecallExperimentAttempt, ...]


class RecallExperimentRunner:
    """Execute injected components without inspecting method/provider identities."""

    def __init__(
        self,
        *,
        input_source: _InputSource,
        generator: QueryGenerator,
        registry: _Registry,
        pool_builder: _PoolBuilder,
        stages: _Stages,
        evaluator: _PreparedEvaluator,
        writer: _ArtifactWriter,
        event_sink: Callable[[str], None] | None = None,
    ) -> None:
        self._input_source = input_source
        self._generator = generator
        self._registry = registry
        self._pool_builder = pool_builder
        self._stages = stages
        self._evaluator = evaluator
        self._writer = writer
        self._event_sink = event_sink

    async def run(self, request: RecallExperimentRequest) -> RecallExperimentResult:
        dataset = self._input_source.load_queries(request.sample)
        self._event("load-and-verify-inputs")
        prepared = self._evaluator.preflight(dataset)
        self._event("evaluator-preflight-and-gold-seed-isolation")
        self._writer.start_run(
            request.run_id,
            recipe_lock=request.recipe_lock,
            sample_manifest=_sample_manifest(request.sample_manifest, dataset),
        )

        contexts = tuple(_generation_contexts(prepared))
        attempts: list[RecallExperimentAttempt] = []
        valid_repeat_ordinal = 0
        for ordinal in range(1, request.max_attempts + 1):
            if valid_repeat_ordinal == request.repeat_count:
                break
            attempt_id = f"attempt-{ordinal:02d}"
            pools, failed = await self._run_attempt(
                attempt_id, contexts, request, valid_repeat_ordinal + 1
            )
            if failed:
                attempts.append(
                    RecallExperimentAttempt(
                        attempt_id=attempt_id,
                        attempt_status="failed",
                        valid_repeat_ordinal=None,
                    )
                )
                continue
            self._event("evaluate-recall")
            result = self._evaluator.evaluate(prepared, pools)
            valid_repeat_ordinal += 1
            attempts.append(
                RecallExperimentAttempt(
                    attempt_id=attempt_id,
                    attempt_status="succeeded",
                    valid_repeat_ordinal=valid_repeat_ordinal,
                    result=result,
                )
            )

        completed = RecallExperimentResult(run_id=request.run_id, attempts=tuple(attempts))
        self._event("write-report")
        self._writer.write_report(_report_payload(completed))
        return completed

    async def _run_attempt(
        self,
        attempt_id: str,
        contexts: Sequence[RecallGenerationContext],
        request: RecallExperimentRequest,
        valid_repeat_ordinal: int,
    ) -> tuple[list[CandidatePool], bool]:
        pools: list[CandidatePool] = []
        for context in contexts:
            self._event("build-generation-context")
            generation = await self._generator.generate(context)
            self._event("generate-and-validate")
            _validate_generation(generation, context, request)
            self._event("write-generation")
            self._writer.write_generation(
                attempt_id,
                context.query_id,
                generation,
                attempt_status="succeeded",
                valid_repeat_ordinal=None,
            )

            self._event("build-gold-free-retrieval-context")
            retrieval_context = RetrievalExecutionContext(
                query_id=context.query_id,
                provider_filters=dict(request.provider_filters),
                max_results_per_action=request.max_results_per_action,
                seed_candidates=list(context.seed_candidates),
            )
            self._event("resolve-and-execute-actions")
            action_results: list[RetrievalActionResult] = []
            for action in generation.action_batch.actions:
                handler = self._registry.resolve(action.action_type)
                if not hasattr(handler, "execute"):
                    raise TypeError("registered retrieval handler lacks execute")
                result = await _execute(handler, action, retrieval_context)
                action_results.append(result)
            infrastructure_failure = any(result.infrastructure_failure for result in action_results)
            self._event("write-retrieval")
            self._writer.write_retrieval(
                attempt_id,
                context.query_id,
                {"results": [_payload(result) for result in action_results]},
                attempt_status="failed" if infrastructure_failure else "succeeded",
                valid_repeat_ordinal=None,
                errors=[_payload(error) for result in action_results for error in result.errors],
            )
            if infrastructure_failure:
                return pools, True

            self._event("build-candidate-pool")
            pool = self._pool_builder.build(context.query_id, action_results)
            self._event("apply-empty-stages")
            pool = self._stages.apply(pool, retrieval_context)
            self._event("write-candidate-pool")
            self._writer.write_candidate_pool(
                attempt_id,
                context.query_id,
                _payload(pool),
                attempt_status="succeeded",
                valid_repeat_ordinal=valid_repeat_ordinal,
            )
            pools.append(pool)
        return pools, False

    def _event(self, name: str) -> None:
        if self._event_sink is not None:
            self._event_sink(name)


async def _execute(
    handler: object, action: object, context: RetrievalExecutionContext
) -> RetrievalActionResult:
    return await _handler(handler).execute(action, context)


def _handler(value: object) -> _ActionHandler:
    if not hasattr(value, "execute"):
        raise TypeError("registered retrieval handler lacks execute")
    return value


def _generation_contexts(prepared: object) -> Sequence[RecallGenerationContext]:
    contexts = getattr(prepared, "generation_contexts", None)
    if not isinstance(contexts, tuple) or not all(isinstance(item, RecallGenerationContext) for item in contexts):
        raise TypeError("evaluator preflight did not return generation-safe contexts")
    return contexts


def _validate_generation(
    generation: GenerationResult, context: RecallGenerationContext, request: RecallExperimentRequest
) -> None:
    if generation.query_id != context.query_id:
        raise ValueError("generation result query ID does not match context")
    validated = validate_action_batch(
        generation.action_batch.model_dump(mode="json"),
        context,
        allowed_actions=request.allowed_actions,
        max_actions=request.max_actions,
    )
    if validated != generation.action_batch:
        raise ValueError("generation result must already be normalized")


def _sample_manifest(manifest: Mapping[str, object], dataset: object) -> dict[str, object]:
    payload = dict(manifest)
    source_hashes = getattr(dataset, "source_hashes", None)
    if isinstance(source_hashes, dict):
        payload["input_hashes"] = source_hashes
    return payload


def _payload(value: object) -> object:
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    return value


def _report_payload(result: RecallExperimentResult) -> dict[str, object]:
    return {
        "run_id": result.run_id,
        "attempts": [
            {
                "attempt_id": attempt.attempt_id,
                "attempt_status": attempt.attempt_status,
                "valid_repeat_ordinal": attempt.valid_repeat_ordinal,
                "result": _payload(attempt.result),
            }
            for attempt in result.attempts
        ],
    }


__all__ = [
    "RecallExperimentAttempt",
    "RecallExperimentRequest",
    "RecallExperimentResult",
    "RecallExperimentRunner",
]
