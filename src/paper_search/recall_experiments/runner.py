"""Generic orchestration for isolated candidate-recall experiment components."""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol

from paper_search.recall_experiments.contracts import (
    CandidatePool,
    RecallGenerationContext,
    RetrievalActionResult,
    RetrievalExecutionContext,
    SeedCandidate,
)
from paper_search.recall_experiments.generation.base import (
    EvidenceSteeredQueryGenerator,
    GenerationResult,
    QueryGenerator,
)
from paper_search.recall_experiments.generation.deepseek import RecallGenerationFailure
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
    execution_identity: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.repeat_count < 1:
            raise ValueError("repeat_count must be positive")
        if self.repeat_count > 3:
            raise ValueError("repeat_count must not exceed 3")
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
    attempt_status: Literal["succeeded", "failed"]
    valid_repeat_ordinal: int | None
    result: object | None = None
    failure_code: str | None = None
    generation_provenance: tuple[Mapping[str, object], ...] = ()

    def __post_init__(self) -> None:
        if self.attempt_status not in {"succeeded", "failed"}:
            raise ValueError("attempt_status must be succeeded or failed")
        if (self.attempt_status == "failed") != (self.failure_code is not None):
            raise ValueError("failed attempt status and failure_code must be recorded together")
        if self.valid_repeat_ordinal is not None and (
            type(self.valid_repeat_ordinal) is not int
            or not 1 <= self.valid_repeat_ordinal <= 3
        ):
            raise ValueError("valid_repeat_ordinal must be None or an integer from 1 through 3")


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
            pools, failure_code, generation_provenance = await self._run_attempt(
                attempt_id, contexts, request, valid_repeat_ordinal + 1
            )
            if failure_code is not None:
                attempts.append(
                    RecallExperimentAttempt(
                        attempt_id=attempt_id,
                        attempt_status="failed",
                        valid_repeat_ordinal=None,
                        failure_code=failure_code,
                        generation_provenance=tuple(generation_provenance),
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
                    generation_provenance=tuple(generation_provenance),
                )
            )

        completed = RecallExperimentResult(run_id=request.run_id, attempts=tuple(attempts))
        self._event("write-report")
        self._writer.write_report(_report_payload(completed, request.execution_identity))
        return completed

    async def _run_attempt(
        self,
        attempt_id: str,
        contexts: Sequence[RecallGenerationContext],
        request: RecallExperimentRequest,
        valid_repeat_ordinal: int,
    ) -> tuple[list[CandidatePool], str | None, list[Mapping[str, object]]]:
        pools: list[CandidatePool] = []
        generation_provenance: list[Mapping[str, object]] = []
        for context in contexts:
            self._event("build-generation-context")
            try:
                generation = await self._generator.generate(context)
            except RecallGenerationFailure as failure:
                self._event("generate-and-validate")
                self._event("write-generation")
                self._writer.write_generation(
                    attempt_id,
                    context.query_id,
                    {
                        "failure_code": failure.code,
                        "errors": [_payload(error) for error in failure.errors],
                        "llm_call_receipts": [
                            _payload(receipt) for receipt in failure.call_receipts
                        ],
                        "repair_count": sum(
                            receipt.call_kind == "repair" for receipt in failure.call_receipts
                        ),
                    },
                    attempt_status="failed",
                    valid_repeat_ordinal=None,
                )
                generation_provenance.append(
                    {
                        "query_id": context.query_id,
                        "failure_code": failure.code,
                        "llm_call_receipts": [
                            _payload(receipt) for receipt in failure.call_receipts
                        ],
                        "repair_count": sum(
                            receipt.call_kind == "repair" for receipt in failure.call_receipts
                        ),
                    }
                )
                return pools, failure.code, generation_provenance
            self._event("generate-and-validate")
            _validate_generation(generation, context, request)
            evidence_generator = (
                self._generator
                if isinstance(self._generator, EvidenceSteeredQueryGenerator)
                else None
            )
            if evidence_generator is None:
                generation_provenance.append(_generation_provenance(generation))
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
            action_results = await self._execute_actions(
                generation.action_batch.actions, retrieval_context
            )
            if evidence_generator is not None and not any(
                result.infrastructure_failure for result in action_results
            ):
                anchor_generation = generation
                refinement_seeds = _evidence_seeds(
                    context.seed_candidates, action_results
                )
                refinement_context = context.model_copy(
                    update={"seed_candidates": refinement_seeds}
                )
                retrieval_context = retrieval_context.model_copy(
                    update={"seed_candidates": refinement_seeds}
                )
                try:
                    generation = await evidence_generator.refine(
                        refinement_context, anchor_generation, action_results
                    )
                except RecallGenerationFailure as failure:
                    self._event("refine-and-validate")
                    receipts = [*anchor_generation.call_receipts, *failure.call_receipts]
                    generation = anchor_generation.model_copy(
                        update={
                            "provenance": {
                                **anchor_generation.provenance,
                                "refinement_fallback": failure.code,
                            },
                            "call_receipts": receipts,
                            "repair_count": sum(
                                receipt.call_kind == "repair" for receipt in receipts
                            ),
                        }
                    )
                else:
                    self._event("refine-and-validate")
                    _validate_generation(generation, refinement_context, request)
                    _validate_anchor_prefix(anchor_generation, generation)
                    remaining_actions = generation.action_batch.actions[
                        len(anchor_generation.action_batch.actions) :
                    ]
                    action_results.extend(
                        await self._execute_actions(remaining_actions, retrieval_context)
                    )
            if evidence_generator is not None:
                generation_provenance.append(_generation_provenance(generation))
                self._event("write-generation")
                self._writer.write_generation(
                    attempt_id,
                    context.query_id,
                    generation,
                    attempt_status="succeeded",
                    valid_repeat_ordinal=None,
                )
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
                failure_code = _retrieval_failure_code(action_results)
                return pools, failure_code, generation_provenance

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
        return pools, None, generation_provenance

    async def _execute_actions(
        self, actions: Sequence[object], context: RetrievalExecutionContext
    ) -> list[RetrievalActionResult]:
        results: list[RetrievalActionResult] = []
        for action in actions:
            handler = self._registry.resolve(getattr(action, "action_type"))
            if not hasattr(handler, "execute"):
                raise TypeError("registered retrieval handler lacks execute")
            results.append(await _execute(handler, action, context))
        return results

    def _event(self, name: str) -> None:
        if self._event_sink is not None:
            self._event_sink(name)


async def _execute(
    handler: object, action: object, context: RetrievalExecutionContext
) -> RetrievalActionResult:
    result = await _handler(handler).execute(action, context)
    if (
        result.action_id != getattr(action, "action_id", None)
        or result.action_type != getattr(action, "action_type", None)
    ):
        raise ValueError("retrieval result does not match executed action")
    return result


def _handler(value: object) -> _ActionHandler:
    if not hasattr(value, "execute"):
        raise TypeError("registered retrieval handler lacks execute")
    return value


def _generation_contexts(prepared: object) -> Sequence[RecallGenerationContext]:
    contexts = getattr(prepared, "generation_contexts", None)
    if not isinstance(contexts, tuple) or not all(isinstance(item, RecallGenerationContext) for item in contexts):
        raise TypeError("evaluator preflight did not return generation-safe contexts")
    return contexts


def _evidence_seeds(
    configured: Sequence[SeedCandidate],
    results: Sequence[RetrievalActionResult],
) -> list[SeedCandidate]:
    selected = list(configured)
    seen = {item.paper.canonical_id for item in selected}
    for result in results:
        for paper in result.hits:
            if paper.canonical_id in seen:
                continue
            seen.add(paper.canonical_id)
            selected.append(SeedCandidate(paper=paper))
    return selected


def _validate_generation(
    generation: GenerationResult, context: RecallGenerationContext, request: RecallExperimentRequest
) -> None:
    if generation.query_id != context.query_id:
        raise ValueError("generation result query ID does not match context")
    try:
        artifact_text = generation.artifact_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("generation artifact bytes must be UTF-8") from error
    validated = validate_action_batch(
        artifact_text, context, allowed_actions=request.allowed_actions, max_actions=request.max_actions
    )
    if validated != generation.action_batch:
        raise ValueError("generation artifact bytes do not match the action batch")


def _validate_anchor_prefix(
    anchor: GenerationResult, refined: GenerationResult
) -> None:
    anchor_actions = anchor.action_batch.actions
    refined_actions = refined.action_batch.actions
    if len(refined_actions) < len(anchor_actions) or refined_actions[: len(anchor_actions)] != anchor_actions:
        raise ValueError("refined generation must preserve the immutable anchor action prefix")


def _generation_provenance(generation: GenerationResult) -> Mapping[str, object]:
    return {
        "query_id": generation.query_id,
        **generation.provenance,
        "llm_call_receipts": [
            _payload(receipt) for receipt in generation.call_receipts
        ],
        "repair_count": generation.repair_count,
    }


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


def _report_payload(
    result: RecallExperimentResult, execution_identity: Mapping[str, object]
) -> dict[str, object]:
    return {
        "schema_version": "candidate-recall-report-v1",
        "run_id": result.run_id,
        "execution_identity": dict(execution_identity),
        "attempts": [
            {
                "attempt_id": attempt.attempt_id,
                "attempt_status": attempt.attempt_status,
                "valid_repeat_ordinal": attempt.valid_repeat_ordinal,
                "result": _payload(attempt.result),
                "failure_code": attempt.failure_code,
                "generation_provenance": [dict(item) for item in attempt.generation_provenance],
            }
            for attempt in result.attempts
        ],
    }


def _retrieval_failure_code(results: Sequence[RetrievalActionResult]) -> str:
    codes = [error.code for result in results for error in result.errors]
    if codes and all(code == "snapshot_unavailable" for code in codes):
        return "snapshot_unavailable"
    return "retrieval_infrastructure_failure"


__all__ = [
    "RecallExperimentAttempt",
    "RecallExperimentRequest",
    "RecallExperimentResult",
    "RecallExperimentRunner",
]
