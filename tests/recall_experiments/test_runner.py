from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from paper_search.domain.models import Paper, QuerySpec
from paper_search.recall_experiments.contracts import (
    CandidatePool,
    RecallActionBatch,
    RecallGenerationContext,
    RetrievalActionResult,
)
from paper_search.recall_experiments.generation.base import GenerationResult
from paper_search.recall_experiments.generation.fixed import FixedActionGenerator
from paper_search.recall_experiments.generation.deepseek import RecallGenerationFailure
from paper_search.recall_experiments.generation.manual import ManualActionGenerator
from paper_search.recall_experiments.runner import (
    RecallExperimentAttempt,
    RecallExperimentRequest,
    RecallExperimentRunner,
)


class _InputSource:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def load_queries(self, sample: object) -> object:
        return object()


class _Evaluator:
    def __init__(self, events: list[str], context: RecallGenerationContext) -> None:
        self.events = events
        self.context = context

    def preflight(self, dataset: object) -> object:
        return SimpleNamespace(generation_contexts=(self.context,))

    def evaluate(self, prepared: object, pools: list[CandidatePool]) -> object:
        return {"pool_count": len(pools)}


class _Generator:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    async def generate(self, context: RecallGenerationContext) -> GenerationResult:
        batch = RecallActionBatch.model_validate(
            {
                "actions": [
                    {
                        "action_id": "search-1",
                        "action_type": "text_search",
                        "strategy": "existing-action",
                        "payload": {"query_text": "graph retrieval"},
                    }
                ]
            }
        )
        return GenerationResult(
            query_id=context.query_id,
            action_batch=batch,
            artifact_bytes=batch.model_dump_json().encode("utf-8"),
        )


class _Registry:
    def __init__(self, events: list[str], handler: object | None) -> None:
        self.events = events
        self.handler = handler

    def resolve(self, action_type: str) -> object:
        if self.handler is None:
            raise KeyError(action_type)
        return self.handler


class _Handler:
    async def execute(self, action: object, context: object) -> RetrievalActionResult:
        assert not hasattr(context, "gold_documents")
        return RetrievalActionResult(
            action_id="search-1",
            action_type="text_search",
            hits=[Paper(canonical_id="doi:10.1000/hit", title="Hit", sources=["openalex"])],
        )


class _PoolBuilder:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def build(self, query_id: str, results: list[RetrievalActionResult]) -> CandidatePool:
        return CandidatePool(query_id=query_id, policy_version="production-dedup-v1")


class _Stages:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def apply(self, pool: CandidatePool, context: object) -> CandidatePool:
        return pool


class _Writer:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.retrieval_writes = 0

    def start_run(self, *args: object, **kwargs: object) -> None:
        return None

    def write_generation(self, *args: object, **kwargs: object) -> None:
        return None

    def write_retrieval(self, *args: object, **kwargs: object) -> None:
        self.retrieval_writes += 1
        return None

    def write_candidate_pool(self, *args: object, **kwargs: object) -> None:
        return None

    def write_report(self, *args: object, **kwargs: object) -> None:
        return None


def _request() -> RecallExperimentRequest:
    return RecallExperimentRequest(
        run_id="run-01",
        sample=object(),
        recipe_lock={},
        sample_manifest={},
        allowed_actions={"text_search"},
        max_actions=1,
        max_results_per_action=5,
        repeat_count=1,
    )


def _context() -> RecallGenerationContext:
    return RecallGenerationContext(
        query_id="query-1",
        original_query="graph retrieval",
        query_spec=QuerySpec(original_query="graph retrieval", research_goal="graph retrieval"),
        gold_documents=[],
    )


def test_runner_follows_the_gold_safe_branch_free_event_order() -> None:
    events: list[str] = []
    writer = _Writer(events)
    runner = RecallExperimentRunner(
        input_source=_InputSource(events),
        generator=_Generator(events),
        registry=_Registry(events, _Handler()),
        pool_builder=_PoolBuilder(events),
        stages=_Stages(events),
        evaluator=_Evaluator(events, _context()),
        writer=writer,
        event_sink=events.append,
    )

    result = asyncio.run(runner.run(_request()))

    assert result.attempts[0].attempt_id == "attempt-01"
    assert events == [
        "load-and-verify-inputs",
        "evaluator-preflight-and-gold-seed-isolation",
        "build-generation-context",
        "generate-and-validate",
        "write-generation",
        "build-gold-free-retrieval-context",
        "resolve-and-execute-actions",
        "write-retrieval",
        "build-candidate-pool",
        "apply-empty-stages",
        "write-candidate-pool",
        "evaluate-recall",
        "write-report",
    ]


def test_runner_refines_only_after_anchor_results_and_executes_the_combined_batch() -> None:
    events: list[str] = []
    executed_queries: list[str] = []
    refinement_titles: list[str] = []

    class EvidenceSteeredGenerator(_Generator):
        async def refine(
            self,
            context: RecallGenerationContext,
            anchor_generation: GenerationResult,
            first_round_results: list[RetrievalActionResult],
        ) -> GenerationResult:
            assert executed_queries == ["graph retrieval"]
            assert [action.action_id for action in anchor_generation.action_batch.actions] == [
                "search-1"
            ]
            refinement_titles.extend(
                paper.title for result in first_round_results for paper in result.hits
            )
            batch = RecallActionBatch.model_validate(
                {
                    "actions": [
                        anchor_generation.action_batch.actions[0].model_dump(mode="json"),
                        {
                            "action_id": "search-2",
                            "action_type": "text_search",
                            "strategy": "first-round-evidence",
                            "payload": {"query_text": "dense graph retrieval"},
                        },
                    ]
                }
            )
            return GenerationResult(
                query_id=context.query_id,
                action_batch=batch,
                artifact_bytes=batch.model_dump_json().encode("utf-8"),
            )

    class RecordingHandler:
        async def execute(self, action: object, context: object) -> RetrievalActionResult:
            del context
            query_text = str(getattr(getattr(action, "payload"), "query_text"))
            executed_queries.append(query_text)
            hits = (
                [
                    Paper(
                        canonical_id="doi:10.1000/anchor-hit",
                        title="Dense retrieval for graphs",
                        sources=["openalex"],
                    )
                ]
                if len(executed_queries) == 1
                else []
            )
            return RetrievalActionResult(
                action_id=str(getattr(action, "action_id")),
                action_type="text_search",
                hits=hits,
            )

    request = RecallExperimentRequest(
        **{**_request().__dict__, "max_actions": 3}
    )
    runner = RecallExperimentRunner(
        input_source=_InputSource(events),
        generator=EvidenceSteeredGenerator(events),
        registry=_Registry(events, RecordingHandler()),
        pool_builder=_PoolBuilder(events),
        stages=_Stages(events),
        evaluator=_Evaluator(events, _context()),
        writer=_Writer(events),
    )

    asyncio.run(runner.run(request))

    assert refinement_titles == ["Dense retrieval for graphs"]
    assert executed_queries == ["graph retrieval", "dense graph retrieval"]


def test_runner_exposes_first_round_hits_as_gold_free_refinement_seeds() -> None:
    events: list[str] = []
    seed = Paper(
        canonical_id="openalex:W1234567890",
        title="Dense retrieval for graphs",
        openalex_id="W1234567890",
        sources=["openalex"],
    )

    class GraphGenerator(_Generator):
        async def refine(
            self,
            context: RecallGenerationContext,
            anchor_generation: GenerationResult,
            first_round_results: list[RetrievalActionResult],
        ) -> GenerationResult:
            del first_round_results
            assert [item.paper for item in context.seed_candidates] == [seed]
            batch = RecallActionBatch.model_validate(
                {
                    "actions": [
                        anchor_generation.action_batch.actions[0].model_dump(mode="json"),
                        {
                            "action_id": "graph-1",
                            "action_type": "citation_expand",
                            "strategy": "structured-graph:references",
                            "payload": {
                                "seed_canonical_id": seed.canonical_id,
                                "direction": "references",
                                "limit": 5,
                            },
                        },
                    ]
                }
            )
            return GenerationResult(
                query_id=context.query_id,
                action_batch=batch,
                artifact_bytes=batch.model_dump_json().encode("utf-8"),
            )

    class GraphHandler:
        async def execute(
            self, action: object, context: object
        ) -> RetrievalActionResult:
            action_id = str(getattr(action, "action_id"))
            action_type = str(getattr(action, "action_type"))
            if action_type == "text_search":
                return RetrievalActionResult(
                    action_id=action_id,
                    action_type="text_search",
                    hits=[seed],
                )
            assert [item.paper for item in getattr(context, "seed_candidates")] == [
                seed
            ]
            return RetrievalActionResult(
                action_id=action_id,
                action_type="citation_expand",
            )

    request = RecallExperimentRequest(
        **{
            **_request().__dict__,
            "allowed_actions": {"text_search", "citation_expand"},
            "max_actions": 2,
        }
    )
    runner = RecallExperimentRunner(
        input_source=_InputSource(events),
        generator=GraphGenerator(events),
        registry=_Registry(events, GraphHandler()),
        pool_builder=_PoolBuilder(events),
        stages=_Stages(events),
        evaluator=_Evaluator(events, _context()),
        writer=_Writer(events),
    )

    result = asyncio.run(runner.run(request))

    assert result.attempts[0].attempt_status == "succeeded"


def test_runner_keeps_anchor_when_optional_refinement_fails() -> None:
    events: list[str] = []
    executed_queries: list[str] = []

    class FailingRefinementGenerator(_Generator):
        async def refine(
            self,
            context: RecallGenerationContext,
            anchor_generation: GenerationResult,
            first_round_results: list[RetrievalActionResult],
        ) -> GenerationResult:
            del context, anchor_generation, first_round_results
            raise RecallGenerationFailure("generation_failure", [])

    class RecordingHandler:
        async def execute(self, action: object, context: object) -> RetrievalActionResult:
            del context
            query_text = str(getattr(getattr(action, "payload"), "query_text"))
            executed_queries.append(query_text)
            return RetrievalActionResult(
                action_id=str(getattr(action, "action_id")),
                action_type="text_search",
                hits=[
                    Paper(
                        canonical_id="doi:10.1000/anchor-hit",
                        title="Anchor hit",
                        sources=["openalex"],
                    )
                ],
            )

    runner = RecallExperimentRunner(
        input_source=_InputSource(events),
        generator=FailingRefinementGenerator(events),
        registry=_Registry(events, RecordingHandler()),
        pool_builder=_PoolBuilder(events),
        stages=_Stages(events),
        evaluator=_Evaluator(events, _context()),
        writer=_Writer(events),
    )

    result = asyncio.run(runner.run(_request()))

    assert result.attempts[0].attempt_status == "succeeded"
    assert executed_queries == ["graph retrieval"]
    provenance = result.attempts[0].generation_provenance[0]
    assert provenance["refinement_fallback"] == "generation_failure"


def test_runner_rejects_retrieval_result_mislabeled_for_executed_action() -> None:
    events: list[str] = []

    class MislabeledHandler:
        async def execute(self, action: object, context: object) -> RetrievalActionResult:
            del action, context
            return RetrievalActionResult(
                action_id="different-action",
                action_type="text_search",
                hits=[],
            )

    runner = RecallExperimentRunner(
        input_source=_InputSource(events),
        generator=_Generator(events),
        registry=_Registry(events, MislabeledHandler()),
        pool_builder=_PoolBuilder(events),
        stages=_Stages(events),
        evaluator=_Evaluator(events, _context()),
        writer=_Writer(events),
    )

    with pytest.raises(ValueError, match="retrieval result does not match executed action"):
        asyncio.run(runner.run(_request()))


def test_unknown_handler_fails_before_a_retrieval_artifact_is_written() -> None:
    events: list[str] = []
    writer = _Writer(events)
    runner = RecallExperimentRunner(
        input_source=_InputSource(events),
        generator=_Generator(events),
        registry=_Registry(events, None),
        pool_builder=_PoolBuilder(events),
        stages=_Stages(events),
        evaluator=_Evaluator(events, _context()),
        writer=writer,
        event_sink=events.append,
    )

    with pytest.raises(KeyError):
        asyncio.run(runner.run(_request()))

    assert writer.retrieval_writes == 0


def test_runner_writes_completed_phase_statuses_instead_of_permanent_running_statuses() -> None:
    events: list[str] = []

    class RecordingWriter(_Writer):
        def __init__(self, captured_events: list[str]) -> None:
            super().__init__(captured_events)
            self.statuses: list[str] = []

        def write_generation(self, *args: object, **kwargs: object) -> None:
            self.statuses.append(str(kwargs["attempt_status"]))

        def write_retrieval(self, *args: object, **kwargs: object) -> None:
            self.statuses.append(str(kwargs["attempt_status"]))

        def write_candidate_pool(self, *args: object, **kwargs: object) -> None:
            self.statuses.append(str(kwargs["attempt_status"]))

    writer = RecordingWriter(events)
    runner = RecallExperimentRunner(
        input_source=_InputSource(events),
        generator=_Generator(events),
        registry=_Registry(events, _Handler()),
        pool_builder=_PoolBuilder(events),
        stages=_Stages(events),
        evaluator=_Evaluator(events, _context()),
        writer=writer,
    )

    asyncio.run(runner.run(_request()))

    assert writer.statuses == ["succeeded", "succeeded", "succeeded"]


def test_runner_rejects_generation_bytes_that_disagree_with_the_action_batch() -> None:
    events: list[str] = []
    writer = _Writer(events)

    class MismatchedGenerator(_Generator):
        async def generate(self, context: RecallGenerationContext) -> GenerationResult:
            result = await super().generate(context)
            return result.model_copy(update={"artifact_bytes": b'{"actions":[]}'})

    runner = RecallExperimentRunner(
        input_source=_InputSource(events),
        generator=MismatchedGenerator(events),
        registry=_Registry(events, _Handler()),
        pool_builder=_PoolBuilder(events),
        stages=_Stages(events),
        evaluator=_Evaluator(events, _context()),
        writer=writer,
    )

    with pytest.raises(ValueError, match="artifact bytes"):
        asyncio.run(runner.run(_request()))


@pytest.mark.parametrize("generator_kind", ["fixed", "manual"])
def test_runner_rejects_fixed_or_manual_empty_bytes_with_nonempty_action_batch(
    tmp_path, generator_kind: str
) -> None:
    empty = {"query-1": {"actions": []}}
    if generator_kind == "fixed":
        source = FixedActionGenerator(
            empty,
            expected_query_ids=["query-1"],
            allowed_actions={"text_search"},
            max_actions=1,
        )
    else:
        actions_path = tmp_path / "manual.json"
        actions_path.write_text(json.dumps(empty), encoding="utf-8")
        source = ManualActionGenerator(
            actions_path,
            expected_query_ids=["query-1"],
            allowed_actions={"text_search"},
            max_actions=1,
        )

    class ForgedGenerator:
        async def generate(self, context: RecallGenerationContext) -> GenerationResult:
            empty_result = await source.generate(context)
            nonempty = await _Generator([]).generate(context)
            return empty_result.model_copy(update={"action_batch": nonempty.action_batch})

    events: list[str] = []
    runner = RecallExperimentRunner(
        input_source=_InputSource(events),
        generator=ForgedGenerator(),
        registry=_Registry(events, _Handler()),
        pool_builder=_PoolBuilder(events),
        stages=_Stages(events),
        evaluator=_Evaluator(events, _context()),
        writer=_Writer(events),
    )

    with pytest.raises(ValueError, match="artifact bytes"):
        asyncio.run(runner.run(_request()))


def test_runner_rejects_repeat_counts_above_the_supported_ordinal_range() -> None:
    with pytest.raises(ValueError, match="repeat_count"):
        RecallExperimentRequest(
            run_id="run-01",
            sample=object(),
            recipe_lock={},
            sample_manifest={},
            allowed_actions={"text_search"},
            max_actions=1,
            max_results_per_action=1,
            repeat_count=4,
        )


@pytest.mark.parametrize("ordinal", [0, 4])
def test_attempt_model_rejects_out_of_range_repeat_ordinals(ordinal: int) -> None:
    with pytest.raises(ValueError, match="valid_repeat_ordinal"):
        RecallExperimentAttempt(
            attempt_id="attempt-01",
            attempt_status="succeeded",
            valid_repeat_ordinal=ordinal,
        )


def test_attempt_model_rejects_failure_code_as_attempt_status() -> None:
    with pytest.raises(ValueError, match="attempt_status"):
        RecallExperimentAttempt(
            attempt_id="attempt-01",
            attempt_status="generation_failure",
            valid_repeat_ordinal=None,
            failure_code="generation_failure",
        )


def test_generation_failure_is_recorded_and_later_attempts_continue() -> None:
    events: list[str] = []

    class FailOnceGenerator(_Generator):
        calls = 0

        async def generate(self, context: RecallGenerationContext) -> GenerationResult:
            self.calls += 1
            if self.calls == 1:
                raise RecallGenerationFailure("generation_failure", [])
            return await super().generate(context)

    class RecordingWriter(_Writer):
        def __init__(self, captured_events: list[str]) -> None:
            super().__init__(captured_events)
            self.generation_failures: list[dict[str, object]] = []
            self.report: dict[str, object] | None = None

        def write_generation(self, *args: object, **kwargs: object) -> None:
            if kwargs["attempt_status"] == "failed":
                self.generation_failures.append(dict(args[2]))

        def write_report(self, report: dict[str, object]) -> None:
            self.report = report

    writer = RecordingWriter(events)
    runner = RecallExperimentRunner(
        input_source=_InputSource(events),
        generator=FailOnceGenerator(events),
        registry=_Registry(events, _Handler()),
        pool_builder=_PoolBuilder(events),
        stages=_Stages(events),
        evaluator=_Evaluator(events, _context()),
        writer=writer,
    )
    request = RecallExperimentRequest(
        **{**_request().__dict__, "max_repeat_attempts": 2}
    )

    result = asyncio.run(runner.run(request))

    assert [(attempt.attempt_id, attempt.attempt_status) for attempt in result.attempts] == [
        ("attempt-01", "failed"),
        ("attempt-02", "succeeded"),
    ]
    assert writer.generation_failures == [
        {
            "failure_code": "generation_failure",
            "errors": [],
            "llm_call_receipts": [],
            "repair_count": 0,
        }
    ]
    assert writer.report is not None
    assert writer.report["attempts"][0]["attempt_status"] == "failed"
    assert writer.report["attempts"][0]["failure_code"] == "generation_failure"
