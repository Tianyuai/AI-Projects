"""Recall-only scoring and compatibility checks for frozen candidate pools.

Identifier-map bytes deliberately cross this module's boundary only in
``CandidateRecallEvaluator.preflight``.  Everything passed to generation and
retrieval is identifier-safe; resolved Gold associations remain private to the
prepared context and evaluator.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import median
from typing import Literal

from pydantic import Field, model_validator

from paper_search.domain.models import DomainModel, QuerySpec
from paper_search.evaluation.dataset import IdentifierMap
from paper_search.recall_experiments.contracts import (
    CandidatePool,
    GoldDocument,
    RecallGenerationContext,
)
from paper_search.recall_experiments.inputs.base import FrozenRecallDataset
from paper_search.recall_experiments.inputs.gold_catalog import SealedGoldDocumentCatalog
from paper_search.recall_experiments.paper_identity import EvidenceDrivenIdentifierResolver
from paper_search.recall_experiments.recipes import RecallMethodRecipe, SampleBinding


TerminalConclusion = Literal[
    "passed",
    "failed",
    "not_comparable",
    "insufficient_valid_repeats",
    "insufficient_historical_evidence",
]
_ATTEMPT_ID = re.compile(r"attempt-(0[1-5])$")


@dataclass(frozen=True)
class _PrivateScoringData:
    gold_by_query: Mapping[str, frozenset[str]]


@dataclass(frozen=True)
class PreparedEvaluationContext:
    """Separate safe generation inputs from evaluator-private resolved Gold."""

    generation_contexts: tuple[RecallGenerationContext, ...]
    _scoring: _PrivateScoringData
    _resolver: EvidenceDrivenIdentifierResolver


class PerQueryCandidateRecall(DomainModel):
    query_id: str
    candidate_pool_ids: list[str]
    gold_hit_ids: list[str]
    gold_association_count: int = Field(ge=1)
    gold_hit_count: int = Field(ge=0)
    candidate_recall: float = Field(ge=0, le=1)


class RecallRepeatResult(DomainModel):
    """A recall-only result for one semantically successful execution."""

    candidate_pool_policy_version: str
    per_query: list[PerQueryCandidateRecall]
    gold_association_count: int = Field(ge=1)
    gold_hit_count: int = Field(ge=0)
    macro_candidate_recall: float = Field(ge=0, le=1)


class HistoricalReplayEvidence(DomainModel):
    """The subset of historical evidence that a comparison can prove."""

    candidate_pool_policy_version: str
    gold_association_count: int = Field(ge=1)
    gold_hit_count: int = Field(ge=0)
    macro_candidate_recall: float = Field(ge=0, le=1)
    per_query: list[PerQueryCandidateRecall] | None = None

    @classmethod
    def from_repeat(cls, result: RecallRepeatResult) -> HistoricalReplayEvidence:
        return cls(
            candidate_pool_policy_version=result.candidate_pool_policy_version,
            gold_association_count=result.gold_association_count,
            gold_hit_count=result.gold_hit_count,
            macro_candidate_recall=result.macro_candidate_recall,
            per_query=result.per_query,
        )


class HistoricalReplayComparison(DomainModel):
    conclusion: TerminalConclusion
    per_query_comparison: Literal["passed", "failed", "not_provable", "not_comparable"]


class RecallAttempt(DomainModel):
    """One scheduled execution; a semantic result gets a distinct repeat ordinal."""

    attempt_id: str
    infrastructure_failure: bool = False
    valid_repeat_ordinal: int | None = Field(default=None, ge=1, le=3)
    result: RecallRepeatResult | None = None
    historical_gold_retention: float | None = Field(default=None, ge=0, le=1)
    passes_tolerance: bool | None = None

    @model_validator(mode="after")
    def validate_attempt_shape(self) -> RecallAttempt:
        if _ATTEMPT_ID.fullmatch(self.attempt_id) is None:
            raise ValueError("attempt_id must be attempt-01 through attempt-05")
        if self.infrastructure_failure:
            if self.valid_repeat_ordinal is not None or self.result is not None:
                raise ValueError("infrastructure failures cannot receive a valid repeat ordinal")
            return self
        if self.valid_repeat_ordinal is None or self.result is None:
            raise ValueError("semantic attempts require result and valid_repeat_ordinal")
        return self


class RegeneratedComparison(DomainModel):
    conclusion: TerminalConclusion
    attempts: list[RecallAttempt]
    valid_repeat_count: int = Field(ge=0)
    passing_repeat_count: int = Field(ge=0)
    hit_count_summary: dict[str, float] | None = None
    macro_candidate_recall_summary: dict[str, float] | None = None


class CandidateRecallEvaluator:
    """Score complete candidate pools without invoking ranking-oriented metrics."""

    def __init__(
        self,
        recipe: RecallMethodRecipe | None = None,
        *,
        sample: SampleBinding | None = None,
        blind_sample: SampleBinding | None = None,
        gold_catalog: SealedGoldDocumentCatalog | None = None,
    ) -> None:
        self._recipe = recipe
        self._sample = sample
        self._blind_sample = blind_sample
        self._gold_catalog = gold_catalog

    def preflight(self, dataset: FrozenRecallDataset) -> PreparedEvaluationContext:
        """Resolve aliases once and prove all evaluator-only frozen invariants."""
        identifier_map = IdentifierMap.from_bytes(dataset.evaluation_materials.identifier_map_bytes)
        resolver = EvidenceDrivenIdentifierResolver(identifier_map)
        material_queries = dataset.evaluation_materials.gold_records
        dataset_ids = [query.query_id for query in dataset.queries]
        material_ids = [query.query_id for query in material_queries]
        if dataset_ids != material_ids:
            raise ValueError("Gold denominator query IDs do not match selected dataset")
        if len(dataset_ids) != len(set(dataset_ids)):
            raise ValueError("Gold denominator contains duplicate query IDs")
        if self._sample is not None and set(self._sample.query_ids) != set(dataset_ids):
            raise ValueError("sample query IDs do not match selected Gold denominator")
        if self._blind_sample is not None and self._sample is not None:
            if set(self._sample.query_ids).intersection(self._blind_sample.query_ids):
                raise ValueError("Oracle and Blind query IDs overlap")

        gold_by_query: dict[str, frozenset[str]] = {}
        for query in material_queries:
            resolved = [resolver.resolve(value) for value in query.relevant_paper_ids]
            if not resolved:
                raise ValueError("frozen data forbids zero-Gold queries")
            if len(resolved) != len(set(resolved)):
                raise ValueError("duplicate resolved Gold association")
            gold_by_query[query.query_id] = frozenset(resolved)

        for query in dataset.queries:
            if frozenset(resolver.resolve(value) for value in query.relevant_paper_ids) != gold_by_query[
                query.query_id
            ]:
                raise ValueError("selected dataset Gold denominator does not match private scoring data")

        all_gold = frozenset().union(*gold_by_query.values())
        for seed in dataset.seed_candidates:
            if resolver.paper_identities(seed.paper).intersection(all_gold):
                raise ValueError("seed candidate resolves to a Gold ID")

        documents_by_query = self._validated_oracle_documents(gold_by_query, resolver)
        generation_contexts = tuple(
            RecallGenerationContext(
                query_id=query.query_id,
                original_query=query.query,
                query_spec=QuerySpec(original_query=query.query, research_goal=query.query),
                seed_candidates=dataset.seed_candidates,
                gold_documents=documents_by_query.get(query.query_id, []),
            )
            for query in dataset.queries
        )
        return PreparedEvaluationContext(
            generation_contexts=generation_contexts,
            _scoring=_PrivateScoringData(gold_by_query=gold_by_query),
            _resolver=resolver,
        )

    def evaluate(
        self,
        dataset: FrozenRecallDataset | PreparedEvaluationContext,
        pools: Sequence[CandidatePool],
    ) -> RecallRepeatResult:
        prepared = dataset if isinstance(dataset, PreparedEvaluationContext) else self.preflight(dataset)
        expected_ids = [context.query_id for context in prepared.generation_contexts]
        pools_by_query = {pool.query_id: pool for pool in pools}
        if len(pools_by_query) != len(pools) or set(pools_by_query) != set(expected_ids):
            raise ValueError("candidate pools must contain exactly one pool for every Gold query")
        policies = {pool.policy_version for pool in pools}
        if len(policies) != 1:
            raise ValueError("candidate pools must use one locked policy version")
        policy_version = next(iter(policies))
        if (
            self._recipe is not None
            and policy_version != self._recipe.candidate_pool.policy_version
        ):
            raise ValueError("candidate pool policy does not match recipe candidate pool policy")

        per_query: list[PerQueryCandidateRecall] = []
        for query_id in expected_ids:
            pool = pools_by_query[query_id]
            candidate_ids = frozenset(
                prepared._resolver.primary_paper_id(entry.paper) for entry in pool.entries
            )
            candidate_identities = frozenset().union(
                *(prepared._resolver.paper_identities(entry.paper) for entry in pool.entries)
            )
            gold = prepared._scoring.gold_by_query[query_id]
            hits = candidate_identities.intersection(gold)
            per_query.append(
                PerQueryCandidateRecall(
                    query_id=query_id,
                    candidate_pool_ids=sorted(candidate_ids),
                    gold_hit_ids=sorted(hits),
                    gold_association_count=len(gold),
                    gold_hit_count=len(hits),
                    candidate_recall=len(hits) / len(gold),
                )
            )
        total_gold = sum(row.gold_association_count for row in per_query)
        total_hits = sum(row.gold_hit_count for row in per_query)
        return RecallRepeatResult(
            candidate_pool_policy_version=policy_version,
            per_query=per_query,
            gold_association_count=total_gold,
            gold_hit_count=total_hits,
            macro_candidate_recall=sum(row.candidate_recall for row in per_query) / len(per_query),
        )

    def _validated_oracle_documents(
        self,
        gold_by_query: Mapping[str, frozenset[str]],
        resolver: EvidenceDrivenIdentifierResolver,
    ) -> dict[str, list[GoldDocument]]:
        if self._recipe is None or self._recipe.generator.gold_visibility != "oracle":
            return {}
        if self._gold_catalog is None or self._gold_catalog.status != "complete":
            raise ValueError("oracle generation requires a complete sealed Gold catalog")
        catalog_pairs = {
            (record.query_id, resolver.resolve(record.gold_paper_id))
            for record in self._gold_catalog.records
            if record.has_title()
        }
        expected_pairs = {
            (query_id, gold_id) for query_id, ids in gold_by_query.items() for gold_id in ids
        }
        if not expected_pairs.issubset(catalog_pairs):
            raise ValueError("Oracle Gold association lacks a title in the sealed catalog")
        return {
            query_id: self._gold_catalog.to_generation_documents(query_id)
            for query_id in gold_by_query
        }


def compare_exact_replay(
    current: RecallRepeatResult, historical: HistoricalReplayEvidence | RecallRepeatResult | None
) -> HistoricalReplayComparison:
    """Compare only evidence whose historic granularity makes it provable."""
    if historical is None:
        return HistoricalReplayComparison(
            conclusion="insufficient_historical_evidence", per_query_comparison="not_provable"
        )
    evidence = _as_historical_evidence(historical)
    if current.candidate_pool_policy_version != evidence.candidate_pool_policy_version:
        return HistoricalReplayComparison(
            conclusion="not_comparable", per_query_comparison="not_comparable"
        )
    aggregate_matches = (
        current.gold_association_count == evidence.gold_association_count
        and current.gold_hit_count == evidence.gold_hit_count
        and current.macro_candidate_recall == evidence.macro_candidate_recall
    )
    if evidence.per_query is None:
        return HistoricalReplayComparison(
            conclusion="passed" if aggregate_matches else "failed",
            per_query_comparison="not_provable",
        )
    current_by_id = {row.query_id: row for row in current.per_query}
    historical_by_id = {row.query_id: row for row in evidence.per_query}
    per_query_matches = set(current_by_id) == set(historical_by_id) and all(
        set(current_by_id[query_id].candidate_pool_ids)
        == set(historical_by_id[query_id].candidate_pool_ids)
        and set(current_by_id[query_id].gold_hit_ids)
        == set(historical_by_id[query_id].gold_hit_ids)
        for query_id in current_by_id
    )
    passed = aggregate_matches and per_query_matches
    return HistoricalReplayComparison(
        conclusion="passed" if passed else "failed",
        per_query_comparison="passed" if per_query_matches else "failed",
    )


def compare_regenerated(
    repeats: Sequence[RecallAttempt],
    historical: HistoricalReplayEvidence | RecallRepeatResult | None,
    policy: str,
) -> RegeneratedComparison:
    """Apply the fixed 3-valid-within-5 repeat policy to regenerated results."""
    attempts = list(repeats)
    if len(attempts) > 5:
        raise ValueError("regenerated comparison permits at most five scheduled attempts")
    _validate_attempt_sequence(attempts)
    valid = [attempt for attempt in attempts if not attempt.infrastructure_failure]
    summaries = _summaries(valid)
    if historical is None:
        return _comparison(
            "insufficient_historical_evidence", attempts, valid, 0, summaries
        )
    evidence = _as_historical_evidence(historical)
    if evidence.candidate_pool_policy_version != policy or any(
        attempt.result is not None and attempt.result.candidate_pool_policy_version != policy
        for attempt in valid
    ):
        return _comparison("not_comparable", attempts, valid, 0, summaries)
    if len(valid) < 3:
        return _comparison("insufficient_valid_repeats", attempts, valid, 0, summaries)
    if evidence.per_query is None:
        return _comparison("insufficient_historical_evidence", attempts, valid, 0, summaries)

    historical_associations = _hit_associations(evidence.per_query)
    if not historical_associations:
        return _comparison("insufficient_historical_evidence", attempts, valid, 0, summaries)
    evaluated_attempts: list[RecallAttempt] = []
    passing = 0
    for attempt in attempts:
        if attempt.infrastructure_failure:
            evaluated_attempts.append(attempt)
            continue
        assert attempt.result is not None
        retention = (
            len(_hit_associations(attempt.result.per_query).intersection(historical_associations))
            / len(historical_associations)
        )
        passes = (
            abs(attempt.result.gold_hit_count - evidence.gold_hit_count) <= 1
            and abs(attempt.result.macro_candidate_recall - evidence.macro_candidate_recall) <= 0.02
            and retention >= 0.90
        )
        passing += int(passes)
        evaluated_attempts.append(
            attempt.model_copy(
                update={"historical_gold_retention": retention, "passes_tolerance": passes}
            )
        )
    return _comparison(
        "passed" if passing >= 2 else "failed", evaluated_attempts, valid, passing, summaries
    )


def _as_historical_evidence(
    value: HistoricalReplayEvidence | RecallRepeatResult,
) -> HistoricalReplayEvidence:
    return value if isinstance(value, HistoricalReplayEvidence) else HistoricalReplayEvidence.from_repeat(value)


def _validate_attempt_sequence(attempts: Sequence[RecallAttempt]) -> None:
    attempt_ids = [attempt.attempt_id for attempt in attempts]
    expected_ids = [f"attempt-{ordinal:02d}" for ordinal in range(1, len(attempts) + 1)]
    if attempt_ids != expected_ids:
        raise ValueError("scheduled attempts must be an ordered contiguous prefix")
    ordinals = [attempt.valid_repeat_ordinal for attempt in attempts if not attempt.infrastructure_failure]
    if ordinals != list(range(1, len(ordinals) + 1)):
        raise ValueError("valid repeat ordinals must be consecutive and distinct")


def _hit_associations(rows: Sequence[PerQueryCandidateRecall]) -> set[tuple[str, str]]:
    return {(row.query_id, identifier) for row in rows for identifier in row.gold_hit_ids}


def _summaries(valid: Sequence[RecallAttempt]) -> tuple[dict[str, float] | None, dict[str, float] | None]:
    if not valid:
        return None, None
    hits = [attempt.result.gold_hit_count for attempt in valid if attempt.result is not None]
    recalls = [attempt.result.macro_candidate_recall for attempt in valid if attempt.result is not None]
    return _summary(hits), _summary(recalls)


def _summary(values: Sequence[int | float]) -> dict[str, float]:
    return {"min": float(min(values)), "median": float(median(values)), "max": float(max(values))}


def _comparison(
    conclusion: TerminalConclusion,
    attempts: Sequence[RecallAttempt],
    valid: Sequence[RecallAttempt],
    passing: int,
    summaries: tuple[dict[str, float] | None, dict[str, float] | None],
) -> RegeneratedComparison:
    return RegeneratedComparison(
        conclusion=conclusion,
        attempts=list(attempts),
        valid_repeat_count=len(valid),
        passing_repeat_count=passing,
        hit_count_summary=summaries[0],
        macro_candidate_recall_summary=summaries[1],
    )


__all__ = [
    "CandidateRecallEvaluator",
    "EvidenceDrivenIdentifierResolver",
    "HistoricalReplayComparison",
    "HistoricalReplayEvidence",
    "PerQueryCandidateRecall",
    "PreparedEvaluationContext",
    "RecallAttempt",
    "RecallRepeatResult",
    "RegeneratedComparison",
    "compare_exact_replay",
    "compare_regenerated",
]
