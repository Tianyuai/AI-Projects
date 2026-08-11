"""Pure sealed-query recomposition helpers and aggregate scoring."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from pydantic import Field, field_validator, model_validator

from paper_search.domain.models import (
    DomainModel,
    NonEmptyStr,
    NonNegativeInt,
    Paper,
    ProviderResult,
    QuerySpec,
    Sha256,
    UsageActual,
)
from paper_search.evaluation.dataset import (
    EvaluationQuery,
    IdentifierMap,
    PredictionRecord,
    normalize_paper_id,
)
from paper_search.evaluation.metrics import evaluate
from paper_search.evaluation.ranking_metrics import evaluate_ranking
from paper_search.processing.filter import apply_hard_filters
from paper_search.ranking.fusion import fuse_provider_results


RecompositionMethod = Literal["append_v2", "round_robin_slots", "rrf_slots_k60"]
RecompositionConclusion = Literal[
    "integrity_failure",
    "no_usable_recomposition_signal",
    "signal_insufficient",
    "legacy_benchmark_met",
]
RecompositionReasonCode = Literal[
    "experiment_integrity_failed",
    "no_variant_passed_signal_gate",
    "usable_signal_below_legacy_benchmark",
    "legacy_benchmark_met",
]

_REASON_CODE_BY_CONCLUSION: dict[RecompositionConclusion, RecompositionReasonCode] = {
    "integrity_failure": "experiment_integrity_failed",
    "no_usable_recomposition_signal": "no_variant_passed_signal_gate",
    "signal_insufficient": "usable_signal_below_legacy_benchmark",
    "legacy_benchmark_met": "legacy_benchmark_met",
}

_FIXED_METHODS: tuple[RecompositionMethod, ...] = (
    "append_v2",
    "round_robin_slots",
    "rrf_slots_k60",
)
_TOP_K = 50
_USABLE_SELECTED_THRESHOLD = 19


class RecompositionInput(DomainModel):
    query_id: str
    query_spec: QuerySpec
    baseline_slots: tuple[tuple[Paper, ...], ...]
    addition_slots: tuple[tuple[Paper, ...], ...]
    retrieved_paper_ids: tuple[str, ...]
    post_filter_paper_ids: tuple[str, ...]

    @field_validator("retrieved_paper_ids", "post_filter_paper_ids")
    @classmethod
    def normalize_stage_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(normalize_paper_id(value) for value in values)


class RecompositionProjection(DomainModel):
    method: RecompositionMethod
    retrieved_ids: tuple[str, ...]
    post_filter_ids: tuple[str, ...]
    selected_ids: tuple[str, ...]

    @field_validator("retrieved_ids", "post_filter_ids", "selected_ids")
    @classmethod
    def normalize_projection_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(normalize_paper_id(value) for value in values)

    @model_validator(mode="after")
    def validate_unique_ids(self) -> RecompositionProjection:
        for field_name, values in (
            ("retrieved_ids", self.retrieved_ids),
            ("post_filter_ids", self.post_filter_ids),
            ("selected_ids", self.selected_ids),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{field_name} must not contain duplicate IDs")
        return self


class SealedQueryRecompositionRow(DomainModel):
    method: RecompositionMethod
    true_positive_count: NonNegativeInt
    total_gold_associations: NonNegativeInt
    not_retrieved: NonNegativeInt
    filtered_out: NonNegativeInt
    ranked_outside_top50: NonNegativeInt
    selected_top50: NonNegativeInt
    macro_f1: float = Field(ge=0, le=1, allow_inf_nan=False)
    macro_recall: float = Field(ge=0, le=1, allow_inf_nan=False)
    micro_recall: float = Field(ge=0, le=1, allow_inf_nan=False)
    mrr: float = Field(ge=0, le=1, allow_inf_nan=False)
    ndcg: float = Field(ge=0, le=1, allow_inf_nan=False)
    retains_append_selected_gold: bool
    retrieved_streams_unchanged: bool
    post_filter_streams_unchanged: bool
    usable_signal: bool

    @model_validator(mode="after")
    def validate_stage_conservation(self) -> SealedQueryRecompositionRow:
        if self.total_gold_associations != (
            self.not_retrieved
            + self.filtered_out
            + self.ranked_outside_top50
            + self.selected_top50
        ):
            raise ValueError("pipeline stages must conserve gold associations")
        return self


def _row_has_usable_signal(
    row: SealedQueryRecompositionRow,
    append: SealedQueryRecompositionRow,
) -> bool:
    return all(
        (
            row.selected_top50 > _USABLE_SELECTED_THRESHOLD,
            row.retains_append_selected_gold,
            row.macro_f1 >= append.macro_f1,
            row.macro_recall >= append.macro_recall,
            row.mrr >= append.mrr,
            row.ndcg >= append.ndcg,
            row.filtered_out == 0,
            row.retrieved_streams_unchanged,
            row.post_filter_streams_unchanged,
        )
    )


def _is_canonical_integrity_row(row: SealedQueryRecompositionRow) -> bool:
    return all(
        (
            row.true_positive_count == 0,
            row.not_retrieved == row.total_gold_associations,
            row.filtered_out == 0,
            row.ranked_outside_top50 == 0,
            row.selected_top50 == 0,
            row.macro_f1 == 0.0,
            row.macro_recall == 0.0,
            row.micro_recall == 0.0,
            row.mrr == 0.0,
            row.ndcg == 0.0,
            not row.retains_append_selected_gold,
            not row.retrieved_streams_unchanged,
            not row.post_filter_streams_unchanged,
            not row.usable_signal,
        )
    )


class SealedQueryRecompositionReport(DomainModel):
    schema_version: Literal["sealed-query-recomposition-offline-v1"] = (
        "sealed-query-recomposition-offline-v1"
    )
    input_hashes: dict[NonEmptyStr, Sha256]
    current_formal_selected: NonNegativeInt
    legacy_title_selected: NonNegativeInt
    rows: tuple[SealedQueryRecompositionRow, ...] = Field(min_length=3, max_length=3)
    conclusion: RecompositionConclusion
    reason_codes: tuple[RecompositionReasonCode] = Field(min_length=1, max_length=1)

    @model_validator(mode="after")
    def validate_fixed_rows(self) -> SealedQueryRecompositionReport:
        if tuple(row.method for row in self.rows) != _FIXED_METHODS:
            raise ValueError("recomposition rows must use the fixed order")
        totals = {row.total_gold_associations for row in self.rows}
        if len(totals) != 1:
            raise ValueError("recomposition rows must use the same gold denominator")
        if self.reason_codes != (_REASON_CODE_BY_CONCLUSION[self.conclusion],):
            raise ValueError("reason code must match conclusion")
        if self.conclusion == "integrity_failure":
            if not all(_is_canonical_integrity_row(row) for row in self.rows):
                raise ValueError(
                    "integrity_failure rows must use canonical aggregate failure shape"
                )
            return self
        if any(
            not row.retrieved_streams_unchanged
            or not row.post_filter_streams_unchanged
            or _is_canonical_integrity_row(row)
            for row in self.rows
        ):
            raise ValueError(
                "failed stream invariants or canonical failure rows require "
                "integrity_failure conclusion"
            )
        append = self.rows[0]
        if any(
            row.usable_signal != _row_has_usable_signal(row, append)
            for row in self.rows
        ):
            raise ValueError("usable_signal must match preregistered rule")
        if self.conclusion != _conclusion(self.rows, self.legacy_title_selected):
            raise ValueError("conclusion must match usable rows and legacy benchmark")
        return self


@dataclass(frozen=True)
class _ScoredProjection:
    true_positive_count: int
    total_gold_associations: int
    not_retrieved: int
    filtered_out: int
    ranked_outside_top50: int
    selected_top50: int
    macro_f1: float
    macro_recall: float
    micro_recall: float
    mrr: float
    ndcg: float
    selected_gold_associations: frozenset[tuple[str, str]]


def compose_append(slots: Sequence[Sequence[Paper]]) -> tuple[Paper, ...]:
    """Append slot contents in order, keeping the first canonical ID occurrence."""
    selected: list[Paper] = []
    seen: set[str] = set()
    for slot in slots:
        for paper in slot:
            paper_id = normalize_paper_id(paper.canonical_id)
            if paper_id in seen:
                continue
            seen.add(paper_id)
            selected.append(paper)
    return tuple(selected)


def compose_round_robin(slots: Sequence[Sequence[Paper]]) -> tuple[Paper, ...]:
    """Interleave slots by rank, keeping each canonical ID at first occurrence."""
    selected: list[Paper] = []
    seen: set[str] = set()
    max_slot_length = max((len(slot) for slot in slots), default=0)
    for rank in range(max_slot_length):
        for slot in slots:
            if rank >= len(slot):
                continue
            paper = slot[rank]
            paper_id = normalize_paper_id(paper.canonical_id)
            if paper_id in seen:
                continue
            seen.add(paper_id)
            selected.append(paper)
    return tuple(selected)


def compose_rrf(slots: Sequence[Sequence[Paper]]) -> tuple[Paper, ...]:
    """Fuse slots with the existing deterministic RRF helper and fixed k=60."""
    results: dict[str, ProviderResult[list[Paper]]] = {}
    for index, slot in enumerate(slots):
        if not slot:
            continue
        provider = f"slot_{index:06d}"
        results[provider] = ProviderResult[list[Paper]](
            data=list(slot),
            usage=UsageActual(),
            provenance={
                "provider": provider,
                "endpoint": "sealed-offline-recomposition",
                "model_id": "sealed-offline-recomposition",
                "requested_at": "2026-08-11T00:00:00+00:00",
                "response_hash": "sha256:" + f"{index:064x}"[-64:],
            },
            cache_hit=True,
            latency_ms=0,
            errors=[],
        )
    return tuple(item.paper for item in fuse_provider_results(results, method="rrf", rrf_k=60))


def project_all(
    inputs: Sequence[RecompositionInput],
) -> dict[RecompositionMethod, dict[str, RecompositionProjection]]:
    """Project the three fixed recompositions over verified sealed streams."""
    projections: dict[RecompositionMethod, dict[str, RecompositionProjection]] = {
        method: {} for method in _FIXED_METHODS
    }
    seen_query_ids: set[str] = set()
    for record in inputs:
        if record.query_id in seen_query_ids:
            raise ValueError(f"duplicate query_id: {record.query_id}")
        seen_query_ids.add(record.query_id)

        slots = (*record.baseline_slots, *record.addition_slots)
        addition_papers = tuple(
            paper for slot in record.addition_slots for paper in slot
        )
        retrieved_ids = _stable_normalized_union(
            record.retrieved_paper_ids,
            (paper.canonical_id for paper in addition_papers),
        )
        accepted_additions = apply_hard_filters(
            addition_papers, record.query_spec
        ).accepted
        post_filter_ids = _stable_normalized_union(
            record.post_filter_paper_ids,
            (item.paper.canonical_id for item in accepted_additions),
        )
        selected_by_method: dict[RecompositionMethod, tuple[Paper, ...]] = {
            "append_v2": compose_append(slots),
            "round_robin_slots": compose_round_robin(slots),
            "rrf_slots_k60": compose_rrf(slots),
        }
        accepted_ids = set(post_filter_ids)
        for method in _FIXED_METHODS:
            projections[method][record.query_id] = RecompositionProjection(
                method=method,
                retrieved_ids=retrieved_ids,
                post_filter_ids=post_filter_ids,
                selected_ids=_accepted_top50_ids(selected_by_method[method], accepted_ids),
            )
    return projections


def _stable_normalized_union(*groups: Iterable[str]) -> tuple[str, ...]:
    values: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for value in group:
            normalized = normalize_paper_id(value)
            if normalized in seen:
                continue
            seen.add(normalized)
            values.append(normalized)
    return tuple(values)


def build_report(
    *,
    gold: Sequence[EvaluationQuery],
    identifier_map: IdentifierMap,
    projections: Mapping[RecompositionMethod, Mapping[str, RecompositionProjection]],
    input_hashes: Mapping[str, str],
    current_formal_selected: int,
    legacy_title_selected: int,
) -> SealedQueryRecompositionReport:
    """Build an aggregate-only sealed recomposition report from in-memory inputs."""
    hashes = dict(input_hashes)
    if not _projection_integrity_passes(gold, identifier_map, projections):
        return _integrity_failure_report(
            gold=gold,
            identifier_map=identifier_map,
            input_hashes=hashes,
            current_formal_selected=current_formal_selected,
            legacy_title_selected=legacy_title_selected,
        )

    try:
        scored = {
            method: _score_projection(gold, identifier_map, projections[method])
            for method in _FIXED_METHODS
        }
    except ValueError:
        return _integrity_failure_report(
            gold=gold,
            identifier_map=identifier_map,
            input_hashes=hashes,
            current_formal_selected=current_formal_selected,
            legacy_title_selected=legacy_title_selected,
        )

    append_score = scored["append_v2"]
    rows: list[SealedQueryRecompositionRow] = []
    for method in _FIXED_METHODS:
        score = scored[method]
        retains_append = append_score.selected_gold_associations.issubset(
            score.selected_gold_associations
        )
        no_regression = all(
            (
                score.macro_f1 >= append_score.macro_f1,
                score.macro_recall >= append_score.macro_recall,
                score.mrr >= append_score.mrr,
                score.ndcg >= append_score.ndcg,
            )
        )
        usable = (
            score.selected_top50 > _USABLE_SELECTED_THRESHOLD
            and no_regression
            and retains_append
            and score.filtered_out == 0
        )
        rows.append(
            SealedQueryRecompositionRow(
                method=method,
                true_positive_count=score.true_positive_count,
                total_gold_associations=score.total_gold_associations,
                not_retrieved=score.not_retrieved,
                filtered_out=score.filtered_out,
                ranked_outside_top50=score.ranked_outside_top50,
                selected_top50=score.selected_top50,
                macro_f1=score.macro_f1,
                macro_recall=score.macro_recall,
                micro_recall=score.micro_recall,
                mrr=score.mrr,
                ndcg=score.ndcg,
                retains_append_selected_gold=retains_append,
                retrieved_streams_unchanged=True,
                post_filter_streams_unchanged=True,
                usable_signal=usable,
            )
        )

    conclusion = _conclusion(tuple(rows), legacy_title_selected)
    return SealedQueryRecompositionReport(
        input_hashes=hashes,
        current_formal_selected=current_formal_selected,
        legacy_title_selected=legacy_title_selected,
        rows=tuple(rows),
        conclusion=conclusion,
        reason_codes=(_REASON_CODE_BY_CONCLUSION[conclusion],),
    )


def _accepted_top50_ids(papers: Sequence[Paper], accepted_ids: set[str]) -> tuple[str, ...]:
    selected: list[str] = []
    seen: set[str] = set()
    for paper in papers:
        paper_id = normalize_paper_id(paper.canonical_id)
        if paper_id not in accepted_ids or paper_id in seen:
            continue
        seen.add(paper_id)
        selected.append(paper_id)
        if len(selected) == _TOP_K:
            break
    return tuple(selected)


def _projection_integrity_passes(
    gold: Sequence[EvaluationQuery],
    identifier_map: IdentifierMap,
    projections: Mapping[RecompositionMethod, Mapping[str, RecompositionProjection]],
) -> bool:
    if set(projections) != set(_FIXED_METHODS):
        return False
    expected_query_ids = tuple(record.query_id for record in gold)
    if len(expected_query_ids) != len(set(expected_query_ids)):
        return False
    reference_retrieved: dict[str, frozenset[str]] | None = None
    reference_post_filter: dict[str, frozenset[str]] | None = None
    for method in _FIXED_METHODS:
        by_query = projections[method]
        if set(by_query) != set(expected_query_ids):
            return False
        retrieved_by_query: dict[str, frozenset[str]] = {}
        post_filter_by_query: dict[str, frozenset[str]] = {}
        for query_id in expected_query_ids:
            projection = by_query[query_id]
            if projection.method != method:
                return False
            retrieved = frozenset(identifier_map.resolve(value) for value in projection.retrieved_ids)
            post_filter = frozenset(
                identifier_map.resolve(value) for value in projection.post_filter_ids
            )
            selected = frozenset(identifier_map.resolve(value) for value in projection.selected_ids)
            if not selected <= post_filter or not post_filter <= retrieved:
                return False
            retrieved_by_query[query_id] = retrieved
            post_filter_by_query[query_id] = post_filter
        if reference_retrieved is None:
            reference_retrieved = retrieved_by_query
            reference_post_filter = post_filter_by_query
        elif (
            retrieved_by_query != reference_retrieved
            or post_filter_by_query != reference_post_filter
        ):
            return False
    return True


def _score_projection(
    gold: Sequence[EvaluationQuery],
    identifier_map: IdentifierMap,
    projections: Mapping[str, RecompositionProjection],
) -> _ScoredProjection:
    stage_counts = {
        "not_retrieved": 0,
        "filtered_out": 0,
        "ranked_outside_top50": 0,
        "selected_top50": 0,
    }
    selected_gold: set[tuple[str, str]] = set()
    predictions: list[PredictionRecord] = []
    for record in gold:
        projection = projections[record.query_id]
        retrieved = {identifier_map.resolve(value) for value in projection.retrieved_ids}
        post_filter = {
            identifier_map.resolve(value) for value in projection.post_filter_ids
        }
        selected = {identifier_map.resolve(value) for value in projection.selected_ids}
        for gold_id in {identifier_map.resolve(value) for value in record.relevant_paper_ids}:
            association = (record.query_id, gold_id)
            if gold_id in selected:
                stage_counts["selected_top50"] += 1
                selected_gold.add(association)
            elif gold_id in post_filter:
                stage_counts["ranked_outside_top50"] += 1
            elif gold_id in retrieved:
                stage_counts["filtered_out"] += 1
            else:
                stage_counts["not_retrieved"] += 1
        predictions.append(
            PredictionRecord(
                query_id=record.query_id,
                predicted_paper_ids=list(projection.selected_ids),
            )
        )
    metrics = evaluate(gold, predictions, id_map=identifier_map)
    ranking = evaluate_ranking(gold, predictions, id_map=identifier_map)
    return _ScoredProjection(
        true_positive_count=sum(
            query.true_positive_count for query in metrics.per_query.values()
        ),
        total_gold_associations=sum(stage_counts.values()),
        not_retrieved=stage_counts["not_retrieved"],
        filtered_out=stage_counts["filtered_out"],
        ranked_outside_top50=stage_counts["ranked_outside_top50"],
        selected_top50=stage_counts["selected_top50"],
        macro_f1=metrics.summary.macro_f1,
        macro_recall=metrics.summary.macro_recall,
        micro_recall=metrics.summary.micro_recall,
        mrr=ranking.summary.macro_mrr,
        ndcg=ranking.summary.macro_ndcg,
        selected_gold_associations=frozenset(selected_gold),
    )


def _total_gold_associations(
    gold: Sequence[EvaluationQuery], identifier_map: IdentifierMap
) -> int:
    total = 0
    for record in gold:
        total += len({identifier_map.resolve(value) for value in record.relevant_paper_ids})
    return total


def _integrity_failure_report(
    *,
    gold: Sequence[EvaluationQuery],
    identifier_map: IdentifierMap,
    input_hashes: Mapping[str, str],
    current_formal_selected: int,
    legacy_title_selected: int,
) -> SealedQueryRecompositionReport:
    total = _total_gold_associations(gold, identifier_map)
    rows = tuple(
        SealedQueryRecompositionRow(
            method=method,
            true_positive_count=0,
            total_gold_associations=total,
            not_retrieved=total,
            filtered_out=0,
            ranked_outside_top50=0,
            selected_top50=0,
            macro_f1=0.0,
            macro_recall=0.0,
            micro_recall=0.0,
            mrr=0.0,
            ndcg=0.0,
            retains_append_selected_gold=False,
            retrieved_streams_unchanged=False,
            post_filter_streams_unchanged=False,
            usable_signal=False,
        )
        for method in _FIXED_METHODS
    )
    return SealedQueryRecompositionReport(
        input_hashes=dict(input_hashes),
        current_formal_selected=current_formal_selected,
        legacy_title_selected=legacy_title_selected,
        rows=rows,
        conclusion="integrity_failure",
        reason_codes=("experiment_integrity_failed",),
    )


def _conclusion(
    rows: Sequence[SealedQueryRecompositionRow], legacy_title_selected: int
) -> RecompositionConclusion:
    usable_selected = [row.selected_top50 for row in rows if row.usable_signal]
    if not usable_selected:
        return "no_usable_recomposition_signal"
    if max(usable_selected) < legacy_title_selected:
        return "signal_insufficient"
    return "legacy_benchmark_met"


__all__ = [
    "RecompositionInput",
    "RecompositionMethod",
    "RecompositionProjection",
    "SealedQueryRecompositionReport",
    "SealedQueryRecompositionRow",
    "build_report",
    "compose_append",
    "compose_round_robin",
    "compose_rrf",
    "project_all",
]
