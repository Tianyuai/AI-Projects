"""Paired Oracle diagnostics for controlled candidate-family experiments."""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Literal

from pydantic import Field

from paper_search.domain.models import DomainModel, NonEmptyStr, UnitFloat
from paper_search.learning.provider_action_labels import ProviderActionLabel


Family = Literal["baseline", "semantic", "boolean_phrase", "prf"]


class CandidateFamilyBatchEvidence(DomainModel):
    batch_id: NonEmptyStr
    query_count: int = Field(strict=True, ge=0)
    unavailable_query_count: int = Field(strict=True, ge=0)
    baseline_all_candidate_macro_recall: UnitFloat
    v2_all_candidate_macro_recall: UnitFloat
    baseline_oracle_at_3_macro_recall: UnitFloat
    v2_oracle_at_3_macro_recall: UnitFloat
    semantic_incremental_macro_recall: UnitFloat
    boolean_phrase_incremental_macro_recall: UnitFloat
    prf_incremental_macro_recall: UnitFloat


def _family(action_id: str) -> Family:
    if "semantic-original" in action_id:
        return "semantic"
    if "boolean-relaxed" in action_id or "phrase-proximity" in action_id:
        return "boolean_phrase"
    if "candidate-prf-" in action_id:
        return "prf"
    return "baseline"


def _union(rows: list[ProviderActionLabel]) -> set[str]:
    return set().union(*(set(row.gold_hit_ids) for row in rows)) if rows else set()


def _oracle_at_3(rows: list[ProviderActionLabel]) -> set[str]:
    choices = [
        selected
        for size in range(1, min(3, len(rows)) + 1)
        for selected in combinations(rows, size)
    ]
    if not choices:
        return set()
    return max(
        (_union(list(selected)) for selected in choices),
        key=lambda hits: (len(hits), tuple(sorted(hits))),
    )


def summarize_candidate_family_batch(
    labels: list[ProviderActionLabel],
    *,
    batch_id: str,
) -> CandidateFamilyBatchEvidence:
    validated = [ProviderActionLabel.model_validate(row) for row in labels]
    if not validated:
        raise ValueError("candidate family labels are empty")
    if any(row.provider != "openalex" for row in validated):
        raise ValueError("candidate family diagnostics require OpenAlex-only labels")
    grouped: dict[str, list[ProviderActionLabel]] = defaultdict(list)
    for row in validated:
        grouped[row.query_id].append(row)
    metrics: list[tuple[float, float, float, float, float, float, float]] = []
    unavailable = 0
    for rows in grouped.values():
        if any(row.retrieval_status == "unavailable" for row in rows):
            unavailable += 1
            continue
        gold_count = rows[0].gold_association_count
        assert gold_count is not None
        families = {
            name: [row for row in rows if _family(row.action.action_id) == name]
            for name in ("baseline", "semantic", "boolean_phrase", "prf")
        }
        baseline_hits = _union(families["baseline"])
        all_hits = _union(rows)
        metrics.append(
            (
                len(baseline_hits) / gold_count,
                len(all_hits) / gold_count,
                len(_oracle_at_3(families["baseline"])) / gold_count,
                len(_oracle_at_3(rows)) / gold_count,
                len(_union(families["semantic"]).difference(baseline_hits))
                / gold_count,
                len(_union(families["boolean_phrase"]).difference(baseline_hits))
                / gold_count,
                len(_union(families["prf"]).difference(baseline_hits)) / gold_count,
            )
        )
    if not metrics:
        raise ValueError("candidate family labels contain no complete query")
    def macro(index: int) -> float:
        return sum(row[index] for row in metrics) / len(metrics)

    return CandidateFamilyBatchEvidence(
        batch_id=batch_id,
        query_count=len(metrics),
        unavailable_query_count=unavailable,
        baseline_all_candidate_macro_recall=macro(0),
        v2_all_candidate_macro_recall=macro(1),
        baseline_oracle_at_3_macro_recall=macro(2),
        v2_oracle_at_3_macro_recall=macro(3),
        semantic_incremental_macro_recall=macro(4),
        boolean_phrase_incremental_macro_recall=macro(5),
        prf_incremental_macro_recall=macro(6),
    )


__all__ = [
    "CandidateFamilyBatchEvidence",
    "summarize_candidate_family_batch",
]
