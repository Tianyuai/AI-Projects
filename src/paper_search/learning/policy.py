"""Bounded action ranking with a confidence gate for LLM fallback."""

from __future__ import annotations

import unicodedata
from typing import Protocol

from paper_search.domain.models import UnitFloat
from paper_search.learning.contracts import (
    PolicyActionCandidate,
    QueryPolicyInput,
    QueryPolicyOutput,
    RankedPolicyAction,
)


class ActionScorer(Protocol):
    model_id: str

    def score(
        self,
        request: QueryPolicyInput,
        candidates: list[PolicyActionCandidate],
    ) -> list[float]: ...


class RuleActionScorer:
    """CPU-only placeholder proving the same interface used by a trained scorer."""

    model_id = "rule-action-scorer-v1"

    def score(
        self,
        request: QueryPolicyInput,
        candidates: list[PolicyActionCandidate],
    ) -> list[float]:
        del request
        return [1.0 if candidate.origin == "original_query" else 0.5 for candidate in candidates]


class BoundedQueryPolicy:
    def __init__(
        self,
        scorer: ActionScorer,
        *,
        confidence_threshold: UnitFloat,
    ) -> None:
        self._scorer = scorer
        self._confidence_threshold = confidence_threshold

    @property
    def model_id(self) -> str:
        return self._scorer.model_id

    @staticmethod
    def _candidates(request: QueryPolicyInput) -> list[PolicyActionCandidate]:
        anchor = PolicyActionCandidate(
            action_id="policy-anchor",
            action_type="text_search",
            text=request.original_query,
            origin="original_query",
            provider_hint="either",
        )
        selected: list[PolicyActionCandidate] = []
        seen: set[tuple[str, str, str]] = set()
        for candidate in [anchor, *request.seed_actions]:
            if candidate.action_type not in request.allowed_action_types:
                continue
            text = " ".join(unicodedata.normalize("NFKC", candidate.text).split())
            key = (
                candidate.action_type,
                candidate.search_mode,
                text.casefold(),
            )
            if key in seen:
                continue
            seen.add(key)
            selected.append(candidate.model_copy(update={"text": text}))
        if not selected:
            raise ValueError("policy could not construct an allowed original-query anchor")
        return selected

    def plan(self, request: QueryPolicyInput) -> QueryPolicyOutput:
        request = QueryPolicyInput.model_validate(request)
        candidates = self._candidates(request)
        scores = self._scorer.score(request, candidates)
        if len(scores) != len(candidates):
            raise ValueError("action ranker must return one score per candidate")
        ranked = [
            RankedPolicyAction(action=candidate, score=score)
            for candidate, score in zip(candidates, scores, strict=True)
        ]
        ranked.sort(
            key=lambda item: (
                -item.score,
                0 if item.action.origin == "original_query" else 1,
                item.action.action_id,
            )
        )
        selected = ranked[: request.max_actions]
        anchor = next(item for item in ranked if item.action.origin == "original_query")
        if anchor not in selected:
            selected[-1] = anchor
            selected.sort(
                key=lambda item: (
                    -item.score,
                    0 if item.action.origin == "original_query" else 1,
                    item.action.action_id,
                )
            )
        expansion_scores = [
            item.score
            for item in selected
            if item.action.origin != "original_query"
        ]
        confidence = (
            max(expansion_scores) if expansion_scores else anchor.score
        )
        fallback_required = confidence < self._confidence_threshold
        return QueryPolicyOutput(
            query_kind=request.query_kind,
            ranked_actions=selected,
            confidence=confidence,
            fallback_required=fallback_required,
            fallback_reason=(
                "confidence_below_threshold" if fallback_required else None
            ),
            model_id=self._scorer.model_id,
        )


__all__ = ["ActionScorer", "BoundedQueryPolicy", "RuleActionScorer"]
