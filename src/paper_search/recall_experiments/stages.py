"""A future extension seam for candidate-pool stages."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from typing import Protocol

from paper_search.domain.models import DomainModel
from paper_search.recall_experiments.contracts import CandidatePool


class StageResult(DomainModel):
    """The candidate pool produced by one explicit future stage."""

    pool: CandidatePool


class CandidateStage(Protocol):
    """Transform a candidate pool without owning generation or retrieval."""

    def apply(self, pool: CandidatePool, context: object) -> StageResult: ...


class CandidateStagePipeline:
    """Apply explicitly supplied stages; Phase 1 supplies none."""

    def __init__(self, stages: Sequence[CandidateStage] = ()) -> None:
        self._stages = tuple(stages)

    def apply(self, pool: CandidatePool, context: object) -> CandidatePool:
        current = pool
        for stage in self._stages:
            current = stage.apply(current, deepcopy(context)).pool
        return current


__all__ = ["CandidateStage", "CandidateStagePipeline", "StageResult"]
