"""Versioned, raw candidate-pool projections for recall experiments."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from paper_search.domain.models import Paper
from paper_search.processing.deduplicate import deduplicate_papers
from paper_search.recall_experiments.contracts import (
    CandidatePool,
    CandidatePoolEntry,
    CandidateSourceEvidence,
    RetrievalActionResult,
)


CandidatePoolPolicy = Literal["production-dedup-v1", "canonical-id-first-v1"]
_SUPPORTED_POLICIES = frozenset(("production-dedup-v1", "canonical-id-first-v1"))


@dataclass(frozen=True)
class _CollectedHit:
    paper: Paper
    evidence: CandidateSourceEvidence


class CandidatePoolBuilder:
    """Build an unfiltered candidate pool using one locked projection policy."""

    def __init__(self, policy_version: CandidatePoolPolicy | str) -> None:
        if policy_version not in _SUPPORTED_POLICIES:
            raise ValueError(f"unknown candidate pool policy: {policy_version}")
        self._policy_version: CandidatePoolPolicy = policy_version  # type: ignore[assignment]

    def build(
        self,
        query_id: str,
        action_results: Sequence[RetrievalActionResult],
    ) -> CandidatePool:
        """Project handler hits in their received action and provider-rank order."""
        collected = _collect_hits(action_results)
        if self._policy_version == "production-dedup-v1":
            entries = _production_entries(collected)
        else:
            entries = _legacy_entries(collected)
        return CandidatePool(query_id=query_id, policy_version=self._policy_version, entries=entries)


def _collect_hits(action_results: Sequence[RetrievalActionResult]) -> list[_CollectedHit]:
    collected: list[_CollectedHit] = []
    for result in action_results:
        evidence = CandidateSourceEvidence(
            action_id=result.action_id,
            action_type=result.action_type,
            provenance=result.provenance,
        )
        collected.extend(_CollectedHit(paper=paper, evidence=evidence) for paper in result.hits)
    return collected


def _production_entries(collected: Sequence[_CollectedHit]) -> list[CandidatePoolEntry]:
    flattened = [hit.paper for hit in collected]
    deduplicated = deduplicate_papers(flattened, id_map=None)
    members_by_representative = {
        decision.representative_id: set(decision.member_ids)
        for decision in deduplicated.decisions
    }
    return [
        CandidatePoolEntry(
            paper=paper,
            source_evidence=[
                hit.evidence
                for hit in collected
                if hit.paper.canonical_id
                in members_by_representative.get(paper.canonical_id, {paper.canonical_id})
            ],
        )
        for paper in deduplicated.papers
    ]


def _legacy_entries(collected: Sequence[_CollectedHit]) -> list[CandidatePoolEntry]:
    papers: list[Paper] = []
    evidence_by_canonical_id: dict[str, list[CandidateSourceEvidence]] = {}
    for hit in collected:
        evidence = evidence_by_canonical_id.setdefault(hit.paper.canonical_id, [])
        evidence.append(hit.evidence)
        if len(evidence) == 1:
            papers.append(hit.paper)
    return [
        CandidatePoolEntry(paper=paper, source_evidence=evidence_by_canonical_id[paper.canonical_id])
        for paper in papers
    ]


__all__ = ["CandidatePoolBuilder", "CandidatePoolPolicy"]
