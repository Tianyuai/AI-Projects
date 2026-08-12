"""Public, identifier-safe contracts for replaceable frozen recall inputs."""

from __future__ import annotations

from typing import Protocol

from paper_search.domain.models import DomainModel, Sha256
from paper_search.evaluation.business_results import BusinessResultRecord
from paper_search.evaluation.dataset import EvaluationQuery
from paper_search.evaluation.execution_adapter import EvaluationExecutionRecord
from paper_search.recall_experiments.contracts import SeedCandidate
from paper_search.recall_experiments.recipes import (
    HistoricalBaselineBinding,
    SampleBinding,
)


class OpaqueEvaluationMaterials(DomainModel):
    """Private scoring material; identifier-map bytes are evaluator-only."""

    gold_records: list[EvaluationQuery]
    identifier_map_bytes: bytes
    identifier_map_sha256: Sha256


class FrozenRecallDataset(DomainModel):
    """A selected frozen query slice plus evaluator-private material."""

    queries: list[EvaluationQuery]
    source_hashes: dict[str, Sha256]
    evaluation_materials: OpaqueEvaluationMaterials
    seed_candidates: list[SeedCandidate]


class HistoricalRecallBaseline(DomainModel):
    """Historical result records aligned with one exact frozen denominator."""

    query_ids: list[str]
    gold_association_count: int
    business_results: list[BusinessResultRecord]
    executions: list[EvaluationExecutionRecord]
    source_hashes: dict[str, Sha256]


class FrozenInputSource(Protocol):
    """Load a frozen sample without resolving its identifier map."""

    def load_queries(self, sample_binding: SampleBinding) -> FrozenRecallDataset: ...

    def load_historical_baseline(
        self, binding: HistoricalBaselineBinding
    ) -> HistoricalRecallBaseline | None: ...
