"""Offline formal-run input adapter with exact-byte source verification."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TypeVar

from paper_search.domain.models import DomainModel, Sha256
from paper_search.evaluation.business_results import BusinessResultRecord
from paper_search.evaluation.dataset import EvaluationQuery
from paper_search.evaluation.execution_adapter import EvaluationExecutionRecord
from paper_search.recall_experiments.inputs.base import (
    FrozenRecallDataset,
    HistoricalRecallBaseline,
    OpaqueEvaluationMaterials,
)
from paper_search.recall_experiments.recipes import (
    ArtifactBinding,
    HistoricalBaselineBinding,
    SampleBinding,
)


class FormalRunInputSource:
    """Load selected formal-run rows while keeping identifier data opaque."""

    def __init__(self, workspace_root: Path | None = None) -> None:
        self._workspace_root = (workspace_root or Path.cwd()).resolve()

    def load_queries(self, sample_binding: SampleBinding) -> FrozenRecallDataset:
        binding = sample_binding.frozen_inputs
        if binding is None:
            raise ValueError("sample binding lacks formal frozen inputs")
        gold_bytes = self._read_bound_bytes(binding.gold_associations, "gold_associations")
        identifier_map_bytes = self._read_bound_bytes(binding.identifier_map, "identifier_map")
        paper_source_hashes = {
            f"bound_paper_source_{index}": paper_source.sha256
            for index, paper_source in enumerate(binding.bound_paper_sources)
        }
        for index, paper_source in enumerate(binding.bound_paper_sources):
            self._read_bound_bytes(paper_source, f"bound_paper_source_{index}")
        gold_records = _parse_jsonl_bytes(gold_bytes, EvaluationQuery)
        selected = _select_source_order(gold_records, sample_binding.query_ids)
        return FrozenRecallDataset(
            queries=selected,
            source_hashes={
                "gold_associations": binding.gold_associations.sha256,
                "identifier_map": binding.identifier_map.sha256,
                **paper_source_hashes,
            },
            evaluation_materials=OpaqueEvaluationMaterials(
                gold_records=selected,
                identifier_map_bytes=identifier_map_bytes,
                identifier_map_sha256=binding.identifier_map.sha256,
            ),
            seed_candidates=binding.seed_candidates,
        )

    def load_historical_baseline(
        self, binding: HistoricalBaselineBinding
    ) -> HistoricalRecallBaseline:
        gold_records = _parse_jsonl_bytes(
            self._read_bound_bytes(binding.gold_associations, "gold_associations"), EvaluationQuery
        )
        selected = _select_source_order(gold_records, binding.query_ids)
        business_results = _parse_jsonl_bytes(
            self._read_bound_bytes(binding.business_results, "business_results"),
            BusinessResultRecord,
        )
        executions = _parse_jsonl_bytes(
            self._read_bound_bytes(binding.executions, "executions"),
            EvaluationExecutionRecord,
        )
        expected_ids = [record.query_id for record in selected]
        if [record.query_id for record in business_results] != expected_ids:
            raise ValueError("historical business-result query IDs do not match Gold associations")
        if [record.query_id for record in executions] != expected_ids:
            raise ValueError("historical execution query IDs do not match Gold associations")
        return HistoricalRecallBaseline(
            query_ids=expected_ids,
            gold_association_count=sum(len(record.relevant_paper_ids) for record in selected),
            business_results=business_results,
            executions=executions,
            source_hashes={
                "gold_associations": binding.gold_associations.sha256,
                "business_results": binding.business_results.sha256,
                "executions": binding.executions.sha256,
            },
        )

    def _read_bound_bytes(self, binding: ArtifactBinding, label: str) -> bytes:
        path = self._path(binding.path)
        content = path.read_bytes()
        actual = _sha256(content)
        if actual != binding.sha256:
            raise ValueError(f"{label} hash mismatch")
        return content

    def _path(self, relative_path: str) -> Path:
        path = (self._workspace_root / relative_path).resolve(strict=True)
        if not path.is_relative_to(self._workspace_root):
            raise ValueError("frozen input path escapes workspace root")
        return path


def _select_source_order(
    records: list[EvaluationQuery], configured_query_ids: list[str]
) -> list[EvaluationQuery]:
    configured = set(configured_query_ids)
    selected = [record for record in records if record.query_id in configured]
    found = {record.query_id for record in selected}
    if found != configured:
        missing = sorted(configured.difference(found))
        raise ValueError(f"configured query IDs are absent from Gold associations: {missing}")
    return selected


def _sha256(content: bytes) -> Sha256:
    return f"sha256:{hashlib.sha256(content).hexdigest()}"


_RecordT = TypeVar("_RecordT", bound=DomainModel)


def _parse_jsonl_bytes(content: bytes, model: type[_RecordT]) -> list[_RecordT]:
    return [model.model_validate_json(line) for line in content.splitlines() if line.strip()]
