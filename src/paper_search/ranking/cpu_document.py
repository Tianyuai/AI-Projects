"""Candidate-identity-preserving adapter for the CPU document ranker."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol, cast

from paper_search.domain.models import FusedPaper
from paper_search.learning.cpu_document_ranker import (
    CpuPairwiseDocumentRanker,
    DocumentCandidateEvidence,
)


class _DocumentRanker(Protocol):
    model_id: str

    def rank(
        self,
        query: str,
        candidates: Sequence[DocumentCandidateEvidence],
    ) -> list[DocumentCandidateEvidence]: ...


class DocumentRankingStage(Protocol):
    """Replaceable post-fusion ranker that may only permute candidates."""

    model_id: str

    def rank(
        self,
        query: str,
        candidates: Sequence[FusedPaper],
    ) -> list[FusedPaper]: ...


class CpuDocumentRankingStage:
    """Adapt a learned ranker to fused papers without changing their evidence."""

    def __init__(self, ranker: _DocumentRanker) -> None:
        self.ranker = ranker
        self.model_id = ranker.model_id

    def rank(
        self,
        query: str,
        candidates: Sequence[FusedPaper],
    ) -> list[FusedPaper]:
        rows = list(candidates)
        ids = [item.paper.canonical_id for item in rows]
        if len(ids) != len(set(ids)):
            raise ValueError("document ranking input candidate identity is not unique")
        evidence = [
            DocumentCandidateEvidence(
                paper=item.paper,
                baseline_score=item.score,
                source_ranks=item.source_ranks,
            )
            for item in rows
        ]
        ranked = self.ranker.rank(query, evidence)
        ranked_ids = [item.paper.canonical_id for item in ranked]
        if len(ranked_ids) != len(ids) or set(ranked_ids) != set(ids):
            raise ValueError("document ranker changed candidate identity")
        fused_by_id = {item.paper.canonical_id: item for item in rows}
        return [fused_by_id[identifier] for identifier in ranked_ids]


def _manifest_int(manifest: Mapping[str, object], name: str) -> int:
    value = manifest.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"CPU document ranker manifest {name} must be an integer")
    return value


def _manifest_float(manifest: Mapping[str, object], name: str) -> float:
    value = manifest.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"CPU document ranker manifest {name} must be numeric")
    return float(value)


def load_cpu_document_ranking_stage(
    manifest_path: Path,
    weights_path: Path,
) -> CpuDocumentRankingStage:
    """Load a frozen CPU ranker after validating its deployment manifest and hash."""

    return load_cpu_document_ranking_stage_bytes(
        manifest_path.read_bytes(),
        weights_path.read_bytes(),
    )


def load_cpu_document_ranking_stage_bytes(
    manifest_bytes: bytes,
    weights_bytes: bytes,
) -> CpuDocumentRankingStage:
    """Load a frozen CPU ranker from content-addressed input-lock bytes."""

    raw_manifest = json.loads(manifest_bytes)
    if not isinstance(raw_manifest, dict):
        raise ValueError("CPU document ranker manifest must be an object")
    manifest = cast(dict[str, object], raw_manifest)
    if manifest.get("schema_version") != "cpu-pairwise-document-ranker-manifest-v1":
        raise ValueError("unsupported CPU document ranker manifest schema")
    if manifest.get("model_id") != CpuPairwiseDocumentRanker.model_id:
        raise ValueError("unsupported CPU document ranker model id")
    actual_hash = "sha256:" + hashlib.sha256(weights_bytes).hexdigest()
    if manifest.get("model_sha256") != actual_hash:
        raise ValueError("CPU document ranker model hash mismatch")
    ranker = CpuPairwiseDocumentRanker.load_bytes(
        weights_bytes,
        dimension=_manifest_int(manifest, "dimension"),
        epochs=_manifest_int(manifest, "epochs"),
        learning_rate=_manifest_float(manifest, "learning_rate"),
        l2=_manifest_float(manifest, "l2"),
        learned_weight=_manifest_float(manifest, "learned_weight"),
        hard_negative_limit=_manifest_int(manifest, "hard_negative_limit"),
        seed=_manifest_int(manifest, "seed"),
    )
    return CpuDocumentRankingStage(ranker)


__all__ = [
    "CpuDocumentRankingStage",
    "DocumentRankingStage",
    "load_cpu_document_ranking_stage",
    "load_cpu_document_ranking_stage_bytes",
]
