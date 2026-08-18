from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from paper_search.domain.models import FusedPaper, Paper
from paper_search.learning.cpu_document_ranker import CpuPairwiseDocumentRanker
from paper_search.ranking.cpu_document import (
    CpuDocumentRankingStage,
    load_cpu_document_ranking_stage,
    load_cpu_document_ranking_stage_bytes,
)


def _fused(identifier: str, title: str, *, score: float, rank: int) -> FusedPaper:
    return FusedPaper(
        paper=Paper(
            canonical_id=identifier,
            openalex_id=identifier,
            title=title,
            is_retracted=False,
        ),
        score=score,
        source_ranks={"lexical-anchor": rank},
    )


class _ReverseRanker:
    model_id = "reverse-fixture-v1"

    def rank(self, query, candidates):
        assert query == "graph retrieval"
        return list(reversed(candidates))


class _DroppingRanker:
    model_id = "dropping-fixture-v1"

    def rank(self, query, candidates):
        return list(candidates[:-1])


def test_cpu_document_stage_reorders_without_replacing_fused_evidence() -> None:
    first = _fused("openalex:W1", "first", score=0.2, rank=1)
    second = _fused("openalex:W2", "second", score=0.1, rank=2)
    stage = CpuDocumentRankingStage(_ReverseRanker())

    ranked = stage.rank("graph retrieval", [first, second])

    assert ranked == [second, first]
    assert {item.paper.canonical_id for item in ranked} == {
        item.paper.canonical_id for item in [first, second]
    }


def test_cpu_document_stage_rejects_candidate_identity_changes() -> None:
    stage = CpuDocumentRankingStage(_DroppingRanker())
    candidates = [
        _fused("openalex:W1", "first", score=0.2, rank=1),
        _fused("openalex:W2", "second", score=0.1, rank=2),
    ]

    with pytest.raises(ValueError, match="candidate identity"):
        stage.rank("graph retrieval", candidates)


def test_cpu_document_stage_loader_verifies_hash_and_restores_configuration(
    tmp_path,
) -> None:
    weights_path = tmp_path / "ranker.f64"
    weights = np.zeros(64, dtype="<f8")
    weights[3] = 1.25
    weights_path.write_bytes(weights.tobytes())
    digest = hashlib.sha256(weights_path.read_bytes()).hexdigest()
    manifest_path = tmp_path / "ranker.json"
    manifest = {
        "schema_version": "cpu-pairwise-document-ranker-manifest-v1",
        "model_id": CpuPairwiseDocumentRanker.model_id,
        "model_sha256": f"sha256:{digest}",
        "dimension": 64,
        "epochs": 3,
        "learning_rate": 0.04,
        "l2": 0.00001,
        "learned_weight": 0.5,
        "hard_negative_limit": 11,
        "seed": 17,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    stage = load_cpu_document_ranking_stage(manifest_path, weights_path)

    assert stage.model_id == CpuPairwiseDocumentRanker.model_id
    assert stage.ranker.dimension == 64
    assert stage.ranker.epochs == 3
    assert stage.ranker.learned_weight == 0.5
    assert np.array_equal(stage.ranker.weights, weights.astype(np.float64))

    manifest["model_sha256"] = "sha256:" + "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_cpu_document_ranking_stage(manifest_path, weights_path)


def test_cpu_document_stage_loads_from_verified_bytes_without_paths() -> None:
    weights = np.zeros(64, dtype="<f8")
    weights[7] = 2.5
    weights_bytes = weights.tobytes()
    manifest_bytes = json.dumps(
        {
            "schema_version": "cpu-pairwise-document-ranker-manifest-v1",
            "model_id": CpuPairwiseDocumentRanker.model_id,
            "model_sha256": f"sha256:{hashlib.sha256(weights_bytes).hexdigest()}",
            "dimension": 64,
            "epochs": 3,
            "learning_rate": 0.04,
            "l2": 0.00001,
            "learned_weight": 0.5,
            "hard_negative_limit": 11,
            "seed": 17,
        }
    ).encode("utf-8")

    stage = load_cpu_document_ranking_stage_bytes(manifest_bytes, weights_bytes)

    assert stage.model_id == CpuPairwiseDocumentRanker.model_id
    assert np.array_equal(stage.ranker.weights, weights.astype(np.float64))
