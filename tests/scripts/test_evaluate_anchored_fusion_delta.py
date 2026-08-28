from __future__ import annotations

from pathlib import Path
import subprocess
import sys

from paper_search.domain.models import Paper
from paper_search.learning.cpu_document_ranker import (
    DocumentCandidateEvidence,
    DocumentRankingQuery,
)
from scripts import evaluate_anchored_fusion_delta as evaluation


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "evaluate_anchored_fusion_delta.py"
)


def test_cli_exposes_only_the_frozen_delta_gate_inputs() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    for option in (
        "--production-shard-dir",
        "--oof-run-dir",
        "--oof-report",
        "--context-freeze-manifest",
        "--alphas",
        "--reliability-scales",
        "--task-provenance-scale",
        "--candidate-view",
        "--trainable-families",
        "--selection-fold",
        "--output",
    ):
        assert option in completed.stdout


def test_online_only_view_removes_pasa_only_candidates_but_keeps_online_support() -> None:
    pasa_only = DocumentCandidateEvidence(
        paper=Paper(
            canonical_id="arxiv:2301.00001",
            arxiv_id="2301.00001",
            title="PASA only",
        ),
        baseline_score=1.0,
        source_ranks={"pasa-local-original@receipt": 1},
    )
    mixed = DocumentCandidateEvidence(
        paper=Paper(
            canonical_id="arxiv:2301.00002",
            arxiv_id="2301.00002",
            title="Mixed",
        ),
        baseline_score=0.9,
        source_ranks={"pasa-local-original@receipt": 2, "policy-1@receipt": 5},
    )
    online = DocumentCandidateEvidence(
        paper=Paper(canonical_id="openalex:W3", openalex_id="W3", title="Online"),
        baseline_score=0.8,
        source_ranks={"policy-2@receipt": 3},
    )
    query = DocumentRankingQuery(
        query_id="q1",
        query="test query",
        gold_paper_ids=["arxiv:2301.00001", "arxiv:2301.00002"],
        candidates=[pasa_only, mixed, online],
    )

    filtered, removed_count = evaluation._online_retrieval_candidate_view(query)

    assert [row.paper.canonical_id for row in filtered.candidates] == [
        "arxiv:2301.00002",
        "openalex:W3",
    ]
    assert removed_count == 1
    assert len(query.candidates) == 3
