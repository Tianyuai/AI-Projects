from __future__ import annotations

import json
from pathlib import Path

import pytest

import paper_search.learning as learning
from paper_search.domain.models import Paper


def test_write_pasa_supplement_receipt_is_offline_namespaced_and_immutable(
    tmp_path: Path,
) -> None:
    papers = [
        Paper(
            canonical_id="arxiv:2301.01234",
            arxiv_id="2301.01234",
            title="Graph retrieval",
            sources=["pasa_paper_database"],
        )
    ]

    paths = learning.write_pasa_supplement_receipt(
        output_root=tmp_path,
        query_id="q1",
        query="graph retrieval",
        papers=papers,
        index_sha256="sha256:" + "a" * 64,
    )
    repeated = learning.write_pasa_supplement_receipt(
        output_root=tmp_path,
        query_id="q1",
        query="graph retrieval",
        papers=papers,
        index_sha256="sha256:" + "a" * 64,
    )

    assert repeated == paths
    generation = json.loads(paths["generation"].read_text(encoding="utf-8"))
    retrieval = json.loads(paths["retrieval"].read_text(encoding="utf-8"))
    assert generation["generation_provenance"] == {
        "candidate_source_namespace": "pasa_paper_database",
        "gold_visibility": "training_labels_only",
        "index_sha256": "sha256:" + "a" * 64,
        "llm_request_count": 0,
        "network_request_count": 0,
    }
    assert retrieval["results"][0]["hits"][0]["canonical_id"] == (
        "arxiv:2301.01234"
    )
    assert retrieval["usage"]["search_api_calls"] == 0

    with pytest.raises(FileExistsError, match="immutable PASA receipt"):
        learning.write_pasa_supplement_receipt(
            output_root=tmp_path,
            query_id="q1",
            query="different query",
            papers=papers,
            index_sha256="sha256:" + "a" * 64,
        )


def test_write_augmented_handoff_appends_only_offline_root(tmp_path: Path) -> None:
    existing = tmp_path / "existing"
    supplement = tmp_path / "pasa"
    existing.mkdir()
    supplement.mkdir()
    base = tmp_path / "base.json"
    base.write_text(
        json.dumps(
            {
                "schema_version": "openalex-ranking-training-handoff-v1",
                "cumulative_unique_ready_query_ids": ["q1"],
                "ordered_receipt_roots": [str(existing.resolve())],
                "conflicts": [],
                "online_llm_requests": 0,
                "test_partition_touched": False,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "augmented.json"

    payload = learning.write_pasa_augmented_handoff(
        base_handoff_path=base,
        supplement_root=supplement,
        output_path=output,
        supplement_query_count=1,
        index_sha256="sha256:" + "b" * 64,
    )

    assert payload["ordered_receipt_roots"] == [
        str(existing.resolve()),
        str(supplement.resolve()),
    ]
    assert payload["pasa_offline_supplement"]["network_request_count"] == 0
    assert payload["test_partition_touched"] is False


def test_write_mixed_receipt_binds_positive_negative_leakage_guard(
    tmp_path: Path,
) -> None:
    papers = [
        Paper(canonical_id="arxiv:1", arxiv_id="1", title="Gold"),
        Paper(canonical_id="arxiv:2", arxiv_id="2", title="Negative"),
    ]

    paths = learning.write_pasa_supplement_receipt(
        output_root=tmp_path,
        query_id="q1",
        query="dataset retrieval",
        papers=papers,
        index_sha256="sha256:" + "c" * 64,
        mixed_candidate_audit={
            "positive_candidate_count": 1,
            "lexical_negative_candidate_count": 1,
            "mixed_positive_negative": True,
        },
    )

    generation = json.loads(paths["generation"].read_text(encoding="utf-8"))
    provenance = generation["generation_provenance"]
    assert provenance["candidate_policy"] == "mixed_lexical_plus_gold_training"
    assert provenance["source_label_leakage_guard"] == (
        "positive_and_negative_share_action"
    )
    assert provenance["mixed_candidate_audit"]["positive_candidate_count"] == 1

    with pytest.raises(ValueError, match="both positive and negative"):
        learning.write_pasa_supplement_receipt(
            output_root=tmp_path / "invalid",
            query_id="q2",
            query="dataset retrieval",
            papers=papers[:1],
            index_sha256="sha256:" + "c" * 64,
            mixed_candidate_audit={
                "positive_candidate_count": 1,
                "lexical_negative_candidate_count": 0,
                "mixed_positive_negative": False,
            },
        )
