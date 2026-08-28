from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import pytest

from paper_search.evaluation.predictions import paper_evaluation_id
from paper_search.learning.cpu_document_ranker import (
    DocumentCandidateEvidence,
    DocumentRankingQuery,
)
from paper_search.learning.large_scale_fusion_training import (
    FusionTrainingCheckpoint,
    FusionTrainingPackage,
    FrozenCandidateOverlayEntry,
    apply_frozen_candidate_overlay,
    build_document_ranking_query,
    index_training_receipts,
    load_training_package,
    query_has_gold_candidate,
    read_query_shard,
    read_fusion_checkpoint,
    with_context_label_files,
    write_query_shard,
    write_fusion_checkpoint,
)
from paper_search.retrieval.pasa_paper_database import (
    PASA_TRAINING_CONSTRAINT_INJECTED_SOURCE,
    PASA_TRAINING_GOLD_INJECTED_SOURCE,
)
from paper_search.learning.query_constraint_annotations import query_sha256
from scripts.train_large_scale_fusion import (
    _fit_input_sha256,
    _prepare_shards,
    _production_replay_batch_index,
    _revalidated_production_replay_queries,
)


def _jsonl(rows: list[dict[str, object]]) -> bytes:
    return b"".join(
        (json.dumps(row, sort_keys=True) + "\n").encode("utf-8") for row in rows
    )


def _bundle(task_rows: list[dict[str, object]], constraint_rows: list[dict[str, object]]) -> bytes:
    components = (b"baseline", b"fusion", _jsonl(task_rows), _jsonl(constraint_rows))
    return struct.pack("<8sQQQQ", b"F5PROD1\0", *(len(row) for row in components)) + b"".join(components)


def _task(query_id: str, query: str) -> dict[str, object]:
    return {
        "query_id": query_id,
        "query_sha256": "sha256:" + hashlib.sha256(query.encode()).hexdigest(),
        "role": "training",
        "split": "auto_train",
        "tasks": [],
        "ambiguous_fields": [],
        "task_label_status": "unresolved",
    }


def _constraint(query_id: str, query: str) -> dict[str, object]:
    return {
        "query_id": query_id,
        "query_sha256": "sha256:" + hashlib.sha256(query.encode()).hexdigest(),
        "role": "training",
        "split": "auto_train",
        "labels": [],
        "status": "partial",
    }


def test_load_training_package_preserves_receipt_order_and_requires_frozen_labels(
    tmp_path: Path,
) -> None:
    roots = [tmp_path / "receipt-a", tmp_path / "receipt-b"]
    for root in roots:
        root.mkdir()
    partition = tmp_path / "train.jsonl"
    rows = [
        {
            "query_id": "q1",
            "query": "first query",
                "gold_paper_ids": ["doi:10.1000/one"],
            "role": "training",
            "split": "auto_train",
        },
        {
            "query_id": "q2",
            "query": "second query",
                "gold_paper_ids": ["doi:10.1000/two"],
            "role": "training",
            "split": "auto_train",
        },
    ]
    partition.write_bytes(_jsonl(rows))
    handoff = tmp_path / "handoff.json"
    handoff.write_text(
        json.dumps(
            {
                "schema_version": "openalex-ranking-training-handoff-v1",
                "cumulative_unique_ready_query_ids": ["q2", "q1"],
                "ordered_receipt_roots": [str(root) for root in roots],
                "high_recall_candidate_supplement": {
                    "receipt_roots": [str(roots[1])],
                    "llm_request_count": 0,
                    "test_partition_touched": False,
                },
                "conflicts": ["q2"],
                "online_llm_requests": 0,
                "test_partition_touched": False,
            }
        ),
        encoding="utf-8",
    )
    bundle = tmp_path / "weights.bundle"
    bundle.write_bytes(
        _bundle(
            [_task("q1", "first query"), _task("q2", "second query")],
            [
                _constraint("q1", "first query"),
                _constraint("q2", "second query"),
            ],
        )
    )

    package = load_training_package(
        handoff_path=handoff,
        partition_path=partition,
        production_bundle_path=bundle,
    )

    assert package.query_ids == ("q2", "q1")
    assert package.ordered_receipt_roots == tuple(root.resolve() for root in roots)
    assert package.additive_receipt_roots == (roots[1].resolve(),)
    assert package.conflicting_query_ids == ("q2",)
    assert package.task_label_count == 2
    assert package.constraint_label_count == 2


def test_load_training_package_rejects_any_ready_query_without_both_label_families(
    tmp_path: Path,
) -> None:
    root = tmp_path / "receipt"
    root.mkdir()
    partition = tmp_path / "train.jsonl"
    partition.write_bytes(
        _jsonl(
            [
                {
                    "query_id": "q1",
                    "query": "first query",
                    "gold_paper_ids": ["doi:10.1000/one"],
                    "role": "training",
                    "split": "auto_train",
                }
            ]
        )
    )
    handoff = tmp_path / "handoff.json"
    handoff.write_text(
        json.dumps(
            {
                "schema_version": "openalex-ranking-training-handoff-v1",
                "cumulative_unique_ready_query_ids": ["q1"],
                "ordered_receipt_roots": [str(root)],
                "conflicts": [],
                "online_llm_requests": 0,
                "test_partition_touched": False,
            }
        ),
        encoding="utf-8",
    )
    bundle = tmp_path / "weights.bundle"
    bundle.write_bytes(_bundle([_task("q1", "first query")], []))

    with pytest.raises(ValueError, match="constraint label coverage"):
        load_training_package(
            handoff_path=handoff,
            partition_path=partition,
            production_bundle_path=bundle,
        )


def test_fusion_checkpoint_round_trip_preserves_next_batch_and_weights(
    tmp_path: Path,
) -> None:
    checkpoint = FusionTrainingCheckpoint(
        input_sha256="sha256:" + "a" * 64,
        epoch_index=3,
        next_batch_index=7,
        batch_count=12,
        pair_counts={"reliability": 19, "task_provenance": 5},
        query_counts={"reliability": 11, "task_provenance": 4},
        weights={
            "reliability": np.array([1.5, -2.0], dtype=np.float64),
            "task_provenance": np.array([0.25, 0.75], dtype=np.float64),
        },
        replay_pair_counts={"reliability": 7, "task_provenance": 3},
        replay_query_counts={"reliability": 5, "task_provenance": 2},
    )

    write_fusion_checkpoint(tmp_path, checkpoint)
    restored = read_fusion_checkpoint(tmp_path)

    assert restored.input_sha256 == checkpoint.input_sha256
    assert restored.epoch_index == 3
    assert restored.next_batch_index == 7
    assert restored.pair_counts == checkpoint.pair_counts
    assert restored.query_counts == checkpoint.query_counts
    assert restored.replay_pair_counts == checkpoint.replay_pair_counts
    assert restored.replay_query_counts == checkpoint.replay_query_counts
    for family in checkpoint.weights:
        np.testing.assert_array_equal(restored.weights[family], checkpoint.weights[family])


def test_fit_identity_binds_family_budgets_warm_start_and_replay() -> None:
    package = "sha256:" + "a" * 64
    budgets = {
        "entity": 48,
        "hard_constraint": 128,
        "reliability": 96,
        "task_provenance": 112,
    }

    assert _fit_input_sha256(package, pair_budget_by_family=budgets) != (
        _fit_input_sha256(
            package,
            pair_budget_by_family={**budgets, "reliability": 97},
        )
    )
    assert _fit_input_sha256(
        package,
        pair_budget_by_family=budgets,
        activation_manifest_sha256="sha256:" + "b" * 64,
    ) != _fit_input_sha256(
        package,
        pair_budget_by_family=budgets,
        activation_manifest_sha256="sha256:" + "c" * 64,
    )
    assert _fit_input_sha256(
        package,
        pair_budget_by_family=budgets,
        warm_start_weights_sha256="sha256:" + "d" * 64,
        replay_manifest_sha256="sha256:" + "e" * 64,
        replay_every_batches=4,
    ) != _fit_input_sha256(
        package,
        pair_budget_by_family=budgets,
        warm_start_weights_sha256="sha256:" + "f" * 64,
        replay_manifest_sha256="sha256:" + "e" * 64,
        replay_every_batches=4,
    )
    assert _fit_input_sha256(
        package,
        pair_budget_by_family=budgets,
        method_usage_evidence_schema_version="method-usage-v1",
    ) != _fit_input_sha256(
        package,
        pair_budget_by_family=budgets,
        method_usage_evidence_schema_version="method-usage-v2",
    )


def test_production_replay_rebinds_current_gold_and_drops_injected_candidates() -> None:
    query = "papers using graph neural networks"
    natural_gold = DocumentCandidateEvidence(
        paper={
            "canonical_id": "openalex:W1",
            "openalex_id": "W1",
            "title": "Graph neural networks",
        },
        baseline_score=1.0,
        source_ranks={"openalex-original": 1},
    )
    stale_gold = DocumentCandidateEvidence(
        paper={
            "canonical_id": "openalex:W2",
            "openalex_id": "W2",
            "title": "Stale Gold",
        },
        baseline_score=0.5,
        source_ranks={"openalex-original": 2},
    )
    injected = DocumentCandidateEvidence(
        paper={
            "canonical_id": "arxiv:2301.01234",
            "arxiv_id": "2301.01234",
            "title": "Injected",
            "sources": [PASA_TRAINING_GOLD_INJECTED_SOURCE],
        },
        baseline_score=0.25,
        source_ranks={"pasa-local-original": 3},
    )
    old = DocumentRankingQuery(
        query_id="q1",
        query=query,
        gold_paper_ids=["openalex:W2"],
        candidates=[natural_gold, stale_gold, injected],
    )
    package = FusionTrainingPackage(
        handoff_path=Path("handoff.json"),
        partition_path=Path("partition.jsonl"),
        production_bundle_path=Path("bundle.bin"),
        query_ids=("q1",),
        rows_by_query_id={
            "q1": {
                "query_id": "q1",
                "query": query,
                "gold_paper_ids": ["openalex:W1"],
                "role": "training",
                "split": "auto_train",
            }
        },
        ordered_receipt_roots=(),
        conflicting_query_ids=(),
        task_labels_bytes=b"",
        constraint_labels_bytes=b"",
        task_label_count=0,
        constraint_label_count=0,
        input_sha256="sha256:" + "a" * 64,
    )

    replay = _revalidated_production_replay_queries([old], package)

    assert len(replay) == 1
    assert replay[0].gold_paper_ids == ["openalex:W1"]
    assert [candidate.paper.canonical_id for candidate in replay[0].candidates] == [
        "openalex:W1",
        "openalex:W2",
    ]


def test_full_production_replay_interleaves_every_old_shard_exactly_once() -> None:
    indexes = [
        _production_replay_batch_index(
            batch_index,
            training_batch_count=335,
            replay_batch_count=287,
            replay_every_batches=1,
        )
        for batch_index in range(335)
    ]

    assert [index for index in indexes if index is not None] == list(range(287))


def test_context_label_files_replace_frozen_training_context_and_identity(
    tmp_path: Path,
) -> None:
    query = "papers using graph neural networks"
    package = FusionTrainingPackage(
        handoff_path=tmp_path / "handoff.json",
        partition_path=tmp_path / "train.jsonl",
        production_bundle_path=tmp_path / "weights.bundle",
        query_ids=("q1",),
        rows_by_query_id={
            "q1": {
                "query_id": "q1",
                "query": query,
                "role": "training",
                "split": "auto_train",
            }
        },
        ordered_receipt_roots=(),
        conflicting_query_ids=(),
        task_labels_bytes=_jsonl([_task("q1", query)]),
        constraint_labels_bytes=_jsonl([_constraint("q1", query)]),
        task_label_count=1,
        constraint_label_count=1,
        input_sha256="sha256:" + "a" * 64,
    )
    task = _task("q1", query)
    task["tasks"] = [{"normalized_value": "graph learning", "confidence": 0.9}]
    task["task_label_status"] = "runtime_deterministic"
    constraint = _constraint("q1", query)
    constraint.update(
        {
            "labels": ["method"],
            "methods": ["graph neural networks"],
            "label_sources": {"method": "local_deterministic"},
            "label_confidence": {"method": 0.9},
            "evidence": {"method": ["graph neural networks"]},
            "status": "accepted",
        }
    )
    task_path = tmp_path / "tasks.jsonl"
    constraint_path = tmp_path / "constraints.jsonl"
    task_path.write_bytes(_jsonl([task]))
    constraint_path.write_bytes(_jsonl([constraint]))

    updated = with_context_label_files(
        package,
        task_labels_path=task_path,
        constraint_labels_path=constraint_path,
    )

    assert updated.task_labels_bytes == task_path.read_bytes()
    assert updated.constraint_labels_bytes == constraint_path.read_bytes()
    assert updated.input_sha256 != package.input_sha256


def _paper(
    canonical_id: str, title: str, *, arxiv_id: str | None = None
) -> dict[str, object]:
    row: dict[str, object] = {
        "canonical_id": canonical_id,
        "title": title,
        "abstract": "compact abstract",
        "publication_year": 2024,
        "citation_count": 2,
    }
    if arxiv_id is not None:
        row["arxiv_id"] = arxiv_id
    return row


def _receipt_pair(
    root: Path,
    *,
    query_id: str,
    actions: list[dict[str, object]],
    results: list[dict[str, object]],
    candidate_source_namespace: str | None = None,
    retrieval_provenance: dict[str, object] | None = None,
) -> None:
    generation = root / "openalex" / "batch-0001" / "generation" / "attempt-01"
    retrieval = root / "openalex" / "batch-0001" / "retrieval" / "attempt-01"
    generation.mkdir(parents=True)
    retrieval.mkdir(parents=True)
    generation_payload: dict[str, object] = {
        "query_id": query_id,
        "attempt_status": "succeeded",
        "actions": actions,
    }
    if candidate_source_namespace is not None:
        generation_payload["generation_provenance"] = {
            "candidate_source_namespace": candidate_source_namespace
        }
    (generation / f"{query_id}.json").write_text(
        json.dumps(generation_payload),
        encoding="utf-8",
    )
    retrieval_payload: dict[str, object] = {
        "query_id": query_id,
        "attempt_status": "succeeded",
        "results": results,
    }
    if retrieval_provenance is not None:
        retrieval_payload["retrieval_provenance"] = retrieval_provenance
    (retrieval / f"{query_id}.json").write_text(
        json.dumps(retrieval_payload),
        encoding="utf-8",
    )


def test_receipt_index_uses_frozen_root_order_and_deduplicates_action_identity(
    tmp_path: Path,
) -> None:
    roots = [tmp_path / "old", tmp_path / "new"]
    for root in roots:
        root.mkdir()
    query = "find the relevant paper"
    _receipt_pair(
        roots[0],
        query_id="q1",
        actions=[
            {
                "action_id": "ceiling-candidate-anchor",
                "action_type": "text_search",
                "payload": {"query_text": query},
            },
            {
                "action_id": "old-semantic-original",
                "action_type": "text_search",
                "payload": {"query_text": query, "search_mode": "semantic"},
            },
        ],
        results=[
            {
                "action_id": "ceiling-candidate-anchor",
                "hits": [_paper("doi:10.1000/negative", "negative")],
            },
            {
                "action_id": "old-semantic-original",
                "hits": [_paper("doi:10.1000/old", "old duplicate")],
            },
        ],
    )
    _receipt_pair(
        roots[1],
        query_id="q1",
        actions=[
            {
                "action_id": "new-semantic-original",
                "action_type": "text_search",
                "payload": {"query_text": query, "search_mode": "semantic"},
            }
        ],
        results=[
            {
                "action_id": "new-semantic-original",
                "hits": [
                    _paper(
                        "doi:10.48550/arxiv.2401.00001",
                        "gold replacement",
                        arxiv_id="2401.00001",
                    )
                ],
            }
        ],
    )
    package = FusionTrainingPackage(
        handoff_path=tmp_path / "handoff.json",
        partition_path=tmp_path / "train.jsonl",
        production_bundle_path=tmp_path / "weights.bundle",
        query_ids=("q1",),
        rows_by_query_id={
            "q1": {
                "query_id": "q1",
                "query": query,
                "gold_paper_ids": ["arxiv:2401.00001"],
                "role": "training",
                "split": "auto_train",
            }
        },
        ordered_receipt_roots=tuple(root.resolve() for root in roots),
        conflicting_query_ids=(),
        task_labels_bytes=b"",
        constraint_labels_bytes=b"",
        task_label_count=0,
        constraint_label_count=0,
        input_sha256="sha256:" + "a" * 64,
    )

    index = index_training_receipts(package)
    ranking_query = build_document_ranking_query(package, "q1", index["q1"])

    candidate_ids = {row.paper.canonical_id for row in ranking_query.candidates}
    assert candidate_ids == {
        "doi:10.1000/negative",
        "doi:10.48550/arxiv.2401.00001",
    }
    assert "doi:10.1000/old" not in candidate_ids
    assert set(ranking_query.gold_paper_ids) & {
        paper_evaluation_id(candidate.paper) for candidate in ranking_query.candidates
    }
    assert query_has_gold_candidate(ranking_query) is True

    shard = tmp_path / "shard-00000.jsonl.gz"
    write_query_shard(shard, [ranking_query])
    restored = read_query_shard(shard)

    assert restored == [ranking_query]


def test_training_query_can_merge_repeated_action_hits_monotonically(
    tmp_path: Path,
) -> None:
    roots = [tmp_path / "successful", tmp_path / "empty-retry"]
    for root in roots:
        root.mkdir()
    query = "find a stable candidate"
    action = {
        "action_id": "anchor",
        "action_type": "text_search",
        "payload": {"query_text": query},
    }
    _receipt_pair(
        roots[0],
        query_id="q1",
        actions=[action],
        results=[{"action_id": "anchor", "hits": [_paper("openalex:W1", "one")]}],
    )
    _receipt_pair(
        roots[1],
        query_id="q1",
        actions=[action],
        results=[{"action_id": "anchor", "hits": []}],
    )
    package = FusionTrainingPackage(
        handoff_path=tmp_path / "handoff.json",
        partition_path=tmp_path / "train.jsonl",
        production_bundle_path=tmp_path / "weights.bundle",
        query_ids=("q1",),
        rows_by_query_id={
            "q1": {
                "query_id": "q1",
                "query": query,
                "gold_paper_ids": ["openalex:W1"],
                "role": "training",
                "split": "auto_train",
            }
        },
        ordered_receipt_roots=tuple(root.resolve() for root in roots),
        conflicting_query_ids=(),
        task_labels_bytes=b"",
        constraint_labels_bytes=b"",
        task_label_count=0,
        constraint_label_count=0,
        input_sha256="sha256:" + "a" * 64,
    )

    index = index_training_receipts(package)
    ranking_query = build_document_ranking_query(
        package,
        "q1",
        index["q1"],
        additive_receipt_roots=(roots[1].resolve(),),
    )

    assert [row.paper.canonical_id for row in ranking_query.candidates] == [
        "openalex:W1"
    ]


def test_additive_recall_fairly_merges_new_candidates_without_reinforcing_baseline(
    tmp_path: Path,
) -> None:
    roots = [tmp_path / "baseline", tmp_path / "supplement"]
    for root in roots:
        root.mkdir()
    query = "find a stable candidate"
    _receipt_pair(
        roots[0],
        query_id="q1",
        actions=[
            {
                "action_id": "baseline-anchor",
                "action_type": "text_search",
                "payload": {"query_text": query},
            }
        ],
        results=[
            {
                "action_id": "baseline-anchor",
                "hits": [
                    _paper("openalex:W1", "existing candidate"),
                    _paper("openalex:W2", "baseline neighbor"),
                ],
            }
        ],
    )
    _receipt_pair(
        roots[1],
        query_id="q1",
        actions=[
            {
                "action_id": "cross-vocabulary-supplement",
                "action_type": "text_search",
                "payload": {"query_text": "stable cross vocabulary candidate"},
            }
        ],
        results=[
            {
                "action_id": "cross-vocabulary-supplement",
                "hits": [
                    _paper("openalex:W3", "new supplemental candidate"),
                    _paper("openalex:W1", "existing candidate"),
                ],
            }
        ],
    )
    package = FusionTrainingPackage(
        handoff_path=tmp_path / "handoff.json",
        partition_path=tmp_path / "train.jsonl",
        production_bundle_path=tmp_path / "weights.bundle",
        query_ids=("q1",),
        rows_by_query_id={
            "q1": {
                "query_id": "q1",
                "query": query,
                "gold_paper_ids": ["openalex:W1"],
                "role": "training",
                "split": "auto_train",
            }
        },
        ordered_receipt_roots=tuple(root.resolve() for root in roots),
        conflicting_query_ids=(),
        task_labels_bytes=b"",
        constraint_labels_bytes=b"",
        task_label_count=0,
        constraint_label_count=0,
        input_sha256="sha256:" + "a" * 64,
    )

    index = index_training_receipts(package)
    baseline = build_document_ranking_query(
        package,
        "q1",
        (index["q1"][0],),
    )
    augmented = build_document_ranking_query(
        package,
        "q1",
        index["q1"],
        additive_receipt_roots=(roots[1].resolve(),),
        non_reinforcing_additive=True,
    )

    baseline_by_id = {
        candidate.paper.canonical_id: candidate for candidate in baseline.candidates
    }
    augmented_by_id = {
        candidate.paper.canonical_id: candidate for candidate in augmented.candidates
    }
    assert [candidate.paper.canonical_id for candidate in augmented.candidates] == [
        "openalex:W1",
        "openalex:W3",
        "openalex:W2",
    ]
    assert augmented_by_id["openalex:W1"] == baseline_by_id["openalex:W1"]
    assert augmented_by_id["openalex:W2"] == baseline_by_id["openalex:W2"]
    assert augmented_by_id["openalex:W3"].support_count == 1


def test_training_shards_honor_sealed_additive_receipt_roots(tmp_path: Path) -> None:
    roots = [tmp_path / "successful", tmp_path / "empty-retry"]
    for root in roots:
        root.mkdir()
    query = "find a stable candidate"
    action = {
        "action_id": "anchor",
        "action_type": "text_search",
        "payload": {"query_text": query},
    }
    _receipt_pair(
        roots[0],
        query_id="q1",
        actions=[action],
        results=[{"action_id": "anchor", "hits": [_paper("openalex:W1", "one")]}],
    )
    _receipt_pair(
        roots[1],
        query_id="q1",
        actions=[action],
        results=[{"action_id": "anchor", "hits": []}],
    )
    package = FusionTrainingPackage(
        handoff_path=tmp_path / "handoff.json",
        partition_path=tmp_path / "train.jsonl",
        production_bundle_path=tmp_path / "weights.bundle",
        query_ids=("q1",),
        rows_by_query_id={
            "q1": {
                "query_id": "q1",
                "query": query,
                "gold_paper_ids": ["openalex:W1"],
                "role": "training",
                "split": "auto_train",
            }
        },
        ordered_receipt_roots=tuple(root.resolve() for root in roots),
        conflicting_query_ids=(),
        task_labels_bytes=b"",
        constraint_labels_bytes=b"",
        task_label_count=0,
        constraint_label_count=0,
        input_sha256="sha256:" + "a" * 64,
        additive_receipt_roots=(roots[1].resolve(),),
    )

    _prepare_shards(package=package, shard_dir=tmp_path / "shards", batch_size=1)
    restored = read_query_shard(tmp_path / "shards" / "shard-00000.jsonl.gz")

    assert [row.paper.canonical_id for row in restored[0].candidates] == [
        "openalex:W1"
    ]


def test_query_gold_match_uses_all_normalized_candidate_identity_aliases() -> None:
    query = DocumentRankingQuery(
        query_id="q-alias",
        query="find the aliased paper",
        gold_paper_ids=["openalex:W123"],
        candidates=[
            {
                "paper": {
                    "canonical_id": "doi:10.1000/publisher-version",
                    "doi": "10.1000/publisher-version",
                    "openalex_id": "W123",
                    "title": "Published version",
                },
                "baseline_score": 1.0,
                "source_ranks": {"openalex": 1},
            }
        ],
    )

    assert query_has_gold_candidate(query) is True


def test_query_gold_match_bridges_arxiv_and_datacite_doi_aliases() -> None:
    query = DocumentRankingQuery(
        query_id="q-datacite",
        query="find the preprint",
        gold_paper_ids=["arxiv:2301.01234"],
        candidates=[
            {
                "paper": {
                    "canonical_id": "doi:10.48550/arxiv.2301.01234",
                    "doi": "10.48550/arxiv.2301.01234",
                    "title": "Preprint",
                },
                "baseline_score": 1.0,
                "source_ranks": {"openalex": 1},
            }
        ],
    )

    assert query_has_gold_candidate(query) is True


def test_candidate_source_namespace_keeps_offline_pasa_action_additive(
    tmp_path: Path,
) -> None:
    roots = [tmp_path / "openalex", tmp_path / "pasa"]
    for root in roots:
        root.mkdir()
    query_text = "graph retrieval"
    action = {
        "action_id": "anchor",
        "action_type": "text_search",
        "payload": {"query_text": query_text},
    }
    _receipt_pair(
        roots[0],
        query_id="q1",
        actions=[action],
        results=[{"action_id": "anchor", "hits": [_paper("openalex:W1", "one")]}],
    )
    _receipt_pair(
        roots[1],
        query_id="q1",
        actions=[action],
        results=[
            {
                "action_id": "anchor",
                "hits": [
                    {
                        **_paper("arxiv:2301.01234", "two"),
                        "sources": [
                            "pasa_paper_database",
                            "pasa_training_gold_injected",
                        ],
                    }
                ],
            }
        ],
        candidate_source_namespace="pasa_paper_database",
    )
    package = FusionTrainingPackage(
        handoff_path=tmp_path / "handoff.json",
        partition_path=tmp_path / "train.jsonl",
        production_bundle_path=tmp_path / "weights.bundle",
        query_ids=("q1",),
        rows_by_query_id={
            "q1": {
                "query_id": "q1",
                "query": query_text,
                "gold_paper_ids": ["arxiv:2301.01234"],
                "role": "training",
                "split": "auto_train",
            }
        },
        ordered_receipt_roots=tuple(root.resolve() for root in roots),
        conflicting_query_ids=(),
        task_labels_bytes=b"",
        constraint_labels_bytes=b"",
        task_label_count=0,
        constraint_label_count=0,
        input_sha256="sha256:" + "a" * 64,
    )

    index = index_training_receipts(package)
    ranking_query = build_document_ranking_query(package, "q1", index["q1"])

    assert {candidate.paper.canonical_id for candidate in ranking_query.candidates} == {
        "openalex:W1",
        "arxiv:2301.01234",
    }
    injected = next(
        candidate
        for candidate in ranking_query.candidates
        if candidate.paper.canonical_id == "arxiv:2301.01234"
    )
    assert "pasa_training_gold_injected" in injected.paper.sources


def test_legacy_mixed_pasa_gold_is_marked_during_hydration(tmp_path: Path) -> None:
    root = tmp_path / "legacy-pasa"
    root.mkdir()
    query_text = "papers using graph networks"
    action = {
        "action_id": "pasa-local-original",
        "action_type": "text_search",
        "payload": {"query_text": query_text},
    }
    _receipt_pair(
        root,
        query_id="q1",
        actions=[action],
        results=[
            {
                "action_id": "pasa-local-original",
                "hits": [
                    {
                        **_paper(
                            "arxiv:2301.01234",
                            "graph networks",
                            arxiv_id="2301.01234",
                        ),
                        "sources": ["pasa_paper_database"],
                    }
                ],
            }
        ],
        candidate_source_namespace="pasa_paper_database",
        retrieval_provenance={
            "candidate_policy": "mixed_lexical_plus_gold_training",
            "mixed_candidate_audit": {"direct_gold_candidate_count": 1},
        },
    )
    package = FusionTrainingPackage(
        handoff_path=tmp_path / "handoff.json",
        partition_path=tmp_path / "train.jsonl",
        production_bundle_path=tmp_path / "weights.bundle",
        query_ids=("q1",),
        rows_by_query_id={
            "q1": {
                "query_id": "q1",
                "query": query_text,
                "gold_paper_ids": ["arxiv:2301.01234"],
                "role": "training",
                "split": "auto_train",
            }
        },
        ordered_receipt_roots=(root.resolve(),),
        conflicting_query_ids=(),
        task_labels_bytes=b"",
        constraint_labels_bytes=b"",
        task_label_count=0,
        constraint_label_count=0,
        input_sha256="sha256:" + "a" * 64,
    )

    index = index_training_receipts(package)
    ranking_query = build_document_ranking_query(package, "q1", index["q1"])

    assert "pasa_training_gold_injected" in ranking_query.candidates[0].paper.sources


def test_frozen_candidate_overlay_is_append_only_and_hard_constraint_only() -> None:
    query = DocumentRankingQuery.model_validate(
        {
            "query_id": "q1",
            "query": "vision without adversarial training",
            "gold_paper_ids": ["openalex:W1"],
            "candidates": [
                {
                    "paper": {"canonical_id": "openalex:W1", "title": "Gold"},
                    "baseline_score": 1.0,
                    "source_ranks": {"openalex": 1},
                }
            ],
        }
    )
    overlay = DocumentCandidateEvidence.model_validate(
        {
            "paper": {
                "canonical_id": "arxiv:2301.00001",
                "arxiv_id": "2301.00001",
                "title": "Uses adversarial training",
                "sources": [PASA_TRAINING_CONSTRAINT_INJECTED_SOURCE],
            },
            "baseline_score": 0.5,
            "source_ranks": {"pasa-overlay": 1},
        }
    )
    entry = FrozenCandidateOverlayEntry(
        query_sha256=query_sha256(query.query), candidates=(overlay,)
    )

    expanded = apply_frozen_candidate_overlay(query, entry)

    assert [row.paper.canonical_id for row in expanded.candidates] == [
        "openalex:W1",
        "arxiv:2301.00001",
    ]
    with pytest.raises(ValueError, match="overlaps immutable base"):
        apply_frozen_candidate_overlay(
            query,
            FrozenCandidateOverlayEntry(
                query_sha256=query_sha256(query.query),
                candidates=(query.candidates[0],),
            ),
        )
    with pytest.raises(ValueError, match="hard-constraint-only marker"):
        apply_frozen_candidate_overlay(
            query,
            FrozenCandidateOverlayEntry(
                query_sha256=query_sha256(query.query),
                candidates=(
                    overlay.model_copy(
                        update={
                            "paper": overlay.paper.model_copy(update={"sources": []})
                        }
                    ),
                ),
            ),
        )
