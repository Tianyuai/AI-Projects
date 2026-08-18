from __future__ import annotations

from paper_search.learning.gold_retrievability_audit import (
    AuditSampleItem,
    FrozenAuditManifest,
    shard_frozen_audit_manifest,
)


def _item(index: int) -> AuditSampleItem:
    return AuditSampleItem(
        query_id=f"q{index}",
        stratum="intent=which_what|length=q1|gold=1",
        intent_family="which_what",
        length_bucket="q1",
        gold_count_bucket="1",
        query_token_count=5,
        gold_paper_count=1,
        fold=index % 3 + 1,
        selection_sha256=f"sha256:{index:064x}",
    )


def test_sharding_skips_completed_queries_and_preserves_remaining_union() -> None:
    manifest = FrozenAuditManifest(
        schema_version="gold-retrievability-audit-freeze-v1",
        dataset="pasa",
        split="auto_train",
        role="training",
        revision="revision",
        source_path="partition.jsonl",
        source_sha256=f"sha256:{'a' * 64}",
        seed="seed",
        sampling_method="proportional_hamilton_sha256",
        stratification_fields=(
            "intent_family",
            "query_length_quartile",
            "gold_count_bucket",
        ),
        population_query_count=10,
        sample_query_count=6,
        length_cut_points=(5, 10, 15),
        stratum_population_counts={"intent=which_what|length=q1|gold=1": 10},
        stratum_sample_counts={"intent=which_what|length=q1|gold=1": 6},
        fold_counts={1: 2, 2: 2, 3: 2},
        sample=[_item(index) for index in range(6)],
    )

    shards = shard_frozen_audit_manifest(
        manifest, skip_query_ids={"q0"}, shard_count=2
    )

    assert [shard.sample_query_count for shard in shards] == [3, 2]
    ids = [{item.query_id for item in shard.sample} for shard in shards]
    assert ids[0].isdisjoint(ids[1])
    assert ids[0] | ids[1] == {"q1", "q2", "q3", "q4", "q5"}
