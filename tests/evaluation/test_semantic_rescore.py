from __future__ import annotations

from pathlib import Path

import pytest

from paper_search.control.pricing import load_quality_gate_policy
from paper_search.evaluation.dataset import EvaluationQuery, IdentifierMap
from paper_search.evaluation.semantic_rescore import (
    GenerationHashes,
    SemanticRescoreReport,
    SourceProjection,
    build_rescore_report,
    decide_bottleneck,
    score_source,
)


POLICY = load_quality_gate_policy(Path("configs/quality_gates_v1.yaml"))
HASH = "sha256:" + "a" * 64
GOLD = (
    EvaluationQuery(
        query_id="q1",
        query="first",
        relevant_paper_ids=[
            "arxiv:2401.00001",
            "doi:10.48550/arxiv.2401.00001",
        ],
    ),
    EvaluationQuery(
        query_id="q2",
        query="second",
        relevant_paper_ids=["arxiv:2401.00001"],
    ),
)
VERIFIED_MAP = IdentifierMap.from_bytes(
    b'{"arxiv:2401.00001":"doi:10.48550/arxiv.2401.00001",'
    b'"openalex:W7":"doi:10.48550/arxiv.2401.00001"}'
)


def _source(
    *,
    label: str = "formal_baseline_2026_08_10",
    kind: str = "formal_run",
    verification_status: str = "formal_validated",
    capture_replay_status: str = "not_applicable",
    query_ids: tuple[str, ...] = ("q1", "q2"),
    retrieved: dict[str, tuple[str, ...]] | None = None,
    post_filter: dict[str, tuple[str, ...]] | None = None,
    selected: dict[str, tuple[str, ...]] | None = None,
) -> SourceProjection:
    return SourceProjection(
        label=label,
        kind=kind,
        verification_status=verification_status,
        capture_replay_status=capture_replay_status,
        binding_hashes={"run": HASH},
        query_ids=query_ids,
        retrieved_paper_ids=retrieved
        or {
            "q1": ("doi:10.48550/arxiv.2401.00001",),
            "q2": ("openalex:W7",),
        },
        post_filter_paper_ids=post_filter
        or {
            "q1": ("doi:10.48550/arxiv.2401.00001",),
            "q2": ("openalex:W7",),
        },
        selected_paper_ids=selected
        or {
            "q1": ("doi:10.48550/arxiv.2401.00001",),
            "q2": ("openalex:W7",),
        },
    )


FORMAL_SOURCE = _source()


def _generation_hashes() -> GenerationHashes:
    return GenerationHashes(
        public_audit_sha256=HASH,
        gold_sha256=HASH,
        identity_evidence_sha256=HASH,
        snapshot_manifest_sha256=HASH,
        private_audit_sha256=HASH,
        candidate_map_sha256=HASH,
    )


def _four_sources() -> tuple[SourceProjection, ...]:
    return (
        FORMAL_SOURCE,
        _source(label="formal_baseline_2026_08_09"),
        _source(
            label="legacy_title_2026_08_05",
            kind="legacy_hash_bound_run",
            verification_status="legacy_hash_bound",
        ),
        _source(
            label="query_evolution_prompt_v2",
            kind="sealed_probe",
            verification_status="probe_verified",
            capture_replay_status="matched",
        ),
    )


def test_score_source_conserves_resolved_associations() -> None:
    row = score_source(GOLD, VERIFIED_MAP, FORMAL_SOURCE, policy=POLICY)
    stages = row.pipeline_stages

    assert stages.total_gold_associations == (
        stages.not_retrieved
        + stages.filtered_out
        + stages.ranked_outside_top50
        + stages.selected_top50
    )
    assert stages.total_gold_associations == 2
    assert stages.selected_top50 == 2
    assert row.true_positive_count == 2
    assert row.direct_same_arxiv_hit_count == 1
    assert [check.rule_id for check in row.metric_quality_checks] == [
        "hard-filter-recall-loss",
        "macro-recall-positive",
        "micro-recall-positive",
    ]


def test_score_source_collapses_aliases_within_a_query_but_not_across_queries() -> None:
    row = score_source(GOLD, VERIFIED_MAP, FORMAL_SOURCE, policy=POLICY)

    assert row.pipeline_stages.total_gold_associations == 2
    assert row.true_positive_count == 2


def test_score_source_assigns_exact_stage_precedence() -> None:
    source = _source(
        retrieved={
            "q1": ("openalex:W7",),
            "q2": ("openalex:W7",),
        },
        post_filter={"q1": ("openalex:W7",), "q2": ()},
        selected={"q1": (), "q2": ()},
    )

    row = score_source(GOLD, VERIFIED_MAP, source, policy=POLICY)

    assert row.pipeline_stages.filtered_out == 1
    assert row.pipeline_stages.ranked_outside_top50 == 1
    assert row.pipeline_stages.selected_top50 == 0
    assert row.pipeline_stages.not_retrieved == 0


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (_source(query_ids=("q2", "q1")), "query IDs"),
        (
            _source(
                post_filter={"q1": (), "q2": ()},
                selected={
                    "q1": ("doi:10.48550/arxiv.2401.00001",),
                    "q2": (),
                },
            ),
            "selected IDs must be a subset",
        ),
    ],
)
def test_score_source_rejects_incompatible_source_projection(
    source: SourceProjection,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        score_source(GOLD, VERIFIED_MAP, source, policy=POLICY)


def test_build_report_requires_fixed_source_order_and_equal_denominators() -> None:
    report = build_rescore_report(
        gold=GOLD,
        identifier_map=VERIFIED_MAP,
        sources=_four_sources(),
        policy=POLICY,
        generation_hashes=_generation_hashes(),
        quality_policy_sha256=HASH,
    )

    assert [row.label for row in report.runs] == [
        "formal_baseline_2026_08_10",
        "formal_baseline_2026_08_09",
        "legacy_title_2026_08_05",
        "query_evolution_prompt_v2",
    ]
    payload = report.model_dump(mode="python")
    payload["runs"][1]["pipeline_stages"]["total_gold_associations"] = 1
    payload["runs"][1]["pipeline_stages"]["selected_top50"] = 1
    with pytest.raises(ValueError, match="same total_gold_associations"):
        SemanticRescoreReport.model_validate(payload)
    with pytest.raises(ValueError, match="fixed order"):
        build_rescore_report(
            gold=GOLD,
            identifier_map=VERIFIED_MAP,
            sources=tuple(reversed(_four_sources())),
            policy=POLICY,
            generation_hashes=_generation_hashes(),
            quality_policy_sha256=HASH,
        )


def test_decide_bottleneck_reports_designated_tie() -> None:
    tied = score_source(
        GOLD,
        VERIFIED_MAP,
        _source(
            retrieved={"q1": (), "q2": ("openalex:W7",)},
            post_filter={"q1": (), "q2": ()},
            selected={"q1": (), "q2": ()},
        ),
        policy=POLICY,
    )
    decision = decide_bottleneck((tied,))

    assert decision.primary_loss_stage is None
    assert decision.next_direction is None
    assert decision.reason_codes == ("largest_loss_tie",)


def test_decide_bottleneck_reports_cross_source_sensitivity() -> None:
    designated = score_source(
        GOLD,
        VERIFIED_MAP,
        _source(
            retrieved={"q1": (), "q2": ("openalex:W7",)},
            post_filter={"q1": (), "q2": ("openalex:W7",)},
            selected={"q1": (), "q2": ("openalex:W7",)},
        ),
        policy=POLICY,
    )
    comparison = score_source(
        GOLD,
        VERIFIED_MAP,
        _source(
            label="formal_baseline_2026_08_09",
            retrieved={
                "q1": ("openalex:W7",),
                "q2": ("openalex:W7",),
            },
            post_filter={"q1": (), "q2": ("openalex:W7",)},
            selected={"q1": (), "q2": ("openalex:W7",)},
        ),
        policy=POLICY,
    )

    decision = decide_bottleneck((designated, comparison))

    assert decision.primary_loss_stage == "not_retrieved"
    assert decision.next_direction is None
    assert decision.reason_codes == ("source_sensitivity",)
