"""Offline gate for v3 LLM actions used only as low-confidence supplements."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

_ROOT = Path(__file__).resolve().parents[1]
_SOURCE = str(_ROOT / "src")
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if _SOURCE not in sys.path:
    sys.path.insert(0, _SOURCE)

from paper_search.application.composition import (  # noqa: E402
    _locked_document_ranker,
    load_locked_identifier_map,
)
from paper_search.application.locks import load_verified_input_lock  # noqa: E402
from paper_search.domain.models import FusedPaper, Paper, ProviderResult  # noqa: E402
from paper_search.learning.adaptive_openalex_recall import (  # noqa: E402
    assess_openalex_recall_confidence,
)
from paper_search.learning.cpu_document_ranker import (  # noqa: E402
    DocumentCandidateEvidence,
)
from paper_search.processing.deduplicate import deduplicate_papers  # noqa: E402
from paper_search.processing.filter import apply_hard_filters  # noqa: E402
from paper_search.query.low_confidence_supplement import (  # noqa: E402
    LowConfidenceLLMActionSelection,
    select_low_confidence_llm_action,
)
from paper_search.ranking.fusion import fuse_provider_results  # noqa: E402
from scripts.evaluate_protected_action_mix import (  # noqa: E402
    _canonical_bytes,
    _decode_observations,
    _execution,
    _first_gold_rank,
    _load_json,
    _load_jsonl,
    _plan,
    _query_spec,
    _ranked_papers,
    _sha256,
    paper_identity_record,
    round_robin_cap_action_results,
)


_DEFAULT_VALIDATION = Path(
    "artifacts/llm-openalex-validations/"
    "protected-action-single-call-live-24-20260828"
)
_POLICY_VERSION = "low-confidence-llm-lexical-supplement-v1"


def promotion_decision(metrics: Mapping[str, int]) -> dict[str, object]:
    """Apply the pre-agreed minimal promotion gate."""

    query_count = metrics.get("query_count", 0)
    failed: list[str] = []
    exact_requirements = {
        "baseline_reconstruction_exact_query_count": query_count,
        "production_f5_query_count": query_count,
        "gold_blind_selection_query_count": query_count,
        "independent_quota_query_count": query_count,
        "supplemented_gold_pool_regressed_query_count": 0,
    }
    for name, expected in exact_requirements.items():
        if metrics.get(name) != expected:
            failed.append(name.removesuffix("_query_count"))
    for cutoff in (5, 10, 20):
        if metrics.get(f"supplemented_top{cutoff}_hit_query_count", 0) < metrics.get(
            f"baseline_top{cutoff}_hit_query_count", 0
        ):
            failed.append(f"top{cutoff}_regression")
    for cutoff in (10, 20):
        if metrics.get(
            f"unconstrained_supplemented_top{cutoff}_hit_query_count", 0
        ) < metrics.get(f"unconstrained_baseline_top{cutoff}_hit_query_count", 0):
            failed.append(f"unconstrained_top{cutoff}_regression")
    strict_gain = any(
        metrics.get(f"supplemented_{name}_hit_query_count", 0)
        > metrics.get(f"baseline_{name}_hit_query_count", 0)
        for name in ("gold_pool", "top5", "top10", "top20")
    )
    if not strict_gain:
        failed.append("no_strict_gain")
    return {
        "passed": not failed,
        "decision": (
            "eligible_for_production_promotion"
            if not failed
            else "keep_current_production"
        ),
        "failed_gates": failed,
        "metrics": dict(metrics),
    }


def _source_results(
    observations: Sequence[object],
    *,
    step: str,
    source_query_id: str | None = None,
    prefix: str | None = None,
) -> dict[str, ProviderResult[list[Paper]]]:
    output: dict[str, ProviderResult[list[Paper]]] = {}
    for item in observations:
        item_step = getattr(item, "step")
        item_query_id = getattr(item, "subquery_id")
        if item_step != step or (
            source_query_id is not None and item_query_id != source_query_id
        ):
            continue
        provider = str(getattr(item, "provider"))
        search_mode = str(getattr(item, "search_mode"))
        target_id = prefix or str(item_query_id)
        output[f"{provider}:{target_id}:{search_mode}"] = cast(
            ProviderResult[list[Paper]], getattr(item, "result")
        )
    return output


def _fair_merge(
    baseline: Sequence[FusedPaper],
    supplemental: Sequence[FusedPaper],
    *,
    identifier_map: object,
) -> list[FusedPaper]:
    baseline_papers = [item.paper for item in baseline]
    merged = list(baseline)
    for candidate in supplemental:
        if len(
            deduplicate_papers(
                [*baseline_papers, candidate.paper],
                id_map=identifier_map,
            ).papers
        ) == len(baseline_papers):
            continue
        merged.append(candidate)
    return sorted(merged, key=lambda item: item.score, reverse=True)


def _current_baseline_fused(
    observations: Sequence[object],
    *,
    identifier_map: object,
    max_raw_candidates: int,
    max_additional_raw_candidates: int,
    max_total_raw_candidates: int,
    max_deduplicated_candidates: int,
) -> list[FusedPaper]:
    primary = round_robin_cap_action_results(
        _source_results(observations, step="retrieve"),
        max_raw_candidates,
    )
    existing_supplement = round_robin_cap_action_results(
        _source_results(observations, step="retrieve_supplement"),
        max_additional_raw_candidates,
    )
    combined = round_robin_cap_action_results(
        {**primary, **existing_supplement},
        max_total_raw_candidates,
    )
    primary = {
        source_id: result
        for source_id, result in combined.items()
        if source_id not in existing_supplement
    }
    existing_supplement = {
        source_id: result
        for source_id, result in combined.items()
        if source_id in existing_supplement
    }
    baseline_fused = fuse_provider_results(
        primary,
        method="rrf",
        id_map=identifier_map,
    )
    if existing_supplement:
        baseline_fused = _fair_merge(
            baseline_fused,
            fuse_provider_results(
                existing_supplement,
                method="rrf",
                id_map=identifier_map,
            ),
            identifier_map=identifier_map,
        )
    return baseline_fused[:max_deduplicated_candidates]


def _rank(
    fused: Sequence[FusedPaper],
    *,
    spec: object,
    document_ranker: object,
    identifier_map: object,
) -> list[FusedPaper]:
    query_spec = cast(object, spec)
    papers = deduplicate_papers(
        [item.paper for item in fused],
        id_map=identifier_map,
    ).papers
    filtered = apply_hard_filters(papers, query_spec)  # type: ignore[arg-type]
    accepted_by_id = {item.paper.canonical_id: item for item in filtered.accepted}
    selected = [
        item.model_copy(
            update={
                "score": item.score
                * accepted_by_id[item.paper.canonical_id].score_multiplier
            }
        )
        for item in fused
        if item.paper.canonical_id in accepted_by_id
    ]
    selected.sort(key=lambda item: (-item.score, item.paper.canonical_id))
    contextual_rank = getattr(document_ranker, "rank_with_context", None)
    if not callable(contextual_rank):
        raise ValueError("production document ranker lacks unified context")
    ranked = contextual_rank(
        query_spec.original_query,  # type: ignore[attr-defined]
        selected,
        query_spec=query_spec,
    )
    if len(ranked) != len(selected) or {
        item.paper.canonical_id for item in ranked
    } != {item.paper.canonical_id for item in selected}:
        raise ValueError("production ranker changed candidate identity")
    return list(ranked)


def _candidate_evidence(fused: Sequence[FusedPaper]) -> list[DocumentCandidateEvidence]:
    return [
        DocumentCandidateEvidence(
            paper=item.paper,
            baseline_score=item.score,
            source_ranks=item.source_ranks,
        )
        for item in fused
    ]


def _selection_record(
    *,
    case_id: str,
    stratum: str,
    decision: object,
    selection: LowConfidenceLLMActionSelection | None,
    source_count: int,
) -> dict[str, object]:
    return {
        "case_id": case_id,
        "stratum": stratum,
        "policy_version": _POLICY_VERSION,
        "gold_loaded": False,
        "confidence_decision": decision.model_dump(mode="json"),  # type: ignore[attr-defined]
        "selected_action": (
            None
            if selection is None
            else {
                "source_query_id": selection.source_query_id,
                "text": selection.action.text,
                "search_mode": selection.action.search_mode,
                "provider_hint": selection.action.provider_hint,
                "novel_phrase_count": selection.novel_phrase_count,
                "novel_term_count": selection.novel_term_count,
            }
        ),
        "selected_source_count": source_count,
    }


def evaluate(
    *,
    workspace: Path,
    validation_root: Path,
    production_lock_path: Path,
    output_root: Path,
) -> dict[str, object]:
    sample = _load_json(validation_root / "sample-manifest.json")
    raw_cases = sample.get("cases")
    if not isinstance(raw_cases, list):
        raise ValueError("sample cases are unavailable")
    cases = [cast(dict[str, object], item) for item in raw_cases if isinstance(item, dict)]
    baseline_records = [
        _load_json(path)
        for path in sorted((validation_root / "baseline" / "results").glob("*.json"))
    ]
    candidate_records = [
        _load_json(path)
        for path in sorted((validation_root / "candidate" / "results").glob("*.json"))
    ]
    if not (len(cases) == len(baseline_records) == len(candidate_records) == 24):
        raise ValueError("paired 24-query frozen receipts are incomplete")

    verified = load_verified_input_lock(
        production_lock_path,
        artifact_root=workspace,
    )
    lock = verified.lock
    document_ranker = _locked_document_ranker(
        lock,
        verified.artifact_bytes,
        verified.ranker_artifact_failures,
    )
    if document_ranker is None:
        raise ValueError("production document ranker is unavailable")
    identifier_map, alias_count = load_locked_identifier_map(
        lock,
        verified.artifact_bytes,
    )
    if identifier_map is None:
        raise ValueError("production identifier aliases are unavailable")
    binding = lock.baseline.cross_vocabulary_supplement
    if binding is None:
        raise ValueError("current production supplement binding is unavailable")
    retrieval = lock.baseline.retrieval

    selection_rows: list[dict[str, object]] = []
    ranked_rows: list[dict[str, object]] = []
    for case, baseline, candidate in zip(
        cases, baseline_records, candidate_records, strict=True
    ):
        case_id = str(case["case_id"])
        if baseline.get("case_id") != case_id or candidate.get("case_id") != case_id:
            raise ValueError("paired case order changed")
        spec = _query_spec(baseline)
        baseline_observations = _decode_observations(
            baseline,
            validation_root=validation_root,
        )
        candidate_observations = _decode_observations(
            candidate,
            validation_root=validation_root,
        )
        baseline_fused = _current_baseline_fused(
            baseline_observations,
            identifier_map=identifier_map,
            max_raw_candidates=retrieval.max_raw_candidates,
            max_additional_raw_candidates=binding.max_additional_raw_candidates,
            max_total_raw_candidates=binding.max_total_raw_candidates,
            max_deduplicated_candidates=retrieval.max_deduplicated_candidates,
        )
        reconstructed = _rank(
            baseline_fused,
            spec=spec,
            document_ranker=document_ranker,
            identifier_map=identifier_map,
        )
        actual_baseline_ids = [
            paper.canonical_id
            for paper in _execution(baseline).pre_truncation_candidates
        ]
        reconstructed_ids = [item.paper.canonical_id for item in reconstructed]
        baseline_exact = reconstructed_ids == actual_baseline_ids

        confidence = assess_openalex_recall_confidence(
            spec,
            _candidate_evidence(baseline_fused),
        )
        selection = select_low_confidence_llm_action(
            spec,
            _plan(baseline),
            _plan(candidate),
            confidence,
        )
        supplemental_results: dict[str, ProviderResult[list[Paper]]] = {}
        if selection is not None:
            supplemental_results = _source_results(
                candidate_observations,
                step="retrieve",
                source_query_id=selection.source_query_id,
                prefix="llm-low-confidence-v3",
            )
            if not supplemental_results:
                raise ValueError(
                    f"selected action lacks frozen retrieval evidence: {case_id}"
                )
            supplemental_results = round_robin_cap_action_results(
                supplemental_results,
                binding.max_additional_raw_candidates,
            )
        if supplemental_results:
            llm_fused = fuse_provider_results(
                supplemental_results,
                method="rrf",
                id_map=identifier_map,
            )
            combined_fused = _fair_merge(
                baseline_fused,
                llm_fused,
                identifier_map=identifier_map,
            )
        else:
            combined_fused = list(baseline_fused)
        independent_cap = (
            retrieval.max_deduplicated_candidates
            + binding.max_additional_raw_candidates
        )
        combined_fused = combined_fused[:independent_cap]
        supplemented = _rank(
            combined_fused,
            spec=spec,
            document_ranker=document_ranker,
            identifier_map=identifier_map,
        )
        supplemented_ids = [item.paper.canonical_id for item in supplemented]
        baseline_members_retained = set(actual_baseline_ids).issubset(supplemented_ids)
        baseline_evidence_immutable = all(
            next(
                item
                for item in combined_fused
                if item.paper.canonical_id == baseline_item.paper.canonical_id
            ).model_dump(mode="json")
            == baseline_item.model_dump(mode="json")
            for baseline_item in baseline_fused
        )
        selection_rows.append(
            _selection_record(
                case_id=case_id,
                stratum=str(case["stratum"]),
                decision=confidence,
                selection=selection,
                source_count=len(supplemental_results),
            )
        )
        ranked_rows.append(
            {
                "case_id": case_id,
                "stratum": str(case["stratum"]),
                "baseline_reconstruction_exact": baseline_exact,
                "baseline_members_retained": baseline_members_retained,
                "baseline_evidence_immutable": baseline_evidence_immutable,
                "deployment_role": getattr(document_ranker, "deployment_role", None),
                "baseline_pool_count": len(actual_baseline_ids),
                "supplemented_pool_count": len(supplemented_ids),
                "selected_action": selection is not None,
                "papers": [paper_identity_record(item.paper) for item in supplemented],
            }
        )

    selection_artifact = {
        "schema_version": "low-confidence-llm-supplement-selection-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "policy_version": _POLICY_VERSION,
        "gold_loaded": False,
        "network_calls": 0,
        "llm_calls": 0,
        "rows": selection_rows,
    }
    selection_bytes = _canonical_bytes(selection_artifact)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "selection-receipt.json").write_bytes(selection_bytes)
    (output_root / "supplemented-ranked-pools.jsonl").write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in ranked_rows
        ),
        encoding="utf-8",
    )

    sources = sample.get("sources")
    partition_source = sources.get("partition") if isinstance(sources, Mapping) else None
    partition_value = (
        partition_source.get("path") if isinstance(partition_source, Mapping) else None
    )
    if not isinstance(partition_value, str):
        raise ValueError("frozen partition path is unavailable")
    partition_path = (workspace / partition_value).resolve(strict=True)
    if not partition_path.is_relative_to(workspace.resolve(strict=True)):
        raise ValueError("partition path escaped the workspace")
    partition = {str(row["query_id"]): row for row in _load_jsonl(partition_path)}

    report_rows: list[dict[str, object]] = []
    for case, baseline, ranked, selection in zip(
        cases, baseline_records, ranked_rows, selection_rows, strict=True
    ):
        case_id = str(case["case_id"])
        source = partition.get(case_id)
        if source is None:
            raise ValueError(f"query is absent from frozen partition: {case_id}")
        gold = source.get("gold_paper_ids")
        if not isinstance(gold, list) or not all(isinstance(item, str) for item in gold):
            raise ValueError(f"query lacks frozen Gold identifiers: {case_id}")
        supplemented_papers = [
            Paper.model_validate(item)
            for item in cast(list[dict[str, object]], ranked["papers"])
        ]
        baseline_pool_rank = _first_gold_rank(
            _execution(baseline).pre_truncation_candidates,
            gold,
            identifier_map=identifier_map,
        )
        supplemented_pool_rank = _first_gold_rank(
            supplemented_papers,
            gold,
            identifier_map=identifier_map,
        )
        baseline_rank = _first_gold_rank(
            _ranked_papers(baseline),
            gold,
            identifier_map=identifier_map,
        )
        supplemented_rank = _first_gold_rank(
            supplemented_papers,
            gold,
            identifier_map=identifier_map,
        )
        report_rows.append(
            {
                "case_id": case_id,
                "stratum": str(case["stratum"]),
                "baseline_gold_in_pool": baseline_pool_rank is not None,
                "supplemented_gold_in_pool": supplemented_pool_rank is not None,
                "supplemented_gold_pool_improved": baseline_pool_rank is None
                and supplemented_pool_rank is not None,
                "supplemented_gold_pool_regressed": baseline_pool_rank is not None
                and supplemented_pool_rank is None,
                "baseline_first_gold_rank": baseline_rank,
                "supplemented_first_gold_rank": supplemented_rank,
                "baseline_reconstruction_exact": ranked[
                    "baseline_reconstruction_exact"
                ],
                "baseline_members_retained": ranked["baseline_members_retained"],
                "baseline_evidence_immutable": ranked["baseline_evidence_immutable"],
                "deployment_role": ranked["deployment_role"],
                "selected_action": ranked["selected_action"],
                "confidence_reason_codes": cast(
                    Mapping[str, object], selection["confidence_decision"]
                ).get("reason_codes", []),
            }
        )

    metrics: dict[str, int] = {
        "query_count": len(report_rows),
        "baseline_reconstruction_exact_query_count": sum(
            int(item["baseline_reconstruction_exact"] is True) for item in report_rows
        ),
        "production_f5_query_count": sum(
            int(item["deployment_role"] == "F5-gated-fusion") for item in report_rows
        ),
        "gold_blind_selection_query_count": len(selection_rows),
        "independent_quota_query_count": sum(
            int(
                item["baseline_members_retained"] is True
                and item["baseline_evidence_immutable"] is True
            )
            for item in report_rows
        ),
        "selected_action_query_count": sum(
            int(item["selected_action"] is True) for item in report_rows
        ),
        "supplemented_gold_pool_improved_query_count": sum(
            int(item["supplemented_gold_pool_improved"] is True)
            for item in report_rows
        ),
        "supplemented_gold_pool_regressed_query_count": sum(
            int(item["supplemented_gold_pool_regressed"] is True)
            for item in report_rows
        ),
    }
    for condition in ("baseline", "supplemented"):
        metrics[f"{condition}_gold_pool_hit_query_count"] = sum(
            int(item[f"{condition}_gold_in_pool"] is True) for item in report_rows
        )
        for cutoff in (5, 10, 20):
            metrics[f"{condition}_top{cutoff}_hit_query_count"] = sum(
                int(
                    isinstance(item[f"{condition}_first_gold_rank"], int)
                    and cast(int, item[f"{condition}_first_gold_rank"]) <= cutoff
                )
                for item in report_rows
            )
    unconstrained = [item for item in report_rows if item["stratum"] == "unconstrained"]
    for condition in ("baseline", "supplemented"):
        for cutoff in (10, 20):
            metrics[
                f"unconstrained_{condition}_top{cutoff}_hit_query_count"
            ] = sum(
                int(
                    isinstance(item[f"{condition}_first_gold_rank"], int)
                    and cast(int, item[f"{condition}_first_gold_rank"]) <= cutoff
                )
                for item in unconstrained
            )
    decision = promotion_decision(metrics)
    report = {
        "schema_version": "low-confidence-llm-supplement-offline-gate-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "comparison": "current-production-versus-v3-low-confidence-supplement",
        "sample_role": "reused-disjoint-live-canary-offline-recombination",
        "network_calls": 0,
        "llm_calls": 0,
        "gold_loaded_after_selection_and_ranking": True,
        "selection_receipt_sha256": _sha256(selection_bytes),
        "identifier_alias_count": alias_count,
        "production_lock_file_sha256": _sha256(production_lock_path.read_bytes()),
        "production_lock_modified": False,
        "independent_supplemental_candidate_quota": binding.max_additional_raw_candidates,
        "maximum_deduplicated_candidates": (
            retrieval.max_deduplicated_candidates
            + binding.max_additional_raw_candidates
        ),
        "metrics": metrics,
        "decision": decision,
        "cases": report_rows,
        "safety": {
            "gold_used_for_selection": False,
            "online_request_count": 0,
            "llm_request_count": 0,
            "final_test_touched": False,
            "production_lock_modified": False,
        },
    }
    (output_root / "offline-gate.json").write_bytes(_canonical_bytes(report))
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--validation-root", type=Path, default=_DEFAULT_VALIDATION)
    parser.add_argument(
        "--production-lock",
        type=Path,
        default=Path("deliverables/evaluator/live-evaluator.lock.yaml"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=_DEFAULT_VALIDATION / "low-confidence-llm-supplement-offline-v1",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    workspace = args.workspace.resolve(strict=True)
    validation_root = (workspace / args.validation_root).resolve(strict=True)
    production_lock = (workspace / args.production_lock).resolve(strict=True)
    output_root = (workspace / args.output_root).resolve()
    for path in (validation_root, production_lock):
        if not path.is_relative_to(workspace):
            raise ValueError("input escaped the workspace")
    if not output_root.is_relative_to(workspace):
        raise ValueError("output escaped the workspace")
    report = evaluate(
        workspace=workspace,
        validation_root=validation_root,
        production_lock_path=production_lock,
        output_root=output_root,
    )
    print(json.dumps(report["decision"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["evaluate", "promotion_decision"]
