"""Aggregate-only offline analysis for title-candidate retention strategies."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import AbstractSet, Any

from paper_search.domain.models import (
    Paper,
    ProviderResult,
    QuerySpec,
    UsageActual,
)
from paper_search.evaluation.dataset import (
    EvaluationQuery,
    IdentifierMap,
    PredictionRecord,
    read_jsonl,
    sha256_file,
)
from paper_search.evaluation.metrics import evaluate
from paper_search.evaluation.ranking_metrics import evaluate_ranking
from paper_search.processing.filter import apply_hard_filters
from paper_search.retrieval.openalex import decode_openalex_page
from paper_search.ranking.fusion import fuse_provider_results


_FORBIDDEN_OUTPUT_KEYS = frozenset(
    {"query_id", "paper_id", "title", "request_id", "response"}
)
_TITLE_WEIGHTS = (1.25, 1.5, 2.0, 3.0)
_TITLE_SLOT_MINIMUMS = (1, 2, 3, 5, 10)
_FILTER_PATH = "src/paper_search/processing/filter.py"


def _provider_result(papers: Sequence[Paper]) -> ProviderResult[list[Paper]]:
    return ProviderResult[list[Paper]](
        data=list(papers),
        usage=UsageActual(),
        provenance={
            "provider": "offline",
            "endpoint": "offline",
            "model_id": "offline",
            "requested_at": datetime(2026, 8, 9, tzinfo=UTC).isoformat(),
            "response_hash": "sha256:offline",
        },
        cache_hit=True,
        latency_ms=0,
        errors=[],
    )


def weighted_rrf_ids(
    openalex: Sequence[Paper],
    titles: Sequence[Paper],
    eligible_ids: AbstractSet[str],
    *,
    title_weight: float,
    limit: int = 50,
) -> list[str]:
    """Return eligible IDs ranked by RRF with one title-source weight."""

    if not math.isfinite(title_weight) or title_weight <= 0:
        raise ValueError("title_weight must be finite and positive")
    if type(limit) is not int or limit <= 0:
        raise ValueError("limit must be a positive integer")
    fused = fuse_provider_results(
        {
            "openalex": _provider_result(openalex),
            "title_candidates": _provider_result(titles),
        },
        method="rrf",
        rrf_k=60,
    )
    scored: list[tuple[float, str]] = []
    for item in fused:
        score = sum(
            (title_weight if source == "title_candidates" else 1.0)
            / (60 + rank)
            for source, rank in item.source_ranks.items()
        )
        scored.append((score, item.paper.canonical_id))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [paper_id for _, paper_id in scored if paper_id in eligible_ids][
        :limit
    ]


def reserve_title_slots(
    baseline_ids: Sequence[str],
    title_ids: Sequence[str],
    eligible_ids: AbstractSet[str],
    *,
    minimum: int,
    limit: int = 50,
) -> list[str]:
    """Guarantee a minimum title-source count without bypassing eligibility."""

    if type(minimum) is not int or minimum < 0:
        raise ValueError("minimum must be a nonnegative integer")
    if type(limit) is not int or limit <= 0 or minimum > limit:
        raise ValueError("limit must be positive and at least minimum")
    ranked_titles = list(
        dict.fromkeys(
            paper_id for paper_id in title_ids if paper_id in eligible_ids
        )
    )
    title_set = set(ranked_titles)
    selected = list(
        dict.fromkeys(
            paper_id for paper_id in baseline_ids if paper_id in eligible_ids
        )
    )[:limit]
    needed = max(0, minimum - sum(item in title_set for item in selected))
    additions = [
        item for item in ranked_titles if item not in selected
    ][:needed]
    for addition in additions:
        if len(selected) < limit:
            selected.append(addition)
            continue
        replacement = next(
            (
                index
                for index in range(len(selected) - 1, -1, -1)
                if selected[index] not in title_set
            ),
            None,
        )
        if replacement is None:
            break
        selected[replacement] = addition
    return selected


def retains_baseline_golds(
    gold_by_query: Mapping[str, AbstractSet[str]],
    baseline: Mapping[str, Sequence[str]],
    candidate: Mapping[str, Sequence[str]],
    id_map: IdentifierMap,
) -> bool:
    """Return whether every historical exact-gold hit survives per query."""

    if set(baseline) != set(gold_by_query) or set(candidate) != set(
        gold_by_query
    ):
        raise ValueError("query sets must match")
    for query_id, gold_ids in gold_by_query.items():
        resolved_gold = {id_map.resolve(item) for item in gold_ids}
        baseline_hits = {
            id_map.resolve(item) for item in baseline[query_id]
        } & resolved_gold
        candidate_hits = {
            id_map.resolve(item) for item in candidate[query_id]
        } & resolved_gold
        if not baseline_hits <= candidate_hits:
            return False
    return True


def assert_aggregate_only(payload: object) -> None:
    """Reject record-level keys from a public diagnostic payload."""

    if isinstance(payload, dict):
        forbidden = _FORBIDDEN_OUTPUT_KEYS & set(payload)
        if forbidden:
            raise ValueError("aggregate-only payload contains private keys")
        for value in payload.values():
            assert_aggregate_only(value)
    elif isinstance(payload, list):
        for value in payload:
            assert_aggregate_only(value)


def _read_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            raise ValueError("blank JSONL line")
        row = json.loads(raw_line)
        if not isinstance(row, dict) or not isinstance(row.get("query_id"), str):
            raise ValueError("invalid query record")
        query_id = row["query_id"]
        if query_id in seen:
            raise ValueError("duplicate query record")
        seen.add(query_id)
        rows.append(row)
    return rows


def _snapshot_bytes(run: Path, ref: Mapping[str, object]) -> bytes:
    raw_path = ref.get("snapshot_path")
    if not isinstance(raw_path, str):
        raise ValueError("snapshot path is unavailable")
    snapshot_root = (run / "snapshots").resolve()
    path = (snapshot_root / raw_path).resolve()
    if not path.is_relative_to(snapshot_root):
        raise ValueError("snapshot path escapes run directory")
    return path.read_bytes()


def _decode_llm_object(content: bytes) -> dict[str, object] | None:
    try:
        envelope = json.loads(content)
        text = envelope["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError, json.JSONDecodeError):
        return None
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        return None
    return decoded if isinstance(decoded, dict) else None


def _title_diagnostic_index(
    run: Path,
    diagnostics: Sequence[Mapping[str, object]],
) -> int:
    for index, diagnostic in enumerate(diagnostics):
        if diagnostic.get("dependency") != "llm":
            continue
        refs = diagnostic.get("snapshot_refs")
        if not isinstance(refs, list) or not refs:
            continue
        ref = refs[0]
        if not isinstance(ref, dict):
            continue
        decoded = _decode_llm_object(_snapshot_bytes(run, ref))
        if decoded is not None and "titles" in decoded:
            return index
    raise ValueError("title generation diagnostic is unavailable")


def _papers_from_diagnostic(
    run: Path,
    diagnostic: Mapping[str, object],
) -> list[Paper]:
    refs = diagnostic.get("snapshot_refs")
    if not isinstance(refs, list):
        raise ValueError("snapshot refs are invalid")
    papers: list[Paper] = []
    for ref in refs:
        if not isinstance(ref, dict):
            raise ValueError("snapshot ref is invalid")
        decoded = decode_openalex_page(_snapshot_bytes(run, ref), limit=300)
        papers.extend(decoded.papers)
    return papers


def _deduplicate_ids(papers: Sequence[Paper]) -> list[Paper]:
    result: list[Paper] = []
    seen: set[str] = set()
    for paper in papers:
        if paper.canonical_id in seen:
            continue
        seen.add(paper.canonical_id)
        result.append(paper)
    return result


def _filter_is_unchanged(run: Path) -> bool:
    run_record = json.loads((run / "run.json").read_text(encoding="utf-8"))
    source_sha = run_record.get("source_git_sha")
    if not isinstance(source_sha, str) or not source_sha:
        return False
    repository = Path(__file__).resolve().parents[1]
    committed = subprocess.run(
        [
            "git",
            "diff",
            "--quiet",
            f"{source_sha}..HEAD",
            "--",
            _FILTER_PATH,
        ],
        cwd=repository,
        check=False,
    )
    working = subprocess.run(
        ["git", "diff", "--quiet", "--", _FILTER_PATH],
        cwd=repository,
        check=False,
    )
    return committed.returncode == 0 and working.returncode == 0


def _query_sources(
    run: Path,
    execution: Mapping[str, object],
    business: Mapping[str, object],
) -> tuple[list[Paper], list[Paper], list[Paper], set[str], set[str], int, int]:
    raw_diagnostics = execution.get("diagnostics")
    if not isinstance(raw_diagnostics, list) or any(
        not isinstance(item, dict) for item in raw_diagnostics
    ):
        raise ValueError("execution diagnostics are invalid")
    diagnostics = list(raw_diagnostics)
    title_index = _title_diagnostic_index(run, diagnostics)

    openalex: list[Paper] = []
    for diagnostic in diagnostics[:title_index]:
        if diagnostic.get("dependency") == "openalex":
            openalex.extend(_papers_from_diagnostic(run, diagnostic))

    title_historical: list[Paper] = []
    title_repaired: list[Paper] = []
    recovered_from_errors: list[Paper] = []
    error_bearing_responses = 0
    for diagnostic in diagnostics[title_index + 1 :]:
        if diagnostic.get("dependency") != "openalex":
            continue
        papers = _papers_from_diagnostic(run, diagnostic)
        title_repaired.extend(papers)
        errors = diagnostic.get("errors")
        has_errors = isinstance(errors, list) and bool(errors)
        if has_errors:
            error_bearing_responses += 1
            recovered_from_errors.extend(papers)
        else:
            title_historical.extend(papers)

    openalex = _deduplicate_ids(openalex)
    title_historical = _deduplicate_ids(title_historical)
    title_repaired = _deduplicate_ids(title_repaired)
    recovered_from_errors = _deduplicate_ids(recovered_from_errors)
    raw_eligible = execution.get("post_filter_paper_ids")
    if not isinstance(raw_eligible, list) or any(
        not isinstance(item, str) for item in raw_eligible
    ):
        raise ValueError("post-filter IDs are invalid")
    historical_eligible = set(raw_eligible)
    raw_retrieved = execution.get("retrieved_paper_ids", raw_eligible)
    if not isinstance(raw_retrieved, list) or any(
        not isinstance(item, str) for item in raw_retrieved
    ):
        raise ValueError("retrieved IDs are invalid")
    historical_retrieved = set(raw_retrieved)

    query_analysis = business.get("query_analysis")
    if not isinstance(query_analysis, dict):
        raise ValueError("query analysis is invalid")
    spec = QuerySpec.model_validate(query_analysis.get("query_spec"))
    new_papers = [
        paper
        for paper in recovered_from_errors
        if paper.canonical_id not in historical_retrieved
    ]
    accepted = apply_hard_filters(new_papers, spec).accepted
    repaired_eligible = historical_eligible | {
        item.paper.canonical_id for item in accepted
    }
    return (
        openalex,
        title_historical,
        title_repaired,
        historical_eligible,
        repaired_eligible,
        error_bearing_responses,
        len(recovered_from_errors),
    )


def _variant_summary(
    name: str,
    predictions_by_query: Mapping[str, Sequence[str]],
    gold: Sequence[EvaluationQuery],
    id_map: IdentifierMap,
) -> dict[str, object]:
    predictions = [
        PredictionRecord(
            query_id=record.query_id,
            predicted_paper_ids=list(predictions_by_query[record.query_id]),
        )
        for record in gold
    ]
    metrics = evaluate(gold, predictions, id_map=id_map)
    ranking = evaluate_ranking(gold, predictions, id_map=id_map)
    exact_gold_count = sum(
        item.true_positive_count for item in metrics.per_query.values()
    )
    hit_query_count = sum(
        item.true_positive_count > 0 for item in metrics.per_query.values()
    )
    summary = metrics.summary
    return {
        "name": name,
        "selected_count": sum(
            len(values) for values in predictions_by_query.values()
        ),
        "exact_gold_count": exact_gold_count,
        "hit_query_count": hit_query_count,
        "metrics": {
            "macro_precision": summary.macro_precision,
            "macro_recall": summary.macro_recall,
            "macro_f1": summary.macro_f1,
            "micro_precision": summary.micro_precision,
            "micro_recall": summary.micro_recall,
            "micro_f1": summary.micro_f1,
            "macro_recall_at_5": summary.macro_recall_at_5,
            "macro_recall_at_10": summary.macro_recall_at_10,
            "macro_recall_at_20": summary.macro_recall_at_20,
            "macro_mrr": ranking.summary.macro_mrr,
            "macro_ndcg": ranking.summary.macro_ndcg,
        },
    }


def _promotion_result(
    candidate: dict[str, object],
    historical: dict[str, object],
    *,
    retains_golds: bool,
) -> tuple[bool, list[str]]:
    candidate_metrics = candidate["metrics"]
    historical_metrics = historical["metrics"]
    if not isinstance(candidate_metrics, dict) or not isinstance(
        historical_metrics, dict
    ):
        raise ValueError("variant metrics are invalid")
    reasons: list[str] = []
    if candidate_metrics["macro_f1"] <= historical_metrics["macro_f1"]:
        reasons.append("macro_f1_not_improved")
    if not retains_golds:
        reasons.append("baseline_gold_lost")
    for name in (
        "macro_recall_at_5",
        "macro_recall_at_10",
        "macro_recall_at_20",
        "macro_mrr",
        "macro_ndcg",
    ):
        if candidate_metrics[name] < historical_metrics[name]:
            reasons.append(f"{name}_regressed")
    return not reasons, reasons


def analyze_run(
    run: Path,
    gold_path: Path,
    id_map_path: Path,
    *,
    expected_query_count: int = 60,
    expected_total_selected: int = 2908,
    require_unchanged_filter: bool = True,
) -> dict[str, object]:
    """Reconstruct one sealed run and evaluate aggregate-only variants."""

    if require_unchanged_filter and not _filter_is_unchanged(run):
        raise ValueError("hard-filter implementation changed")
    executions = _read_jsonl_objects(run / "executions.jsonl")
    businesses = _read_jsonl_objects(run / "business-results.jsonl")
    execution_by_query = {item["query_id"]: item for item in executions}
    business_by_query = {item["query_id"]: item for item in businesses}
    if set(execution_by_query) != set(business_by_query):
        raise ValueError("run query sets do not match")
    if len(executions) != expected_query_count:
        raise ValueError("unexpected query count")
    gold = read_jsonl(gold_path, EvaluationQuery)
    id_map = IdentifierMap.from_path(id_map_path)
    if {item.query_id for item in gold} != set(execution_by_query):
        raise ValueError("gold query set does not match run")
    gold_by_query = {
        item.query_id: set(item.relevant_paper_ids) for item in gold
    }

    historical: dict[str, list[str]] = {}
    repaired: dict[str, list[str]] = {}
    weighted: dict[float, dict[str, list[str]]] = {
        weight: {} for weight in _TITLE_WEIGHTS
    }
    reserved: dict[int, dict[str, list[str]]] = {
        minimum: {} for minimum in _TITLE_SLOT_MINIMUMS
    }
    repaired_titles: dict[str, list[str]] = {}
    historical_eligible_by_query: dict[str, set[str]] = {}
    repaired_eligible_by_query: dict[str, set[str]] = {}
    exact_sequences = 0
    error_bearing_responses = 0
    recovered_valid_papers = 0
    newly_eligible_papers = 0

    for query_id, execution in execution_by_query.items():
        business = business_by_query[query_id]
        (
            openalex,
            title_historical,
            title_repaired,
            historical_eligible,
            repaired_eligible,
            query_error_responses,
            query_recovered_papers,
        ) = _query_sources(run, execution, business)
        historical[query_id] = weighted_rrf_ids(
            openalex,
            title_historical,
            historical_eligible,
            title_weight=1.0,
        )
        expected = business.get("selected_paper_ids")
        if not isinstance(expected, list) or any(
            not isinstance(item, str) for item in expected
        ):
            raise ValueError("selected IDs are invalid")
        if historical[query_id] != expected:
            raise ValueError("historical Top-50 reconstruction mismatch")
        exact_sequences += 1
        repaired[query_id] = weighted_rrf_ids(
            openalex,
            title_repaired,
            repaired_eligible,
            title_weight=1.0,
        )
        for weight in _TITLE_WEIGHTS:
            weighted[weight][query_id] = weighted_rrf_ids(
                openalex,
                title_repaired,
                repaired_eligible,
                title_weight=weight,
            )
        repaired_titles[query_id] = [
            paper.canonical_id for paper in title_repaired
        ]
        historical_eligible_by_query[query_id] = historical_eligible
        repaired_eligible_by_query[query_id] = repaired_eligible
        error_bearing_responses += query_error_responses
        recovered_valid_papers += query_recovered_papers
        newly_eligible_papers += len(repaired_eligible - historical_eligible)

    total_selected = sum(len(values) for values in historical.values())
    if total_selected != expected_total_selected:
        raise ValueError("historical Top-50 reconstruction mismatch")
    for minimum in _TITLE_SLOT_MINIMUMS:
        for query_id in historical:
            reserved[minimum][query_id] = reserve_title_slots(
                repaired[query_id],
                repaired_titles[query_id],
                repaired_eligible_by_query[query_id],
                minimum=minimum,
            )

    def pool_gold_hits(
        pools: Mapping[str, AbstractSet[str]],
    ) -> tuple[int, int]:
        total = 0
        queries = 0
        for query_id, pool in pools.items():
            resolved_gold = {
                id_map.resolve(item) for item in gold_by_query[query_id]
            }
            hits = {id_map.resolve(item) for item in pool} & resolved_gold
            total += len(hits)
            queries += bool(hits)
        return total, queries

    newly_eligible_by_query = {
        query_id: repaired_eligible_by_query[query_id]
        - historical_eligible_by_query[query_id]
        for query_id in historical
    }
    historical_pool_gold, _ = pool_gold_hits(historical_eligible_by_query)
    repaired_pool_gold, _ = pool_gold_hits(repaired_eligible_by_query)
    newly_eligible_gold, newly_eligible_gold_queries = pool_gold_hits(
        newly_eligible_by_query
    )
    variants: list[dict[str, object]] = []
    historical_summary = _variant_summary(
        "historical_rrf", historical, gold, id_map
    )
    historical_summary.update(
        {
            "changed_sequence_queries": 0,
            "changed_set_queries": 0,
            "promotable": False,
            "reason_codes": ["reference_variant"],
        }
    )
    variants.append(historical_summary)

    candidate_inputs: list[
        tuple[str, Mapping[str, Sequence[str]], dict[str, object]]
    ] = [("repaired_rrf", repaired, {})]
    candidate_inputs.extend(
        (
            f"weighted_rrf_title_{str(weight).replace('.', '_')}",
            weighted[weight],
            {"title_weight": weight},
        )
        for weight in _TITLE_WEIGHTS
    )
    candidate_inputs.extend(
        (
            f"reserved_title_slots_{minimum}",
            reserved[minimum],
            {"minimum_title_slots": minimum},
        )
        for minimum in _TITLE_SLOT_MINIMUMS
    )
    for name, predictions, parameters in candidate_inputs:
        summary = _variant_summary(name, predictions, gold, id_map)
        changed_sequence_queries = sum(
            list(predictions[query_id]) != historical[query_id]
            for query_id in historical
        )
        changed_set_queries = sum(
            set(predictions[query_id]) != set(historical[query_id])
            for query_id in historical
        )
        retains_golds = retains_baseline_golds(
            gold_by_query,
            historical,
            predictions,
            id_map,
        )
        promotable, reasons = _promotion_result(
            summary,
            historical_summary,
            retains_golds=retains_golds,
        )
        summary.update(
            {
                "parameters": parameters,
                "changed_sequence_queries": changed_sequence_queries,
                "changed_set_queries": changed_set_queries,
                "retains_baseline_golds": retains_golds,
                "promotable": promotable,
                "reason_codes": reasons,
            }
        )
        variants.append(summary)

    promotable_variants = [
        item for item in variants if item.get("promotable") is True
    ]
    promotable_variants.sort(
        key=lambda item: (
            -item["metrics"]["macro_f1"],
            -item["metrics"]["macro_recall"],
            -item["metrics"]["macro_ndcg"],
            -item["metrics"]["macro_mrr"],
            item["name"],
        )
    )
    run_record = json.loads((run / "run.json").read_text(encoding="utf-8"))
    payload: dict[str, object] = {
        "schema_version": "title-retention-offline-v1",
        "run_id": run_record.get("run_id", run.name),
        "source_git_sha": run_record.get("source_git_sha"),
        "input_hashes": {
            "executions_sha256": sha256_file(run / "executions.jsonl"),
            "business_results_sha256": sha256_file(
                run / "business-results.jsonl"
            ),
            "gold_sha256": sha256_file(gold_path),
            "identifier_map_sha256": sha256_file(id_map_path),
        },
        "reconstruction": {
            "exact_query_sequences": exact_sequences,
            "query_count": len(executions),
            "total_selected": total_selected,
        },
        "partial_success": {
            "error_bearing_responses": error_bearing_responses,
            "recovered_valid_papers": recovered_valid_papers,
            "newly_eligible_papers": newly_eligible_papers,
            "historical_pool_exact_gold": historical_pool_gold,
            "repaired_pool_exact_gold": repaired_pool_gold,
            "newly_eligible_exact_gold": newly_eligible_gold,
            "newly_eligible_gold_queries": newly_eligible_gold_queries,
        },
        "variants": variants,
        "recommended_variant": (
            promotable_variants[0]["name"] if promotable_variants else None
        ),
    }
    assert_aggregate_only(payload)
    return payload


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(
                payload,
                temporary,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            temporary.write("\n")
        temporary_path.replace(path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze sealed title-candidate retention offline"
    )
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--id-map", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        payload = analyze_run(args.run, args.gold, args.id_map)
        _write_json_atomic(args.out, payload)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"title retention analysis failed: {error}", file=sys.stderr)
        return 2
    summary = payload["reconstruction"]
    print(
        json.dumps(
            {
                "schema_version": payload["schema_version"],
                "reconstruction": summary,
                "recommended_variant": payload["recommended_variant"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
