"""Audit online candidate recall ceiling versus conditional production ranking."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
_SOURCE = str(_ROOT / "src")
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if _SOURCE not in sys.path:
    sys.path.insert(0, _SOURCE)

from paper_search.evaluation.predictions import (  # noqa: E402
    paper_evaluation_id,
    paper_matches_evaluation_ids,
)
from paper_search.learning.f5_production_deployment import (  # noqa: E402
    load_f5_production_ranker_bytes,
)
from paper_search.learning.large_scale_fusion_training import (  # noqa: E402
    read_query_shard,
)
from scripts.evaluate_anchored_fusion_delta import (  # noqa: E402
    _online_retrieval_candidate_view,
)


_CUTOFFS = (5, 10, 20)


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / denominator if denominator else 0.0


def _summarize_subset(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    query_count = len(records)
    gold_count = sum(int(row["gold_count"]) for row in records)
    candidate_gold_count = sum(len(row["gold_ranks"]) for row in records)
    candidate_hit_count = sum(bool(row["gold_ranks"]) for row in records)
    output: dict[str, Any] = {
        "query_count": query_count,
        "gold_association_count": gold_count,
        "candidate_gold_association_count": candidate_gold_count,
        "candidate_gold_hit_query_count": candidate_hit_count,
        "candidate_gold_miss_query_count": query_count - candidate_hit_count,
        "candidate_gold_hit_query_rate": _ratio(candidate_hit_count, query_count),
        "candidate_micro_recall": _ratio(candidate_gold_count, gold_count),
    }
    for cutoff in _CUTOFFS:
        query_hits = sum(
            any(int(rank) <= cutoff for rank in row["gold_ranks"])
            for row in records
        )
        association_hits = sum(
            sum(int(rank) <= cutoff for rank in row["gold_ranks"])
            for row in records
        )
        output[f"gold_in_top_{cutoff}_query_count"] = query_hits
        output[f"gold_in_top_{cutoff}_query_rate"] = _ratio(
            query_hits, query_count
        )
        output[f"top_{cutoff}_query_rate_given_candidate_hit"] = _ratio(
            query_hits, candidate_hit_count
        )
        output[f"top_{cutoff}_micro_recall"] = _ratio(
            association_hits, gold_count
        )
        output[f"candidate_hit_but_below_top_{cutoff}_query_count"] = (
            candidate_hit_count - query_hits
        )
    recall_misses = int(output["candidate_gold_miss_query_count"])
    ranking_gaps = int(output["candidate_hit_but_below_top_20_query_count"])
    unresolved = recall_misses + ranking_gaps
    if unresolved == 0:
        dominant = "none"
    elif recall_misses > ranking_gaps:
        dominant = "recall"
    elif ranking_gaps > recall_misses:
        dominant = "ranking"
    else:
        dominant = "balanced"
    output["dominant_bottleneck_at_20"] = dominant
    output["recall_share_of_queries_unresolved_at_20"] = _ratio(
        recall_misses, unresolved
    )
    return output


def summarize_records(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    normalized: list[dict[str, Any]] = []
    query_ids: set[str] = set()
    for raw in records:
        query_id = str(raw.get("query_id", ""))
        if not query_id:
            raise ValueError("recall audit row has no query id")
        if query_id in query_ids:
            raise ValueError(f"duplicate query id: {query_id}")
        query_ids.add(query_id)
        gold_count = int(raw.get("gold_count", 0))
        ranks = [int(value) for value in raw.get("gold_ranks", [])]
        if (
            gold_count <= 0
            or len(ranks) > gold_count
            or len(ranks) != len(set(ranks))
            or any(rank <= 0 for rank in ranks)
        ):
            raise ValueError(f"invalid Gold ranks: {query_id}")
        labels = sorted({str(value) for value in raw.get("labels", [])})
        normalized.append(
            {
                "query_id": query_id,
                "gold_count": gold_count,
                "gold_ranks": sorted(ranks),
                "labels": labels,
            }
        )

    by_stratum: dict[str, dict[str, Any]] = {}
    for stratum in ("method", "negation", "dataset", "year"):
        by_stratum[stratum] = _summarize_subset(
            [row for row in normalized if stratum in row["labels"]]
        )
    by_stratum["unconstrained"] = _summarize_subset(
        [row for row in normalized if not row["labels"]]
    )
    by_stratum["any_constraint"] = _summarize_subset(
        [row for row in normalized if row["labels"]]
    )
    return {
        "overall": _summarize_subset(normalized),
        "by_stratum": by_stratum,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _object(path: Path, *, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_records(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(
            filename="", mode="wb", compresslevel=6, mtime=0, fileobj=raw
        ) as stream:
            for row in rows:
                stream.write(
                    json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
                    + b"\n"
                )
    temporary.replace(path)


def _read_records(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _constraint_labels(path: Path) -> dict[str, list[str]]:
    output: dict[str, list[str]] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            query_id = str(row.get("query_id", ""))
            if not query_id or query_id in output:
                raise ValueError("constraint labels have missing or duplicate query ids")
            output[query_id] = sorted(
                {str(value) for value in row.get("labels", [])}
            )
    return output


def _input_state(
    *, context_manifest_path: Path, production_selection_path: Path
) -> dict[str, Any]:
    context = _object(context_manifest_path, label="context freeze manifest")
    if (
        context.get("schema_version") != "directed-fusion-context-freeze-v3"
        or context.get("test_partition_touched") is not False
        or context.get("production_lock_modified") is not False
        or context.get("query_count") != 21429
    ):
        raise ValueError("context freeze is not the exact isolated 21,429 package")
    source = context.get("source_candidate_package")
    outputs = context.get("outputs")
    if not isinstance(source, dict) or not isinstance(outputs, dict):
        raise ValueError("context freeze is missing candidate or label evidence")
    if source.get("all_shard_hashes_verified_twice") is not True:
        raise ValueError("candidate shards lack prior double hash verification")
    shard_manifest_path = Path(str(source["manifest_path"]))
    if _sha256(shard_manifest_path) != source.get("manifest_sha256"):
        raise ValueError("candidate shard manifest hash mismatch")
    shard_manifest = _object(shard_manifest_path, label="candidate shard manifest")
    if (
        shard_manifest.get("query_count") != 21429
        or shard_manifest.get("test_partition_touched") is not False
    ):
        raise ValueError("candidate shard manifest scope mismatch")

    constraint_labels_path = context_manifest_path.parent / "constraint-labels.merged.jsonl"
    if _sha256(constraint_labels_path) != outputs.get("constraint_labels_sha256"):
        raise ValueError("constraint label hash mismatch")

    selection = _object(production_selection_path, label="production selection")
    if (
        selection.get("production_default") != "F5-gated-fusion"
        or selection.get("training_query_count") != 18314
        or selection.get("test_partition_touched") is not False
    ):
        raise ValueError("production selection is not the promoted 18,314 F5")
    model_root = production_selection_path.parent
    production_manifest_path = model_root / str(selection["default_manifest"])
    production_bundle_path = model_root / str(selection["default_weights"])
    if (
        _sha256(production_manifest_path) != selection.get("default_manifest_sha256")
        or _sha256(production_bundle_path) != selection.get("default_weights_sha256")
    ):
        raise ValueError("production F5 selection hash mismatch")
    completed = shard_manifest.get("completed_shards")
    if not isinstance(completed, list) or len(completed) != int(
        shard_manifest.get("batch_count", 0)
    ):
        raise ValueError("candidate shards are incomplete")
    return {
        "context": context,
        "context_manifest_path": context_manifest_path,
        "constraint_labels_path": constraint_labels_path,
        "production_selection_path": production_selection_path,
        "production_manifest_path": production_manifest_path,
        "production_bundle_path": production_bundle_path,
        "shard_dir": shard_manifest_path.parent,
        "shard_manifest": shard_manifest,
        "shard_manifest_path": shard_manifest_path,
    }


def _progress_identity(state: Mapping[str, Any]) -> dict[str, str]:
    return {
        "context_manifest_sha256": _sha256(state["context_manifest_path"]),
        "constraint_labels_sha256": _sha256(state["constraint_labels_path"]),
        "production_selection_sha256": _sha256(
            state["production_selection_path"]
        ),
        "production_manifest_sha256": _sha256(state["production_manifest_path"]),
        "production_bundle_sha256": _sha256(state["production_bundle_path"]),
        "candidate_shard_manifest_sha256": _sha256(
            state["shard_manifest_path"]
        ),
    }


def _process(
    *, state: Mapping[str, Any], output_dir: Path
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.json"
    identity = _progress_identity(state)
    progress: dict[str, Any] = {
        "schema_version": "online-recall-ceiling-progress-v1",
        "inputs": identity,
        "completed_shards": {},
        "test_partition_touched": False,
        "online_requests_made": 0,
        "llm_requests_made": 0,
    }
    if progress_path.is_file():
        progress = _object(progress_path, label="recall audit progress")
        if progress.get("inputs") != identity:
            raise ValueError("recall audit checkpoint belongs to different inputs")
    completed = progress.get("completed_shards")
    if not isinstance(completed, dict):
        raise ValueError("recall audit checkpoint is invalid")

    labels_by_query_id = _constraint_labels(state["constraint_labels_path"])
    ranker = load_f5_production_ranker_bytes(
        state["production_manifest_path"].read_bytes(),
        state["production_bundle_path"].read_bytes(),
    )
    source_rows = sorted(
        state["shard_manifest"]["completed_shards"],
        key=lambda row: int(row["batch_index"]),
    )
    for position, source_row in enumerate(source_rows, start=1):
        batch_index = int(source_row["batch_index"])
        key = str(batch_index)
        output_path = output_dir / f"rank-shard-{batch_index:05d}.jsonl.gz"
        saved = completed.get(key)
        if isinstance(saved, dict) and output_path.is_file():
            if (
                saved.get("source_sha256") != source_row.get("sha256")
                or saved.get("output_sha256") != _sha256(output_path)
            ):
                raise ValueError(f"recall audit checkpoint mismatch: {batch_index}")
            continue

        source_path = state["shard_dir"] / f"shard-{batch_index:05d}.jsonl.gz"
        queries = read_query_shard(source_path)
        if len(queries) != int(source_row["query_count"]):
            raise ValueError(f"candidate shard query count mismatch: {batch_index}")
        rows: list[dict[str, Any]] = []
        removed_candidate_count = 0
        queries_with_removed = 0
        for query in queries:
            online_query, removed = _online_retrieval_candidate_view(query)
            removed_candidate_count += removed
            queries_with_removed += int(removed > 0)
            ranked = ranker.rank(online_query.query, online_query.candidates)
            input_ids = [
                paper_evaluation_id(candidate.paper)
                for candidate in online_query.candidates
            ]
            output_ids = [paper_evaluation_id(candidate.paper) for candidate in ranked]
            if len(input_ids) != len(output_ids) or set(input_ids) != set(output_ids):
                raise ValueError("production F5 changed online candidate identity")
            if query.query_id not in labels_by_query_id:
                raise ValueError(f"missing frozen constraint labels: {query.query_id}")
            gold = set(online_query.gold_paper_ids)
            gold_ranks = [
                rank
                for rank, candidate in enumerate(ranked, start=1)
                if paper_matches_evaluation_ids(candidate.paper, gold)
            ]
            rows.append(
                {
                    "query_id": online_query.query_id,
                    "gold_count": len(gold),
                    "gold_ranks": gold_ranks,
                    "labels": labels_by_query_id[online_query.query_id],
                    "online_candidate_count": len(online_query.candidates),
                    "removed_training_only_candidate_count": removed,
                }
            )
        _write_records(output_path, rows)
        completed[key] = {
            "batch_index": batch_index,
            "query_count": len(rows),
            "source_sha256": source_row["sha256"],
            "output_path": str(output_path),
            "output_sha256": _sha256(output_path),
            "removed_training_only_candidate_count": removed_candidate_count,
            "query_count_with_removed_training_only_candidates": queries_with_removed,
        }
        _write_json(progress_path, progress)
        if position % 10 == 0 or position == len(source_rows):
            print(
                json.dumps(
                    {
                        "stage": "online_recall_audit",
                        "completed_shard": position,
                        "shard_count": len(source_rows),
                        "batch_index": batch_index,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    all_rows: list[dict[str, Any]] = []
    for source_row in source_rows:
        batch_index = int(source_row["batch_index"])
        all_rows.extend(
            _read_records(output_dir / f"rank-shard-{batch_index:05d}.jsonl.gz")
        )
    expected_query_count = int(state["shard_manifest"]["query_count"])
    if len(all_rows) != expected_query_count:
        raise ValueError("recall audit did not cover the full frozen package")
    summary = summarize_records(all_rows)
    overall = summary["overall"]
    context = state["context"]
    source = context["source_candidate_package"]
    report = {
        "schema_version": "online-recall-ceiling-conditional-ranking-v1",
        "scope": "strict-ready-21429-auto-train-current-online-supported-candidates",
        "candidate_view": "exclude-pasa-only-training-candidates-retain-online-supported",
        "ranking_model": "current-promoted-production-f5-18314-full-fit-diagnostic",
        "ranking_evidence_note": (
            "Diagnostic decomposition on auto_train; not an independent held-out score."
        ),
        "query_count": expected_query_count,
        "summary": summary,
        "mixed_candidate_reference": {
            "gold_hit_query_count": int(source["expanded_gold_hit_query_count"]),
            "online_only_gold_hit_query_count": int(
                overall["candidate_gold_hit_query_count"]
            ),
            "mixed_minus_online_only_hit_query_count": int(
                source["expanded_gold_hit_query_count"]
            )
            - int(overall["candidate_gold_hit_query_count"]),
        },
        "training_only_candidate_filter": {
            "removed_candidate_count": sum(
                int(row["removed_training_only_candidate_count"])
                for row in completed.values()
            ),
            "query_count_with_removed_candidates": sum(
                int(row["query_count_with_removed_training_only_candidates"])
                for row in completed.values()
            ),
        },
        "decision": {
            "dominant_bottleneck_at_20": overall["dominant_bottleneck_at_20"],
            "recommended_next_branch": (
                "openalex-s2-supplemental-recall"
                if overall["dominant_bottleneck_at_20"] == "recall"
                else "conditional-ranking-diagnosis"
            ),
            "reuse_auto_dev538_for_parameter_selection": False,
            "method_pair_retraining_started": False,
        },
        "inputs": {
            **identity,
            "context_manifest_path": str(state["context_manifest_path"]),
            "constraint_labels_path": str(state["constraint_labels_path"]),
            "production_selection_path": str(state["production_selection_path"]),
            "candidate_shard_manifest_path": str(state["shard_manifest_path"]),
        },
        "candidate_pool_identity_unchanged": True,
        "production_lock_modified": False,
        "development_labels_used_for_training": False,
        "test_partition_touched": False,
        "online_requests_made": 0,
        "llm_requests_made": 0,
    }
    _write_json(output_dir / "summary.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context-freeze-manifest", type=Path, required=True)
    parser.add_argument("--production-selection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    state = _input_state(
        context_manifest_path=args.context_freeze_manifest,
        production_selection_path=args.production_selection,
    )
    report = _process(state=state, output_dir=args.output_dir)
    print(
        json.dumps(
            {
                "report": str(args.output_dir / "summary.json"),
                "query_count": report["query_count"],
                "decision": report["decision"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
