"""Freeze an expanded receipt-backed candidate package and effective-pair audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence

_ROOT = Path(__file__).resolve().parents[1]
for _IMPORT_ROOT in (_ROOT, _ROOT / "src"):
    if str(_IMPORT_ROOT) not in sys.path:
        sys.path.insert(0, str(_IMPORT_ROOT))

from paper_search.learning.f5_production_deployment import (  # noqa: E402
    load_f5_production_ranker_bytes,
)
from paper_search.learning.gated_feature_fusion_ranker import (  # noqa: E402
    UnifiedFusionContextResolver,
)
from paper_search.learning.large_scale_fusion_training import (  # noqa: E402
    build_document_ranking_query,
    index_training_receipts,
    load_training_package,
)
from scripts.augment_fusion_training_with_pasa import (  # noqa: E402
    _coverage,
    _delta,
    _query_audit,
)


def _sha256(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _expanded_handoff(
    base: dict[str, Any],
    *,
    receipt_roots: tuple[Path, ...],
    validation_result_sha256: str,
) -> dict[str, Any]:
    if base.get("schema_version") != "openalex-ranking-training-handoff-v1":
        raise ValueError("unsupported base handoff schema")
    if base.get("test_partition_touched") is not False:
        raise ValueError("base handoff touched the test partition")
    if base.get("online_llm_requests") != 0:
        raise ValueError("base handoff used online LLM requests")
    roots = list(dict.fromkeys(str(path.resolve()) for path in receipt_roots))
    prior_supplement = base.get("high_recall_candidate_supplement")
    prior_additive_roots: list[str] = []
    if isinstance(prior_supplement, dict):
        raw_prior_roots = prior_supplement.get("receipt_roots")
        if isinstance(raw_prior_roots, list):
            prior_additive_roots = [
                str(Path(value).resolve())
                for value in raw_prior_roots
                if isinstance(value, str) and value
            ]
    new_additive_roots = roots[len(base["ordered_receipt_roots"]) :]
    additive_roots = list(
        dict.fromkeys([*prior_additive_roots, *new_additive_roots])
    )
    output = dict(base)
    output["ordered_receipt_roots"] = roots
    output["high_recall_candidate_supplement"] = {
        "validation_result_sha256": validation_result_sha256,
        "receipt_roots": additive_roots,
        "validation_consumed_after_gate": True,
        "strict_ready_ceiling_unchanged": True,
        "llm_request_count": 0,
        "test_partition_touched": False,
    }
    return output


class _LocalContextStore:
    def __init__(self, resolver: UnifiedFusionContextResolver) -> None:
        self._resolver = resolver

    def for_training_query(self, query: str):  # type: ignore[no-untyped-def]
        return self._resolver.for_local_query(query)


def _audit_query(
    package: Any,
    indexed: dict[str, tuple[Path, ...]],
    ranker: Any,
    query_id: str,
    additive_receipt_roots: tuple[Path, ...],
) -> dict[str, object]:
    try:
        query = build_document_ranking_query(
            package,
            query_id,
            indexed[query_id],
            additive_receipt_roots=additive_receipt_roots,
        )
    except ValueError as error:
        paths = ", ".join(str(path) for path in indexed[query_id])
        raise ValueError(f"{error}; receipt_paths=[{paths}]") from error
    return _query_audit(query, ranker)


def _read_audit_rows(directory: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(directory.glob("batch-*.jsonl")):
        rows.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return rows


def _historical_baseline(payload: dict[str, Any]) -> dict[str, Any]:
    """Preserve the original baseline when freezing an already-expanded package."""

    coverage = payload.get("coverage")
    if isinstance(coverage, dict):
        return coverage
    historical_before = payload.get("historical_before")
    if isinstance(historical_before, dict):
        return historical_before
    raise ValueError("baseline coverage lacks coverage or historical_before")


def _selected_batch_indexes(
    batch_count: int, *, start: int, end: int | None
) -> range:
    """Return a validated half-open batch slice for checkpoint workers."""

    stop = batch_count if end is None else end
    if start < 0 or stop < start or stop > batch_count:
        raise ValueError(
            f"invalid batch slice: start={start}, end={stop}, count={batch_count}"
        )
    return range(start, stop)


def _supplemented_query_ids(
    indexed: dict[str, tuple[Path, ...]], additive_roots: tuple[Path, ...]
) -> set[str]:
    roots = tuple(root.resolve() for root in additive_roots)
    return {
        query_id
        for query_id, paths in indexed.items()
        if any(path.is_relative_to(root) for path in paths for root in roots)
    }


def _non_additive_paths(
    paths: tuple[Path, ...], additive_roots: tuple[Path, ...]
) -> tuple[Path, ...]:
    roots = tuple(root.resolve() for root in additive_roots)
    return tuple(
        path
        for path in paths
        if not any(path.is_relative_to(root) for root in roots)
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-handoff", type=Path, required=True)
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--production-manifest", type=Path, required=True)
    parser.add_argument("--production-bundle", type=Path, required=True)
    parser.add_argument("--receipt-root", type=Path, action="append", required=True)
    parser.add_argument("--validation-result", type=Path, required=True)
    parser.add_argument("--baseline-coverage", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--start-batch", type=int, default=0)
    parser.add_argument("--end-batch", type=int)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--refresh-supplemented", action="store_true")
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")

    base = json.loads(args.base_handoff.read_text(encoding="utf-8"))
    validation = json.loads(args.validation_result.read_text(encoding="utf-8"))
    if validation.get("decision", {}).get("validation_gate_passed") is not True:
        raise ValueError("recall validation gate did not pass")
    if validation.get("scope", {}).get("llm_calls") != 0:
        raise ValueError("recall validation used LLM calls")
    all_roots = tuple(
        [Path(value) for value in base["ordered_receipt_roots"]]
        + list(args.receipt_root)
    )
    expanded = _expanded_handoff(
        base,
        receipt_roots=all_roots,
        validation_result_sha256=_sha256(args.validation_result.read_bytes()),
    )
    handoff_path = args.output_dir / "ranking-training-handoff-expanded.json"
    handoff_bytes = _canonical_bytes(expanded)
    if handoff_path.is_file():
        if handoff_path.read_bytes() != handoff_bytes:
            raise ValueError("expanded handoff checkpoint does not match")
    else:
        _write_atomic(handoff_path, handoff_bytes)

    package = load_training_package(
        handoff_path=handoff_path,
        partition_path=args.partition,
        production_bundle_path=args.production_bundle,
    )
    indexed = index_training_receipts(package)
    additive_roots = tuple(path.resolve() for path in args.receipt_root)
    supplemented_query_ids = _supplemented_query_ids(indexed, additive_roots)
    ranker = load_f5_production_ranker_bytes(
        args.production_manifest.read_bytes(), args.production_bundle.read_bytes()
    )
    resolver = ranker.context_store
    if not isinstance(resolver, UnifiedFusionContextResolver):
        raise ValueError("production ranker lacks unified local context")
    ranker.context_store = _LocalContextStore(resolver)

    audit_dir = args.output_dir / "effective-pair-audit-batches"
    batch_count = (len(package.query_ids) + args.batch_size - 1) // args.batch_size
    batch_indexes = _selected_batch_indexes(
        batch_count, start=args.start_batch, end=args.end_batch
    )
    for batch_index in batch_indexes:
        query_ids = package.query_ids[
            batch_index * args.batch_size : (batch_index + 1) * args.batch_size
        ]
        path = audit_dir / f"batch-{batch_index:05d}.jsonl"
        existing_rows: dict[str, dict[str, object]] = {}
        if path.is_file():
            parsed_rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            observed = [row["query_id"] for row in parsed_rows]
            if observed != list(query_ids):
                raise ValueError(f"audit checkpoint query mismatch: {path}")
            if not args.refresh_supplemented or not (
                set(query_ids) & supplemented_query_ids
            ):
                continue
            existing_rows = {str(row["query_id"]): row for row in parsed_rows}
        rows = []
        for query_id in query_ids:
            if existing_rows and query_id not in supplemented_query_ids:
                rows.append(existing_rows[query_id])
                continue
            rows.append(
                _audit_query(
                    package,
                    indexed,
                    ranker,
                    query_id,
                    additive_roots,
                )
            )
        _write_atomic(
            path,
            b"".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8")
                + b"\n"
                for row in rows
            ),
        )
        print(
            json.dumps(
                {
                    "completed_batch": batch_index + 1,
                    "batch_count": batch_count,
                    "processed_query_count": batch_index * args.batch_size
                    + len(query_ids),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    if args.audit_only:
        return 0

    rows = _read_audit_rows(audit_dir)
    if len(rows) != len(package.query_ids):
        raise ValueError("effective-pair audit coverage is incomplete")
    after = _coverage(rows)
    historical_baseline = _historical_baseline(
        json.loads(args.baseline_coverage.read_text(encoding="utf-8"))
    )

    base_indexed = {
        query_id: _non_additive_paths(indexed[query_id], additive_roots)
        for query_id in supplemented_query_ids
    }
    missing_base = [query_id for query_id, paths in base_indexed.items() if not paths]
    if missing_base:
        raise ValueError(f"supplemented query lacks a base receipt: {missing_base[0]}")
    aligned_dir = args.output_dir / "aligned-base-supplement-audit-batches"
    for batch_index in range(batch_count):
        query_ids = package.query_ids[
            batch_index * args.batch_size : (batch_index + 1) * args.batch_size
        ]
        selected_ids = [
            query_id for query_id in query_ids if query_id in supplemented_query_ids
        ]
        if not selected_ids:
            continue
        path = aligned_dir / f"batch-{batch_index:05d}.jsonl"
        if path.is_file():
            observed = [
                json.loads(line)["query_id"]
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if observed != selected_ids:
                raise ValueError(f"aligned checkpoint query mismatch: {path}")
            continue
        aligned_batch = [
            _audit_query(package, base_indexed, ranker, query_id, ())
            for query_id in selected_ids
        ]
        _write_atomic(
            path,
            b"".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8")
                + b"\n"
                for row in aligned_batch
            ),
        )
        print(
            json.dumps(
                {
                    "aligned_base_completed_batch": batch_index + 1,
                    "batch_count": batch_count,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    aligned_supplement_rows = _read_audit_rows(aligned_dir)
    if len(aligned_supplement_rows) != len(supplemented_query_ids):
        raise ValueError("aligned base audit coverage is incomplete")
    aligned_rows_by_id = {str(row["query_id"]): row for row in rows}
    aligned_rows_by_id.update(
        {str(row["query_id"]): row for row in aligned_supplement_rows}
    )
    aligned_before = _coverage(
        [aligned_rows_by_id[query_id] for query_id in package.query_ids]
    )
    comparison = {
        "schema_version": "expanded-high-recall-effective-pair-coverage-v2",
        "strict_ready_query_count": len(package.query_ids),
        "strict_ready_ceiling_unchanged": True,
        "historical_before": historical_baseline,
        "before": aligned_before,
        "after": after,
        "delta": _delta(aligned_before, after),
        "context_alignment": {
            "before": "production_unified_local_parser",
            "after": "production_unified_local_parser",
            "supplemented_query_count": len(supplemented_query_ids),
        },
        "inputs": {
            "expanded_handoff_sha256": _sha256(handoff_bytes),
            "validation_result_sha256": _sha256(args.validation_result.read_bytes()),
            "production_manifest_sha256": _sha256(
                args.production_manifest.read_bytes()
            ),
            "production_bundle_sha256": _sha256(args.production_bundle.read_bytes()),
        },
        "validation_consumed_after_gate": True,
        "training_started": False,
        "production_lock_modified": False,
        "network_request_count": 0,
        "llm_request_count": 0,
        "test_partition_touched": False,
    }
    _write_atomic(
        args.output_dir / "effective-pair-coverage.json",
        _canonical_bytes(comparison),
    )
    manifest = {
        "schema_version": "expanded-receipt-backed-candidate-package-v1",
        "query_count": len(package.query_ids),
        "ordered_receipt_root_count": len(package.ordered_receipt_roots),
        "handoff_sha256": _sha256(handoff_bytes),
        "audit_batch_count": batch_count,
        "effective_pair_coverage_sha256": _sha256(
            _canonical_bytes(comparison)
        ),
        "candidate_membership_materialization": "deterministic_on_load",
        "deduplication_policy": "production_identifier_aliases",
        "base_receipt_merge_policy": "later_sealed_root_wins",
        "supplement_receipt_merge_policy": "monotonic_action_identity_union",
        "aligned_context": "production_unified_local_parser",
        "supplemented_query_count": len(supplemented_query_ids),
        "online_requests_made": 0,
        "llm_requests_made": 0,
        "training_started": False,
        "production_lock_modified": False,
        "test_partition_touched": False,
    }
    _write_atomic(args.output_dir / "manifest.json", _canonical_bytes(manifest))
    print(json.dumps(comparison, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
