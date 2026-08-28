"""Compare production/candidate B0/F4/F5 artifacts on frozen auto_dev."""

from __future__ import annotations

import argparse
import copy
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

from paper_search.learning.document_ranking_receipts import (  # noqa: E402
    load_folded_document_ranking_evaluation_queries,
)
from paper_search.learning.anchored_fusion import (  # noqa: E402
    scale_anchored_family_weights,
)
from paper_search.learning.f5_production_deployment import (  # noqa: E402
    load_f5_production_ranker_bytes,
)
from paper_search.learning.fusion_model_comparison import (  # noqa: E402
    evaluate_fusion_model_set,
)
from paper_search.learning.gated_feature_fusion_ranker import (  # noqa: E402
    load_gated_feature_fusion_ranker_bytes,
)


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _load_rankers(args: argparse.Namespace) -> dict[str, Any]:
    production_f5 = load_f5_production_ranker_bytes(
        args.production_f5_manifest.read_bytes(),
        args.production_f5_weights.read_bytes(),
    )
    baseline = production_f5.baseline_ranker
    context = production_f5.context_store
    if args.anchored_reliability_scale is not None:
        anchored = copy.copy(production_f5)
        anchored.weights = scale_anchored_family_weights(
            production_f5.weights,
            family="task_provenance",
            scale=args.anchored_task_provenance_scale,
        )
        anchored.weights = scale_anchored_family_weights(
            anchored.weights,
            family="reliability",
            scale=args.anchored_reliability_scale,
        )
        return {
            "B0": baseline,
            "F5-production-18314": production_f5,
            "F5-production-anchored": anchored,
        }

    production_f4 = load_f5_production_ranker_bytes(
        args.production_f4_manifest.read_bytes(),
        args.production_f4_weights.read_bytes(),
    )
    candidate_f5 = load_gated_feature_fusion_ranker_bytes(
        args.candidate_f5_manifest.read_bytes(),
        args.candidate_f5_weights.read_bytes(),
        baseline_ranker=baseline,
        context_store=context,
    )
    candidate_f4 = load_gated_feature_fusion_ranker_bytes(
        args.candidate_f4_manifest.read_bytes(),
        args.candidate_f4_weights.read_bytes(),
        baseline_ranker=baseline,
        context_store=context,
    )
    return {
        "B0": baseline,
        "F4-production-18314": production_f4,
        "F5-production-18314": production_f5,
        "F4-candidate-21429": candidate_f4,
        "F5-candidate-21429": candidate_f5,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "data/training_private/manifests/"
            "pasa-auto-dev-extended-independent538-v1.json"
        ),
    )
    parser.add_argument(
        "--partition",
        type=Path,
        default=Path("data/training_private/freeze-v1/partitions/pasa_auto_dev.jsonl"),
    )
    parser.add_argument(
        "--receipt-root",
        type=Path,
        action="append",
        default=None,
    )
    parser.add_argument("--required-action-id", action="append", default=[])
    parser.add_argument("--required-candidate-policy")
    parser.add_argument(
        "--production-f5-manifest",
        type=Path,
        default=Path(
            "artifacts/models/"
            "gated-feature-fusion-18314-unified-context-v3-v1/manifest.json"
        ),
    )
    parser.add_argument(
        "--production-f5-weights",
        type=Path,
        default=Path(
            "artifacts/models/"
            "gated-feature-fusion-18314-unified-context-v3-v1/weights.bundle"
        ),
    )
    parser.add_argument(
        "--production-f4-manifest",
        type=Path,
        default=Path(
            "artifacts/models/"
            "reliability-fusion-18314-unified-context-v3-v1/manifest.json"
        ),
    )
    parser.add_argument(
        "--production-f4-weights",
        type=Path,
        default=Path(
            "artifacts/models/"
            "reliability-fusion-18314-unified-context-v3-v1/weights.bundle"
        ),
    )
    parser.add_argument(
        "--candidate-f5-manifest",
        type=Path,
        default=Path(
            "artifacts/models/"
            "gated-feature-fusion-21429-openalex-pasa-high-recall-v2-fast64-"
            "context-v4-v2/manifest.json"
        ),
    )
    parser.add_argument(
        "--candidate-f5-weights",
        type=Path,
        default=Path(
            "artifacts/models/"
            "gated-feature-fusion-21429-openalex-pasa-high-recall-v2-fast64-"
            "context-v4-v2/weights.bin"
        ),
    )
    parser.add_argument(
        "--candidate-f4-manifest",
        type=Path,
        default=Path(
            "artifacts/models/"
            "gated-feature-fusion-21429-openalex-pasa-high-recall-v2-fast64-"
            "context-v4-v2-f4/manifest.json"
        ),
    )
    parser.add_argument(
        "--candidate-f4-weights",
        type=Path,
        default=Path(
            "artifacts/models/"
            "gated-feature-fusion-21429-openalex-pasa-high-recall-v2-fast64-"
            "context-v4-v2-f4/weights.bin"
        ),
    )
    parser.add_argument("--anchored-reliability-scale", type=float)
    parser.add_argument(
        "--anchored-task-provenance-scale", type=float, default=1.0
    )
    parser.add_argument("--gate-model", default="F5-candidate-21429")
    parser.add_argument("--production-model", default="F5-production-18314")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/training_private/evaluations/"
            "fusion-training-packages-production18314-vs-candidate21429-"
            "auto-dev538-v1.json"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if (
        args.anchored_reliability_scale is not None
        and not 0.0 <= args.anchored_reliability_scale <= 1.0
    ) or not 0.0 <= args.anchored_task_provenance_scale <= 1.0:
        raise ValueError("anchored independent evaluation scales are invalid")
    anchored_mode = args.anchored_reliability_scale is not None
    gate_model = (
        "F5-production-anchored"
        if anchored_mode and args.gate_model == "F5-candidate-21429"
        else args.gate_model
    )
    receipt_roots = args.receipt_root or [
        Path(
            "data/training_private/online_recall/"
            "core4-semantic-boolean-paired90-candidate-v1"
        ),
        Path(
            "data/training_private/online_recall/"
            "auto-dev-core4-backfill-20260820-smoke-v1"
        ),
        Path(
            "data/training_private/online_recall/"
            "auto-dev-core4-backfill-20260820-paired300-v4"
        ),
        Path(
            "data/training_private/online_recall/"
            "auto-dev-core4-backfill-20260820-paired588-v1"
        ),
        Path(
            "data/training_private/online_recall/"
            "auto-dev-core4-backfill-20260820-coverage-replay-v1"
        ),
    ]
    queries = load_folded_document_ranking_evaluation_queries(
        manifest_path=args.manifest,
        partition_path=args.partition,
        receipt_roots=receipt_roots,
        required_action_ids=(
            frozenset(args.required_action_id) if args.required_action_id else None
        ),
        required_candidate_policy=args.required_candidate_policy,
    )
    live_rankers = _load_rankers(args)
    replay_rankers = _load_rankers(args)
    report = evaluate_fusion_model_set(
        queries,
        rankers=live_rankers,
        replay_rankers=replay_rankers,
        gate_model=gate_model,
        production_model=args.production_model,
    )
    inputs: Mapping[str, Path] = {
        "sample_manifest": args.manifest,
        "partition": args.partition,
        "production_f5_manifest": args.production_f5_manifest,
        "production_f5_weights": args.production_f5_weights,
    }
    if not anchored_mode:
        inputs = {
            **inputs,
            "production_f4_manifest": args.production_f4_manifest,
            "production_f4_weights": args.production_f4_weights,
            "candidate_f5_manifest": args.candidate_f5_manifest,
            "candidate_f5_weights": args.candidate_f5_weights,
            "candidate_f4_manifest": args.candidate_f4_manifest,
            "candidate_f4_weights": args.candidate_f4_weights,
        }
    report["inputs"] = {
        name: {"path": str(path), "sha256": _sha256(path)}
        for name, path in inputs.items()
    }
    report["live_replay_gate"]["scope"] = (
        "fresh-artifact-loads-over-identical-frozen-receipts"
    )
    if anchored_mode:
        report["anchored_candidate"] = {
            "base_model": "F5-production-18314",
            "reliability_scale": args.anchored_reliability_scale,
            "task_provenance_scale": args.anchored_task_provenance_scale,
            "entity_scale": 1.0,
            "hard_constraint_scale": 1.0,
            "new_weights_fitted": False,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(json.dumps({**report, "report": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
