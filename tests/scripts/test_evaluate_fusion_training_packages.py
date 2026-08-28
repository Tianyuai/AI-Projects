from __future__ import annotations

from scripts.evaluate_fusion_training_packages import build_parser


def test_fusion_comparison_cli_defaults_to_frozen_auto_dev() -> None:
    args = build_parser().parse_args([])

    assert args.manifest.name == "pasa-auto-dev-extended-independent538-v1.json"
    assert args.partition.name == "pasa_auto_dev.jsonl"
    assert args.gate_model == "F5-candidate-21429"
    assert args.production_model == "F5-production-18314"
    assert (
        args.output.name
        == "fusion-training-packages-production18314-vs-candidate21429-auto-dev538-v1.json"
    )


def test_fusion_comparison_cli_accepts_production_anchored_scales() -> None:
    args = build_parser().parse_args(
        [
            "--anchored-reliability-scale",
            "0",
            "--anchored-task-provenance-scale",
            "0",
        ]
    )

    assert args.anchored_reliability_scale == 0.0
    assert args.anchored_task_provenance_scale == 0.0
