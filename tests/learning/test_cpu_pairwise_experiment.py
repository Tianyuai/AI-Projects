from __future__ import annotations

import json
from pathlib import Path

from paper_search.learning.contracts import PolicyActionCandidate
from paper_search.learning.cpu_pairwise_experiment import (
    run_cpu_pairwise_experiment,
)
from paper_search.learning.provider_action_labels import ProviderActionLabel


def _action(action_id: str, text: str, *, origin: str) -> PolicyActionCandidate:
    return PolicyActionCandidate(
        action_id=action_id,
        action_type="text_search",
        text=text,
        origin=origin,
        provider_hint="openalex",
    )


def _rows(*, role: str, split: str, prefix: str) -> list[ProviderActionLabel]:
    rows: list[ProviderActionLabel] = []
    for index in range(2):
        query = f"Find graph retrieval method {index}"
        for action, hit in (
            (_action("anchor", query, origin="original_query"), False),
            (_action("good", "graph diffusion retrieval", origin="deterministic_rule"), True),
            (_action("bad", "protein structure", origin="deterministic_rule"), False),
        ):
            rows.append(
                ProviderActionLabel(
                    dataset="pasa",
                    split=split,
                    role=role,
                    query_id=f"{prefix}-{index}",
                    query=query,
                    provider="openalex",
                    action=action,
                    retrieval_status="available",
                    gold_association_count=1,
                    gold_hit_ids=("doi:10.1/hit",) if hit else (),
                    gold_hit_count=int(hit),
                    action_recall=float(hit),
                    novel_over_anchor_hit_count=int(hit),
                )
            )
    return rows


def _write(path: Path, rows: list[ProviderActionLabel]) -> None:
    path.write_text(
        "".join(row.model_dump_json() + "\n" for row in rows),
        encoding="utf-8",
    )


def test_pairwise_experiment_freezes_deterministic_hashed_bundle(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    calibration = tmp_path / "calibration.jsonl"
    evaluation = tmp_path / "evaluation.jsonl"
    _write(train, _rows(role="training", split="auto_train", prefix="train"))
    _write(
        calibration,
        _rows(role="development", split="auto_dev", prefix="calibration"),
    )
    _write(
        evaluation,
        _rows(role="development", split="auto_dev", prefix="evaluation"),
    )
    first_model = tmp_path / "first.f64"
    first_manifest = tmp_path / "first.json"
    second_model = tmp_path / "second.f64"
    second_manifest = tmp_path / "second.json"

    first = run_cpu_pairwise_experiment(
        train_path=train,
        calibration_path=calibration,
        evaluation_path=evaluation,
        model_path=first_model,
        manifest_path=first_manifest,
        dimension=256,
        epochs=3,
        seed=7,
    )
    second = run_cpu_pairwise_experiment(
        train_path=train,
        calibration_path=calibration,
        evaluation_path=evaluation,
        model_path=second_model,
        manifest_path=second_manifest,
        dimension=256,
        epochs=3,
        seed=7,
    )

    assert first.model_sha256 == second.model_sha256
    assert first_model.read_bytes() == second_model.read_bytes()
    assert json.loads(first_manifest.read_text(encoding="utf-8")) == json.loads(
        second_manifest.read_text(encoding="utf-8")
    )
    assert first.target_provider == "openalex"
    assert first.threshold_selected_on == "pasa/auto_dev/calibration"
    assert first.train_query_count == 2
    assert first.calibration_query_count == 2
    assert first.evaluation_query_count == 2
    assert first.preference_pair_count > 0
    assert 0.0 <= first.confidence_threshold <= 1.0
