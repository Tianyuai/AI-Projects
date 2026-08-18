"""Deterministic train/calibrate/evaluate workflow for the CPU pairwise ranker."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

from pydantic import Field

from paper_search.domain.models import DomainModel, UnitFloat
from paper_search.evaluation.dataset import write_frozen_bytes
from paper_search.learning.candidate_ceiling_decision import (
    ProviderCeilingBatchEvidence,
    summarize_candidate_ceiling_batch,
)
from paper_search.learning.contracts import QueryPolicyInput
from paper_search.learning.cpu_baseline import select_f1_threshold
from paper_search.learning.cpu_pairwise_ranker import (
    CpuPairwiseActionRanker,
    provider_action_reward,
)
from paper_search.learning.policy import BoundedQueryPolicy, RuleActionScorer
from paper_search.learning.provider_action_labels import ProviderActionLabel
from paper_search.learning.routing import RuleQueryRouter


class CpuPairwiseExperimentManifest(DomainModel):
    schema_version: str = "cpu-pairwise-action-ranker-experiment-v1"
    model_id: str = CpuPairwiseActionRanker.model_id
    target_provider: str = "openalex"
    dimension: int = Field(strict=True, gt=0)
    epochs: int = Field(strict=True, gt=0)
    learning_rate: float = Field(gt=0)
    l2: float = Field(ge=0)
    seed: int
    confidence_threshold: UnitFloat
    threshold_selected_on: str = "pasa/auto_dev/calibration"
    train_label_count: int = Field(strict=True, ge=0)
    train_query_count: int = Field(strict=True, ge=0)
    train_preference_query_count: int = Field(strict=True, ge=0)
    calibration_label_count: int = Field(strict=True, ge=0)
    calibration_query_count: int = Field(strict=True, ge=0)
    calibration_preference_query_count: int = Field(strict=True, ge=0)
    evaluation_label_count: int = Field(strict=True, ge=0)
    evaluation_query_count: int = Field(strict=True, ge=0)
    evaluation_preference_query_count: int = Field(strict=True, ge=0)
    preference_pair_count: int = Field(strict=True, gt=0)
    rule_evaluation_macro_recall: UnitFloat
    learned_evaluation_macro_recall: UnitFloat
    oracle_at_3_evaluation_macro_recall: UnitFloat
    all_candidate_evaluation_macro_recall: UnitFloat
    train_labels_sha256: str
    calibration_labels_sha256: str
    evaluation_labels_sha256: str
    model_sha256: str


def _sha256(content: bytes) -> str:
    return "sha256:" + hashlib.sha256(content).hexdigest()


def _load(path: Path) -> tuple[list[ProviderActionLabel], bytes]:
    content = path.read_bytes()
    rows = [
        ProviderActionLabel.model_validate(json.loads(line))
        for line in content.decode("utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"provider action label file is empty: {path}")
    return rows, content


def _query_count(rows: list[ProviderActionLabel]) -> int:
    return len({row.query_id for row in rows})


def _preference_query_count(rows: list[ProviderActionLabel]) -> int:
    rewards: dict[str, set[float]] = defaultdict(set)
    for row in rows:
        if row.retrieval_status == "available":
            rewards[row.query_id].add(provider_action_reward(row))
    return sum(len(values) > 1 for values in rewards.values())


def _validate_partition(
    rows: list[ProviderActionLabel],
    *,
    role: str,
    split: str,
    name: str,
) -> None:
    if any(row.role != role for row in rows):
        raise ValueError(f"{name} labels contain the wrong dataset role")
    if any(row.provider != "openalex" for row in rows):
        raise ValueError(f"{name} labels must contain only OpenAlex observations")
    if any(row.dataset != "pasa" or row.split != split for row in rows):
        raise ValueError(f"{name} labels must come from PaSa {split}")


def _calibration_examples(
    ranker: CpuPairwiseActionRanker,
    rows: list[ProviderActionLabel],
) -> tuple[list[bool], list[float]]:
    grouped: dict[str, list[ProviderActionLabel]] = defaultdict(list)
    for row in rows:
        if row.retrieval_status == "available":
            grouped[row.query_id].append(row)
    labels: list[bool] = []
    probabilities: list[float] = []
    router = RuleQueryRouter()
    for query_rows in grouped.values():
        anchors = [
            row for row in query_rows if row.action.origin == "original_query"
        ]
        if len(anchors) != 1:
            raise ValueError("calibration query requires exactly one available anchor")
        anchor_reward = provider_action_reward(anchors[0])
        candidates = [
            row for row in query_rows if row.action.origin != "original_query"
        ]
        routed = router.route(query_rows[0].query)
        request = QueryPolicyInput(
            query_id=query_rows[0].query_id,
            original_query=query_rows[0].query,
            query_kind=routed.query_kind,
            query_spec=routed.query_spec,
            seed_actions=[row.action for row in candidates],
            allowed_action_types=["text_search", "title_search"],
            max_actions=3,
        )
        probabilities.extend(
            ranker.score(request, [row.action for row in candidates])
        )
        labels.extend(
            provider_action_reward(row) > anchor_reward for row in candidates
        )
    if not labels:
        raise ValueError("calibration labels contain no non-anchor action")
    return labels, probabilities


def _openalex_metrics(
    labels: list[ProviderActionLabel],
    policy: BoundedQueryPolicy,
    *,
    batch_id: str,
) -> ProviderCeilingBatchEvidence:
    summary = summarize_candidate_ceiling_batch(
        labels, batch_id=batch_id, policy=policy
    )
    return next(item for item in summary.providers if item.provider == "openalex")


def run_cpu_pairwise_experiment(
    *,
    train_path: Path,
    calibration_path: Path,
    evaluation_path: Path,
    model_path: Path,
    manifest_path: Path,
    dimension: int = 16384,
    epochs: int = 12,
    learning_rate: float = 0.08,
    l2: float = 1e-6,
    seed: int = 17,
) -> CpuPairwiseExperimentManifest:
    train, train_content = _load(train_path)
    calibration, calibration_content = _load(calibration_path)
    evaluation, evaluation_content = _load(evaluation_path)
    _validate_partition(
        train, role="training", split="auto_train", name="training"
    )
    _validate_partition(
        calibration,
        role="development",
        split="auto_dev",
        name="calibration",
    )
    _validate_partition(
        evaluation,
        role="development",
        split="auto_dev",
        name="evaluation",
    )
    ids = [
        {row.query_id for row in train},
        {row.query_id for row in calibration},
        {row.query_id for row in evaluation},
    ]
    if ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]:
        raise ValueError("train, calibration, and evaluation queries must be disjoint")

    ranker = CpuPairwiseActionRanker(
        target_provider="openalex",
        dimension=dimension,
        epochs=epochs,
        learning_rate=learning_rate,
        l2=l2,
        seed=seed,
    )
    pair_count = ranker.fit(train)
    binary_labels, probabilities = _calibration_examples(ranker, calibration)
    threshold = select_f1_threshold(binary_labels, probabilities)
    learned_policy = BoundedQueryPolicy(
        ranker, confidence_threshold=threshold
    )
    rule_policy = BoundedQueryPolicy(
        RuleActionScorer(), confidence_threshold=0.0
    )
    learned = _openalex_metrics(
        evaluation, learned_policy, batch_id="auto-dev-evaluation-learned"
    )
    rule = _openalex_metrics(
        evaluation, rule_policy, batch_id="auto-dev-evaluation-rule"
    )
    ranker.save(model_path)
    model_content = model_path.read_bytes()
    manifest = CpuPairwiseExperimentManifest(
        dimension=dimension,
        epochs=epochs,
        learning_rate=learning_rate,
        l2=l2,
        seed=seed,
        confidence_threshold=threshold,
        train_label_count=len(train),
        train_query_count=_query_count(train),
        train_preference_query_count=_preference_query_count(train),
        calibration_label_count=len(calibration),
        calibration_query_count=_query_count(calibration),
        calibration_preference_query_count=_preference_query_count(calibration),
        evaluation_label_count=len(evaluation),
        evaluation_query_count=_query_count(evaluation),
        evaluation_preference_query_count=_preference_query_count(evaluation),
        preference_pair_count=pair_count,
        rule_evaluation_macro_recall=rule.current_top3_macro_recall,
        learned_evaluation_macro_recall=learned.current_top3_macro_recall,
        oracle_at_3_evaluation_macro_recall=learned.oracle_at_3_macro_recall,
        all_candidate_evaluation_macro_recall=learned.all_candidate_macro_recall,
        train_labels_sha256=_sha256(train_content),
        calibration_labels_sha256=_sha256(calibration_content),
        evaluation_labels_sha256=_sha256(evaluation_content),
        model_sha256=_sha256(model_content),
    )
    content = (
        json.dumps(
            manifest.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")
    write_frozen_bytes(manifest_path, content)
    return manifest


__all__ = ["CpuPairwiseExperimentManifest", "run_cpu_pairwise_experiment"]
