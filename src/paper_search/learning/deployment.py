"""Fail-closed loading and composition hooks for the CPU action ranker."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

from paper_search.learning.adapters import AnalyzerFallback, QueryPolicyAnalyzerAdapter
from paper_search.learning.cpu_action_ranker import CpuActionRanker
from paper_search.learning.cpu_pairwise_ranker import CpuPairwiseActionRanker
from paper_search.learning.policy import BoundedQueryPolicy


def _sha256(content: bytes) -> str:
    return "sha256:" + hashlib.sha256(content).hexdigest()


def load_cpu_action_policy(
    *,
    model_path: Path,
    result_path: Path,
) -> BoundedQueryPolicy:
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        dimension = int(result["dimension"])
        threshold = float(result["learned"]["threshold"])
        expected_hash = str(result["model_sha256"])
    except (OSError, UnicodeError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        raise ValueError("invalid CPU action ranker result manifest") from None
    try:
        model_content = model_path.read_bytes()
    except OSError:
        raise ValueError("CPU action ranker weights are unavailable") from None
    if _sha256(model_content) != expected_hash:
        raise ValueError("CPU action ranker artifact hash mismatch")
    ranker = CpuActionRanker.load(
        model_path,
        dimension=dimension,
        confidence_threshold=threshold,
    )
    return BoundedQueryPolicy(ranker, confidence_threshold=threshold)


def build_cpu_action_analyzer_decorator(
    *,
    model_path: Path,
    result_path: Path,
    max_actions: int,
) -> Callable[[AnalyzerFallback], QueryPolicyAnalyzerAdapter]:
    policy = load_cpu_action_policy(
        model_path=model_path,
        result_path=result_path,
    )

    def decorate(fallback: AnalyzerFallback) -> QueryPolicyAnalyzerAdapter:
        return QueryPolicyAnalyzerAdapter(
            policy,
            max_actions=max_actions,
            fallback=fallback,
        )

    return decorate


def load_cpu_pairwise_action_policy(
    *,
    model_path: Path,
    manifest_path: Path,
) -> BoundedQueryPolicy:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["schema_version"] != "cpu-pairwise-action-ranker-experiment-v1":
            raise ValueError
        if manifest["model_id"] != CpuPairwiseActionRanker.model_id:
            raise ValueError
        target_provider = str(manifest["target_provider"])
        if target_provider != "openalex":
            raise ValueError
        dimension = int(manifest["dimension"])
        epochs = int(manifest["epochs"])
        learning_rate = float(manifest["learning_rate"])
        l2 = float(manifest["l2"])
        seed = int(manifest["seed"])
        threshold = float(manifest["confidence_threshold"])
        expected_hash = str(manifest["model_sha256"])
        if dimension <= 0 or epochs <= 0 or not 0.0 <= threshold <= 1.0:
            raise ValueError
    except (OSError, UnicodeError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        raise ValueError("invalid CPU pairwise ranker manifest") from None
    try:
        model_content = model_path.read_bytes()
    except OSError:
        raise ValueError("CPU pairwise ranker weights are unavailable") from None
    if _sha256(model_content) != expected_hash:
        raise ValueError("CPU pairwise ranker artifact hash mismatch")
    ranker = CpuPairwiseActionRanker.load(
        model_path,
        target_provider="openalex",
        dimension=dimension,
        epochs=epochs,
        learning_rate=learning_rate,
        l2=l2,
        seed=seed,
    )
    return BoundedQueryPolicy(ranker, confidence_threshold=threshold)


def build_cpu_pairwise_action_analyzer_decorator(
    *,
    model_path: Path,
    manifest_path: Path,
    max_actions: int,
) -> Callable[[AnalyzerFallback], QueryPolicyAnalyzerAdapter]:
    policy = load_cpu_pairwise_action_policy(
        model_path=model_path,
        manifest_path=manifest_path,
    )

    def decorate(fallback: AnalyzerFallback) -> QueryPolicyAnalyzerAdapter:
        return QueryPolicyAnalyzerAdapter(
            policy,
            max_actions=max_actions,
            fallback=fallback,
        )

    return decorate


__all__ = [
    "build_cpu_action_analyzer_decorator",
    "build_cpu_pairwise_action_analyzer_decorator",
    "load_cpu_action_policy",
    "load_cpu_pairwise_action_policy",
]
