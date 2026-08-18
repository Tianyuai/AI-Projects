"""Load provider-observed action labels from immutable canary artifacts."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from paper_search.evaluation.dataset import write_frozen_bytes
from paper_search.learning.contracts import PolicyActionCandidate
from paper_search.learning.data_isolation import DatasetRole
from paper_search.learning.provider_action_labels import (
    Provider,
    ProviderActionLabel,
    ProviderActionObservation,
    build_provider_action_labels,
)
from paper_search.recall_experiments.contracts import (
    RecallActionBatch,
    RecallSearchAction,
    RetrievalActionResult,
    TextSearchAction,
)


def _normalized(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _action_text(action: RecallSearchAction) -> str:
    if action.action_type == "text_search":
        return action.payload.query_text
    if action.action_type == "title_search":
        return action.payload.title_text
    raise ValueError("citation expansion cannot train the text action ranker")


def _candidate(
    action: RecallSearchAction,
    *,
    query: str,
    provider: Provider,
) -> PolicyActionCandidate:
    text = _action_text(action)
    search_mode = (
        action.payload.search_mode
        if isinstance(action, TextSearchAction)
        else "lexical"
    )
    return PolicyActionCandidate(
        action_id=action.action_id,
        action_type=action.action_type,
        text=text,
        origin=(
            "original_query"
            if search_mode == "lexical"
            and _normalized(text) == _normalized(query)
            else "deterministic_rule"
        ),
        provider_hint=provider,
        search_mode=search_mode,
    )


def _load_partition(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        raw = json.loads(line)
        if not isinstance(raw, dict):
            raise ValueError(f"partition line {line_number} must be an object")
        query_id = raw.get("query_id")
        if not isinstance(query_id, str) or not query_id:
            raise ValueError(f"partition line {line_number} has no query ID")
        if query_id in rows:
            raise ValueError(f"duplicate partition query ID: {query_id}")
        rows[query_id] = cast(dict[str, Any], raw)
    if not rows:
        raise ValueError("partition is empty")
    return rows


def _successful_or_failed_receipt(batch: Path, query_id: str) -> dict[str, Any]:
    paths = sorted((batch / "retrieval").glob(f"attempt-*/{query_id}.json"))
    if not paths:
        raise ValueError(f"missing retrieval receipt for {query_id} in {batch}")
    succeeded: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    for path in paths:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("attempt_status") == "succeeded":
            succeeded.append(raw)
        elif raw.get("attempt_status") == "failed":
            failed.append(raw)
    selected = succeeded or failed[-1:]
    if len(selected) != 1:
        raise ValueError(f"ambiguous retrieval receipts for {query_id} in {batch}")
    return selected[0]


def _restore_redacted_usage(raw: object) -> object:
    if not isinstance(raw, dict):
        return raw
    restored = dict(raw)
    usage = restored.get("usage")
    if not isinstance(usage, dict):
        return restored
    restored_usage = dict(usage)
    for key in ("input_tokens", "output_tokens"):
        if restored_usage.get(key) == "[REDACTED]":
            restored_usage.pop(key)
    restored["usage"] = restored_usage
    return restored


def load_provider_action_labels_from_canary_runs(
    *,
    partition_path: Path,
    provider_run_roots: Mapping[Provider, Path],
) -> list[ProviderActionLabel]:
    """Rebuild labels without network calls from saved action-level receipts."""
    partition = _load_partition(partition_path)
    labels: list[ProviderActionLabel] = []
    seen: set[tuple[str, Provider]] = set()
    for provider, root in provider_run_roots.items():
        for report_path in sorted(root.glob("*/canary-report.json")):
            batch = report_path.parent
            report = json.loads(report_path.read_text(encoding="utf-8"))
            actions_by_query = report.get("actions_by_query")
            if not isinstance(actions_by_query, dict):
                raise ValueError(f"invalid action map in {report_path}")
            for query_id, raw_actions in actions_by_query.items():
                key = (query_id, provider)
                if key in seen:
                    raise ValueError(f"duplicate provider receipt: {query_id}/{provider}")
                seen.add(key)
                row = partition.get(query_id)
                if row is None:
                    raise ValueError(f"canary query is absent from partition: {query_id}")
                actions = RecallActionBatch.model_validate(
                    {"actions": raw_actions}
                ).actions
                receipt = _successful_or_failed_receipt(batch, query_id)
                raw_results = receipt.get("results")
                if not isinstance(raw_results, list):
                    raise ValueError(f"retrieval receipt has no results: {query_id}")
                results = [
                    RetrievalActionResult.model_validate(_restore_redacted_usage(item))
                    for item in raw_results
                ]
                results_by_id = {result.action_id: result for result in results}
                if len(results_by_id) != len(results):
                    raise ValueError(f"duplicate action result for {query_id}")
                if set(results_by_id) != {action.action_id for action in actions}:
                    raise ValueError(f"action/result mismatch for {query_id}")
                observations = []
                for action in actions:
                    result = results_by_id[action.action_id]
                    observations.append(
                        ProviderActionObservation(
                            provider=provider,
                            action=_candidate(
                                action,
                                query=cast(str, row["query"]),
                                provider=provider,
                            ),
                            hits=result.hits,
                            usage=result.usage,
                            errors=result.errors,
                            infrastructure_failure=result.infrastructure_failure,
                        )
                    )
                role = cast(DatasetRole, row["role"])
                labels.extend(
                    build_provider_action_labels(
                        dataset=cast(str, row["dataset"]),
                        split=cast(str, row["split"]),
                        role=role,
                        query_id=query_id,
                        query=cast(str, row["query"]),
                        gold_paper_ids=cast(list[str], row["gold_paper_ids"]),
                        observations=observations,
                    )
                )
    if not labels:
        raise ValueError("no provider action labels were loaded")
    return labels


def freeze_provider_action_labels(
    labels: list[ProviderActionLabel],
    path: Path,
) -> str:
    validated = [ProviderActionLabel.model_validate(label) for label in labels]
    if not validated:
        raise ValueError("provider action labels are empty")
    ordered = sorted(
        validated,
        key=lambda label: (
            label.dataset,
            label.split,
            label.query_id,
            label.provider,
            label.action.action_id,
        ),
    )
    content = "".join(
        json.dumps(
            label.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
        for label in ordered
    ).encode("utf-8")
    write_frozen_bytes(path, content)
    return "sha256:" + hashlib.sha256(content).hexdigest()


def merge_provider_action_label_sets(
    label_sets: list[list[ProviderActionLabel]],
    *,
    later_sets_override: bool = False,
) -> list[ProviderActionLabel]:
    """Merge compatible action receipts and recompute novelty from one anchor."""
    merged: dict[tuple[str, str, str, Provider, str], ProviderActionLabel] = {}
    query_metadata: dict[tuple[str, str, str, Provider], tuple[str, str, str]] = {}
    for labels in label_sets:
        for raw_label in labels:
            label = ProviderActionLabel.model_validate(raw_label)
            query_key = (
                label.dataset,
                label.split,
                label.query_id,
                label.provider,
            )
            metadata = (label.role, label.query, label.dataset)
            previous_metadata = query_metadata.setdefault(query_key, metadata)
            if previous_metadata != metadata:
                raise ValueError(f"incompatible query metadata: {label.query_id}")
            action_key = (*query_key, label.action.action_id)
            if action_key in merged and not later_sets_override:
                raise ValueError(
                    f"duplicate provider action label: "
                    f"{label.query_id}/{label.provider}/{label.action.action_id}"
                )
            merged[action_key] = label

    grouped: dict[tuple[str, str, str, Provider], list[ProviderActionLabel]] = {}
    for key, label in merged.items():
        grouped.setdefault(key[:-1], []).append(label)

    output: list[ProviderActionLabel] = []
    for query_key, labels in grouped.items():
        anchors = [
            label
            for label in labels
            if label.retrieval_status == "available"
            and label.action.origin == "original_query"
        ]
        if len(anchors) != 1:
            raise ValueError(
                "merged provider labels require exactly one available anchor: "
                f"{query_key[2]}/{query_key[3]}"
            )
        anchor_hits = set(anchors[0].gold_hit_ids)
        for label in labels:
            if label.retrieval_status == "available":
                label = label.model_copy(
                    update={
                        "novel_over_anchor_hit_count": len(
                            set(label.gold_hit_ids).difference(anchor_hits)
                        )
                    }
                )
            output.append(label)
    return sorted(
        output,
        key=lambda label: (
            label.dataset,
            label.split,
            label.query_id,
            label.provider,
            label.action.action_id,
        ),
    )


__all__ = [
    "freeze_provider_action_labels",
    "load_provider_action_labels_from_canary_runs",
    "merge_provider_action_label_sets",
]
