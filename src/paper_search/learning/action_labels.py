"""Leakage-safe weak supervision for complete retrieval actions."""

from __future__ import annotations

import re
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Literal, cast

from pydantic import Field

from paper_search.domain.models import DomainModel, NonEmptyStr, Sha256
from paper_search.evaluation.dataset import write_frozen_bytes
from paper_search.learning.candidates import DeterministicActionCandidateGenerator
from paper_search.learning.contracts import PolicyActionCandidate, QueryKind
from paper_search.learning.data_isolation import DatasetRole
from paper_search.learning.routing import RuleQueryRouter


def _tokens(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", value.casefold()))


class ActionWeakLabel(DomainModel):
    dataset: NonEmptyStr
    split: NonEmptyStr
    role: Literal["training", "development"]
    query_id: NonEmptyStr
    query: NonEmptyStr
    query_kind: QueryKind
    action: PolicyActionCandidate
    label: Literal["positive", "hard_negative"]


class ActionLabelManifest(DomainModel):
    schema_version: Literal["query-action-labels-v1"] = "query-action-labels-v1"
    dataset: NonEmptyStr
    split: NonEmptyStr
    role: Literal["training", "development"]
    query_count: int = Field(strict=True, ge=0)
    action_count: int = Field(strict=True, ge=0)
    positive_count: int = Field(strict=True, ge=0)
    hard_negative_count: int = Field(strict=True, ge=0)
    output_sha256: Sha256


def build_action_labels(
    *,
    dataset: str,
    split: str,
    role: DatasetRole,
    query_id: str,
    query: str,
    query_kind: QueryKind,
    candidates: list[PolicyActionCandidate],
    gold_titles: list[str],
) -> list[ActionWeakLabel]:
    if role == "final_test":
        raise ValueError("final_test cannot produce action labels")
    output_role: Literal["training", "development"] = role
    gold_tokens = {token for title in gold_titles for token in _tokens(title)}
    result: list[ActionWeakLabel] = []
    for raw_candidate in candidates:
        candidate = PolicyActionCandidate.model_validate(raw_candidate)
        action_tokens = _tokens(candidate.text)
        overlap = len(action_tokens.intersection(gold_tokens))
        precision = overlap / len(action_tokens) if action_tokens else 0.0
        positive = candidate.origin == "original_query" or (
            overlap >= 2 and precision >= 0.5
        )
        result.append(
            ActionWeakLabel(
                dataset=dataset,
                split=split,
                role=output_role,
                query_id=query_id,
                query=query,
                query_kind=query_kind,
                action=candidate,
                label="positive" if positive else "hard_negative",
            )
        )
    return result


def freeze_action_labels(
    *,
    partition_path: Path,
    source_path: Path,
    output_path: Path,
    max_candidates: int = 8,
) -> ActionLabelManifest:
    partition_rows = [
        json.loads(line)
        for line in partition_path.read_text(encoding="utf-8").splitlines()
    ]
    if not partition_rows:
        raise ValueError("frozen partition is empty")
    dataset = str(partition_rows[0]["dataset"])
    split = str(partition_rows[0]["split"])
    raw_role = str(partition_rows[0]["role"])
    if raw_role not in {"training", "development"}:
        raise ValueError("final_test cannot produce action labels")
    role = cast(Literal["training", "development"], raw_role)
    allowed = {str(row["query_id"]): str(row["query"]) for row in partition_rows}
    source_by_id: dict[str, dict[str, object]] = {}
    for line in source_path.read_text(encoding="utf-8").splitlines():
        raw = json.loads(line)
        source_by_id[str(raw["qid"])] = raw
    missing = set(allowed).difference(source_by_id)
    if missing:
        raise ValueError(f"source records missing frozen query IDs: {len(missing)}")
    router = RuleQueryRouter()
    generator = DeterministicActionCandidateGenerator(
        max_candidates=max_candidates
    )
    labels: list[ActionWeakLabel] = []
    for query_id, query in allowed.items():
        raw = source_by_id[query_id]
        if str(raw["question"]).strip() != query:
            raise ValueError(f"query mismatch for frozen ID: {query_id}")
        answers = raw.get("answer")
        if not isinstance(answers, list) or not all(
            isinstance(value, str) and value.strip() for value in answers
        ):
            raise ValueError(f"invalid Gold titles for frozen ID: {query_id}")
        routed = router.route(query)
        candidates = generator.generate(
            routed.query_spec,
            query_kind=routed.query_kind,
        )
        labels.extend(
            build_action_labels(
                dataset=dataset,
                split=split,
                role=role,
                query_id=query_id,
                query=query,
                query_kind=routed.query_kind,
                candidates=candidates,
                gold_titles=answers,
            )
        )
    content = (
        "".join(
            json.dumps(
                row.model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for row in labels
        )
    ).encode("utf-8")
    write_frozen_bytes(output_path, content)
    counts = Counter(row.label for row in labels)
    return ActionLabelManifest(
        dataset=dataset,
        split=split,
        role=role,
        query_count=len(allowed),
        action_count=len(labels),
        positive_count=counts["positive"],
        hard_negative_count=counts["hard_negative"],
        output_sha256="sha256:" + hashlib.sha256(content).hexdigest(),
    )


__all__ = [
    "ActionLabelManifest",
    "ActionWeakLabel",
    "build_action_labels",
    "freeze_action_labels",
]
