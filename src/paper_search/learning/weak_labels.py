"""Leakage-safe weak labels for query-term retention on CPU."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Literal, cast

from pydantic import Field

from paper_search.domain.models import DomainModel, NonEmptyStr, Sha256
from paper_search.evaluation.dataset import write_frozen_bytes
from paper_search.learning.data_isolation import DatasetRole


_QUESTION_STOPWORDS = frozenset(
    {
        "a",
        "about",
        "an",
        "and",
        "any",
        "are",
        "can",
        "could",
        "does",
        "for",
        "from",
        "give",
        "have",
        "in",
        "is",
        "me",
        "mention",
        "of",
        "on",
        "paper",
        "papers",
        "please",
        "provide",
        "research",
        "show",
        "some",
        "studies",
        "study",
        "that",
        "the",
        "to",
        "what",
        "which",
        "who",
        "with",
        "work",
        "works",
    }
)


def _tokens(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", value.casefold())


class QueryTermLabel(DomainModel):
    dataset: NonEmptyStr
    split: NonEmptyStr
    role: Literal["training", "development"]
    query_id: NonEmptyStr
    query: NonEmptyStr
    action_type: Literal["text_search"] = "text_search"
    action_text: NonEmptyStr
    origin: Literal["query_term"] = "query_term"
    label: Literal["positive", "hard_negative"]
    query_term_index: int = Field(strict=True, ge=0)


class QueryTermLabelManifest(DomainModel):
    schema_version: Literal["query-term-labels-v1"] = "query-term-labels-v1"
    dataset: NonEmptyStr
    split: NonEmptyStr
    role: Literal["training", "development"]
    query_count: int = Field(strict=True, ge=0)
    label_count: int = Field(strict=True, ge=0)
    positive_count: int = Field(strict=True, ge=0)
    hard_negative_count: int = Field(strict=True, ge=0)
    output_sha256: Sha256


def build_query_term_labels(
    *,
    dataset: str,
    split: str,
    role: DatasetRole,
    query_id: str,
    query: str,
    gold_titles: list[str],
) -> list[QueryTermLabel]:
    """Use Gold titles for labels but serialize only query-derived terms."""
    if role == "final_test":
        raise ValueError("final_test cannot produce query-term labels")
    gold_tokens = {token for title in gold_titles for token in _tokens(title)}
    seen: set[str] = set()
    labels: list[QueryTermLabel] = []
    for index, token in enumerate(_tokens(query)):
        if len(token) < 3 or token in _QUESTION_STOPWORDS or token in seen:
            continue
        seen.add(token)
        labels.append(
            QueryTermLabel(
                dataset=dataset,
                split=split,
                role=role,
                query_id=query_id,
                query=query,
                action_text=token,
                label="positive" if token in gold_tokens else "hard_negative",
                query_term_index=index,
            )
        )
    return labels


def _jsonl_bytes(rows: list[QueryTermLabel]) -> bytes:
    return (
        "".join(
            json.dumps(
                row.model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for row in rows
        )
    ).encode("utf-8")


def freeze_query_term_labels(
    *,
    partition_path: Path,
    source_path: Path,
    output_path: Path,
) -> QueryTermLabelManifest:
    partition_rows = [
        json.loads(line)
        for line in partition_path.read_text(encoding="utf-8").splitlines()
    ]
    if not partition_rows:
        raise ValueError("frozen partition is empty")
    dataset = str(partition_rows[0]["dataset"])
    split = str(partition_rows[0]["split"])
    role = str(partition_rows[0]["role"])
    if role not in {"training", "development"}:
        raise ValueError("final_test cannot produce query-term labels")
    label_role = cast(Literal["training", "development"], role)
    allowed = {str(row["query_id"]): str(row["query"]) for row in partition_rows}
    source_by_id: dict[str, dict[str, object]] = {}
    for line in source_path.read_text(encoding="utf-8").splitlines():
        raw = json.loads(line)
        source_by_id[str(raw["qid"])] = raw
    missing = sorted(set(allowed).difference(source_by_id))
    if missing:
        raise ValueError(f"source records missing frozen query IDs: {len(missing)}")

    labels: list[QueryTermLabel] = []
    for query_id, query in allowed.items():
        raw = source_by_id[query_id]
        if str(raw["question"]).strip() != query:
            raise ValueError(f"query mismatch for frozen ID: {query_id}")
        answers = raw.get("answer")
        if not isinstance(answers, list) or not all(
            isinstance(value, str) and value.strip() for value in answers
        ):
            raise ValueError(f"invalid Gold titles for frozen ID: {query_id}")
        labels.extend(
            build_query_term_labels(
                dataset=dataset,
                split=split,
                role=label_role,
                query_id=query_id,
                query=query,
                gold_titles=answers,
            )
        )
    content = _jsonl_bytes(labels)
    write_frozen_bytes(output_path, content)
    counts = Counter(row.label for row in labels)
    return QueryTermLabelManifest(
        dataset=dataset,
        split=split,
        role=role,
        query_count=len(allowed),
        label_count=len(labels),
        positive_count=counts["positive"],
        hard_negative_count=counts["hard_negative"],
        output_sha256="sha256:" + hashlib.sha256(content).hexdigest(),
    )


__all__ = [
    "QueryTermLabel",
    "QueryTermLabelManifest",
    "build_query_term_labels",
    "freeze_query_term_labels",
]
