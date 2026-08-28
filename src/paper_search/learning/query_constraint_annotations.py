"""Hybrid, evidence-backed query-constraint annotation freezing."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from paper_search.learning.query_constraint_profile import ConstraintLabel
from paper_search.learning.query_constraint_profile import QueryConstraintProfile
from paper_search.query.parser import rule_fallback


AnnotationRole = Literal["training", "development"]
AnnotationStatus = Literal["accepted", "partial", "review_required"]
LabelSource = Literal["rule", "model", "human_review", "local_deterministic"]
_MODEL_LABELS: tuple[ConstraintLabel, ...] = (
    "conceptual",
    "dataset",
    "method",
    "task",
)
_RULE_LABELS: tuple[ConstraintLabel, ...] = ("negation", "title_like", "year")


def query_sha256(query: str) -> str:
    return "sha256:" + hashlib.sha256(query.encode("utf-8")).hexdigest()


def _normalized(values: Sequence[str]) -> list[str]:
    return sorted({" ".join(value.casefold().split()) for value in values if value.strip()})


def _content_sha256(payload: object) -> str:
    content = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(content).hexdigest()


class ModelConstraintSuggestion(BaseModel):
    """Frozen output contract for a model that only annotates query structure."""

    model_config = ConfigDict(frozen=True)

    query_id: str
    query_sha256: str
    methods: list[str] = Field(default_factory=list)
    datasets: list[str] = Field(default_factory=list)
    tasks: list[str] = Field(default_factory=list)
    conceptual: bool | None = None
    evidence: dict[str, list[str]] = Field(default_factory=dict)
    confidence: dict[str, float] = Field(default_factory=dict)
    model_id: str
    prompt_sha256: str


class FrozenConstraintAnnotation(BaseModel):
    model_config = ConfigDict(frozen=True)

    query_id: str
    role: AnnotationRole
    split: Literal["auto_train", "auto_dev"]
    query_sha256: str
    labels: list[ConstraintLabel]
    methods: list[str] = Field(default_factory=list)
    datasets: list[str] = Field(default_factory=list)
    tasks: list[str] = Field(default_factory=list)
    year_from: int | None = None
    year_to: int | None = None
    exclusions: list[str] = Field(default_factory=list)
    label_sources: dict[str, LabelSource] = Field(default_factory=dict)
    label_confidence: dict[str, float] = Field(default_factory=dict)
    evidence: dict[str, list[str]] = Field(default_factory=dict)
    status: AnnotationStatus


class ConstraintAnnotationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    annotation: FrozenConstraintAnnotation
    review_reasons: list[str]
    needs_model_annotation: bool


def constraint_profile_from_annotation(
    annotation: FrozenConstraintAnnotation,
) -> QueryConstraintProfile:
    """Convert only accepted frozen labels into the runtime profile contract."""

    accepted_confidence = [
        annotation.label_confidence[label]
        for label in annotation.labels
        if label in annotation.label_confidence
    ]
    confidence = min(accepted_confidence) if accepted_confidence else 0.25
    return QueryConstraintProfile(
        labels=list(annotation.labels),
        methods=list(annotation.methods),
        datasets=list(annotation.datasets),
        tasks=list(annotation.tasks),
        exclusions=list(annotation.exclusions),
        year_from=annotation.year_from,
        year_to=annotation.year_to,
        has_negation="negation" in annotation.labels,
        is_title_like="title_like" in annotation.labels,
        is_conceptual="conceptual" in annotation.labels,
        constraint_count=len(annotation.labels),
        confidence=confidence,
    )


class FrozenConstraintProfileStore:
    """Read-only constraint lookup with an explicit train/dev role boundary."""

    def __init__(self, annotations: Sequence[FrozenConstraintAnnotation]) -> None:
        self._profiles: dict[str, QueryConstraintProfile] = {}
        self._annotations: dict[str, FrozenConstraintAnnotation] = {}
        for raw in annotations:
            annotation = FrozenConstraintAnnotation.model_validate(raw)
            profile = constraint_profile_from_annotation(annotation)
            existing = self._profiles.get(annotation.query_sha256)
            if existing is not None and existing != profile:
                raise ValueError("conflicting frozen profiles for one query hash")
            existing_annotation = self._annotations.get(annotation.query_sha256)
            if existing_annotation is not None and (
                existing_annotation.role != annotation.role
                or existing_annotation.split != annotation.split
            ):
                raise ValueError("conflicting frozen constraint roles for one query hash")
            self._profiles[annotation.query_sha256] = profile
            self._annotations[annotation.query_sha256] = annotation

    def profile_for_query(self, query: str) -> QueryConstraintProfile | None:
        """Backward-compatible scoring lookup."""

        return self.for_scoring_query(query)

    def for_scoring_query(self, query: str) -> QueryConstraintProfile | None:
        return self._profiles.get(query_sha256(query))

    def for_training_query(self, query: str) -> QueryConstraintProfile | None:
        digest = query_sha256(query)
        annotation = self._annotations.get(digest)
        if annotation is not None and (
            annotation.role != "training" or annotation.split != "auto_train"
        ):
            raise ValueError("development constraint label cannot be used during fit")
        return self._profiles.get(digest)


def _valid_evidence(query: str, snippets: Sequence[str]) -> bool:
    normalized_query = " ".join(query.casefold().split())
    return bool(snippets) and all(
        " ".join(snippet.casefold().split()) in normalized_query
        for snippet in snippets
        if snippet.strip()
    ) and all(snippet.strip() for snippet in snippets)


def annotate_query_constraints(
    *,
    query_id: str,
    role: str,
    split: str,
    query: str,
    suggestion: ModelConstraintSuggestion | None,
    minimum_model_confidence: float = 0.80,
) -> ConstraintAnnotationResult:
    """Merge conservative rules with one frozen, evidence-backed model output."""

    if role not in {"training", "development"} or split not in {
        "auto_train",
        "auto_dev",
    }:
        raise ValueError("query constraint annotations forbid test roles")
    if (role, split) not in {
        ("training", "auto_train"),
        ("development", "auto_dev"),
    }:
        raise ValueError("query role and split do not match")
    digest = query_sha256(query)
    if suggestion is not None:
        if suggestion.query_id != query_id:
            raise ValueError("model annotation query id mismatch")
        if suggestion.query_sha256 != digest:
            raise ValueError("model annotation query hash mismatch")

    spec = rule_fallback(query)
    labels: set[ConstraintLabel] = set()
    sources: dict[str, LabelSource] = {}
    confidence: dict[str, float] = {}
    evidence: dict[str, list[str]] = {}
    if spec.year_from is not None or spec.year_to is not None:
        labels.add("year")
        sources["year"] = "rule"
        confidence["year"] = 1.0
        evidence["year"] = [
            value for value in (str(spec.year_from), str(spec.year_to)) if value != "None"
        ][:1]
    if spec.exclusions:
        labels.add("negation")
        sources["negation"] = "rule"
        confidence["negation"] = 1.0
        evidence["negation"] = list(spec.exclusions)
    stripped = query.strip()
    if (
        len(stripped) >= 2
        and stripped[0] in {'"', "“", "'"}
        and stripped[-1] in {'"', "”", "'"}
    ) or stripped.casefold().startswith(("find the paper titled", "locate the paper named")):
        labels.add("title_like")
        sources["title_like"] = "rule"
        confidence["title_like"] = 1.0
        evidence["title_like"] = [stripped]

    methods: list[str] = []
    datasets: list[str] = []
    tasks: list[str] = []
    reasons: list[str] = []
    if suggestion is not None:
        proposed = {
            "method": _normalized(suggestion.methods),
            "dataset": _normalized(suggestion.datasets),
            "task": _normalized(suggestion.tasks),
        }
        accepted_entities: dict[str, list[str]] = {}
        for label, values in proposed.items():
            if not values:
                continue
            label_confidence = float(suggestion.confidence.get(label, 0.0))
            snippets = suggestion.evidence.get(label, [])
            if label_confidence < minimum_model_confidence:
                reasons.append(f"{label}_low_confidence")
                continue
            if not _valid_evidence(query, snippets):
                reasons.append(f"{label}_evidence_not_in_query")
                continue
            typed_label = label  # narrowed by the fixed mapping above
            labels.add(typed_label)  # type: ignore[arg-type]
            sources[label] = "model"
            confidence[label] = label_confidence
            evidence[label] = list(snippets)
            accepted_entities[label] = values
        methods = accepted_entities.get("method", [])
        datasets = accepted_entities.get("dataset", [])
        tasks = accepted_entities.get("task", [])

        if suggestion.conceptual is True:
            conceptual_confidence = float(suggestion.confidence.get("conceptual", 0.0))
            conceptual_evidence = suggestion.evidence.get("conceptual", [])
            if conceptual_confidence < minimum_model_confidence:
                reasons.append("conceptual_low_confidence")
            elif not _valid_evidence(query, conceptual_evidence):
                reasons.append("conceptual_evidence_not_in_query")
            elif labels.intersection({"method", "dataset", "task", "year", "negation"}):
                reasons.append("conceptual_structured_conflict")
            else:
                labels.add("conceptual")
                sources["conceptual"] = "model"
                confidence["conceptual"] = conceptual_confidence
                evidence["conceptual"] = list(conceptual_evidence)
        if suggestion.conceptual is True and labels.intersection(
            {"method", "dataset", "task", "year", "negation"}
        ):
            if "conceptual_structured_conflict" not in reasons:
                reasons.append("conceptual_structured_conflict")
    else:
        reasons.append("model_annotation_missing")

    status: AnnotationStatus
    if any(reason != "model_annotation_missing" for reason in reasons):
        status = "review_required"
    elif suggestion is None:
        status = "partial"
    else:
        status = "accepted"
    annotation = FrozenConstraintAnnotation(
        query_id=query_id,
        role=role,
        split=split,
        query_sha256=digest,
        labels=sorted(labels),
        methods=methods,
        datasets=datasets,
        tasks=tasks,
        year_from=spec.year_from,
        year_to=spec.year_to,
        exclusions=_normalized(spec.exclusions),
        label_sources=sources,
        label_confidence=confidence,
        evidence=evidence,
        status=status,
    )
    return ConstraintAnnotationResult(
        annotation=annotation,
        review_reasons=sorted(set(reasons)),
        needs_model_annotation=suggestion is None,
    )


def build_hybrid_constraint_package(
    rows: Sequence[Mapping[str, Any]],
    *,
    suggestions: Mapping[str, ModelConstraintSuggestion],
) -> dict[str, Any]:
    """Build deterministic label, review, model-input, coverage, and manifest rows."""

    ordered = sorted(rows, key=lambda row: str(row["query_id"]))
    if len({str(row["query_id"]) for row in ordered}) != len(ordered):
        raise ValueError("query constraint package query ids must be unique")
    annotations: list[dict[str, Any]] = []
    review_queue: list[dict[str, Any]] = []
    model_input_queue: list[dict[str, Any]] = []
    role_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    for row in ordered:
        query_id = str(row["query_id"])
        query = str(row["query"])
        role = str(row["role"])
        split = str(row["split"])
        result = annotate_query_constraints(
            query_id=query_id,
            role=role,
            split=split,
            query=query,
            suggestion=suggestions.get(query_id),
        )
        annotation = result.annotation.model_dump(mode="json")
        annotations.append(annotation)
        role_counts[role] += 1
        status_counts[result.annotation.status] += 1
        label_counts.update(result.annotation.labels)
        if result.review_reasons:
            review_queue.append(
                {
                    "query_id": query_id,
                    "query_sha256": result.annotation.query_sha256,
                    "reasons": result.review_reasons,
                }
            )
        if result.needs_model_annotation:
            model_input_queue.append(
                {
                    "query_id": query_id,
                    "role": role,
                    "split": split,
                    "query": query,
                    "query_sha256": result.annotation.query_sha256,
                    "requested_labels": list(_MODEL_LABELS),
                }
            )
    coverage = {
        "query_count": len(annotations),
        "accepted_label_count": {
            label: label_counts.get(label, 0)
            for label in (*_MODEL_LABELS, *_RULE_LABELS)
        },
        "status_count": dict(sorted(status_counts.items())),
        "review_queue_count": len(review_queue),
        "model_input_queue_count": len(model_input_queue),
    }
    input_identity = [
        {
            "query_id": str(row["query_id"]),
            "role": str(row["role"]),
            "split": str(row["split"]),
            "query_sha256": query_sha256(str(row["query"])),
        }
        for row in ordered
    ]
    manifest = {
        "schema_version": "hybrid-query-constraint-annotations-v1",
        "query_count": len(annotations),
        "query_count_by_role": dict(sorted(role_counts.items())),
        "input_sha256": _content_sha256(input_identity),
        "labels_sha256": _content_sha256(annotations),
        "review_queue_sha256": _content_sha256(review_queue),
        "model_input_queue_sha256": _content_sha256(model_input_queue),
        "test_partition_touched": False,
        "development_labels_used_for_training": False,
    }
    return {
        "annotations": annotations,
        "review_queue": review_queue,
        "model_input_queue": model_input_queue,
        "coverage": coverage,
        "manifest": manifest,
    }


__all__ = [
    "ConstraintAnnotationResult",
    "FrozenConstraintAnnotation",
    "FrozenConstraintProfileStore",
    "ModelConstraintSuggestion",
    "annotate_query_constraints",
    "build_hybrid_constraint_package",
    "constraint_profile_from_annotation",
    "query_sha256",
]
