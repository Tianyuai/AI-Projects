from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from collections.abc import Sequence
from datetime import date
from hashlib import sha256
from pathlib import Path
from typing import Annotated, Literal

from pydantic import AfterValidator, Field, StringConstraints, ValidationError, model_validator

from paper_search.domain.models import DomainModel, NonEmptyStr, NonNegativeInt, UnitFloat


QueryType = Literal[
    "topic",
    "method",
    "dataset",
    "time_venue",
    "combined",
    "relationship",
    "exclusion",
    "ambiguous",
]
AnnotationKind = Literal["type-domain", "constraints"]
DOMAIN_LABELS = (
    "artificial-intelligence",
    "machine-learning",
    "natural-language-processing",
    "information-retrieval",
    "computer-vision",
    "speech-audio",
    "robotics",
    "data-mining",
    "knowledge-graphs",
    "recommender-systems",
    "human-computer-interaction",
    "software-engineering",
    "computer-systems",
    "networks-security",
    "databases",
    "theory-algorithms",
    "computational-biology",
    "computational-social-science",
    "scientific-computing",
    "multidisciplinary",
    "other",
)


def _validate_domain_label(value: str) -> str:
    if value not in DOMAIN_LABELS:
        raise ValueError("domain must use the frozen domain-labels-v1 vocabulary")
    return value


DomainLabel = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$",
    ),
    AfterValidator(_validate_domain_label),
]


class TypeDomainAnnotationRecord(DomainModel):
    """One minimal human-authored query type/domain label."""

    query_id: NonEmptyStr
    query_type: QueryType
    domain: DomainLabel
    annotator: NonEmptyStr

class AnnotationRecord(DomainModel):
    """One human-authored query constraint annotation."""

    query_id: NonEmptyStr
    research_goal: NonEmptyStr
    must_have: list[NonEmptyStr] = Field(default_factory=list)
    should_have: list[NonEmptyStr] = Field(default_factory=list)
    exclusions: list[NonEmptyStr] = Field(default_factory=list)
    year_from: int | None = None
    year_to: int | None = None
    venues: list[NonEmptyStr] = Field(default_factory=list)
    query_type: QueryType
    domain: DomainLabel
    annotator: NonEmptyStr

    @model_validator(mode="after")
    def validate_year_range(self) -> AnnotationRecord:
        maximum_year = date.today().year + 1
        for name, value in (("year_from", self.year_from), ("year_to", self.year_to)):
            if value is not None and not 1900 <= value <= maximum_year:
                raise ValueError(f"{name} must be between 1900 and {maximum_year}")
        if (
            self.year_from is not None
            and self.year_to is not None
            and self.year_from > self.year_to
        ):
            raise ValueError("year_from must not exceed year_to")
        return self


class AnnotationValidationSummary(DomainModel):
    """Secret-safe result of validating one private annotation file."""

    status: Literal["valid"] = "valid"
    kind: AnnotationKind
    count: NonNegativeInt
    sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    ids_match: bool


class FieldAgreement(DomainModel):
    """Agreement result for one categorical annotation field."""

    kappa: float = Field(ge=-1, le=1, allow_inf_nan=False)
    threshold: UnitFloat
    accepted: bool


class AgreementReport(DomainModel):
    """Cohen's kappa results for aligned human annotation files."""

    compared_query_count: NonNegativeInt
    fields: dict[str, FieldAgreement]


def validate_annotation_file(
    labels_path: Path,
    ids_path: Path,
    *,
    kind: AnnotationKind,
) -> AnnotationValidationSummary:
    """Validate private JSONL without returning paths, IDs, or record values."""
    try:
        if kind not in ("type-domain", "constraints"):
            raise ValueError
        raw_labels = labels_path.read_bytes()
        lines = raw_labels.decode("utf-8").splitlines()
        if not lines or any(not line.strip() for line in lines):
            raise ValueError

        model = TypeDomainAnnotationRecord if kind == "type-domain" else AnnotationRecord
        query_ids: list[str] = []
        for line in lines:
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError
            query_ids.append(model.model_validate(payload).query_id)
        if len(query_ids) != len(set(query_ids)):
            raise ValueError

        expected_payload = json.loads(ids_path.read_bytes().decode("utf-8"))
        if not isinstance(expected_payload, list) or not all(
            isinstance(query_id, str) and query_id.strip() for query_id in expected_payload
        ):
            raise ValueError
        expected_ids = list(expected_payload)
        if len(expected_ids) != len(set(expected_ids)):
            raise ValueError
        if len(query_ids) != len(expected_ids) or set(query_ids) != set(expected_ids):
            raise ValueError
    except (OSError, UnicodeError, json.JSONDecodeError, ValidationError, TypeError, ValueError):
        raise ValueError("private annotations are invalid") from None

    return AnnotationValidationSummary(
        kind=kind,
        count=len(query_ids),
        sha256=f"sha256:{sha256(raw_labels).hexdigest()}",
        ids_match=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Validate one private annotation file and print only a safe summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("type-domain", "constraints"), required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--ids", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        summary = validate_annotation_file(args.labels, args.ids, kind=args.kind)
    except ValueError:
        print("annotation validation failed", file=sys.stderr)
        return 1
    print(summary.model_dump_json())
    return 0


def cohen_kappa(first: Sequence[str], second: Sequence[str]) -> float:
    """Compute Cohen's kappa from two aligned categorical label sequences."""
    if len(first) != len(second):
        raise ValueError("label sequences must have the same length")
    if not first:
        raise ValueError("label sequences must be non-empty")

    sample_count = len(first)
    observed_agreement = sum(
        first_label == second_label
        for first_label, second_label in zip(first, second, strict=True)
    ) / sample_count
    first_counts = Counter(first)
    second_counts = Counter(second)
    expected_agreement = sum(
        (first_counts[label] / sample_count) * (second_counts[label] / sample_count)
        for label in first_counts.keys() | second_counts.keys()
    )

    if math.isclose(expected_agreement, 1.0):
        if math.isclose(observed_agreement, 1.0):
            return 1.0
        raise ValueError("kappa is undefined when expected agreement is one")
    return (observed_agreement - expected_agreement) / (1 - expected_agreement)


def _index_annotations(
    records: Sequence[AnnotationRecord],
    *,
    side: str,
) -> dict[str, AnnotationRecord]:
    indexed: dict[str, AnnotationRecord] = {}
    for record in records:
        if record.query_id in indexed:
            raise ValueError(f"duplicate {side} query_id: {record.query_id}")
        indexed[record.query_id] = record
    return indexed


def compare_annotations(
    left: Sequence[AnnotationRecord],
    right: Sequence[AnnotationRecord],
    *,
    fields: Sequence[str],
) -> AgreementReport:
    """Align two annotation sets by query ID and evaluate categorical fields."""
    left_by_id = _index_annotations(left, side="left")
    right_by_id = _index_annotations(right, side="right")
    if left_by_id.keys() != right_by_id.keys():
        raise ValueError("annotation query ID sets differ")
    if not fields:
        raise ValueError("at least one agreement field is required")

    allowed_fields = {"query_type", "domain"}
    report_fields: dict[str, FieldAgreement] = {}
    query_ids = sorted(left_by_id)
    for field_name in fields:
        if field_name not in allowed_fields:
            raise ValueError(f"unsupported agreement field: {field_name}")
        left_labels = [str(getattr(left_by_id[query_id], field_name)) for query_id in query_ids]
        right_labels = [
            str(getattr(right_by_id[query_id], field_name)) for query_id in query_ids
        ]
        kappa = cohen_kappa(left_labels, right_labels)
        threshold = 0.80
        report_fields[field_name] = FieldAgreement(
            kappa=kappa,
            threshold=threshold,
            accepted=kappa >= threshold,
        )

    return AgreementReport(
        compared_query_count=len(query_ids),
        fields=report_fields,
    )

if __name__ == "__main__":
    raise SystemExit(main())
