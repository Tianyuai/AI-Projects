from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from paper_search.evaluation import annotation as annotation_module
from paper_search.evaluation.annotation import (
    AnnotationRecord,
    TypeDomainAnnotationRecord,
    cohen_kappa,
    compare_annotations,
)


EXPECTED_DOMAIN_LABELS = (
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


def test_domain_vocabulary_artifact_is_frozen_and_bound_to_code() -> None:
    payload = json.loads(Path("data/domain_labels.v1.json").read_text(encoding="utf-8"))

    assert payload["version"] == "domain-labels-v1"
    assert tuple(payload["labels"]) == EXPECTED_DOMAIN_LABELS
    assert len(payload["labels"]) == len(set(payload["labels"]))
    assert set(payload["definitions"]) == set(payload["labels"])
    assert all(payload["definitions"][label].strip() for label in payload["labels"])
    assert annotation_module.DOMAIN_LABELS == EXPECTED_DOMAIN_LABELS


@pytest.mark.parametrize("domain", ["information-retrieval", "other"])
def test_domain_vocabulary_accepts_frozen_values(domain: str) -> None:
    record = TypeDomainAnnotationRecord.model_validate(
        {
            "query_id": "q1",
            "query_type": "method",
            "domain": domain,
            "annotator": "member-b",
        }
    )

    assert record.domain == domain


@pytest.mark.parametrize(
    "domain",
    ["search-systems", "Information-Retrieval", "search systems"],
)
def test_domain_vocabulary_rejects_values_outside_frozen_list(domain: str) -> None:
    with pytest.raises(ValidationError):
        TypeDomainAnnotationRecord.model_validate(
            {
                "query_id": "q1",
                "query_type": "method",
                "domain": domain,
                "annotator": "member-b",
            }
        )


def _annotation(
    query_id: str,
    *,
    query_type: str = "method",
    domain: str = "information-retrieval",
    annotator: str = "member-a",
) -> dict[str, object]:
    return {
        "query_id": query_id,
        "research_goal": "Find efficient scholarly retrieval methods",
        "must_have": ["scholarly retrieval"],
        "should_have": ["efficient inference"],
        "exclusions": [],
        "year_from": 2020,
        "year_to": None,
        "venues": [],
        "query_type": query_type,
        "domain": domain,
        "annotator": annotator,
    }


def test_annotation_schema_accepts_the_approved_fields_and_is_frozen() -> None:
    record = AnnotationRecord.model_validate(_annotation("q1"))

    assert record.query_id == "q1"
    assert record.query_type == "method"
    with pytest.raises(ValidationError):
        record.domain = "changed"


def test_annotation_schema_rejects_invalid_years_and_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="year_from"):
        AnnotationRecord.model_validate(
            {**_annotation("q1"), "year_from": 2030, "year_to": 2020}
        )
    with pytest.raises(ValidationError):
        AnnotationRecord.model_validate({**_annotation("q1"), "invented": True})


@pytest.mark.parametrize(
    ("field", "value"),
    [("query_type", "invented"), ("domain", "Information Retrieval")],
)
def test_annotation_schema_rejects_unapproved_labels(field: str, value: str) -> None:
    with pytest.raises(ValidationError):
        AnnotationRecord.model_validate({**_annotation("q1"), field: value})


def test_type_domain_annotation_accepts_only_the_minimal_frozen_contract() -> None:
    record = TypeDomainAnnotationRecord.model_validate(
        {
            "query_id": "q1",
            "query_type": "method",
            "domain": "information-retrieval",
            "annotator": "member-b",
        }
    )

    assert record.query_id == "q1"
    with pytest.raises(ValidationError):
        record.domain = "changed"
    with pytest.raises(ValidationError):
        TypeDomainAnnotationRecord.model_validate(
            {**record.model_dump(), "research_goal": "not allowed"}
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("query_id", "   "),
        ("annotator", "   "),
        ("query_type", "invented"),
        ("domain", "Information Retrieval"),
    ],
)
def test_type_domain_annotation_rejects_invalid_fields(
    field: str,
    value: str,
) -> None:
    payload = {
        "query_id": "q1",
        "query_type": "method",
        "domain": "information-retrieval",
        "annotator": "member-b",
    }

    with pytest.raises(ValidationError):
        TypeDomainAnnotationRecord.model_validate({**payload, field: value})

def test_cohen_kappa_is_one_for_perfect_agreement() -> None:
    assert cohen_kappa(["method", "topic"], ["method", "topic"]) == 1.0


def test_cohen_kappa_uses_both_raters_marginal_frequencies() -> None:
    assert cohen_kappa(
        ["method", "method", "topic", "topic"],
        ["method", "topic", "topic", "topic"],
    ) == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("first", "second", "reason"),
    [([], [], "non-empty"), (["method"], [], "same length")],
)
def test_cohen_kappa_rejects_invalid_inputs(
    first: list[str],
    second: list[str],
    reason: str,
) -> None:
    with pytest.raises(ValueError, match=reason):
        cohen_kappa(first, second)


def test_compare_annotations_aligns_by_query_id_and_flags_low_agreement() -> None:
    left = [
        AnnotationRecord.model_validate(_annotation("q1", domain="information-retrieval")),
        AnnotationRecord.model_validate(
            _annotation("q2", query_type="topic", domain="computer-vision")
        ),
    ]
    right = [
        AnnotationRecord.model_validate(
            _annotation(
                "q2",
                query_type="topic",
                domain="information-retrieval",
                annotator="member-b",
            )
        ),
        AnnotationRecord.model_validate(_annotation("q1", annotator="member-b")),
    ]

    report = compare_annotations(left, right, fields=("query_type", "domain"))

    assert report.compared_query_count == 2
    assert report.fields["query_type"].accepted is True
    assert report.fields["query_type"].kappa == 1.0
    assert report.fields["domain"].accepted is False
    assert report.fields["domain"].threshold == 0.80


def test_compare_annotations_rejects_missing_and_duplicate_query_ids() -> None:
    q1 = AnnotationRecord.model_validate(_annotation("q1"))
    q2 = AnnotationRecord.model_validate(_annotation("q2", annotator="member-b"))

    with pytest.raises(ValueError, match="query ID sets differ"):
        compare_annotations([q1], [q2], fields=("query_type",))
    with pytest.raises(ValueError, match="duplicate left query_id: q1"):
        compare_annotations([q1, q1], [q1], fields=("query_type",))
