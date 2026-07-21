from __future__ import annotations

import json
import subprocess
import sys
from hashlib import sha256
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

def _write_jsonl(path: Path, records: list[dict[str, object]]) -> bytes:
    payload = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records
    ).encode("utf-8")
    path.write_bytes(payload)
    return payload


def _write_ids(path: Path, query_ids: list[str]) -> None:
    path.write_text(json.dumps(query_ids), encoding="utf-8")


def _type_domain(
    query_id: str,
    *,
    domain: str = "information-retrieval",
) -> dict[str, object]:
    return {
        "query_id": query_id,
        "query_type": "method",
        "domain": domain,
        "annotator": "member-b",
    }


def test_validate_type_domain_file_returns_only_safe_summary(tmp_path: Path) -> None:
    labels = tmp_path / "private.jsonl"
    ids = tmp_path / "safe.ids.json"
    payload = _write_jsonl(labels, [_type_domain("q1"), _type_domain("q2")])
    _write_ids(ids, ["q1", "q2"])

    summary = annotation_module.validate_annotation_file(
        labels,
        ids,
        kind="type-domain",
    )

    assert summary.model_dump() == {
        "status": "valid",
        "kind": "type-domain",
        "count": 2,
        "sha256": f"sha256:{sha256(payload).hexdigest()}",
        "ids_match": True,
    }
    assert "private.jsonl" not in summary.model_dump_json()
    assert "q1" not in summary.model_dump_json()


def test_validate_constraint_file_uses_full_annotation_schema(tmp_path: Path) -> None:
    labels = tmp_path / "constraints.jsonl"
    ids = tmp_path / "safe.ids.json"
    _write_jsonl(labels, [_annotation("q1")])
    _write_ids(ids, ["q1"])

    summary = annotation_module.validate_annotation_file(labels, ids, kind="constraints")

    assert summary.status == "valid"
    assert summary.count == 1


@pytest.mark.parametrize(
    "case",
    [
        "invalid-utf8",
        "blank-line",
        "unknown-field",
        "duplicate-id",
        "missing-id",
        "extra-id",
        "wrong-kind",
        "invalid-domain",
    ],
)
def test_validate_annotation_file_collapses_private_failures(
    tmp_path: Path,
    case: str,
) -> None:
    labels = tmp_path / "private.jsonl"
    ids = tmp_path / "safe.ids.json"
    records = [_type_domain("q1")]
    expected_ids = ["q1"]
    if case == "invalid-utf8":
        labels.write_bytes(b"\xff")
    elif case == "blank-line":
        labels.write_bytes(
            (json.dumps(_type_domain("q1")) + "\n\n").encode("utf-8")
        )
    else:
        if case == "unknown-field":
            records[0]["private-answer"] = "must-not-leak"
        elif case == "duplicate-id":
            records.append(_type_domain("q1"))
        elif case == "missing-id":
            expected_ids.append("q2")
        elif case == "extra-id":
            records.append(_type_domain("q2"))
        elif case == "wrong-kind":
            pass
        elif case == "invalid-domain":
            records[0]["domain"] = "search-systems"
        _write_jsonl(labels, records)
    _write_ids(ids, expected_ids)
    kind = "constraints" if case == "wrong-kind" else "type-domain"

    with pytest.raises(ValueError) as exc_info:
        annotation_module.validate_annotation_file(labels, ids, kind=kind)

    assert str(exc_info.value) == "private annotations are invalid"
    assert exc_info.value.__cause__ is None


def test_annotation_cli_success_outputs_safe_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    labels = tmp_path / "private.jsonl"
    ids = tmp_path / "safe.ids.json"
    _write_jsonl(labels, [_type_domain("q1")])
    _write_ids(ids, ["q1"])

    exit_code = annotation_module.main(
        ["--kind", "type-domain", "--labels", str(labels), "--ids", str(ids)]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert json.loads(captured.out)["status"] == "valid"
    assert captured.err == ""
    assert "q1" not in captured.out
    assert str(labels) not in captured.out


def test_annotation_cli_failure_leaks_no_private_values(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    private_dir = tmp_path / "private-filename-sentinel"
    private_dir.mkdir()
    labels = private_dir / "labels.jsonl"
    ids = tmp_path / "safe.ids.json"
    _write_jsonl(
        labels,
        [
            {
                "query_id": "query-id-sentinel",
                "query_type": "method",
                "domain": "information-retrieval",
                "annotator": "annotator-sentinel",
                "private-answer": "answer-sentinel",
            }
        ],
    )
    _write_ids(ids, ["query-id-sentinel"])

    exit_code = annotation_module.main(
        ["--kind", "type-domain", "--labels", str(labels), "--ids", str(ids)]
    )
    captured = capsys.readouterr()
    combined = captured.out + captured.err

    assert exit_code == 1
    assert captured.out == ""
    assert captured.err == "annotation validation failed\n"
    for sentinel in (
        "private-filename-sentinel",
        "query-id-sentinel",
        "annotator-sentinel",
        "answer-sentinel",
    ):
        assert sentinel not in combined

def test_annotation_module_cli_executes_without_preimport_warning() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "paper_search.evaluation.annotation", "--help"],
        check=False,
        capture_output=True,
        cwd=Path("src"),
        text=True,
    )

    assert result.returncode == 0
    assert "usage:" in result.stdout
    assert result.stderr == ""
