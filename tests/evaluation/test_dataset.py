import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

import paper_search.evaluation.dataset as dataset_module
from paper_search.evaluation.dataset import (
    EvaluationQuery,
    PredictionRecord,
    normalize_paper_id,
    normalize_title,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("https://doi.org/10.1000/ABC", "doi:10.1000/abc"),
        ("doi:10.5555/Example", "doi:10.5555/example"),
        ("10.1234/Bare", "doi:10.1234/bare"),
    ],
)
def test_normalize_doi(raw: str, expected: str) -> None:
    assert normalize_paper_id(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("arXiv:2501.10120v3", "arxiv:2501.10120"),
        ("https://arxiv.org/pdf/1706.03762v5.pdf", "arxiv:1706.03762"),
        ("https://arxiv.org/abs/hep-th/9901001v2", "arxiv:hep-th/9901001"),
    ],
)
def test_normalize_arxiv_id(raw: str, expected: str) -> None:
    assert normalize_paper_id(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("https://openalex.org/w123", "openalex:W123"),
        ("openalex:W456", "openalex:W456"),
        ("https://www.semanticscholar.org/paper/example/ABC123", "s2:ABC123"),
        ("s2:deadbeef", "s2:deadbeef"),
    ],
)
def test_normalize_provider_id(raw: str, expected: str) -> None:
    assert normalize_paper_id(raw) == expected


def test_normalize_title_uses_nfkc_casefold_punctuation_and_whitespace() -> None:
    assert normalize_title("  Ａ Study:  On RAG!  ") == "a study on rag"
    assert normalize_paper_id("  Ａ Study:  On RAG!  ", kind="title") == (
        "title:a study on rag"
    )
    assert normalize_paper_id("title:Graph-Based Retrieval") == "title:graph based retrieval"


@pytest.mark.parametrize(
    "raw",
    ["", "ordinary untyped title", "openalex:123", "doi:not-a-doi", "title:!!!"],
)
def test_normalize_paper_id_rejects_empty_ambiguous_or_invalid_values(raw: str) -> None:
    with pytest.raises(ValueError):
        normalize_paper_id(raw)


def test_evaluation_query_normalizes_ids_and_is_frozen() -> None:
    query = EvaluationQuery(
        query_id=" q1 ",
        query=" RAG evaluation ",
        relevant_paper_ids=["arXiv:2501.10120v2"],
        metadata={"split": "dev"},
    )

    assert query.query_id == "q1"
    assert query.query == "RAG evaluation"
    assert query.relevant_paper_ids == ["arxiv:2501.10120"]
    with pytest.raises(ValidationError):
        query.query = "changed"


def test_evaluation_query_rejects_duplicates_after_normalization() -> None:
    with pytest.raises(ValidationError, match="duplicate"):
        EvaluationQuery(
            query_id="q1",
            query="RAG",
            relevant_paper_ids=["arXiv:2501.10120", "2501.10120v2"],
        )


def test_evaluation_query_rejects_unknown_fields_and_non_json_metadata() -> None:
    with pytest.raises(ValidationError):
        EvaluationQuery(query_id="q1", query="RAG", unexpected=True)
    with pytest.raises(ValidationError):
        EvaluationQuery(query_id="q1", query="RAG", metadata={"bad": object()})


def test_prediction_record_normalizes_but_preserves_ranked_duplicates() -> None:
    record = PredictionRecord(
        query_id="q1",
        predicted_paper_ids=["doi:10.1000/A", "https://doi.org/10.1000/a"],
    )

    assert record.predicted_paper_ids == ["doi:10.1000/a", "doi:10.1000/a"]


def test_read_jsonl_loads_valid_models(tmp_path: Path) -> None:
    path = tmp_path / "gold.jsonl"
    path.write_text(
        '{"query_id":" q1 ","query":" RAG ","relevant_paper_ids":[],"metadata":{}}\n',
        encoding="utf-8",
    )

    records = dataset_module.read_jsonl(path, EvaluationQuery)

    assert records == [EvaluationQuery(query_id="q1", query="RAG")]


def test_read_jsonl_reports_line_and_rejects_unknown_fields(tmp_path: Path) -> None:
    path = tmp_path / "gold.jsonl"
    path.write_text('{"query_id":"q1","query":"x","extra":1}\n', encoding="utf-8")

    with pytest.raises(ValueError, match=r"gold\.jsonl:1"):
        dataset_module.read_jsonl(path, EvaluationQuery)


@pytest.mark.parametrize(
    ("content", "reason"),
    [
        ("\n", "blank line"),
        ("[]\n", "JSON object"),
        ("not-json\n", "invalid JSON"),
    ],
)
def test_read_jsonl_rejects_malformed_lines(
    tmp_path: Path,
    content: str,
    reason: str,
) -> None:
    path = tmp_path / "gold.jsonl"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match=rf"gold\.jsonl:1.*{reason}"):
        dataset_module.read_jsonl(path, EvaluationQuery)


def test_read_jsonl_rejects_duplicate_query_ids(tmp_path: Path) -> None:
    path = tmp_path / "gold.jsonl"
    line = '{"query_id":"q1","query":"x","relevant_paper_ids":[],"metadata":{}}\n'
    path.write_text(line + line, encoding="utf-8")

    with pytest.raises(ValueError, match=r"gold\.jsonl:2.*duplicate query_id: q1"):
        dataset_module.read_jsonl(path, EvaluationQuery)


def test_read_jsonl_reports_invalid_utf8_at_physical_line(tmp_path: Path) -> None:
    path = tmp_path / "gold.jsonl"
    path.write_bytes(
        b'{"query_id":"q1","query":"x"}\n'
        b'{"query_id":"q2","query":"\xff"}\n'
    )

    with pytest.raises(ValueError, match=r"gold\.jsonl:2.*invalid UTF-8"):
        dataset_module.read_jsonl(path, EvaluationQuery)


@pytest.mark.parametrize(
    ("payload", "secret"),
    [
        (
            '{"query_id":"q1","query":"x","relevant_paper_ids":'
            '["API_SECRET_SENTINEL"],"metadata":{}}\n',
            "API_SECRET_SENTINEL",
        ),
        (
            '{"query_id":"q1","query":"x","LEAKED_FIELD_SENTINEL":1}\n',
            "LEAKED_FIELD_SENTINEL",
        ),
    ],
)
def test_read_jsonl_redacts_values_and_unknown_field_names(
    tmp_path: Path,
    payload: str,
    secret: str,
) -> None:
    path = tmp_path / "gold.jsonl"
    path.write_text(payload, encoding="utf-8")

    with pytest.raises(ValueError, match=r"gold\.jsonl:1.*validation failed") as error:
        dataset_module.read_jsonl(path, EvaluationQuery)

    assert secret not in str(error.value)


def test_write_jsonl_atomic_is_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "out.jsonl"
    records = [EvaluationQuery(query_id="q1", query="中文 RAG")]
    expected = (
        '{"metadata":{},"query":"中文 RAG","query_id":"q1",'
        '"relevant_paper_ids":[]}\n'
    ).encode()

    dataset_module.write_jsonl_atomic(path, records)
    first = path.read_bytes()
    dataset_module.write_jsonl_atomic(path, records)

    assert first == expected
    assert path.read_bytes() == first
    assert list(path.parent.glob("*.tmp")) == []


@pytest.mark.parametrize("failure_point", ["serialization", "fsync", "replace"])
def test_write_jsonl_atomic_preserves_destination_and_cleans_temporary_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
) -> None:
    path = tmp_path / "out.jsonl"
    path.write_bytes(b"original\n")
    records = [EvaluationQuery(query_id="q1", query="x")]

    def fail(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise OSError(failure_point)

    if failure_point == "serialization":
        monkeypatch.setattr(dataset_module.json, "dumps", fail)
    elif failure_point == "fsync":
        monkeypatch.setattr(dataset_module.os, "fsync", fail)
    else:
        monkeypatch.setattr(dataset_module.Path, "replace", fail)

    with pytest.raises(OSError, match=failure_point):
        dataset_module.write_jsonl_atomic(path, records)

    assert path.read_bytes() == b"original\n"
    assert list(tmp_path.glob(f".{path.name}.*.tmp")) == []


def test_sha256_file_matches_exact_file_bytes(tmp_path: Path) -> None:
    path = tmp_path / "input.bin"
    content = "科研评测\n".encode()
    path.write_bytes(content)

    assert dataset_module.sha256_file(path) == (
        f"sha256:{hashlib.sha256(content).hexdigest()}"
    )


def test_identifier_map_resolves_normalized_chains(tmp_path: Path) -> None:
    path = tmp_path / "map.json"
    path.write_text(
        '{"https://doi.org/10.1000/A":"arxiv:2501.10120",'
        '"arxiv:2501.10120":"openalex:W1"}',
        encoding="utf-8",
    )

    mapping = dataset_module.IdentifierMap.from_path(path)

    assert mapping.resolve("doi:10.1000/a") == "openalex:W1"
    assert mapping.resolve("arXiv:2501.10120v2") == "openalex:W1"
    assert mapping.resolve("doi:10.2000/unmapped") == "doi:10.2000/unmapped"


def test_identifier_map_resolves_chain_beyond_python_recursion_limit(
    tmp_path: Path,
) -> None:
    path = tmp_path / "long-map.json"
    mapping = {
        f"openalex:W{index}": f"openalex:W{index + 1}"
        for index in range(1, 1501)
    }
    path.write_text(json.dumps(mapping), encoding="utf-8")

    identifier_map = dataset_module.IdentifierMap.from_path(path)

    assert identifier_map.resolve("openalex:W1") == "openalex:W1501"


@pytest.mark.parametrize(
    "payload",
    [
        (
            '{"doi:10.1000/A":"openalex:W1",'
            '"https://doi.org/10.1000/a":"openalex:W2"}'
        ),
        '{"doi:10.1000/a":"openalex:W1","doi:10.1000/a":"openalex:W2"}',
    ],
)
def test_identifier_map_rejects_normalized_and_duplicate_key_conflicts(
    tmp_path: Path,
    payload: str,
) -> None:
    path = tmp_path / "conflict.json"
    path.write_text(payload, encoding="utf-8")

    with pytest.raises(ValueError, match="conflict"):
        dataset_module.IdentifierMap.from_path(path)


@pytest.mark.parametrize(
    "payload",
    [
        '{"doi:10.1000/a":"openalex:W1","openalex:W1":"doi:10.1000/a"}',
        '{"doi:10.1000/a":"doi:10.1000/a"}',
    ],
)
def test_identifier_map_rejects_cycles(tmp_path: Path, payload: str) -> None:
    path = tmp_path / "cycle.json"
    path.write_text(payload, encoding="utf-8")

    with pytest.raises(ValueError, match="cycle"):
        dataset_module.IdentifierMap.from_path(path)


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        ("[]", "JSON object"),
        ('{"openalex:W1":1}', "strings"),
        ("not-json", "invalid JSON"),
        ('{"unknown:1":"openalex:W1"}', "unsupported"),
    ],
)
def test_identifier_map_rejects_invalid_files(
    tmp_path: Path,
    payload: str,
    reason: str,
) -> None:
    path = tmp_path / "invalid-map.json"
    path.write_text(payload, encoding="utf-8")

    with pytest.raises(ValueError, match=reason):
        dataset_module.IdentifierMap.from_path(path)
