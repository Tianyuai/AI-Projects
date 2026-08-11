from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import pytest

from paper_search.domain.models import Paper
from paper_search.recall_experiments.inputs import (
    BoundPaperSource,
    FormalRunInputSource,
    GoldDocumentCatalogBuilder,
    GoldDocumentCatalogSource,
)
from paper_search.recall_experiments.recipes import (
    ArtifactBinding,
    FormalRunInputBinding,
    HistoricalBaselineBinding,
    SampleBinding,
)


def _hash(content: bytes) -> str:
    return f"sha256:{sha256(content).hexdigest()}"


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> ArtifactBinding:
    content = b"".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
        for row in rows
    )
    path.write_bytes(content)
    return ArtifactBinding(path=path.name, sha256=_hash(content))


def _fixture_binding(tmp_path: Path, *, query_ids: list[str]) -> SampleBinding:
    gold = _write_jsonl(
        tmp_path / "gold.jsonl",
        [
            {"query_id": "q-first", "query": "first", "relevant_paper_ids": ["arxiv:2401.00001"]},
            {"query_id": "q-second", "query": "second", "relevant_paper_ids": ["arxiv:2401.00002"]},
            {"query_id": "q-third", "query": "third", "relevant_paper_ids": ["arxiv:2401.00003", "arxiv:2401.00004"]},
        ],
    )
    identifier_map_content = b'{"arxiv:2401.00001":"doi:10.1000/one"}\n'
    identifier_map = ArtifactBinding(
        path="identifier-map.json", sha256=_hash(identifier_map_content)
    )
    (tmp_path / identifier_map.path).write_bytes(identifier_map_content)
    business = _write_jsonl(
        tmp_path / "business-results.jsonl",
        [
            {
                "query_id": query_id,
                "query_analysis": None,
                "selected_paper_ids": [],
                "high_relevance": [],
                "partial_relevance": [],
                "citation_edges": [],
                "is_partial": False,
                "planner_status": None,
                "planner_fallback": False,
                "warnings": [],
                "stop_reason": "completed",
                "hard_failure_code": None,
            }
            for query_id in ("q-first", "q-second", "q-third")
        ],
    )
    executions = _write_jsonl(
        tmp_path / "executions.jsonl",
        [
            {
                "query_id": query_id,
                "run_id": "history",
                "outcome_kind": "success",
                "business_result_sha256": "sha256:" + "0" * 64,
                "usage": {},
                "diagnostics": [],
                "retrieved_paper_ids": [],
                "post_filter_paper_ids": [],
                "is_partial": False,
                "planner_status": None,
                "planner_fallback": False,
                "stop_reason": "completed",
            }
            for query_id in ("q-first", "q-second", "q-third")
        ],
    )
    catalog_content = b'{"query_id":"q-first","gold_paper_id":"arxiv:2401.00001","title":"One"}\n'
    catalog = ArtifactBinding(path="gold-catalog.jsonl", sha256=_hash(catalog_content))
    (tmp_path / catalog.path).write_bytes(catalog_content)
    return SampleBinding(
        sample_id="three-query-canary",
        query_ids=query_ids,
        gold_document_catalog=catalog,
        frozen_inputs=FormalRunInputBinding(
            gold_associations=gold,
            identifier_map=identifier_map,
            seed_candidates=[
                {"paper": {"canonical_id": "seed-two", "title": "Second seed", "sources": ["openalex"]}},
                {"paper": {"canonical_id": "seed-one", "title": "First seed", "sources": ["semantic_scholar"]}},
            ],
            historical_baseline=HistoricalBaselineBinding(
                query_ids=["q-first", "q-second", "q-third"],
                gold_associations=gold,
                business_results=business,
                executions=executions,
            ),
        ),
    )


def test_formal_input_source_selects_explicit_ids_in_frozen_source_order(tmp_path: Path) -> None:
    binding = _fixture_binding(tmp_path, query_ids=["q-third", "q-first", "q-second"])

    dataset = FormalRunInputSource(tmp_path).load_queries(binding)

    assert [query.query_id for query in dataset.queries] == ["q-first", "q-second", "q-third"]
    assert set(dataset.source_hashes) == {"gold_associations", "identifier_map"}
    assert dataset.evaluation_materials.identifier_map_bytes == (
        tmp_path / "identifier-map.json"
    ).read_bytes()
    assert dataset.evaluation_materials.identifier_map_sha256 == binding.frozen_inputs.identifier_map.sha256
    assert [seed.paper.canonical_id for seed in dataset.seed_candidates] == ["seed-two", "seed-one"]


def test_formal_input_source_fails_closed_for_hash_drift_and_missing_configured_id(
    tmp_path: Path,
) -> None:
    binding = _fixture_binding(tmp_path, query_ids=["q-first", "missing"])

    with pytest.raises(ValueError, match="configured query IDs"):
        FormalRunInputSource(tmp_path).load_queries(binding)

    changed = binding.model_copy(
        update={
            "frozen_inputs": binding.frozen_inputs.model_copy(
                update={
                    "identifier_map": ArtifactBinding(
                        path="identifier-map.json", sha256="sha256:" + "f" * 64
                    )
                }
            )
        }
    )
    with pytest.raises(ValueError, match="identifier_map hash mismatch"):
        FormalRunInputSource(tmp_path).load_queries(changed)


def test_identifier_map_stays_opaque_until_evaluator_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binding = _fixture_binding(tmp_path, query_ids=["q-first"])
    import paper_search.evaluation.dataset as evaluation_dataset

    monkeypatch.setattr(
        evaluation_dataset.IdentifierMap,
        "from_bytes",
        classmethod(lambda *_args, **_kwargs: pytest.fail("input adapter parsed identifier map")),
    )

    dataset = FormalRunInputSource(tmp_path).load_queries(binding)

    assert dataset.evaluation_materials.gold_records[0].relevant_paper_ids == ["arxiv:2401.00001"]
    assert "identifier_map" not in dataset.queries[0].model_dump(mode="json")


def test_historical_baseline_requires_exact_query_ids_and_gold_denominator(tmp_path: Path) -> None:
    binding = _fixture_binding(tmp_path, query_ids=["q-first", "q-second", "q-third"])
    baseline_binding = binding.frozen_inputs.historical_baseline
    assert baseline_binding is not None

    baseline = FormalRunInputSource(tmp_path).load_historical_baseline(baseline_binding)

    assert baseline is not None
    assert baseline.query_ids == ["q-first", "q-second", "q-third"]
    assert baseline.gold_association_count == 4

    mismatched = baseline_binding.model_copy(update={"query_ids": ["q-first"]})
    with pytest.raises(ValueError, match="query IDs"):
        FormalRunInputSource(tmp_path).load_historical_baseline(mismatched)


def test_catalog_builder_preserves_available_metadata_and_marks_missing_title_incomplete() -> None:
    from paper_search.evaluation.dataset import EvaluationQuery

    gold = [
        EvaluationQuery(
            query_id="q-one",
            query="one",
            relevant_paper_ids=["arxiv:2401.00001", "arxiv:2401.00002"],
        )
    ]
    sealed = GoldDocumentCatalogBuilder().build(
        gold,
        [
            BoundPaperSource(
                source_id="frozen-snapshot",
                sha256="sha256:" + "1" * 64,
                papers=[
                    Paper(
                        canonical_id="arxiv:2401.00001",
                        title="Available title",
                        abstract=None,
                        authors=[],
                        publication_year=None,
                        sources=["openalex"],
                    )
                ],
            )
        ],
    )

    assert sealed.status == "incomplete"
    with pytest.raises(ValueError, match="oracle_catalog_incomplete"):
        sealed.to_generation_documents("q-one")

    complete = GoldDocumentCatalogBuilder().build(gold[:1], [
        BoundPaperSource(
            source_id="frozen-snapshot",
            sha256="sha256:" + "1" * 64,
            papers=[
                Paper(
                    canonical_id="arxiv:2401.00001",
                    title="Available title",
                    abstract="Preserved abstract",
                    authors=["Ada Author"],
                    publication_year=2024,
                    sources=["openalex"],
                ),
                Paper(
                    canonical_id="arxiv:2401.00002",
                    title="Second title",
                    sources=["semantic_scholar"],
                ),
            ],
        )
    ])
    documents = complete.to_generation_documents("q-one")

    assert documents[0].model_dump(mode="json") == {
        "title": "Available title",
        "abstract": "Preserved abstract",
        "authors": ["Ada Author"],
        "publication_year": 2024,
        "metadata_coverage": {"abstract": True, "authors": True, "year": True},
    }
    assert documents[1].metadata_coverage == {"abstract": False, "authors": False, "year": False}


def test_catalog_source_hash_binds_private_catalog_rows(tmp_path: Path) -> None:
    content = (
        b'{"query_id":"q-one","gold_paper_id":"arxiv:2401.00001",'
        b'"title":"Only title","abstract":null,"authors":[],"publication_year":null}\n'
    )
    (tmp_path / "catalog.jsonl").write_bytes(content)
    binding = ArtifactBinding(path="catalog.jsonl", sha256=_hash(content))

    catalog = GoldDocumentCatalogSource(tmp_path).load(binding)

    assert catalog.status == "complete"
    assert catalog.to_generation_documents("q-one")[0].title == "Only title"
