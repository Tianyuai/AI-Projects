from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import pytest

from paper_search.evaluation.dataset import EvaluationQuery
from paper_search.recall_experiments.inputs import (
    FormalRunInputSource,
    GoldDocumentCatalogBuilder,
    GoldDocumentCatalogSource,
    SealedGoldDocumentCatalog,
)
from paper_search.recall_experiments.recipes import (
    ArtifactBinding,
    FormalRunInputBinding,
    HistoricalBaselineBinding,
    SampleBinding,
    load_sample_binding,
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


def _paper_source(tmp_path: Path, rows: list[dict[str, object]]) -> ArtifactBinding:
    return _write_jsonl(tmp_path / "papers.jsonl", rows)


def _load_sealed_catalog(
    tmp_path: Path, gold: list[EvaluationQuery], paper_rows: list[dict[str, object]]
) -> SealedGoldDocumentCatalog:
    source = _paper_source(tmp_path, paper_rows)
    builder = GoldDocumentCatalogBuilder(tmp_path)
    sealed = builder.build(gold, [source])
    catalog_binding = _write_jsonl(
        tmp_path / "catalog.jsonl",
        [record.model_dump(mode="json") for record in sealed.records],
    )
    manifest_path = tmp_path / "catalog.manifest.json"
    manifest_path.write_bytes(builder.manifest_bytes(catalog_binding, sealed))
    manifest_binding = ArtifactBinding(
        path=manifest_path.name, sha256=_hash(manifest_path.read_bytes())
    )
    return GoldDocumentCatalogSource(tmp_path).load(
        catalog_binding,
        manifest_binding=manifest_binding,
        bound_paper_sources=[source],
        gold_associations=gold,
    )


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


def test_catalog_builder_preserves_available_metadata_and_marks_missing_title_incomplete(
    tmp_path: Path,
) -> None:
    from paper_search.evaluation.dataset import EvaluationQuery

    gold = [
        EvaluationQuery(
            query_id="q-one",
            query="one",
            relevant_paper_ids=["arxiv:2401.00001", "arxiv:2401.00002"],
        )
    ]
    incomplete_source = _paper_source(
        tmp_path,
        [
            {
                "canonical_id": "arxiv:2401.00001",
                "title": "Available title",
                "sources": ["openalex"],
            }
        ],
    )
    sealed = GoldDocumentCatalogBuilder(tmp_path).build(gold, [incomplete_source])

    assert sealed.status == "incomplete"
    assert sealed.source_manifest[0].path == "papers.jsonl"
    assert sealed.source_manifest[0].sha256 == incomplete_source.sha256
    assert sealed.source_manifest_sha256.startswith("sha256:")
    with pytest.raises(ValueError, match="oracle_catalog_incomplete"):
        sealed.to_generation_documents("q-one")

    complete_source = _paper_source(
        tmp_path,
        [
            {
                "canonical_id": "arxiv:2401.00001",
                "title": "Available title",
                "abstract": "Preserved abstract",
                "authors": ["Ada Author"],
                "publication_year": 2024,
                "sources": ["openalex"],
            },
            {
                "canonical_id": "arxiv:2401.00002",
                "title": "Second title",
                "sources": ["semantic_scholar"],
            },
        ],
    )
    complete = GoldDocumentCatalogBuilder(tmp_path).build(gold, [complete_source])
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

    from paper_search.evaluation.dataset import EvaluationQuery

    gold = [
        EvaluationQuery(
            query_id="q-one", query="one", relevant_paper_ids=["arxiv:2401.00001"]
        )
    ]
    catalog = GoldDocumentCatalogSource(tmp_path).load(binding, gold_associations=gold)

    assert catalog.status == "invalid"
    with pytest.raises(ValueError, match="oracle_catalog_invalid"):
        catalog.to_generation_documents("q-one")


def test_catalog_source_requires_matching_sealed_manifest_and_unchanged_sources(
    tmp_path: Path,
) -> None:
    from paper_search.evaluation.dataset import EvaluationQuery

    gold = [EvaluationQuery(query_id="q-one", query="one", relevant_paper_ids=["arxiv:2401.00001"])]
    paper_source = _paper_source(
        tmp_path,
        [{"canonical_id": "arxiv:2401.00001", "title": "Bound title", "sources": ["openalex"]}],
    )
    sealed = GoldDocumentCatalogBuilder(tmp_path).build(gold, [paper_source])
    catalog_binding = _write_jsonl(
        tmp_path / "catalog.jsonl",
        [
            {
                "query_id": "q-one",
                "gold_paper_id": "arxiv:2401.00001",
                "title": "Bound title",
            }
        ],
    )
    manifest_binding = _write_jsonl(
        tmp_path / "catalog.manifest.jsonl",
        [
            {
                "schema_version": "sealed-gold-document-catalog-manifest-v1",
                "catalog": catalog_binding.model_dump(mode="json"),
                "bound_paper_sources": [paper_source.model_dump(mode="json")],
                "sealed_catalog_sha256": sealed.catalog_sha256,
            }
        ],
    )

    catalog = GoldDocumentCatalogSource(tmp_path).load(
        catalog_binding,
        manifest_binding=manifest_binding,
        bound_paper_sources=[paper_source],
        gold_associations=gold,
    )
    assert catalog.status == "complete"

    mismatched = _write_jsonl(
        tmp_path / "catalog.manifest.jsonl",
        [
            {
                "schema_version": "sealed-gold-document-catalog-manifest-v1",
                "catalog": {**catalog_binding.model_dump(mode="json"), "sha256": "sha256:" + "f" * 64},
                "bound_paper_sources": [paper_source.model_dump(mode="json")],
                "sealed_catalog_sha256": sealed.catalog_sha256,
            }
        ],
    )
    invalid = GoldDocumentCatalogSource(tmp_path).load(
        catalog_binding,
        manifest_binding=mismatched,
        bound_paper_sources=[paper_source],
        gold_associations=gold,
    )
    assert invalid.status == "invalid"

    manifest_binding = _write_jsonl(
        tmp_path / "catalog.manifest.jsonl",
        [
            {
                "schema_version": "sealed-gold-document-catalog-manifest-v1",
                "catalog": catalog_binding.model_dump(mode="json"),
                "bound_paper_sources": [paper_source.model_dump(mode="json")],
                "sealed_catalog_sha256": sealed.catalog_sha256,
            }
        ],
    )

    (tmp_path / paper_source.path).write_text(
        '{"canonical_id":"arxiv:2401.00001","title":"Changed","sources":["openalex"]}\n',
        encoding="utf-8",
    )
    changed_source = GoldDocumentCatalogSource(tmp_path).load(
        catalog_binding,
        manifest_binding=manifest_binding,
        bound_paper_sources=[paper_source],
        gold_associations=gold,
    )
    assert changed_source.status == "invalid"


def test_builder_parses_only_verified_content_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from paper_search.evaluation.dataset import EvaluationQuery
    import paper_search.recall_experiments.inputs.gold_catalog as catalog_module

    gold = [EvaluationQuery(query_id="q-one", query="one", relevant_paper_ids=["arxiv:2401.00001"])]
    source = _paper_source(
        tmp_path,
        [{"canonical_id": "arxiv:2401.00001", "title": "Verified", "sources": ["openalex"]}],
    )
    monkeypatch.setattr(
        catalog_module, "read_jsonl", lambda *_args: pytest.fail("reopened source"), raising=False
    )

    catalog = GoldDocumentCatalogBuilder(tmp_path).build(gold, [source])

    assert catalog.status == "complete"


def test_catalog_builder_requires_hash_bound_paper_artifacts(tmp_path: Path) -> None:
    from paper_search.evaluation.dataset import EvaluationQuery

    gold = [EvaluationQuery(query_id="q-one", query="one", relevant_paper_ids=["arxiv:2401.00001"])]
    with pytest.raises(TypeError, match="ArtifactBinding"):
        GoldDocumentCatalogBuilder(tmp_path).build(gold, [{"not": "a bound artifact"}])

    changed = _paper_source(
        tmp_path,
        [{"canonical_id": "arxiv:2401.00001", "title": "Bound", "sources": ["openalex"]}],
    ).model_copy(update={"sha256": "sha256:" + "f" * 64})
    with pytest.raises(ValueError, match="hash mismatch"):
        GoldDocumentCatalogBuilder(tmp_path).build(gold, [changed])

    outside = tmp_path.parent / "outside-papers.jsonl"
    outside.write_text(
        '{"canonical_id":"arxiv:2401.00001","title":"Outside","sources":["openalex"]}\n',
        encoding="utf-8",
    )
    escaped = ArtifactBinding.model_construct(path="../outside-papers.jsonl", sha256=_hash(outside.read_bytes()))
    with pytest.raises(ValueError, match="escapes workspace root"):
        GoldDocumentCatalogBuilder(tmp_path).build(gold, [escaped])


def test_catalog_builder_reads_normalized_papers_from_bound_outcome_artifacts(tmp_path: Path) -> None:
    from paper_search.evaluation.dataset import EvaluationQuery

    gold = [EvaluationQuery(query_id="q-one", query="one", relevant_paper_ids=["arxiv:2401.00001"])]
    source = _write_jsonl(
        tmp_path / "outcomes.jsonl",
        [
            {
                "query_id": "q-one",
                "searches": [
                    {
                        "data": [
                            {
                                "canonical_id": "arxiv:2401.00001",
                                "title": "Nested frozen Paper",
                                "sources": ["openalex"],
                            }
                        ]
                    }
                ],
            }
        ],
    )

    catalog = GoldDocumentCatalogBuilder(tmp_path).build(gold, [source])

    assert catalog.status == "complete"
    assert catalog.to_generation_documents("q-one")[0].title == "Nested frozen Paper"


def test_catalog_source_rejects_missing_extra_and_duplicate_selected_associations(tmp_path: Path) -> None:
    from paper_search.evaluation.dataset import EvaluationQuery

    gold = [
        EvaluationQuery(
            query_id="q-one",
            query="one",
            relevant_paper_ids=["arxiv:2401.00001", "arxiv:2401.00002"],
        )
    ]
    rows = [
        {"query_id": "q-one", "gold_paper_id": "arxiv:2401.00001", "title": "One"},
        {"query_id": "q-one", "gold_paper_id": "arxiv:2401.00001", "title": "Duplicate"},
        {"query_id": "q-other", "gold_paper_id": "arxiv:2401.00003", "title": "Extra"},
    ]
    binding = _write_jsonl(tmp_path / "catalog.jsonl", rows)

    catalog = GoldDocumentCatalogSource(tmp_path).load(binding, gold_associations=gold)

    assert catalog.status == "invalid"
    with pytest.raises(ValueError, match="oracle_catalog_invalid"):
        catalog.to_generation_documents("q-one")


def test_catalog_generation_rejects_identifier_leakage_but_allows_provider_names(tmp_path: Path) -> None:
    from paper_search.evaluation.dataset import EvaluationQuery

    gold = [EvaluationQuery(query_id="q-one", query="one", relevant_paper_ids=["arxiv:2401.00001"])]
    catalog = _load_sealed_catalog(
        tmp_path,
        gold,
        [{"canonical_id": "arxiv:2401.00001", "title": "Paper https://example.test/private", "sources": ["openalex"]}],
    )
    with pytest.raises(ValueError, match="forbidden"):
        catalog.to_generation_documents("q-one")

    safe = _load_sealed_catalog(
        tmp_path,
        gold,
        [{"canonical_id": "arxiv:2401.00001", "title": "OpenAlex and Semantic Scholar in ordinary prose", "sources": ["openalex"]}],
    )
    documents = safe.to_generation_documents("q-one")
    assert documents[0].title.startswith("OpenAlex")


def test_catalog_generation_rejects_identifier_leakage_in_every_visible_text_field(
    tmp_path: Path,
) -> None:
    from paper_search.evaluation.dataset import EvaluationQuery

    gold = [EvaluationQuery(query_id="q-one", query="one", relevant_paper_ids=["arxiv:2401.00001"])]
    for field, value in (
        ("title", "Leaking doi:10.1000/example"),
        ("abstract", "Read https://example.test/private"),
        ("authors", ["S2:private-record"]),
    ):
        row: dict[str, object] = {"canonical_id": "arxiv:2401.00001", "title": "Safe title", "sources": ["openalex"]}
        row[field] = value
        catalog = _load_sealed_catalog(tmp_path, gold, [row])
        with pytest.raises(ValueError, match="forbidden"):
            catalog.to_generation_documents("q-one")


def test_bound_smoke_catalog_is_association_complete_but_oracle_incomplete() -> None:
    workspace = Path(__file__).parents[2]
    sample = load_sample_binding(
        workspace / "configs" / "recall_experiments" / "samples" / "dev-smoke-3.yaml"
    ).binding
    dataset = FormalRunInputSource(workspace).load_queries(sample)
    assert sample.gold_document_catalog is not None
    assert sample.gold_document_catalog_manifest is not None

    catalog = GoldDocumentCatalogSource(workspace).load(
        sample.gold_document_catalog,
        manifest_binding=sample.gold_document_catalog_manifest,
        bound_paper_sources=sample.frozen_inputs.bound_paper_sources,
        gold_associations=dataset.evaluation_materials.gold_records,
    )

    assert catalog.status == "incomplete"
    assert all(record.title is None for record in catalog.records)
