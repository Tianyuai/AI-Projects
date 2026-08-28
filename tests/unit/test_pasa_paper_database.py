from __future__ import annotations

import asyncio
import json
import zipfile
from pathlib import Path

from paper_search.domain.models import Paper
from paper_search.retrieval.pasa_paper_database import (
    PASA_TRAINING_CONSTRAINT_INJECTED_SOURCE,
    PASA_TRAINING_GOLD_INJECTED_SOURCE,
    PasaPaperDatabase,
    PasaPaperDatabaseSearchBackend,
    _arxiv_submission_year,
    build_pasa_paper_index,
)
from paper_search.evaluation.dataset import normalize_title
from paper_search.retrieval import pasa_paper_database


def _source_files(tmp_path: Path) -> tuple[Path, Path]:
    archive = tmp_path / "cs_paper_2nd.zip"
    id_map = tmp_path / "id2paper.json"
    id_map.write_text(
        json.dumps(
            {
                "2301.01234": "Robust Vision Transformers",
                "2302.05678": "Graph Retrieval Systems",
            }
        ),
        encoding="utf-8",
    )
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr(
            "robustvisiontransformers",
            json.dumps(
                {
                    "title": "Robust Vision Transformers",
                    "abstract": "A transformer robust to image corruption.",
                    "sections": [],
                }
            ),
        )
        output.writestr(
            "graphretrievalsystems",
            json.dumps(
                {
                    "title": "Graph Retrieval Systems",
                    "abstract": "Graph diffusion for scholarly search.",
                    "sections": [],
                }
            ),
        )
    return archive, id_map


def test_legacy_arxiv_category_containing_v_keeps_submission_year() -> None:
    assert _arxiv_submission_year("solv-int/9701001") == 1997


def test_build_and_query_pasa_paper_database(tmp_path: Path) -> None:
    archive, id_map = _source_files(tmp_path)
    index = tmp_path / "pasa-paper-database.sqlite3"

    manifest = build_pasa_paper_index(
        archive_path=archive,
        id_map_path=id_map,
        index_path=index,
    )
    database = PasaPaperDatabase(index)

    exact = database.lookup_arxiv("https://arxiv.org/abs/2301.01234v3")
    assert exact is not None
    assert exact.canonical_id == "arxiv:2301.01234"
    assert exact.arxiv_id == "2301.01234"
    assert exact.publication_year == 2023
    assert exact.sources == ["pasa_paper_database"]
    assert [paper.canonical_id for paper in database.search("graph diffusion", 5)] == [
        "arxiv:2302.05678"
    ]
    assert [
        paper.canonical_id
        for paper in database.search_required_phrase_with_term_pairs(
            "robust", ["vision", "transformer", "unrelated"], 5
        )
    ] == ["arxiv:2301.01234"]
    assert manifest["indexed_paper_count"] == 2
    assert manifest["missing_archive_member_count"] == 0
    assert manifest["test_partition_touched"] is False
    assert manifest["archive_sha256"].startswith("sha256:")
    assert manifest["id_map_sha256"].startswith("sha256:")
    assert manifest["index_sha256"].startswith("sha256:")


def test_lookup_normalized_titles_returns_public_metadata_without_gold_input(
    tmp_path: Path,
) -> None:
    archive, id_map = _source_files(tmp_path)
    index = tmp_path / "pasa-paper-database.sqlite3"
    build_pasa_paper_index(
        archive_path=archive,
        id_map_path=id_map,
        index_path=index,
    )

    matches = PasaPaperDatabase(index).lookup_normalized_titles(
        ["Robust Vision-Transformers!", "Unseen title"]
    )

    key = normalize_title("Robust Vision Transformers")
    assert list(matches) == [key]
    assert [paper.canonical_id for paper in matches[key]] == ["arxiv:2301.01234"]


def test_pasa_backend_is_offline_and_preserves_provenance(tmp_path: Path) -> None:
    archive, id_map = _source_files(tmp_path)
    index = tmp_path / "pasa-paper-database.sqlite3"
    build_pasa_paper_index(
        archive_path=archive,
        id_map_path=id_map,
        index_path=index,
    )
    backend = PasaPaperDatabaseSearchBackend(PasaPaperDatabase(index))

    result = asyncio.run(backend.search("local-1", "robust transformer", {}, 10))

    assert [paper.canonical_id for paper in result.hits] == ["arxiv:2301.01234"]
    assert result.usage.search_api_calls == 0
    assert result.usage.llm_calls == 0
    assert result.provenance["provider"] == "pasa_paper_database"
    assert result.provenance["action_id"] == "local-1"
    assert result.infrastructure_failure is False


def test_training_supplement_appends_missing_gold_after_lexical_negatives(
    tmp_path: Path,
) -> None:
    archive, id_map = _source_files(tmp_path)
    index = tmp_path / "pasa-paper-database.sqlite3"
    build_pasa_paper_index(
        archive_path=archive,
        id_map_path=id_map,
        index_path=index,
    )
    database = PasaPaperDatabase(index)

    papers, audit = pasa_paper_database.build_pasa_training_supplement(
        database=database,
        query="graph diffusion",
        gold_paper_ids=["arxiv:2301.01234"],
        search_limit=5,
    )

    assert [paper.canonical_id for paper in papers] == [
        "arxiv:2302.05678",
        "arxiv:2301.01234",
    ]
    assert PASA_TRAINING_GOLD_INJECTED_SOURCE not in papers[0].sources
    assert PASA_TRAINING_GOLD_INJECTED_SOURCE in papers[1].sources
    assert audit == {
        "lexical_candidate_count": 1,
        "direct_gold_candidate_count": 1,
        "supplement_candidate_count": 2,
    }


def test_training_supplement_searches_only_bounded_scientific_content_terms() -> None:
    class RecordingDatabase:
        query = ""

        def search(self, query: str, limit: int) -> list[Paper]:
            self.query = query
            return [
                Paper(
                    canonical_id="arxiv:2301.00001",
                    arxiv_id="2301.00001",
                    title="negative",
                )
            ]

        def lookup_arxiv(self, value: str) -> Paper | None:
            return None

    database = RecordingDatabase()

    pasa_paper_database.build_pasa_training_supplement(
        database=database,
        query=(
            "Could you provide some papers about graph diffusion retrieval with "
            "transformer embeddings ranking evaluation benchmark citations?"
        ),
        gold_paper_ids=[],
        search_limit=5,
    )

    assert database.query == (
        "graph diffusion retrieval transformer embeddings ranking evaluation benchmark"
    )


def test_training_supplement_adds_only_exact_negation_conflict_candidates() -> None:
    class ConstraintDatabase:
        def search(self, query: str, limit: int) -> list[Paper]:
            del limit
            if query == "adversarial training":
                return [
                    Paper(
                        canonical_id="arxiv:2301.00006",
                        arxiv_id="2301.00006",
                        title="Cache replacement using adversarial training",
                    ),
                    Paper(
                        canonical_id="arxiv:2301.00002",
                        arxiv_id="2301.00002",
                        title="Vision retrieval using adversarial training",
                    ),
                    Paper(
                        canonical_id="arxiv:2301.00003",
                        arxiv_id="2301.00003",
                        title="A survey of adversarial training",
                    ),
                    Paper(
                        canonical_id="arxiv:2301.00005",
                        arxiv_id="2301.00005",
                        title="Vision without adversarial training",
                    ),
                ]
            return [
                Paper(
                    canonical_id="arxiv:2301.00004",
                    arxiv_id="2301.00004",
                    title="General vision retrieval",
                )
            ]

        def lookup_arxiv(self, value: str) -> Paper | None:
            if value == "arxiv:2301.00001":
                return Paper(
                    canonical_id=value,
                    arxiv_id="2301.00001",
                    title="Clean vision model",
                )
            return None

    papers, audit = pasa_paper_database.build_pasa_training_supplement(
        database=ConstraintDatabase(),
        query="vision retrieval without adversarial training",
        gold_paper_ids=["arxiv:2301.00001"],
        search_limit=10,
        negative_exclusions=["adversarial training"],
        constraint_negative_limit=5,
    )

    assert [paper.canonical_id for paper in papers] == [
        "arxiv:2301.00004",
        "arxiv:2301.00002",
        "arxiv:2301.00001",
    ]
    assert audit["constraint_negative_candidate_count"] == 1


def test_negation_training_supplement_marks_every_targeted_candidate() -> None:
    class ConstraintDatabase:
        def search_required_phrase_with_term_pairs(
            self, required_phrase: str, topic_terms, limit: int
        ) -> list[Paper]:
            assert required_phrase == "adversarial training"
            assert {"vision", "retrieval"}.issubset(set(topic_terms))
            assert limit == 10
            return [
                Paper(
                    canonical_id="arxiv:2301.00002",
                    arxiv_id="2301.00002",
                    title="Vision retrieval using adversarial training",
                )
            ]

        def lookup_arxiv(self, value: str) -> Paper | None:
            if value == "arxiv:2301.00001":
                return Paper(
                    canonical_id=value,
                    arxiv_id="2301.00001",
                    title="Clean vision retrieval model",
                )
            return None

    papers, audit = pasa_paper_database.build_pasa_negation_training_supplement(
        database=ConstraintDatabase(),
        query="vision retrieval without adversarial training",
        gold_paper_ids=["arxiv:2301.00001"],
        negative_exclusions=["adversarial training"],
        constraint_negative_limit=10,
    )

    assert [paper.canonical_id for paper in papers] == [
        "arxiv:2301.00002",
        "arxiv:2301.00001",
    ]
    assert all(
        PASA_TRAINING_CONSTRAINT_INJECTED_SOURCE in paper.sources for paper in papers
    )
    assert PASA_TRAINING_GOLD_INJECTED_SOURCE in papers[1].sources
    assert audit == {
        "constraint_negative_candidate_count": 1,
        "direct_gold_candidate_count": 1,
        "supplement_candidate_count": 2,
    }


def test_gold_only_training_supplement_never_runs_lexical_search() -> None:
    class ExactOnlyDatabase:
        def search(self, query: str, limit: int) -> list[Paper]:
            raise AssertionError("gold-only supplement must not run FTS")

        def lookup_arxiv(self, value: str) -> Paper | None:
            if value == "arxiv:2301.01234":
                return Paper(
                    canonical_id=value,
                    arxiv_id="2301.01234",
                    title="positive",
                )
            return None

    papers, audit = pasa_paper_database.build_pasa_gold_training_supplement(
        database=ExactOnlyDatabase(),
        gold_paper_ids=["arxiv:2301.01234", "arxiv:2301.99999"],
    )

    assert [paper.canonical_id for paper in papers] == ["arxiv:2301.01234"]
    assert audit == {
        "direct_gold_candidate_count": 1,
        "supplement_candidate_count": 1,
    }


def test_bulk_gold_lookup_normalizes_ids_and_omits_missing_rows(tmp_path: Path) -> None:
    archive, id_map = _source_files(tmp_path)
    index = tmp_path / "pasa-paper-database.sqlite3"
    build_pasa_paper_index(
        archive_path=archive,
        id_map_path=id_map,
        index_path=index,
    )
    database = PasaPaperDatabase(index)

    papers = database.lookup_arxiv_many(
        ["https://arxiv.org/abs/2301.01234v2", "arxiv:2302.05678", "2309.99999"]
    )

    assert list(papers) == ["arxiv:2301.01234", "arxiv:2302.05678"]
    assert papers["arxiv:2301.01234"].title == "Robust Vision Transformers"
