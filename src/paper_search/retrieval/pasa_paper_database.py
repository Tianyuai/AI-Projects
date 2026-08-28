"""Read-only PASA paper database indexing and offline lexical retrieval."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import zipfile
from itertools import combinations
from pathlib import Path
from typing import Any
from collections.abc import Sequence

from paper_search.domain.models import Paper, UsageActual
from paper_search.evaluation.dataset import normalize_paper_id, normalize_title, sha256_file
from paper_search.evaluation.predictions import paper_evaluation_aliases
from paper_search.learning.candidates import query_content_terms
from paper_search.learning.negation_evidence import (
    classify_exclusion_stance,
    negation_topic_terms,
    negation_topic_relevant,
)
from paper_search.recall_experiments.retrieval.backends import BackendSearchResult


SCHEMA_VERSION = "pasa-paper-database-index-v1"
PASA_TRAINING_GOLD_INJECTED_SOURCE = "pasa_training_gold_injected"
PASA_TRAINING_CONSTRAINT_INJECTED_SOURCE = "pasa_training_constraint_injected"
ARXIV_MISSING_YEAR_EVIDENCE_POLICY = "declared-or-arxiv-submission-year-v1"


def _title_key(value: str) -> str:
    return "".join(character for character in value if character.isalpha()).casefold()


def _arxiv_id(value: str) -> str:
    return normalize_paper_id(value, kind="arxiv").removeprefix("arxiv:")


def arxiv_submission_year(value: str) -> int:
    normalized = re.sub(r"v\d+$", "", value.removeprefix("arxiv:"))
    modern = re.fullmatch(r"(\d{2})(\d{2})\.\d{4,5}", normalized)
    legacy = re.fullmatch(r"[^/]+/(\d{2})(\d{2})\d{3}", normalized)
    match = modern or legacy
    if match is None:
        raise ValueError(f"unsupported arXiv identifier: {value}")
    year_value, month_value = (int(part) for part in match.groups())
    if not 1 <= month_value <= 12:
        raise ValueError(f"invalid arXiv identifier month: {value}")
    return 1900 + year_value if year_value >= 91 else 2000 + year_value


# Kept for compatibility with the existing local PASA tests and callers.  New
# training code uses the public name so the derivation policy is explicit.
_arxiv_submission_year = arxiv_submission_year


def effective_publication_year(paper: Paper) -> int | None:
    """Use an explicit year first, then a valid arXiv identifier if available."""

    if paper.publication_year is not None:
        return paper.publication_year
    if paper.arxiv_id is None:
        return None
    try:
        return arxiv_submission_year(paper.arxiv_id)
    except ValueError:
        return None


def _paper(row: sqlite3.Row) -> Paper:
    arxiv_id = str(row["arxiv_id"])
    return Paper(
        canonical_id=f"arxiv:{arxiv_id}",
        arxiv_id=arxiv_id,
        title=str(row["title"]),
        abstract=str(row["abstract"]) if row["abstract"] is not None else None,
        publication_year=arxiv_submission_year(arxiv_id),
        sources=["pasa_paper_database"],
    )


def mark_pasa_training_gold_injected(paper: Paper) -> Paper:
    return paper.model_copy(
        update={
            "sources": list(
                dict.fromkeys(
                    [*paper.sources, PASA_TRAINING_GOLD_INJECTED_SOURCE]
                )
            )
        }
    )


def mark_pasa_training_constraint_injected(paper: Paper) -> Paper:
    return paper.model_copy(
        update={
            "sources": list(
                dict.fromkeys(
                    [*paper.sources, PASA_TRAINING_CONSTRAINT_INJECTED_SOURCE]
                )
            )
        }
    )


def _exclusion_stance(paper: Paper, phrase: str) -> str:
    text = ". ".join(
        value for value in (paper.title, paper.abstract or "") if value
    )
    return classify_exclusion_stance(text, phrase)


def _constraint_candidate_topic_relevant(
    paper: Paper, *, query: str, exclusion: str
) -> bool:
    return negation_topic_relevant(
        query,
        ". ".join(value for value in (paper.title, paper.abstract or "") if value),
        [exclusion],
    )


def _read_id_map(path: Path) -> dict[str, str]:
    payload: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("PASA id2paper must be a JSON object")
    normalized: dict[str, str] = {}
    for raw_id, raw_title in payload.items():
        if not isinstance(raw_id, str) or not isinstance(raw_title, str):
            raise ValueError("PASA id2paper keys and values must be strings")
        arxiv_id = _arxiv_id(raw_id)
        title = " ".join(raw_title.split())
        if not title:
            raise ValueError("PASA id2paper contains an empty title")
        existing = normalized.get(arxiv_id)
        if existing is not None and existing != title:
            raise ValueError(f"PASA id2paper conflict for arxiv:{arxiv_id}")
        normalized[arxiv_id] = title
    if not normalized:
        raise ValueError("PASA id2paper is empty")
    return normalized


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    if path.exists():
        if path.read_bytes() == _canonical_json(manifest):
            return
        raise FileExistsError(f"refusing to overwrite frozen PASA manifest: {path}")
    path.write_bytes(_canonical_json(manifest))


def build_pasa_paper_index(
    *,
    archive_path: Path,
    id_map_path: Path,
    index_path: Path,
) -> dict[str, Any]:
    """Build an immutable FTS5 index from the official PASA ZIP and ID map."""

    archive_path = archive_path.resolve()
    id_map_path = id_map_path.resolve()
    index_path = index_path.resolve()
    if not archive_path.is_file() or not id_map_path.is_file():
        raise FileNotFoundError("PASA paper database source files are unavailable")
    if index_path.exists():
        raise FileExistsError(f"refusing to overwrite PASA index: {index_path}")
    id_map = _read_id_map(id_map_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = index_path.with_name(f".{index_path.name}.tmp")
    temporary.unlink(missing_ok=True)
    indexed = missing = invalid = 0
    try:
        connection = sqlite3.connect(temporary)
        try:
            connection.executescript(
                """
                PRAGMA journal_mode=DELETE;
                PRAGMA synchronous=FULL;
                CREATE TABLE documents (
                    arxiv_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    abstract TEXT,
                    archive_member TEXT NOT NULL,
                    response_hash TEXT NOT NULL
                );
                CREATE VIRTUAL TABLE documents_fts USING fts5(
                    title,
                    abstract,
                    content='documents',
                    content_rowid='rowid',
                    tokenize='unicode61 remove_diacritics 2'
                );
                """
            )
            with zipfile.ZipFile(archive_path, "r") as archive:
                members = set(archive.namelist())
                for arxiv_id, mapped_title in sorted(id_map.items()):
                    member = _title_key(mapped_title)
                    if member not in members:
                        missing += 1
                        continue
                    raw = archive.read(member)
                    try:
                        record = json.loads(raw)
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        invalid += 1
                        continue
                    if not isinstance(record, dict):
                        invalid += 1
                        continue
                    title_value = record.get("title")
                    abstract_value = record.get("abstract")
                    if not isinstance(title_value, str) or not title_value.strip():
                        invalid += 1
                        continue
                    title = " ".join(title_value.split())
                    abstract = (
                        " ".join(abstract_value.split())
                        if isinstance(abstract_value, str) and abstract_value.strip()
                        else None
                    )
                    connection.execute(
                        """
                        INSERT INTO documents(
                            arxiv_id, title, abstract, archive_member, response_hash
                        ) VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            arxiv_id,
                            title,
                            abstract,
                            member,
                            "sha256:" + hashlib.sha256(raw).hexdigest(),
                        ),
                    )
                    indexed += 1
            connection.execute("INSERT INTO documents_fts(documents_fts) VALUES('rebuild')")
            connection.commit()
        finally:
            connection.close()
        temporary.replace(index_path)
    finally:
        temporary.unlink(missing_ok=True)

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "archive_path": str(archive_path),
        "archive_sha256": sha256_file(archive_path),
        "id_map_path": str(id_map_path),
        "id_map_sha256": sha256_file(id_map_path),
        "index_path": str(index_path),
        "index_sha256": sha256_file(index_path),
        "mapped_paper_count": len(id_map),
        "indexed_paper_count": indexed,
        "missing_archive_member_count": missing,
        "invalid_archive_record_count": invalid,
        "online_request_count": 0,
        "llm_request_count": 0,
        "test_partition_touched": False,
    }
    _write_manifest(index_path.with_suffix(index_path.suffix + ".manifest.json"), manifest)
    return manifest


class PasaPaperDatabase:
    """Read an immutable PASA SQLite index without loading the ZIP into memory."""

    def __init__(self, index_path: Path) -> None:
        self.index_path = index_path.resolve()
        if not self.index_path.is_file():
            raise FileNotFoundError(f"PASA paper index is unavailable: {self.index_path}")
        self.index_sha256 = sha256_file(self.index_path)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            f"file:{self.index_path.as_posix()}?mode=ro",
            uri=True,
        )
        connection.row_factory = sqlite3.Row
        return connection

    def lookup_arxiv(self, value: str) -> Paper | None:
        arxiv_id = _arxiv_id(value)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT arxiv_id, title, abstract FROM documents WHERE arxiv_id = ?",
                (arxiv_id,),
            ).fetchone()
        return None if row is None else _paper(row)

    def lookup_arxiv_many(self, values: Sequence[str]) -> dict[str, Paper]:
        """Resolve many normalized arXiv ids with bounded read-only SQL batches."""

        normalized: list[str] = []
        for value in values:
            try:
                arxiv_id = _arxiv_id(value)
            except ValueError:
                continue
            if arxiv_id not in normalized:
                normalized.append(arxiv_id)
        found: dict[str, Paper] = {}
        with self._connect() as connection:
            for start in range(0, len(normalized), 500):
                batch = normalized[start : start + 500]
                placeholders = ",".join("?" for _value in batch)
                rows = connection.execute(
                    "SELECT arxiv_id, title, abstract FROM documents "
                    f"WHERE arxiv_id IN ({placeholders})",
                    batch,
                ).fetchall()
                found.update(
                    (f"arxiv:{paper.arxiv_id}", paper)
                    for paper in (_paper(row) for row in rows)
                    if paper.arxiv_id is not None
                )
        return {
            f"arxiv:{arxiv_id}": found[f"arxiv:{arxiv_id}"]
            for arxiv_id in normalized
            if f"arxiv:{arxiv_id}" in found
        }

    def lookup_normalized_titles(
        self, values: Sequence[str]
    ) -> dict[str, tuple[Paper, ...]]:
        """Resolve public PASA records by normalized title without query labels."""

        requested = list(dict.fromkeys(normalize_title(value) for value in values))
        if not requested:
            return {}
        requested_set = set(requested)
        matched_ids: dict[str, list[str]] = {key: [] for key in requested}
        with self._connect() as connection:
            cursor = connection.execute("SELECT arxiv_id, title FROM documents")
            while rows := cursor.fetchmany(5000):
                for row in rows:
                    key = normalize_title(str(row["title"]))
                    if key in requested_set:
                        matched_ids[key].append(str(row["arxiv_id"]))
        selected_ids = [
            arxiv_id
            for key in requested
            for arxiv_id in matched_ids[key]
        ]
        papers_by_id = self.lookup_arxiv_many(selected_ids)
        return {
            key: tuple(
                papers_by_id[f"arxiv:{arxiv_id}"]
                for arxiv_id in matched_ids[key]
                if f"arxiv:{arxiv_id}" in papers_by_id
            )
            for key in requested
            if matched_ids[key]
        }

    def search(self, query: str, limit: int) -> list[Paper]:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("PASA search limit must be a positive integer")
        terms = list(dict.fromkeys(re.findall(r"[\w-]{2,}", query.casefold())))
        if not terms:
            raise ValueError("PASA search query must contain searchable terms")
        expression = " OR ".join(f'"{term.replace(chr(34), "")}"' for term in terms)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT documents.arxiv_id, documents.title, documents.abstract
                FROM documents_fts
                JOIN documents ON documents.rowid = documents_fts.rowid
                WHERE documents_fts MATCH ?
                ORDER BY bm25(documents_fts), documents.arxiv_id
                LIMIT ?
                """,
                (expression, limit),
            ).fetchall()
        return [_paper(row) for row in rows]

    def search_required_phrase_with_term_pairs(
        self, required_phrase: str, topic_terms: Sequence[str], limit: int
    ) -> list[Paper]:
        """Require one phrase plus at least two topic terms in the local FTS index."""

        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("PASA search limit must be a positive integer")
        required_tokens = re.findall(r"[a-z0-9]+", required_phrase.casefold())
        normalized_topics = list(
            dict.fromkeys(
                token
                for value in topic_terms
                for token in re.findall(r"[a-z0-9]+", value.casefold())
                if len(token) >= 3
            )
        )[:8]
        if not required_tokens or len(normalized_topics) < 2:
            return []
        required = " ".join(required_tokens).replace('"', "")
        pair_expression = " OR ".join(
            f"({left}* AND {right}*)"
            for left, right in combinations(normalized_topics, 2)
        )
        expression = f'"{required}" AND ({pair_expression})'
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT documents.arxiv_id, documents.title, documents.abstract
                FROM documents_fts
                JOIN documents ON documents.rowid = documents_fts.rowid
                WHERE documents_fts MATCH ?
                ORDER BY bm25(documents_fts), documents.arxiv_id
                LIMIT ?
                """,
                (expression, limit),
            ).fetchall()
        return [_paper(row) for row in rows]


class PasaPaperDatabaseSearchBackend:
    """Adapt PASA FTS5 search to the recall experiment backend contract."""

    def __init__(self, database: PasaPaperDatabase) -> None:
        self._database = database

    async def search(
        self,
        action_id: str,
        query: str,
        filters: dict[str, object],
        limit: int,
    ) -> BackendSearchResult:
        if filters.get("_search_mode", "lexical") != "lexical":
            raise ValueError("PASA paper database supports lexical search only")
        unsupported = set(filters).difference({"_search_mode"})
        if unsupported:
            raise ValueError(f"unsupported PASA paper database filters: {sorted(unsupported)}")
        return BackendSearchResult(
            hits=self._database.search(query, limit),
            usage=UsageActual(),
            provenance={
                "provider": "pasa_paper_database",
                "endpoint": "local_fts5",
                "action_id": action_id,
                "index_sha256": self._database.index_sha256,
            },
            infrastructure_failure=False,
        )


def build_pasa_training_supplement(
    *,
    database: PasaPaperDatabase,
    query: str,
    gold_paper_ids: Sequence[str],
    search_limit: int,
    negative_exclusions: Sequence[str] = (),
    constraint_negative_limit: int = 20,
) -> tuple[list[Paper], dict[str, int]]:
    """Build offline lexical negatives, then append missing PASA Gold papers."""

    content_terms = query_content_terms(query)[:8]
    lexical_query = " ".join(content_terms) if content_terms else query
    lexical = database.search(lexical_query, search_limit)
    papers = list(lexical)
    aliases = {
        alias for paper in papers for alias in paper_evaluation_aliases(paper)
    }
    constraint_negative_count = 0
    for exclusion in dict.fromkeys(
        " ".join(value.split()) for value in negative_exclusions if value.strip()
    ):
        for paper in database.search(exclusion, constraint_negative_limit):
            paper_aliases = paper_evaluation_aliases(paper)
            if (
                aliases.intersection(paper_aliases)
                or _exclusion_stance(paper, exclusion) != "conflict"
                or not _constraint_candidate_topic_relevant(
                    paper, query=query, exclusion=exclusion
                )
            ):
                continue
            papers.append(paper)
            aliases.update(paper_aliases)
            constraint_negative_count += 1
            if constraint_negative_count >= constraint_negative_limit:
                break
        if constraint_negative_count >= constraint_negative_limit:
            break
    direct_count = 0
    for gold_id in gold_paper_ids:
        try:
            gold_paper = database.lookup_arxiv(gold_id)
        except ValueError:
            continue
        if gold_paper is None:
            continue
        paper_aliases = paper_evaluation_aliases(gold_paper)
        if aliases.intersection(paper_aliases):
            continue
        gold_paper = mark_pasa_training_gold_injected(gold_paper)
        papers.append(gold_paper)
        aliases.update(paper_aliases)
        direct_count += 1
    audit = {
        "lexical_candidate_count": len(lexical),
        "direct_gold_candidate_count": direct_count,
        "supplement_candidate_count": len(papers),
    }
    if negative_exclusions:
        audit["constraint_negative_candidate_count"] = constraint_negative_count
    return papers, audit


def build_pasa_gold_training_supplement(
    *,
    database: PasaPaperDatabase,
    gold_paper_ids: Sequence[str],
) -> tuple[list[Paper], dict[str, int]]:
    """Resolve training-only Gold papers without running PASA full-text search."""

    papers: list[Paper] = []
    aliases: set[str] = set()
    for gold_id in gold_paper_ids:
        try:
            paper = database.lookup_arxiv(gold_id)
        except ValueError:
            continue
        if paper is None:
            continue
        paper_aliases = paper_evaluation_aliases(paper)
        if aliases.intersection(paper_aliases):
            continue
        papers.append(mark_pasa_training_gold_injected(paper))
        aliases.update(paper_aliases)
    return papers, {
        "direct_gold_candidate_count": len(papers),
        "supplement_candidate_count": len(papers),
    }


def build_pasa_negation_training_supplement(
    *,
    database: PasaPaperDatabase,
    query: str,
    gold_paper_ids: Sequence[str],
    negative_exclusions: Sequence[str],
    constraint_negative_limit: int = 20,
) -> tuple[list[Paper], dict[str, int]]:
    """Build a hard-constraint-only PASA overlay with no source-family supervision."""

    if constraint_negative_limit <= 0:
        raise ValueError("constraint negative limit must be positive")
    papers: list[Paper] = []
    aliases: set[str] = set()
    for exclusion in dict.fromkeys(
        " ".join(value.split()) for value in negative_exclusions if value.strip()
    ):
        topic_terms = negation_topic_terms(query, [exclusion])
        searcher = getattr(database, "search_required_phrase_with_term_pairs", None)
        candidates = (
            searcher(exclusion, topic_terms, constraint_negative_limit)
            if callable(searcher)
            else database.search(exclusion, constraint_negative_limit)
        )
        for paper in candidates:
            paper_aliases = paper_evaluation_aliases(paper)
            if (
                aliases.intersection(paper_aliases)
                or _exclusion_stance(paper, exclusion) != "conflict"
                or not _constraint_candidate_topic_relevant(
                    paper, query=query, exclusion=exclusion
                )
            ):
                continue
            marked = mark_pasa_training_constraint_injected(paper)
            papers.append(marked)
            aliases.update(paper_aliases)
            if len(papers) >= constraint_negative_limit:
                break
        if len(papers) >= constraint_negative_limit:
            break
    constraint_negative_count = len(papers)
    direct_gold_count = 0
    for gold_id in gold_paper_ids:
        try:
            paper = database.lookup_arxiv(gold_id)
        except ValueError:
            continue
        if paper is None:
            continue
        paper_aliases = paper_evaluation_aliases(paper)
        if aliases.intersection(paper_aliases):
            continue
        marked = mark_pasa_training_constraint_injected(
            mark_pasa_training_gold_injected(paper)
        )
        papers.append(marked)
        aliases.update(paper_aliases)
        direct_gold_count += 1
    return papers, {
        "constraint_negative_candidate_count": constraint_negative_count,
        "direct_gold_candidate_count": direct_gold_count,
        "supplement_candidate_count": len(papers),
    }


__all__ = [
    "PasaPaperDatabase",
    "PASA_TRAINING_CONSTRAINT_INJECTED_SOURCE",
    "PASA_TRAINING_GOLD_INJECTED_SOURCE",
    "PasaPaperDatabaseSearchBackend",
    "SCHEMA_VERSION",
    "build_pasa_paper_index",
    "build_pasa_gold_training_supplement",
    "build_pasa_negation_training_supplement",
    "build_pasa_training_supplement",
    "mark_pasa_training_gold_injected",
    "mark_pasa_training_constraint_injected",
]
