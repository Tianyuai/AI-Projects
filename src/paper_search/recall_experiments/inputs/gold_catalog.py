"""Sealed private Gold-document catalogs for Oracle context construction."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

from pydantic import Field, ValidationError

from paper_search.domain.models import DomainModel, NonEmptyStr, Paper, Sha256
from paper_search.evaluation.dataset import EvaluationQuery, normalize_paper_id
from paper_search.recall_experiments.contracts import GoldDocument
from paper_search.recall_experiments.recipes import ArtifactBinding


OracleCatalogStatus = Literal["complete", "incomplete", "invalid"]


class BoundPaperSource(DomainModel):
    """A normalized Paper collection already verified against frozen source bytes."""

    source_id: NonEmptyStr
    sha256: Sha256
    papers: list[Paper]


class SealedGoldDocumentRecord(DomainModel):
    """Private catalog row; identifiers never leave this catalog boundary."""

    query_id: NonEmptyStr
    gold_paper_id: NonEmptyStr
    title: str | None = None
    abstract: str | None = None
    authors: list[NonEmptyStr] = Field(default_factory=list)
    publication_year: int | None = None

    def has_title(self) -> bool:
        return self.title is not None and bool(self.title.strip())

    def to_generation_document(self) -> GoldDocument:
        if not self.has_title():
            raise ValueError("oracle_catalog_incomplete")
        return GoldDocument(
            title=self.title or "",
            abstract=self.abstract,
            authors=self.authors,
            publication_year=self.publication_year,
            metadata_coverage={
                "abstract": bool(self.abstract and self.abstract.strip()),
                "authors": bool(self.authors),
                "year": self.publication_year is not None,
            },
        )


class SealedGoldDocumentCatalog(DomainModel):
    records: list[SealedGoldDocumentRecord]
    source_hashes: dict[str, Sha256]
    status: OracleCatalogStatus

    def to_generation_documents(self, query_id: str) -> list[GoldDocument]:
        rows = [record for record in self.records if record.query_id == query_id]
        if not rows or any(not row.has_title() for row in rows):
            raise ValueError("oracle_catalog_incomplete")
        return [record.to_generation_document() for record in rows]


class GoldDocumentCatalogBuilder:
    """Build a deterministic private catalog without network access."""

    def build(
        self,
        gold_associations: Sequence[EvaluationQuery],
        bound_paper_sources: Sequence[BoundPaperSource]
        | Mapping[str, Sequence[Paper]],
    ) -> SealedGoldDocumentCatalog:
        papers, source_hashes = _paper_index(bound_paper_sources)
        records: list[SealedGoldDocumentRecord] = []
        seen: set[tuple[str, str]] = set()
        for association in gold_associations:
            for gold_id in association.relevant_paper_ids:
                key = (association.query_id, gold_id)
                if key in seen:
                    return SealedGoldDocumentCatalog(
                        records=records, source_hashes=source_hashes, status="invalid"
                    )
                seen.add(key)
                paper = papers.get(gold_id)
                records.append(
                    SealedGoldDocumentRecord(
                        query_id=association.query_id,
                        gold_paper_id=gold_id,
                        title=paper.title if paper is not None else None,
                        abstract=paper.abstract if paper is not None else None,
                        authors=paper.authors if paper is not None else [],
                        publication_year=paper.publication_year if paper is not None else None,
                    )
                )
        status: OracleCatalogStatus = (
            "complete" if all(record.has_title() for record in records) else "incomplete"
        )
        return SealedGoldDocumentCatalog(
            records=records, source_hashes=source_hashes, status=status
        )


class GoldDocumentCatalogSource:
    """Load a hash-bound private catalog only when Oracle data is authorized."""

    def __init__(self, workspace_root: Path | None = None) -> None:
        self._workspace_root = (workspace_root or Path.cwd()).resolve()

    def load(self, binding: ArtifactBinding) -> SealedGoldDocumentCatalog:
        path = (self._workspace_root / binding.path).resolve(strict=True)
        if not path.is_relative_to(self._workspace_root):
            raise ValueError("Gold-document catalog path escapes workspace root")
        content = path.read_bytes()
        if _sha256(content) != binding.sha256:
            raise ValueError("Gold-document catalog hash mismatch")
        records: list[SealedGoldDocumentRecord] = []
        try:
            for line_number, raw_line in enumerate(content.splitlines(), start=1):
                if not raw_line.strip():
                    raise ValueError(f"{path}:{line_number}: blank line is not allowed")
                decoded = json.loads(raw_line)
                if not isinstance(decoded, dict):
                    raise ValueError(f"{path}:{line_number}: expected a JSON object")
                records.append(SealedGoldDocumentRecord.model_validate(decoded))
        except (UnicodeDecodeError, json.JSONDecodeError, ValidationError) as error:
            raise ValueError(f"invalid Gold-document catalog: {path}") from error
        pairs = [(record.query_id, record.gold_paper_id) for record in records]
        status: OracleCatalogStatus
        if not records or len(pairs) != len(set(pairs)):
            status = "invalid"
        elif all(record.has_title() for record in records):
            status = "complete"
        else:
            status = "incomplete"
        return SealedGoldDocumentCatalog(
            records=records, source_hashes={"gold_document_catalog": binding.sha256}, status=status
        )


def _paper_index(
    sources: Sequence[BoundPaperSource] | Mapping[str, Sequence[Paper]],
) -> tuple[dict[str, Paper], dict[str, Sha256]]:
    if isinstance(sources, Mapping):
        groups = [
            BoundPaperSource(
                source_id=source_id,
                sha256=_sha256(
                    json.dumps(
                        [paper.model_dump(mode="json") for paper in papers],
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ),
                papers=list(papers),
            )
            for source_id, papers in sources.items()
        ]
    else:
        groups = list(sources)
    index: dict[str, Paper] = {}
    hashes: dict[str, Sha256] = {}
    for source in groups:
        hashes[source.source_id] = source.sha256
        for paper in source.papers:
            for identifier in _paper_identifiers(paper):
                index.setdefault(identifier, paper)
    return index, hashes


def _paper_identifiers(paper: Paper) -> set[str]:
    values: list[tuple[str, str | None]] = [
        (paper.canonical_id, None),
        (paper.doi or "", "doi"),
        (paper.openalex_id or "", "openalex"),
        (paper.semantic_scholar_id or "", "semantic_scholar"),
    ]
    identifiers: set[str] = set()
    for value, kind in values:
        if not value:
            continue
        try:
            identifiers.add(normalize_paper_id(value, kind=kind))
        except ValueError:
            continue
    return identifiers


def _sha256(content: bytes) -> Sha256:
    return f"sha256:{hashlib.sha256(content).hexdigest()}"
