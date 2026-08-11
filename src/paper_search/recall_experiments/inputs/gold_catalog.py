"""Sealed private Gold-document catalogs for Oracle context construction."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

from pydantic import Field, ValidationError

from paper_search.domain.models import DomainModel, NonEmptyStr, Paper, SafeRelativePath, Sha256
from paper_search.evaluation.dataset import EvaluationQuery, normalize_paper_id, read_jsonl
from paper_search.recall_experiments.contracts import (
    GoldDocument,
    assert_no_forbidden_identifier_keys_or_patterns,
)
from paper_search.recall_experiments.recipes import ArtifactBinding


OracleCatalogStatus = Literal["complete", "incomplete", "invalid"]


class SourceManifestEntry(DomainModel):
    """One exact-byte Paper source that was verified by the catalog boundary."""

    path: SafeRelativePath
    sha256: Sha256


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
        document = GoldDocument(
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
        assert_no_forbidden_identifier_keys_or_patterns(document.model_dump(mode="json"))
        return document


class SealedGoldDocumentCatalog(DomainModel):
    records: list[SealedGoldDocumentRecord]
    source_hashes: dict[str, Sha256]
    source_manifest: list[SourceManifestEntry]
    source_manifest_sha256: Sha256
    catalog_sha256: Sha256
    status: OracleCatalogStatus

    def to_generation_documents(self, query_id: str) -> list[GoldDocument]:
        if self.status == "invalid":
            raise ValueError("oracle_catalog_invalid")
        rows = [record for record in self.records if record.query_id == query_id]
        if not rows or any(not row.has_title() for row in rows):
            raise ValueError("oracle_catalog_incomplete")
        return [record.to_generation_document() for record in rows]


class GoldDocumentCatalogBuilder:
    """Build a deterministic private catalog from verified Paper JSONL artifacts."""

    def __init__(self, workspace_root: Path | None = None) -> None:
        self._workspace_root = (workspace_root or Path.cwd()).resolve()

    def build(
        self,
        gold_associations: Sequence[EvaluationQuery],
        bound_paper_sources: Sequence[ArtifactBinding],
    ) -> SealedGoldDocumentCatalog:
        manifest, papers = self._load_paper_sources(bound_paper_sources)
        records = _records_for_associations(gold_associations, papers)
        return _seal(records, manifest, _catalog_status(records, gold_associations))

    def _load_paper_sources(
        self, bound_paper_sources: Sequence[ArtifactBinding]
    ) -> tuple[list[SourceManifestEntry], dict[str, Paper]]:
        manifest: list[SourceManifestEntry] = []
        papers: dict[str, Paper] = {}
        seen_paths: set[str] = set()
        for binding in bound_paper_sources:
            if not isinstance(binding, ArtifactBinding):
                raise TypeError("bound paper sources must be ArtifactBinding instances")
            if binding.path in seen_paths:
                raise ValueError("duplicate bound paper source path")
            seen_paths.add(binding.path)
            path = _resolve_within(self._workspace_root, binding.path, "bound paper source")
            content = path.read_bytes()
            if _sha256(content) != binding.sha256:
                raise ValueError("bound paper source hash mismatch")
            for paper in _read_normalized_papers(path, content):
                for identifier in _paper_identifiers(paper):
                    papers.setdefault(identifier, paper)
            manifest.append(SourceManifestEntry(path=binding.path, sha256=binding.sha256))
        return manifest, papers


class GoldDocumentCatalogSource:
    """Load a hash-bound private catalog only for association-aware Oracle preflight."""

    def __init__(self, workspace_root: Path | None = None) -> None:
        self._workspace_root = (workspace_root or Path.cwd()).resolve()

    def load(
        self,
        binding: ArtifactBinding,
        *,
        gold_associations: Sequence[EvaluationQuery] | None = None,
    ) -> SealedGoldDocumentCatalog:
        path = _resolve_within(self._workspace_root, binding.path, "Gold-document catalog")
        content = path.read_bytes()
        if _sha256(content) != binding.sha256:
            raise ValueError("Gold-document catalog hash mismatch")
        records = _read_catalog_records(path, content)
        manifest = [SourceManifestEntry(path=binding.path, sha256=binding.sha256)]
        status: OracleCatalogStatus = "invalid"
        if gold_associations is not None:
            status = _catalog_status(records, gold_associations)
        return _seal(records, manifest, status)


def _read_catalog_records(path: Path, content: bytes) -> list[SealedGoldDocumentRecord]:
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
    return records


def _read_normalized_papers(path: Path, content: bytes) -> list[Paper]:
    """Accept direct Paper JSONL or the frozen Query Evolution outcome envelope."""
    try:
        return read_jsonl(path, Paper)
    except ValueError as direct_error:
        papers: list[Paper] = []
        try:
            rows = [json.loads(line) for line in content.splitlines()]
            for row in rows:
                if not isinstance(row, Mapping):
                    raise ValueError("outcome row must be a JSON object")
                searches = row.get("searches")
                if not isinstance(searches, list):
                    raise ValueError("outcome row lacks searches")
                for search in searches:
                    if not isinstance(search, Mapping):
                        raise ValueError("outcome search must be a JSON object")
                    data = search.get("data")
                    if not isinstance(data, list):
                        raise ValueError("outcome search lacks Paper data")
                    papers.extend(Paper.model_validate(record) for record in data)
        except (json.JSONDecodeError, ValidationError, ValueError) as outcome_error:
            raise direct_error from outcome_error
        return papers


def _records_for_associations(
    associations: Sequence[EvaluationQuery], papers: dict[str, Paper]
) -> list[SealedGoldDocumentRecord]:
    records: list[SealedGoldDocumentRecord] = []
    for association in associations:
        for gold_id in association.relevant_paper_ids:
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
    return records


def _catalog_status(
    records: Sequence[SealedGoldDocumentRecord], associations: Sequence[EvaluationQuery]
) -> OracleCatalogStatus:
    expected_pairs = [
        (association.query_id, gold_id)
        for association in associations
        for gold_id in association.relevant_paper_ids
    ]
    actual_pairs = [(record.query_id, record.gold_paper_id) for record in records]
    if (
        not expected_pairs
        or len(expected_pairs) != len(set(expected_pairs))
        or len(actual_pairs) != len(set(actual_pairs))
        or set(actual_pairs) != set(expected_pairs)
    ):
        return "invalid"
    if any(not record.has_title() for record in records):
        return "incomplete"
    return "complete"


def _seal(
    records: list[SealedGoldDocumentRecord],
    source_manifest: list[SourceManifestEntry],
    status: OracleCatalogStatus,
) -> SealedGoldDocumentCatalog:
    manifest_payload = [entry.model_dump(mode="json") for entry in source_manifest]
    source_manifest_sha256 = _sha256(_canonical_bytes(manifest_payload))
    catalog_payload = {
        "records": [record.model_dump(mode="json") for record in records],
        "source_manifest_sha256": source_manifest_sha256,
        "status": status,
    }
    catalog_sha256 = _sha256(_canonical_bytes(catalog_payload))
    return SealedGoldDocumentCatalog(
        records=records,
        source_hashes={entry.path: entry.sha256 for entry in source_manifest},
        source_manifest=source_manifest,
        source_manifest_sha256=source_manifest_sha256,
        catalog_sha256=catalog_sha256,
        status=status,
    )


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


def _resolve_within(root: Path, relative_path: str, label: str) -> Path:
    path = (root / relative_path).resolve(strict=True)
    if not path.is_relative_to(root):
        raise ValueError(f"{label} path escapes workspace root")
    return path


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _sha256(content: bytes) -> Sha256:
    return f"sha256:{hashlib.sha256(content).hexdigest()}"
