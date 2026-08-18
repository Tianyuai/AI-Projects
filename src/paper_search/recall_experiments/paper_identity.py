"""Evidence-driven paper identity resolution for recall evaluation.

Dataset-specific aliases are optional evidence. Unseen arXiv identifiers are
resolved through their deterministic DataCite DOI, while provider records
contribute only identifiers that they explicitly carry.
"""

from __future__ import annotations

from paper_search.domain.models import Paper
from paper_search.evaluation.dataset import IdentifierMap, normalize_paper_id


_ARXIV_DATACITE_PREFIX = "doi:10.48550/arxiv."


def arxiv_datacite_anchor(value: str) -> str:
    """Return the deterministic DataCite DOI for an arXiv ID."""
    normalized = normalize_paper_id(value, kind="arxiv")
    return f"{_ARXIV_DATACITE_PREFIX}{normalized.removeprefix('arxiv:')}"


def _direct_arxiv_anchor(value: str, *, kind: str | None = None) -> str | None:
    normalized = normalize_paper_id(value, kind=kind)
    if normalized.startswith("arxiv:"):
        return arxiv_datacite_anchor(normalized)
    if normalized.startswith(_ARXIV_DATACITE_PREFIX):
        return arxiv_datacite_anchor(normalized.removeprefix(_ARXIV_DATACITE_PREFIX))
    return None


class EvidenceDrivenIdentifierResolver:
    """Match papers using deterministic anchors and explicit provider IDs."""

    def __init__(self, verified_aliases: IdentifierMap | None = None) -> None:
        self._verified_aliases = verified_aliases
        if verified_aliases is not None:
            self._validate_verified_aliases(verified_aliases)

    @staticmethod
    def _validate_verified_aliases(identifier_map: IdentifierMap) -> None:
        for alias, terminal in identifier_map.resolved_pairs():
            if not alias.startswith("arxiv:"):
                continue
            expected = arxiv_datacite_anchor(alias)
            observed = _direct_arxiv_anchor(terminal)
            if observed != expected:
                raise ValueError(
                    "identifier evidence map contains a non-equivalent arXiv relation"
                )

    def resolve(self, value: str, *, kind: str | None = None) -> str:
        """Resolve one ID without requiring it in a frozen alias table."""
        normalized = normalize_paper_id(value, kind=kind)
        direct_anchor = _direct_arxiv_anchor(normalized)
        if direct_anchor is not None:
            return direct_anchor
        if self._verified_aliases is None:
            return normalized
        mapped = self._verified_aliases.resolve(normalized)
        return _direct_arxiv_anchor(mapped) or mapped

    def paper_identities(self, paper: Paper) -> frozenset[str]:
        """Return every identifier explicitly supported by a provider paper."""
        raw_identifiers: list[tuple[str, str | None]] = [(paper.canonical_id, None)]
        if paper.doi is not None:
            raw_identifiers.append((paper.doi, "doi"))
        if paper.openalex_id is not None:
            raw_identifiers.append((paper.openalex_id, "openalex"))
        if paper.semantic_scholar_id is not None:
            raw_identifiers.append((paper.semantic_scholar_id, "semantic_scholar"))
        if paper.arxiv_id is not None:
            raw_identifiers.append((paper.arxiv_id, "arxiv"))

        direct_anchors = {
            anchor
            for value, kind in raw_identifiers
            if (anchor := _direct_arxiv_anchor(value, kind=kind)) is not None
        }
        if len(direct_anchors) > 1:
            raise ValueError("candidate contains conflicting arXiv identity evidence")

        identities = {
            self.resolve(value, kind=kind) for value, kind in raw_identifiers
        }
        mapped_anchors = {
            identity
            for identity in identities
            if identity.startswith(_ARXIV_DATACITE_PREFIX)
        }
        if len(mapped_anchors) > 1:
            raise ValueError("candidate contains conflicting arXiv identity evidence")
        return frozenset(identities)

    def primary_paper_id(self, paper: Paper) -> str:
        """Return one stable report ID while retaining aliases privately."""
        return self.resolve(paper.canonical_id)


__all__ = ["EvidenceDrivenIdentifierResolver", "arxiv_datacite_anchor"]
