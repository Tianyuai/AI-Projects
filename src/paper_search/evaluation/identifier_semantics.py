"""Pure, offline semantic checks for development identifier maps."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from pydantic import Field, ValidationError

from paper_search.domain.models import DomainModel, NonEmptyStr, NonNegativeInt, Sha256
from paper_search.evaluation.dataset import EvaluationQuery, IdentifierMap, normalize_paper_id
from paper_search.storage.dependency_snapshot import (
    DependencyRequestIdentity,
    DependencySnapshotManifestV2,
    DependencySnapshotReader,
)


SemanticState = Literal["verified", "semantic_mismatch", "unresolved"]
ProofKind = Literal[
    "arxiv_datacite_exact",
    "semantic_scholar_exact",
    "openalex_location_exact",
]
_S2_ARXIV_ADAPTER = "semantic-scholar-identity-arxiv-v1"
_S2_DOI_ADAPTER = "semantic-scholar-identity-doi-v1"

_PRIVATE_FIELD_NAMES = frozenset(
    {
        "query_id",
        "query_text",
        "arxiv_id",
        "alias",
        "terminal",
        "snapshot_sha256s",
        "evidence_refs",
        "semantic_scholar_arxiv_paper_id",
        "semantic_scholar_doi_paper_id",
        "semantic_scholar_arxiv_item_index",
        "semantic_scholar_doi_item_index",
        "semantic_scholar_arxiv_external_ids",
        "semantic_scholar_doi_external_ids",
        "openalex_arxiv_ids",
    }
)
_DOI_VALUE = re.compile(r"(?i)(?<![\w/])(?:doi:)?10\.\d{4,9}/\S+")
_ARXIV_VALUE = re.compile(
    r"(?i)(?<![\w/])(?:arxiv:)?(?:\d{4}\.\d{4,5}|[a-z][a-z0-9.-]*/\d{7})(?!\w)"
)
_OPENALEX_VALUE = re.compile(r"(?i)(?<!\w)(?:openalex:)?W\d+(?!\w)")
_QUERY_SENTINEL = re.compile(r"(?i)(?:private[-_ ]?)?query[-_ ]?sentinel")


class IdentityObservation(DomainModel):
    arxiv_id: NonEmptyStr
    alias: NonEmptyStr
    semantic_scholar_arxiv_paper_id: NonEmptyStr | None = None
    semantic_scholar_doi_paper_id: NonEmptyStr | None = None
    semantic_scholar_arxiv_external_ids: dict[NonEmptyStr, NonEmptyStr] = Field(
        default_factory=dict
    )
    semantic_scholar_doi_external_ids: dict[NonEmptyStr, NonEmptyStr] = Field(
        default_factory=dict
    )
    semantic_scholar_arxiv_complete: bool
    semantic_scholar_doi_complete: bool
    openalex_complete: bool
    openalex_arxiv_ids: list[NonEmptyStr] = Field(default_factory=list)
    snapshot_sha256s: list[Sha256] = Field(min_length=1)


class RelationAudit(DomainModel):
    arxiv_id: NonEmptyStr
    alias: NonEmptyStr
    terminal: NonEmptyStr
    state: SemanticState
    proof_kind: ProofKind | None
    reason_code: NonEmptyStr


class IdentifierMapSemanticAudit(DomainModel):
    schema_version: Literal["identifier-map-semantic-audit-v1"]
    scope: Literal["dev"]
    status: Literal["passed", "failed"]
    input_hashes: dict[NonEmptyStr, Sha256]
    gold_group_count: NonNegativeInt
    relation_count: NonNegativeInt
    state_counts: dict[NonEmptyStr, NonNegativeInt]
    proof_counts: dict[NonEmptyStr, NonNegativeInt]
    reason_counts: dict[NonEmptyStr, NonNegativeInt]


@dataclass(frozen=True)
class SemanticAuditBundle:
    report: IdentifierMapSemanticAudit
    private_relations: tuple[RelationAudit, ...]


def arxiv_anchor(arxiv_id: str) -> str:
    normalized = normalize_paper_id(arxiv_id, kind="arxiv")
    return f"doi:10.48550/arxiv.{normalized.removeprefix('arxiv:')}"


def _normalized_external_ids(external_ids: dict[str, str]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for raw_kind, raw_value in external_ids.items():
        kind = raw_kind.casefold()
        if kind not in {"arxiv", "doi"}:
            continue
        try:
            value = normalize_paper_id(raw_value, kind=kind)
        except ValueError:
            continue
        existing = normalized.get(kind)
        if existing is None:
            normalized[kind] = value
        elif existing != value:
            normalized[kind] = "conflict"
    return normalized


def _external_ids_conflict(left: dict[str, str], right: dict[str, str]) -> bool:
    return any(
        left[kind] != right[kind]
        for kind in set(left).intersection(right)
        if left[kind] != "conflict" and right[kind] != "conflict"
    ) or "conflict" in left.values() or "conflict" in right.values()


def _relation(
    *,
    arxiv_id: str,
    alias: str,
    state: SemanticState,
    proof_kind: ProofKind | None,
    reason_code: str,
) -> RelationAudit:
    return RelationAudit(
        arxiv_id=arxiv_id,
        alias=alias,
        terminal=arxiv_anchor(arxiv_id),
        state=state,
        proof_kind=proof_kind,
        reason_code=reason_code,
    )


def _sha256(content: bytes) -> str:
    return f"sha256:{hashlib.sha256(content).hexdigest()}"


def _read_identity_evidence(content: bytes) -> dict[str, object]:
    try:
        evidence = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ValueError("identity evidence is invalid") from None
    if (
        not isinstance(evidence, dict)
        or evidence.get("schema_version") != "identifier-identity-evidence-v1"
        or evidence.get("scope") != "dev"
        or not isinstance(evidence.get("snapshot_manifest_sha256"), str)
        or not isinstance(evidence.get("evidence_refs"), list)
    ):
        raise ValueError("identity evidence is invalid")
    return cast(dict[str, object], evidence)


def _read_dev_arxiv_ids(content: bytes) -> tuple[str, ...]:
    try:
        lines = content.splitlines()
        if not lines or any(not line.strip() for line in lines):
            raise ValueError
        records = [EvaluationQuery.model_validate_json(line) for line in lines]
    except (UnicodeDecodeError, ValidationError, ValueError):
        raise ValueError("development gold is invalid") from None

    arxiv_ids: set[str] = set()
    try:
        for record in records:
            for identifier in record.relevant_paper_ids:
                normalized = normalize_paper_id(identifier)
                if normalized.startswith("arxiv:"):
                    arxiv_ids.add(normalized)
    except ValueError:
        raise ValueError("development gold is invalid") from None
    if not arxiv_ids:
        raise ValueError("development gold is invalid")
    return tuple(sorted(arxiv_ids))


def _snapshot_observations(
    *, evidence: dict[str, object], snapshot_root: Path
) -> tuple[dict[tuple[str, str], IdentityObservation], str]:
    manifest_path = snapshot_root / "snapshot-manifest.json"
    expected_manifest_hash = evidence["snapshot_manifest_sha256"]
    if not isinstance(expected_manifest_hash, str):
        raise ValueError("identity snapshot is invalid")
    try:
        manifest_content = manifest_path.read_bytes()
        manifest = DependencySnapshotManifestV2.model_validate_json(manifest_content)
        reader = DependencySnapshotReader(
            manifest_path,
            snapshot_manifest_sha256=expected_manifest_hash,
        )
        entry_bytes = {}
        entry_requests = {}
        for entry in manifest.entries:
            entry_bytes[entry.entry_id] = reader.read(entry.request).response_bytes
            entry_requests[entry.entry_id] = entry.request
    except (OSError, ValidationError, ValueError, KeyError):
        raise ValueError("identity snapshot is invalid") from None

    observations: dict[tuple[str, str], IdentityObservation] = {}
    raw_refs = evidence["evidence_refs"]
    if not isinstance(raw_refs, list):
        raise ValueError("identity evidence is invalid")
    for raw_ref in raw_refs:
        if not isinstance(raw_ref, dict):
            raise ValueError("identity evidence is invalid")
        arxiv_id = raw_ref.get("arxiv_id")
        alias = raw_ref.get("alias")
        if not isinstance(arxiv_id, str) or not isinstance(alias, str):
            raise ValueError("identity evidence is invalid")
        s2_arxiv_entry_id = raw_ref.get("semantic_scholar_arxiv_entry_id")
        s2_doi_entry_id = raw_ref.get("semantic_scholar_doi_entry_id")
        s2_arxiv_item_index = raw_ref.get("semantic_scholar_arxiv_item_index")
        s2_doi_item_index = raw_ref.get("semantic_scholar_doi_item_index")
        openalex_entry_ids = raw_ref.get("openalex_entry_ids", [])
        if (
            (s2_arxiv_entry_id is not None and not isinstance(s2_arxiv_entry_id, str))
            or (s2_doi_entry_id is not None and not isinstance(s2_doi_entry_id, str))
            or (
                s2_arxiv_item_index is not None
                and (
                    type(s2_arxiv_item_index) is not int
                    or s2_arxiv_item_index < 0
                )
            )
            or (
                s2_doi_item_index is not None
                and (
                    type(s2_doi_item_index) is not int
                    or s2_doi_item_index < 0
                )
            )
            or (s2_arxiv_entry_id is None and s2_arxiv_item_index is not None)
            or (s2_doi_entry_id is None and s2_doi_item_index is not None)
            or not isinstance(openalex_entry_ids, list)
            or not all(isinstance(entry_id, str) for entry_id in openalex_entry_ids)
        ):
            raise ValueError("identity snapshot is invalid")
        referenced_entry_ids = {
            entry_id
            for entry_id in (
                s2_arxiv_entry_id,
                s2_doi_entry_id,
                *openalex_entry_ids,
            )
            if entry_id is not None
        }
        if not referenced_entry_ids.issubset(entry_bytes):
            raise ValueError("identity snapshot is invalid")
        if (
            s2_arxiv_entry_id is not None
            and s2_doi_entry_id is not None
            and s2_arxiv_entry_id == s2_doi_entry_id
        ):
            raise ValueError("identity snapshot is invalid")
        if not _valid_s2_role(
            (
                entry_requests.get(s2_arxiv_entry_id)
                if s2_arxiv_entry_id is not None
                else None
            ),
            expected_adapter=_S2_ARXIV_ADAPTER,
            required=s2_arxiv_entry_id is not None,
        ) or not _valid_s2_role(
            (
                entry_requests.get(s2_doi_entry_id)
                if s2_doi_entry_id is not None
                else None
            ),
            expected_adapter=_S2_DOI_ADAPTER,
            required=s2_doi_entry_id is not None,
        ):
            raise ValueError("identity snapshot is invalid")

        arxiv_s2 = _decode_s2_response(
            entry_bytes[s2_arxiv_entry_id] if s2_arxiv_entry_id is not None else None,
            s2_arxiv_item_index,
        )
        doi_s2 = _decode_s2_response(
            entry_bytes[s2_doi_entry_id] if s2_doi_entry_id is not None else None,
            s2_doi_item_index,
        )
        openalex_ids: list[str] = []
        for entry_id in openalex_entry_ids:
            openalex_ids.extend(_decode_openalex_arxiv_ids(entry_bytes.get(entry_id)))
        try:
            observation = IdentityObservation(
                arxiv_id=arxiv_id,
                alias=alias,
                semantic_scholar_arxiv_paper_id=arxiv_s2[0],
                semantic_scholar_doi_paper_id=doi_s2[0],
                semantic_scholar_arxiv_external_ids=arxiv_s2[1],
                semantic_scholar_doi_external_ids=doi_s2[1],
                semantic_scholar_arxiv_complete=s2_arxiv_entry_id is not None,
                semantic_scholar_doi_complete=s2_doi_entry_id is not None,
                openalex_complete=bool(openalex_entry_ids),
                openalex_arxiv_ids=openalex_ids,
                snapshot_sha256s=[
                    entry.response_sha256
                    for entry in manifest.entries
                    if entry.entry_id
                    in {
                        s2_arxiv_entry_id,
                        s2_doi_entry_id,
                        *openalex_entry_ids,
                    }
                ],
            )
        except ValidationError:
            raise ValueError("identity evidence is invalid") from None
        observations[(observation.arxiv_id, observation.alias)] = observation
    return observations, _sha256(manifest_content)


def _valid_s2_role(
    request: DependencyRequestIdentity | None,
    *,
    expected_adapter: str,
    required: bool,
) -> bool:
    if not required:
        return request is None
    return (
        request is not None
        and request.dependency == "semantic_scholar"
        and request.operation == "batch"
        and request.method == "POST"
        and request.endpoint == "/paper/batch"
        and request.model_or_adapter == expected_adapter
    )


def _decode_s2_response(
    content: bytes | None, item_index: object = None
) -> tuple[str | None, dict[str, str]]:
    if content is None:
        return None, {}
    try:
        payload = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None, {}
    if isinstance(payload, list):
        if (
            type(item_index) is not int
            or item_index < 0
            or item_index >= len(payload)
        ):
            raise ValueError("identity snapshot is invalid")
        payload = payload[item_index]
    elif isinstance(payload, dict) and item_index is not None:
        raise ValueError("identity snapshot is invalid")
    if not isinstance(payload, dict):
        return None, {}
    paper_id = payload.get("paperId")
    external_ids = payload.get("externalIds")
    return (
        paper_id if isinstance(paper_id, str) and paper_id.strip() else None,
        {
            key: value
            for key, value in external_ids.items()
            if isinstance(key, str) and isinstance(value, str)
        }
        if isinstance(external_ids, dict)
        else {},
    )


def _decode_openalex_arxiv_ids(content: bytes | None) -> list[str]:
    if content is None:
        return []
    try:
        payload = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return []
    if not isinstance(payload, dict):
        return []
    results = payload.get("results")
    if isinstance(results, list):
        if len(results) != 1 or not isinstance(results[0], dict):
            return []
        payload = results[0]
    locations = payload.get("locations")
    if not isinstance(locations, list):
        return []
    arxiv_ids: list[str] = []
    for location in locations:
        if not isinstance(location, dict):
            continue
        for field_name in ("landing_page_url", "pdf_url"):
            value = location.get(field_name)
            if not isinstance(value, str):
                continue
            try:
                arxiv_ids.append(normalize_paper_id(value, kind="arxiv"))
            except ValueError:
                continue
    return arxiv_ids


def classify_relation(
    *, alias: str, arxiv_id: str, observation: IdentityObservation | None
) -> RelationAudit:
    """Apply only exact DataCite, two-sided S2, and OpenAlex proof rules."""
    try:
        normalized_arxiv_id = normalize_paper_id(arxiv_id, kind="arxiv")
        normalized_alias = normalize_paper_id(alias)
    except ValueError:
        raise ValueError("identifier relation is invalid") from None

    anchor = arxiv_anchor(normalized_arxiv_id)
    if normalized_alias.startswith("doi:10.48550/arxiv."):
        if normalized_alias == anchor:
            return _relation(
                arxiv_id=normalized_arxiv_id,
                alias=normalized_alias,
                state="verified",
                proof_kind="arxiv_datacite_exact",
                reason_code="arxiv_datacite_exact",
            )
        return _relation(
            arxiv_id=normalized_arxiv_id,
            alias=normalized_alias,
            state="semantic_mismatch",
            proof_kind=None,
            reason_code="arxiv_datacite_mismatch",
        )

    if observation is not None:
        try:
            observation_arxiv_id = normalize_paper_id(observation.arxiv_id, kind="arxiv")
            observation_alias = normalize_paper_id(observation.alias)
        except ValueError:
            observation_arxiv_id = ""
            observation_alias = ""
        if (
            observation_arxiv_id != normalized_arxiv_id
            or observation_alias != normalized_alias
        ):
            return _relation(
                arxiv_id=normalized_arxiv_id,
                alias=normalized_alias,
                state="unresolved",
                proof_kind=None,
                reason_code="observation_binding_mismatch",
            )
        arxiv_external_ids = _normalized_external_ids(
            observation.semantic_scholar_arxiv_external_ids
        )
        doi_external_ids = _normalized_external_ids(
            observation.semantic_scholar_doi_external_ids
        )
        s2_exact = (
            observation.semantic_scholar_arxiv_complete
            and observation.semantic_scholar_doi_complete
            and observation.semantic_scholar_arxiv_paper_id
            and observation.semantic_scholar_arxiv_paper_id
            == observation.semantic_scholar_doi_paper_id
            and normalized_alias.startswith("doi:")
            and arxiv_external_ids.get("doi") == normalized_alias
            and not _external_ids_conflict(arxiv_external_ids, doi_external_ids)
        )
        if s2_exact:
            return _relation(
                arxiv_id=normalized_arxiv_id,
                alias=normalized_alias,
                state="verified",
                proof_kind="semantic_scholar_exact",
                reason_code="semantic_scholar_exact",
            )

        try:
            openalex_arxiv_ids = {
                normalize_paper_id(value, kind="arxiv")
                for value in observation.openalex_arxiv_ids
            }
        except ValueError:
            openalex_arxiv_ids = set()
        if normalized_arxiv_id in openalex_arxiv_ids:
            return _relation(
                arxiv_id=normalized_arxiv_id,
                alias=normalized_alias,
                state="verified",
                proof_kind="openalex_location_exact",
                reason_code="openalex_location_exact",
            )
        if observation.openalex_complete and openalex_arxiv_ids:
            return _relation(
                arxiv_id=normalized_arxiv_id,
                alias=normalized_alias,
                state="semantic_mismatch",
                proof_kind=None,
                reason_code="openalex_location_mismatch",
            )

    return _relation(
        arxiv_id=normalized_arxiv_id,
        alias=normalized_alias,
        state="unresolved",
        proof_kind=None,
        reason_code="insufficient_identity_evidence",
    )


def audit_identifier_map_semantics(
    *, map_bytes: bytes, gold_bytes: bytes, evidence_bytes: bytes, snapshot_root: Path
) -> SemanticAuditBundle:
    """Rebuild observations from sealed snapshots and audit offline."""
    try:
        identifier_map = IdentifierMap.from_bytes(map_bytes)
        gold_arxiv_ids = _read_dev_arxiv_ids(gold_bytes)
        evidence = _read_identity_evidence(evidence_bytes)
    except ValueError:
        raise ValueError("identifier semantic audit inputs are invalid") from None
    observations, manifest_hash = _snapshot_observations(
        evidence=evidence,
        snapshot_root=snapshot_root,
    )

    relations: list[RelationAudit] = []
    pairs = identifier_map.resolved_pairs()
    for arxiv_id in gold_arxiv_ids:
        terminal = identifier_map.resolve(arxiv_id)
        aliases = [alias for alias, target in pairs if target == terminal]
        if not aliases:
            aliases = [terminal]
        for alias in aliases:
            candidate_alias = terminal if alias.startswith("arxiv:") else alias
            observation = observations.get((arxiv_id, candidate_alias))
            relations.append(
                classify_relation(
                    alias=candidate_alias,
                    arxiv_id=arxiv_id,
                    observation=observation,
                )
            )

    ordered_relations = tuple(
        sorted(relations, key=lambda relation: (relation.arxiv_id, relation.alias))
    )
    state_counts = Counter(relation.state for relation in ordered_relations)
    proof_counts = Counter(
        relation.proof_kind
        for relation in ordered_relations
        if relation.proof_kind is not None
    )
    reason_counts = Counter(relation.reason_code for relation in ordered_relations)
    report = IdentifierMapSemanticAudit(
        schema_version="identifier-map-semantic-audit-v1",
        scope="dev",
        status="passed"
        if all(relation.state == "verified" for relation in ordered_relations)
        else "failed",
        input_hashes={
            "map": _sha256(map_bytes),
            "dev_gold": _sha256(gold_bytes),
            "identity_evidence": _sha256(evidence_bytes),
            "snapshot_manifest": manifest_hash,
        },
        gold_group_count=len(gold_arxiv_ids),
        relation_count=len(ordered_relations),
        state_counts=dict(sorted(state_counts.items())),
        proof_counts=dict(sorted(proof_counts.items())),
        reason_counts=dict(sorted(reason_counts.items())),
    )
    return SemanticAuditBundle(report=report, private_relations=ordered_relations)


def assert_public_json_safe(content: bytes) -> None:
    """Reject exact private keys and identifier/query values, not aggregate names."""
    try:
        payload = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ValueError("public JSON is invalid") from None

    def verify(value: object) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                if not isinstance(key, str) or key in _PRIVATE_FIELD_NAMES:
                    raise ValueError("public content contains a private field")
                verify(nested)
        elif isinstance(value, list):
            for nested in value:
                verify(nested)
        elif isinstance(value, str) and (
            _DOI_VALUE.search(value)
            or _ARXIV_VALUE.search(value)
            or _OPENALEX_VALUE.search(value)
            or _QUERY_SENTINEL.search(value)
        ):
            raise ValueError("public content contains a private value")

    verify(payload)


def assert_public_markdown_safe(content: str) -> None:
    """Reject private field tokens and identifier/query values at field boundaries."""
    for field_name in _PRIVATE_FIELD_NAMES:
        if re.search(rf"(?im)^\s*{re.escape(field_name)}\s*:", content):
            raise ValueError("public content contains a private field")
    if (
        _DOI_VALUE.search(content)
        or _ARXIV_VALUE.search(content)
        or _OPENALEX_VALUE.search(content)
        or _QUERY_SENTINEL.search(content)
    ):
        raise ValueError("public content contains a private value")
