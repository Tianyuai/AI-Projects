"""Pure, offline semantic checks for development identifier maps."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from pydantic import Field, ValidationError, model_validator

from paper_search.domain.models import (
    DomainModel,
    NonEmptyStr,
    NonNegativeInt,
    Sha256,
    StrictNonNegativeInt,
)
from paper_search.evaluation.dataset import EvaluationQuery, IdentifierMap, normalize_paper_id
from paper_search.storage.dependency_snapshot import (
    DependencyRequestIdentity,
    DependencySnapshotManifestV2,
    SnapshotEntryV2,
)


SemanticState = Literal["verified", "semantic_mismatch", "unresolved"]
RelationKind = Literal["required_anchor", "provider_candidate"]
ProofKind = Literal[
    "arxiv_datacite_exact",
    "semantic_scholar_exact",
    "openalex_location_exact",
]
ReasonCode = Literal[
    "arxiv_datacite_exact",
    "arxiv_datacite_mismatch",
    "semantic_scholar_exact",
    "openalex_location_exact",
    "openalex_location_mismatch",
    "insufficient_identity_evidence",
    "observation_binding_mismatch",
    "provider_identity_missing",
    "alias_target_conflict",
]
_S2_ARXIV_ADAPTER = "semantic-scholar-identity-arxiv-v1"
_S2_DOI_ADAPTER = "semantic-scholar-identity-doi-v1"
_OPENALEX_ADAPTER = "openalex-identity-v1"

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
    relation_kind: RelationKind = "provider_candidate"
    arxiv_id: NonEmptyStr
    alias: NonEmptyStr
    terminal: NonEmptyStr
    state: SemanticState
    proof_kind: ProofKind | None
    reason_code: ReasonCode

    @model_validator(mode="after")
    def validate_relation_contract(self) -> RelationAudit:
        try:
            normalized_arxiv = normalize_paper_id(self.arxiv_id, kind="arxiv")
            normalized_alias = normalize_paper_id(self.alias)
            normalized_terminal = normalize_paper_id(self.terminal, kind="doi")
        except ValueError:
            raise ValueError("identifier relation is not canonical") from None
        if (
            self.arxiv_id != normalized_arxiv
            or self.alias != normalized_alias
            or self.terminal != normalized_terminal
            or self.terminal != arxiv_anchor(normalized_arxiv)
        ):
            raise ValueError("identifier relation is not canonical")
        if self.relation_kind == "required_anchor":
            if self.alias != self.terminal:
                raise ValueError("required anchor alias must equal terminal")
            if self.state == "verified":
                if (
                    self.proof_kind != "arxiv_datacite_exact"
                    or self.reason_code != "arxiv_datacite_exact"
                ):
                    raise ValueError("verified anchor proof is invalid")
            elif self.proof_kind is not None or self.reason_code != "arxiv_datacite_mismatch":
                raise ValueError("failed anchor proof is invalid")
        elif self.state == "verified":
            if self.proof_kind is None or self.reason_code != self.proof_kind:
                raise ValueError("verified provider proof is invalid")
        elif self.state == "semantic_mismatch":
            if self.proof_kind is not None or self.reason_code not in {
                "arxiv_datacite_mismatch",
                "openalex_location_mismatch",
            }:
                raise ValueError("provider mismatch proof is invalid")
        elif self.proof_kind is not None or self.reason_code not in {
            "insufficient_identity_evidence",
            "observation_binding_mismatch",
            "alias_target_conflict",
        }:
            raise ValueError("unresolved provider proof is invalid")
        return self


class AuditInputHashes(DomainModel):
    dev_gold: Sha256
    identity_evidence: Sha256
    snapshot_manifest: Sha256


class AuditArtifactHashes(DomainModel):
    candidate_map: Sha256
    private_relation_audit: Sha256


class AuditStateCounts(DomainModel):
    verified: StrictNonNegativeInt
    semantic_mismatch: StrictNonNegativeInt
    unresolved: StrictNonNegativeInt


class AuditProofCounts(DomainModel):
    arxiv_datacite_exact: StrictNonNegativeInt
    semantic_scholar_exact: StrictNonNegativeInt
    openalex_location_exact: StrictNonNegativeInt


class AuditReasonCounts(DomainModel):
    arxiv_datacite_exact: StrictNonNegativeInt
    arxiv_datacite_mismatch: StrictNonNegativeInt
    semantic_scholar_exact: StrictNonNegativeInt
    openalex_location_exact: StrictNonNegativeInt
    openalex_location_mismatch: StrictNonNegativeInt
    insufficient_identity_evidence: StrictNonNegativeInt
    observation_binding_mismatch: StrictNonNegativeInt
    provider_identity_missing: StrictNonNegativeInt
    alias_target_conflict: StrictNonNegativeInt


class IdentifierMapSemanticAuditV1(DomainModel):
    schema_version: Literal["identifier-map-semantic-audit-v1"]
    scope: Literal["dev"]
    status: Literal["passed", "failed"]
    input_hashes: dict[NonEmptyStr, Sha256]
    gold_group_count: NonNegativeInt
    relation_count: NonNegativeInt
    state_counts: dict[NonEmptyStr, NonNegativeInt]
    proof_counts: dict[NonEmptyStr, NonNegativeInt]
    reason_counts: dict[NonEmptyStr, NonNegativeInt]


class IdentifierMapSemanticAudit(DomainModel):
    schema_version: Literal["identifier-map-semantic-audit-v2"]
    scope: Literal["dev"]
    status: Literal["passed", "failed"]
    input_hashes: AuditInputHashes
    artifact_hashes: AuditArtifactHashes
    gold_group_count: StrictNonNegativeInt
    required_anchor_count: StrictNonNegativeInt
    verified_anchor_count: StrictNonNegativeInt
    provider_candidate_count: StrictNonNegativeInt
    provider_identity_group_count: StrictNonNegativeInt
    provider_identity_missing_group_count: StrictNonNegativeInt
    relation_count: StrictNonNegativeInt
    state_counts: AuditStateCounts
    proof_counts: AuditProofCounts
    reason_counts: AuditReasonCounts


class PrivateRelationAuditV2(DomainModel):
    schema_version: Literal["identifier-map-private-relation-audit-v2"]
    scope: Literal["dev"]
    relations: tuple[RelationAudit, ...]

    @model_validator(mode="after")
    def validate_relation_order_and_uniqueness(self) -> PrivateRelationAuditV2:
        def order(row: RelationAudit) -> tuple[int, str, str]:
            return (
                0 if row.relation_kind == "required_anchor" else 1,
                row.arxiv_id,
                row.alias,
            )
        keys = [
            (row.relation_kind, row.arxiv_id, row.alias) for row in self.relations
        ]
        if len(set(keys)) != len(keys) or tuple(self.relations) != tuple(
            sorted(self.relations, key=order)
        ):
            raise ValueError("private relation audit rows are not canonical")
        return self


@dataclass(frozen=True)
class SemanticAuditBundle:
    report: IdentifierMapSemanticAuditV1
    private_relations: tuple[RelationAudit, ...]


@dataclass(frozen=True)
class VerifiedIdentifierGeneration:
    audit: IdentifierMapSemanticAudit
    identifier_map: IdentifierMap


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
    relation_kind: RelationKind,
    arxiv_id: str,
    alias: str,
    state: SemanticState,
    proof_kind: ProofKind | None,
    reason_code: str,
) -> RelationAudit:
    return RelationAudit(
        relation_kind=relation_kind,
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


def _snapshot_canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _referenced_snapshot_entry_ids(evidence: dict[str, object]) -> tuple[str, ...]:
    raw_refs = evidence.get("evidence_refs")
    if not isinstance(raw_refs, list):
        raise ValueError("identity evidence is invalid")
    entry_ids: set[str] = set()
    for raw_ref in raw_refs:
        if not isinstance(raw_ref, dict):
            raise ValueError("identity evidence is invalid")
        values = (
            raw_ref.get("semantic_scholar_arxiv_entry_id"),
            raw_ref.get("semantic_scholar_doi_entry_id"),
        )
        openalex_entry_ids = raw_ref.get("openalex_entry_ids", [])
        if (
            any(value is not None and not isinstance(value, str) for value in values)
            or not isinstance(openalex_entry_ids, list)
            or not all(isinstance(entry_id, str) for entry_id in openalex_entry_ids)
        ):
            raise ValueError("identity snapshot is invalid")
        entry_ids.update(value for value in values if value is not None)
        entry_ids.update(openalex_entry_ids)
    return tuple(sorted(entry_ids))


def _snapshot_entries_from_manifest_bytes(
    *, evidence: dict[str, object], manifest_content: bytes
) -> dict[str, SnapshotEntryV2]:
    expected_manifest_hash = evidence["snapshot_manifest_sha256"]
    if not isinstance(expected_manifest_hash, str):
        raise ValueError("identity snapshot is invalid")
    try:
        manifest = DependencySnapshotManifestV2.model_validate_json(manifest_content)
        if _sha256(manifest_content) != expected_manifest_hash:
            raise ValueError
        entries = manifest.entries
        if (
            len({entry.cache_key for entry in entries}) != len(entries)
            or len({entry.entry_id for entry in entries}) != len(entries)
            or entries
            != sorted(
                entries,
                key=lambda entry: (
                    entry.request.dependency,
                    entry.cache_key,
                    entry.entry_id,
                ),
            )
        ):
            raise ValueError
        if manifest.snapshot_set_id != _sha256(
            _snapshot_canonical_json_bytes(
                [entry.model_dump(mode="json", exclude_none=True) for entry in entries]
            )
        ):
            raise ValueError
        for entry in entries:
            expected_cache_key = _sha256(
                _snapshot_canonical_json_bytes(entry.request.model_dump(mode="json"))
            )
            expected_path = (
                f"responses/{entry.request.dependency}/"
                f"{entry.cache_key.removeprefix('sha256:')}.bin"
            )
            if entry.cache_key != expected_cache_key or entry.response_path != expected_path:
                raise ValueError
        return {entry.entry_id: entry for entry in entries}
    except (ValidationError, ValueError):
        raise ValueError("identity snapshot is invalid") from None


def _read_referenced_snapshot_bytes(
    *,
    evidence: dict[str, object],
    snapshot_root: Path,
    manifest_content: bytes,
) -> dict[str, bytes]:
    entries = _snapshot_entries_from_manifest_bytes(
        evidence=evidence, manifest_content=manifest_content
    )
    entry_ids = _referenced_snapshot_entry_ids(evidence)
    if not set(entry_ids).issubset(entries):
        raise ValueError("identity snapshot is invalid")
    response_root = snapshot_root.resolve()
    published_root = response_root / "snapshots"
    if published_root.is_dir():
        response_root = published_root.resolve()
    response_bytes: dict[str, bytes] = {}
    try:
        for entry_id in entry_ids:
            entry = entries[entry_id]
            path = response_root / entry.response_path
            resolved = path.resolve()
            if response_root not in resolved.parents:
                raise ValueError
            cursor = response_root
            for part in Path(entry.response_path).parts:
                cursor /= part
                if cursor.is_symlink():
                    raise ValueError
            content = path.read_bytes()
            if _sha256(content) != entry.response_sha256:
                raise ValueError
            response_bytes[entry_id] = content
    except (OSError, ValueError):
        raise ValueError("identity snapshot is invalid") from None
    return response_bytes


def _snapshot_observations(
    *,
    evidence: dict[str, object],
    snapshot_root: Path,
    manifest_content: bytes | None = None,
    response_bytes: dict[str, bytes] | None = None,
) -> tuple[dict[tuple[str, str], IdentityObservation], str]:
    try:
        if manifest_content is None:
            manifest_content = (snapshot_root / "snapshot-manifest.json").read_bytes()
        entry_by_id = _snapshot_entries_from_manifest_bytes(
            evidence=evidence, manifest_content=manifest_content
        )
        referenced_entry_ids = _referenced_snapshot_entry_ids(evidence)
        if response_bytes is None:
            entry_bytes = _read_referenced_snapshot_bytes(
                evidence=evidence,
                snapshot_root=snapshot_root,
                manifest_content=manifest_content,
            )
        else:
            entry_bytes = response_bytes
        if set(entry_bytes) != set(referenced_entry_ids):
            raise ValueError
        if any(
            _sha256(entry_bytes[entry_id]) != entry_by_id[entry_id].response_sha256
            for entry_id in referenced_entry_ids
        ):
            raise ValueError
        entry_requests = {
            entry_id: entry_by_id[entry_id].request for entry_id in referenced_entry_ids
        }
        manifest_entries = tuple(entry_by_id.values())
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
        relation_entry_ids = {
            entry_id
            for entry_id in (
                s2_arxiv_entry_id,
                s2_doi_entry_id,
                *openalex_entry_ids,
            )
            if entry_id is not None
        }
        if not relation_entry_ids.issubset(entry_bytes):
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
        s2_arxiv_complete = s2_arxiv_entry_id is not None and _s2_arxiv_item_matches(
            arxiv_s2[1],
            arxiv_id=arxiv_id,
            alias=alias,
        )
        s2_doi_complete = s2_doi_entry_id is not None and _s2_doi_item_matches(
            doi_s2[1],
            alias=alias,
        )
        openalex_ids: list[str] = []
        for entry_id in openalex_entry_ids:
            if not _valid_openalex_role(entry_requests.get(entry_id), alias=alias):
                raise ValueError("identity snapshot is invalid")
            openalex_ids.extend(
                _decode_openalex_arxiv_ids(
                    entry_bytes.get(entry_id),
                    expected_alias=alias,
                )
            )
        try:
            observation = IdentityObservation(
                arxiv_id=arxiv_id,
                alias=alias,
                semantic_scholar_arxiv_paper_id=arxiv_s2[0],
                semantic_scholar_doi_paper_id=doi_s2[0],
                semantic_scholar_arxiv_external_ids=arxiv_s2[1],
                semantic_scholar_doi_external_ids=doi_s2[1],
                semantic_scholar_arxiv_complete=s2_arxiv_complete,
                semantic_scholar_doi_complete=s2_doi_complete,
                openalex_complete=bool(openalex_entry_ids),
                openalex_arxiv_ids=openalex_ids,
                snapshot_sha256s=[
                    entry.response_sha256
                    for entry in manifest_entries
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


def _s2_arxiv_item_matches(
    external_ids: dict[str, str], *, arxiv_id: str, alias: str
) -> bool:
    normalized = _normalized_external_ids(external_ids)
    try:
        expected_arxiv_id = normalize_paper_id(arxiv_id, kind="arxiv")
        expected_alias = normalize_paper_id(alias, kind="doi")
    except ValueError:
        return False
    return (
        normalized.get("arxiv") == expected_arxiv_id
        and normalized.get("doi") == expected_alias
    )


def _s2_doi_item_matches(external_ids: dict[str, str], *, alias: str) -> bool:
    normalized = _normalized_external_ids(external_ids)
    try:
        expected_alias = normalize_paper_id(alias, kind="doi")
    except ValueError:
        return False
    return normalized.get("doi") == expected_alias


def _valid_openalex_role(
    request: DependencyRequestIdentity | None, *, alias: str
) -> bool:
    try:
        expected_alias = normalize_paper_id(alias, kind="openalex")
        expected_request = DependencyRequestIdentity.from_canonical_request(
            dependency="openalex",
            operation="search",
            method="GET",
            endpoint="/works",
            model_or_adapter=_OPENALEX_ADAPTER,
            canonical_request={"filter": expected_alias, "per_page": "1"},
        )
    except ValueError:
        return False
    return request == expected_request


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


def _decode_openalex_arxiv_ids(
    content: bytes | None, *, expected_alias: str
) -> list[str]:
    if content is None:
        return []
    try:
        payload = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return []
    if not isinstance(payload, dict):
        return []
    results = payload.get("results")
    if not isinstance(results, list) or len(results) != 1:
        return []
    payload = results[0]
    if not isinstance(payload, dict):
        return []
    work_id = payload.get("id")
    if not isinstance(work_id, str):
        return []
    try:
        if normalize_paper_id(work_id, kind="openalex") != normalize_paper_id(
            expected_alias, kind="openalex"
        ):
            return []
    except ValueError:
        return []
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
    *,
    alias: str,
    arxiv_id: str,
    observation: IdentityObservation | None,
    relation_kind: RelationKind = "provider_candidate",
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
                relation_kind=relation_kind,
                arxiv_id=normalized_arxiv_id,
                alias=normalized_alias,
                state="verified",
                proof_kind="arxiv_datacite_exact",
                reason_code="arxiv_datacite_exact",
            )
        return _relation(
            relation_kind=relation_kind,
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
                relation_kind=relation_kind,
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
                relation_kind=relation_kind,
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
                relation_kind=relation_kind,
                arxiv_id=normalized_arxiv_id,
                alias=normalized_alias,
                state="verified",
                proof_kind="openalex_location_exact",
                reason_code="openalex_location_exact",
            )
        if observation.openalex_complete and openalex_arxiv_ids:
            return _relation(
                relation_kind=relation_kind,
                arxiv_id=normalized_arxiv_id,
                alias=normalized_alias,
                state="semantic_mismatch",
                proof_kind=None,
                reason_code="openalex_location_mismatch",
            )

    return _relation(
        relation_kind=relation_kind,
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
    report = IdentifierMapSemanticAuditV1(
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


def _canonical_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def _parse_json_without_duplicate_keys(content: bytes, *, label: str) -> object:
    def object_pairs_hook(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate JSON keys")
            result[key] = value
        return result

    try:
        return json.loads(content, object_pairs_hook=object_pairs_hook)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        raise ValueError(f"{label} is invalid") from None


def _hash_bytes(content: bytes) -> str:
    return f"sha256:{hashlib.sha256(content).hexdigest()}"


def _private_relation_order(row: RelationAudit) -> tuple[int, str, str]:
    return (
        0 if row.relation_kind == "required_anchor" else 1,
        row.arxiv_id,
        row.alias,
    )


def _relations_from_sealed_snapshot(
    *,
    gold_arxiv_ids: tuple[str, ...],
    observations: dict[tuple[str, str], IdentityObservation],
) -> tuple[RelationAudit, ...]:
    relations = [
        classify_relation(
            alias=arxiv_anchor(arxiv_id),
            arxiv_id=arxiv_id,
            observation=None,
            relation_kind="required_anchor",
        )
        for arxiv_id in gold_arxiv_ids
    ]
    relations.extend(
        classify_relation(alias=alias, arxiv_id=arxiv_id, observation=observation)
        for (arxiv_id, alias), observation in sorted(observations.items())
    )
    targets_by_alias: dict[str, set[str]] = {}
    for relation in relations:
        if relation.relation_kind != "provider_candidate":
            continue
        targets_by_alias.setdefault(relation.alias, set()).add(relation.terminal)
    conflicts = {
        alias for alias, terminals in targets_by_alias.items() if len(terminals) > 1
    }
    return tuple(
        sorted(
            (
                relation.model_copy(
                    update={
                        "state": "unresolved",
                        "proof_kind": None,
                        "reason_code": "alias_target_conflict",
                    }
                )
                if relation.relation_kind == "provider_candidate"
                and relation.alias in conflicts
                else relation
                for relation in relations
            ),
            key=_private_relation_order,
        )
    )


def load_verified_identifier_generation(
    *,
    audit_path: Path,
    gold_path: Path,
    evidence_path: Path,
    snapshot_manifest_path: Path,
    private_audit_path: Path,
    map_path: Path,
) -> VerifiedIdentifierGeneration:
    """Load one passed v2 generation and recheck its cross-artifact contract."""
    try:
        audit_bytes = audit_path.read_bytes()
    except OSError:
        raise ValueError("public audit is invalid") from None
    audit_payload = _parse_json_without_duplicate_keys(
        audit_bytes, label="public audit"
    )
    if _canonical_json_bytes(audit_payload) != audit_bytes:
        raise ValueError("public audit is not canonical")
    assert_public_json_safe(audit_bytes)
    try:
        audit = IdentifierMapSemanticAudit.model_validate(audit_payload)
    except ValidationError:
        raise ValueError("public audit is invalid") from None
    if audit.status != "passed":
        raise ValueError("identifier semantic audit is not passed")

    try:
        gold_bytes = gold_path.read_bytes()
        evidence_bytes = evidence_path.read_bytes()
        manifest_bytes = snapshot_manifest_path.read_bytes()
        private_bytes = private_audit_path.read_bytes()
        map_bytes = map_path.read_bytes()
    except OSError:
        raise ValueError("identifier generation input is invalid") from None

    expected_inputs = {
        "dev_gold": _hash_bytes(gold_bytes),
        "identity_evidence": _hash_bytes(evidence_bytes),
        "snapshot_manifest": _hash_bytes(manifest_bytes),
    }
    if audit.input_hashes.model_dump() != expected_inputs:
        raise ValueError("identifier generation input hash mismatch")
    if _hash_bytes(private_bytes) != audit.artifact_hashes.private_relation_audit:
        raise ValueError("private relation audit hash mismatch")
    if _hash_bytes(map_bytes) != audit.artifact_hashes.candidate_map:
        raise ValueError("candidate map hash mismatch")

    private_payload = _parse_json_without_duplicate_keys(
        private_bytes, label="private relation audit"
    )
    if _canonical_json_bytes(private_payload) != private_bytes:
        raise ValueError("private relation audit is not canonical")
    try:
        private_audit = PrivateRelationAuditV2.model_validate(private_payload)
    except ValidationError:
        raise ValueError("private relation audit is invalid") from None
    if tuple(private_audit.relations) != tuple(
        sorted(private_audit.relations, key=_private_relation_order)
    ):
        raise ValueError("private relation audit order is invalid")

    map_payload = _parse_json_without_duplicate_keys(map_bytes, label="candidate map")
    if not isinstance(map_payload, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in map_payload.items()
    ):
        raise ValueError("candidate map is invalid")
    if _canonical_json_bytes(map_payload) != map_bytes:
        raise ValueError("candidate map is not canonical")
    try:
        for key, value in map_payload.items():
            if normalize_paper_id(key) != key or normalize_paper_id(value) != value:
                raise ValueError
        identifier_map = IdentifierMap.from_bytes(map_bytes)
    except ValueError:
        raise ValueError("candidate map is invalid") from None

    try:
        gold_arxiv_ids = _read_dev_arxiv_ids(gold_bytes)
        gold_ids = set(gold_arxiv_ids)
        evidence = _read_identity_evidence(evidence_bytes)
        manifest = DependencySnapshotManifestV2.model_validate_json(manifest_bytes)
    except (ValidationError, ValueError):
        raise ValueError("identifier generation input is invalid") from None
    if _hash_bytes(manifest_bytes) != evidence["snapshot_manifest_sha256"]:
        raise ValueError("identifier generation input hash mismatch")
    try:
        observations, _ = _snapshot_observations(
            evidence=evidence,
            snapshot_root=snapshot_manifest_path.parent,
            manifest_content=manifest_bytes,
        )
    except ValueError:
        raise ValueError("identifier generation sealed snapshot is invalid") from None

    raw_refs = evidence["evidence_refs"]
    if not isinstance(raw_refs, list):
        raise ValueError("identifier generation input is invalid")
    candidate_keys: list[tuple[str, str]] = []
    for raw_ref in raw_refs:
        if not isinstance(raw_ref, dict):
            raise ValueError("identifier generation input is invalid")
        arxiv_id = raw_ref.get("arxiv_id")
        alias = raw_ref.get("alias")
        if not isinstance(arxiv_id, str) or not isinstance(alias, str):
            raise ValueError("identifier generation input is invalid")
        try:
            candidate_keys.append(
                (
                    normalize_paper_id(arxiv_id, kind="arxiv"),
                    normalize_paper_id(alias),
                )
            )
        except ValueError:
            raise ValueError("identifier generation input is invalid") from None
    if len(set(candidate_keys)) != len(candidate_keys) or candidate_keys != sorted(candidate_keys):
        raise ValueError("identifier generation input is invalid")

    relations = private_audit.relations
    if relations != _relations_from_sealed_snapshot(
        gold_arxiv_ids=gold_arxiv_ids, observations=observations
    ):
        raise ValueError("private relation audit does not match sealed snapshot")
    anchors = tuple(row for row in relations if row.relation_kind == "required_anchor")
    providers = tuple(row for row in relations if row.relation_kind == "provider_candidate")
    if {row.arxiv_id for row in anchors} != gold_ids or len(anchors) != len(gold_ids):
        raise ValueError("private relation audit does not cover anchors")
    if {(row.arxiv_id, row.alias) for row in providers} != set(candidate_keys):
        raise ValueError("private relation audit does not cover evidence refs")
    if any(row.arxiv_id not in gold_ids for row in providers):
        raise ValueError("private relation audit contains an outside-gold relation")

    state_counts = Counter(row.state for row in relations)
    proof_counts = Counter(row.proof_kind for row in relations if row.proof_kind is not None)
    reason_counts = Counter(row.reason_code for row in relations)
    expected_state_counts = {
        "verified": state_counts.get("verified", 0),
        "semantic_mismatch": state_counts.get("semantic_mismatch", 0),
        "unresolved": state_counts.get("unresolved", 0),
    }
    expected_proof_counts = {
        "arxiv_datacite_exact": proof_counts.get("arxiv_datacite_exact", 0),
        "semantic_scholar_exact": proof_counts.get("semantic_scholar_exact", 0),
        "openalex_location_exact": proof_counts.get("openalex_location_exact", 0),
    }
    reason_names: tuple[ReasonCode, ...] = (
        "arxiv_datacite_exact",
        "arxiv_datacite_mismatch",
        "semantic_scholar_exact",
        "openalex_location_exact",
        "openalex_location_mismatch",
        "insufficient_identity_evidence",
        "observation_binding_mismatch",
        "provider_identity_missing",
        "alias_target_conflict",
    )
    expected_reason_counts = {
        reason: reason_counts.get(reason, 0) for reason in reason_names
    }
    verified_anchors = {
        row.arxiv_id
        for row in anchors
        if row.state == "verified" and row.proof_kind == "arxiv_datacite_exact"
    }
    expected_audit_counts = {
        "gold_group_count": len(gold_ids),
        "required_anchor_count": len(anchors),
        "verified_anchor_count": len(verified_anchors),
        "provider_candidate_count": len(providers),
        "provider_identity_group_count": len({row.arxiv_id for row in providers}),
        "provider_identity_missing_group_count": len(
            gold_ids - {row.arxiv_id for row in providers}
        ),
        "relation_count": len(relations),
    }
    if any(
        getattr(audit, field_name) != value
        for field_name, value in expected_audit_counts.items()
    ) or audit.state_counts.model_dump() != expected_state_counts:
        raise ValueError("public audit counts do not match private relations")
    if audit.proof_counts.model_dump() != expected_proof_counts:
        raise ValueError("public audit proof counts do not match private relations")
    if audit.reason_counts.model_dump() != expected_reason_counts:
        raise ValueError("public audit reason counts do not match private relations")
    if any(row.state != "verified" for row in relations):
        raise ValueError("passed audit contains an unverified relation")

    expected_map = {
        row.arxiv_id: row.terminal for row in anchors if row.state == "verified"
    }
    expected_map.update(
        {
            row.alias: row.terminal
            for row in providers
            if row.state == "verified" and row.alias != row.terminal
        }
    )
    if map_payload != dict(sorted(expected_map.items())):
        raise ValueError("candidate map does not match private relations")
    _ = manifest
    return VerifiedIdentifierGeneration(audit=audit, identifier_map=identifier_map)
