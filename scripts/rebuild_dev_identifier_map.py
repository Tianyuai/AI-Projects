"""Rebuild a private dev identifier map from sealed identity evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from paper_search.evaluation.identifier_semantics import (
    IdentifierMapSemanticAudit,
    PrivateRelationAuditV2,
    ReasonCode,
    RelationAudit,
    _read_dev_arxiv_ids,
    _read_identity_evidence,
    _snapshot_observations,
    arxiv_anchor,
    assert_public_json_safe,
    classify_relation,
)
from paper_search.evaluation.dataset import normalize_paper_id


@dataclass(frozen=True)
class RebuiltDevMap:
    map_payload: dict[str, str]
    private_relations: tuple[RelationAudit, ...]
    audit: IdentifierMapSemanticAudit


class SemanticAuditFailure(ValueError):
    """Signal that sealed inputs did not support a passed private dev map."""

    def __init__(
        self,
        public_audit: IdentifierMapSemanticAudit,
        *,
        private_relations: tuple[RelationAudit, ...],
    ) -> None:
        super().__init__("identifier semantic audit failed")
        self.public_audit = public_audit
        self.private_relations = private_relations


def _sha256(content: bytes) -> str:
    return f"sha256:{hashlib.sha256(content).hexdigest()}"


_FIXED_INPUT_HASHES = {
    "dev_gold": "sha256:24009cf03ad069131793b9a190024e239082277bd0e48149a1efbbbb7978e215",
    "identity_evidence": "sha256:e4567d4b7641871ed538c18f5625cd7037e3014065e7311d3a76e81d4e4c61d4",
    "snapshot_manifest": "sha256:a0c0cd67543582e02365a2adfb3464a6f33fa96be5304d4ccac8dd031867943b",
}


def _assert_fixed_baseline_invariant(
    *,
    input_hashes: dict[str, str],
    gold_group_count: int,
    required_anchor_count: int,
    provider_identity_group_count: int,
    provider_identity_missing_group_count: int,
    provider_candidate_count: int,
    relation_count: int,
) -> None:
    if input_hashes != _FIXED_INPUT_HASHES:
        return
    expected = (141, 141, 90, 51, 90, 231)
    actual = (
        gold_group_count,
        required_anchor_count,
        provider_identity_group_count,
        provider_identity_missing_group_count,
        provider_candidate_count,
        relation_count,
    )
    if actual != expected:
        raise ValueError("identifier semantic decoder regression")


def _canonical_json(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
        temporary_path.replace(path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _aggregate_audit(
    *,
    map_bytes: bytes,
    gold_bytes: bytes,
    evidence_bytes: bytes,
    manifest_hash: str,
    gold_group_count: int,
    relations: tuple[RelationAudit, ...],
) -> IdentifierMapSemanticAudit:
    state_counts = Counter(relation.state for relation in relations)
    proof_counts = Counter(
        relation.proof_kind for relation in relations if relation.proof_kind is not None
    )
    reason_counts = Counter(relation.reason_code for relation in relations)
    private_audit = PrivateRelationAuditV2(
        schema_version="identifier-map-private-relation-audit-v2",
        scope="dev",
        relations=relations,
    )
    anchor_relations = tuple(
        relation for relation in relations if relation.relation_kind == "required_anchor"
    )
    provider_relations = tuple(
        relation for relation in relations if relation.relation_kind == "provider_candidate"
    )
    verified_anchor_count = len(
        {
            relation.arxiv_id
            for relation in anchor_relations
            if relation.state == "verified"
            and relation.proof_kind == "arxiv_datacite_exact"
        }
    )
    all_state_counts = {
        "verified": state_counts.get("verified", 0),
        "semantic_mismatch": state_counts.get("semantic_mismatch", 0),
        "unresolved": state_counts.get("unresolved", 0),
    }
    all_proof_counts = {
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
    all_reason_counts = {
        reason: reason_counts.get(reason, 0) for reason in reason_names
    }
    input_hashes = {
        "dev_gold": _sha256(gold_bytes),
        "identity_evidence": _sha256(evidence_bytes),
        "snapshot_manifest": manifest_hash,
    }
    _assert_fixed_baseline_invariant(
        input_hashes=input_hashes,
        gold_group_count=gold_group_count,
        required_anchor_count=len(anchor_relations),
        provider_identity_group_count=len({row.arxiv_id for row in provider_relations}),
        provider_identity_missing_group_count=gold_group_count
        - len({row.arxiv_id for row in provider_relations}),
        provider_candidate_count=len(provider_relations),
        relation_count=len(relations),
    )
    return IdentifierMapSemanticAudit(
        schema_version="identifier-map-semantic-audit-v2",
        scope="dev",
        status=(
            "passed"
            if len(anchor_relations) == gold_group_count
            and verified_anchor_count == gold_group_count
            and relations
            and all(row.state == "verified" for row in relations)
            else "failed"
        ),
        input_hashes=input_hashes,
        artifact_hashes={
            "candidate_map": _sha256(map_bytes),
            "private_relation_audit": _sha256(_canonical_json(private_audit.model_dump(mode="json"))),
        },
        gold_group_count=gold_group_count,
        required_anchor_count=len(anchor_relations),
        verified_anchor_count=verified_anchor_count,
        provider_candidate_count=len(provider_relations),
        provider_identity_group_count=len({row.arxiv_id for row in provider_relations}),
        provider_identity_missing_group_count=gold_group_count
        - len({row.arxiv_id for row in provider_relations}),
        relation_count=len(relations),
        state_counts=all_state_counts,
        proof_counts=all_proof_counts,
        reason_counts=all_reason_counts,
    )


def _resolve_alias_conflicts(relations: list[RelationAudit]) -> list[RelationAudit]:
    targets_by_alias: dict[str, set[str]] = defaultdict(set)
    for relation in relations:
        if relation.relation_kind == "provider_candidate":
            targets_by_alias[relation.alias].add(relation.terminal)
    conflicts = {
        alias for alias, terminals in targets_by_alias.items() if len(terminals) > 1
    }
    if not conflicts:
        return relations
    return [
        relation.model_copy(
            update={
                "state": "unresolved",
                "proof_kind": None,
                "reason_code": "alias_target_conflict",
            }
        )
        if relation.relation_kind == "provider_candidate" and relation.alias in conflicts
        else relation
        for relation in relations
    ]


def rebuild_dev_map(
    *, gold_bytes: bytes, evidence_bytes: bytes, snapshot_root: Path
) -> RebuiltDevMap:
    """Rebuild observations from sealed snapshots, then build the dev map."""
    try:
        gold_arxiv_ids = _read_dev_arxiv_ids(gold_bytes)
        evidence = _read_identity_evidence(evidence_bytes)
    except ValueError:
        raise ValueError("identifier semantic audit inputs are invalid") from None
    observations, manifest_hash = _snapshot_observations(
        evidence=evidence,
        snapshot_root=snapshot_root,
    )
    raw_refs = evidence["evidence_refs"]
    if not isinstance(raw_refs, list):
        raise ValueError("identifier semantic audit inputs are invalid")
    gold_set = set(gold_arxiv_ids)
    candidate_keys: list[tuple[str, str]] = []
    try:
        for raw_ref in raw_refs:
            if not isinstance(raw_ref, dict):
                raise ValueError
            raw_arxiv_id = raw_ref.get("arxiv_id")
            raw_alias = raw_ref.get("alias")
            if not isinstance(raw_arxiv_id, str) or not isinstance(raw_alias, str):
                raise ValueError
            candidate_keys.append(
                (
                    normalize_paper_id(raw_arxiv_id, kind="arxiv"),
                    normalize_paper_id(raw_alias),
                )
            )
    except ValueError:
        raise ValueError("identifier semantic audit inputs are invalid") from None
    if (
        len(set(candidate_keys)) != len(candidate_keys)
        or candidate_keys != sorted(candidate_keys)
        or len(candidate_keys) != len(observations)
        or set(candidate_keys) != set(observations)
        or any(arxiv_id not in gold_set for arxiv_id, _alias in candidate_keys)
    ):
        raise ValueError("identifier semantic audit inputs are invalid")

    relations: list[RelationAudit] = []
    for arxiv_id in gold_arxiv_ids:
        relations.append(
            classify_relation(
                alias=arxiv_anchor(arxiv_id),
                arxiv_id=arxiv_id,
                observation=None,
                relation_kind="required_anchor",
            )
        )
    for (arxiv_id, alias), observation in sorted(observations.items()):
        relations.append(
            classify_relation(
                alias=alias,
                arxiv_id=arxiv_id,
                observation=observation,
            )
        )
    relations = _resolve_alias_conflicts(relations)
    ordered_relations = tuple(
        sorted(
            relations,
            key=lambda row: (
                0 if row.relation_kind == "required_anchor" else 1,
                row.arxiv_id,
                row.alias,
            ),
        )
    )
    map_payload = {arxiv_id: arxiv_anchor(arxiv_id) for arxiv_id in gold_arxiv_ids}
    map_payload.update(
        {
            relation.alias: relation.terminal
            for relation in ordered_relations
            if relation.state == "verified"
            and relation.relation_kind == "provider_candidate"
            and relation.alias != relation.terminal
        }
    )
    map_bytes = _canonical_json(map_payload)
    audit = _aggregate_audit(
        map_bytes=map_bytes,
        gold_bytes=gold_bytes,
        evidence_bytes=evidence_bytes,
        manifest_hash=manifest_hash,
        gold_group_count=len(gold_arxiv_ids),
        relations=ordered_relations,
    )
    if audit.status != "passed":
        raise SemanticAuditFailure(audit, private_relations=ordered_relations)
    return RebuiltDevMap(
        map_payload=dict(sorted(map_payload.items())),
        private_relations=ordered_relations,
        audit=audit,
    )


def publish_semantic_audit(
    audit: IdentifierMapSemanticAudit, *, output_path: Path
) -> None:
    """Atomically write aggregate canonical JSON after privacy validation."""
    content = _canonical_json(audit.model_dump(mode="json"))
    assert_public_json_safe(content)
    _atomic_write(output_path, content)
    assert_public_json_safe(output_path.read_bytes())


def _write_private_outputs(
    result: RebuiltDevMap,
    *,
    out_map: Path,
    out_private_audit: Path,
) -> None:
    _atomic_write(out_map, _canonical_json(result.map_payload))
    _atomic_write(
        out_private_audit,
        _canonical_json(
            PrivateRelationAuditV2(
                schema_version="identifier-map-private-relation-audit-v2",
                scope="dev",
                relations=result.private_relations,
            ).model_dump(mode="json")
        ),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--out-map", type=Path, required=True)
    parser.add_argument("--out-private-audit", type=Path, required=True)
    parser.add_argument("--out-public-audit", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = rebuild_dev_map(
            gold_bytes=args.gold.read_bytes(),
            evidence_bytes=args.evidence.read_bytes(),
            snapshot_root=args.snapshot_root,
        )
    except SemanticAuditFailure as error:
        publish_semantic_audit(error.public_audit, output_path=args.out_public_audit)
        _atomic_write(
            args.out_private_audit,
            _canonical_json(
                PrivateRelationAuditV2(
                    schema_version="identifier-map-private-relation-audit-v2",
                    scope="dev",
                    relations=error.private_relations,
                ).model_dump(mode="json")
            ),
        )
        return 1
    _write_private_outputs(
        result,
        out_map=args.out_map,
        out_private_audit=args.out_private_audit,
    )
    publish_semantic_audit(result.audit, output_path=args.out_public_audit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
