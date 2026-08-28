"""Immutable offline PASA candidate receipts for fusion training augmentation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from paper_search.domain.models import Paper


def _canonical_json(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _write_immutable(path: Path, payload: bytes, *, label: str) -> None:
    if path.exists():
        if path.read_bytes() == payload:
            return
        raise FileExistsError(f"immutable {label} already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def write_pasa_supplement_receipt(
    *,
    output_root: Path,
    query_id: str,
    query: str,
    papers: Sequence[Paper],
    index_sha256: str,
    mixed_candidate_audit: Mapping[str, int | bool] | None = None,
) -> dict[str, Path]:
    """Write one additive, zero-request PASA receipt pair idempotently."""

    normalized_query = " ".join(query.split())
    if not query_id.strip() or not normalized_query or not papers:
        raise ValueError("PASA supplement receipt inputs must be non-empty")
    action_id = "pasa-local-original"
    bucket = hashlib.sha256(query_id.encode("utf-8")).hexdigest()[:4]
    base = output_root / "pasa" / f"bucket-{bucket}"
    generation_path = base / "generation" / "attempt-01" / f"{query_id}.json"
    retrieval_path = base / "retrieval" / "attempt-01" / f"{query_id}.json"
    provenance = {
        "candidate_source_namespace": "pasa_paper_database",
        "gold_visibility": "training_labels_only",
        "index_sha256": index_sha256,
        "llm_request_count": 0,
        "network_request_count": 0,
    }
    if mixed_candidate_audit is not None:
        positive_count = int(mixed_candidate_audit.get("positive_candidate_count", 0))
        negative_count = int(
            mixed_candidate_audit.get("lexical_negative_candidate_count", 0)
        )
        if (
            mixed_candidate_audit.get("mixed_positive_negative") is not True
            or positive_count <= 0
            or negative_count <= 0
        ):
            raise ValueError("mixed PASA receipt requires both positive and negative candidates")
        provenance.update(
            {
                "candidate_policy": "mixed_lexical_plus_gold_training",
                "source_label_leakage_guard": "positive_and_negative_share_action",
                "mixed_candidate_audit": dict(sorted(mixed_candidate_audit.items())),
            }
        )
    generation = {
        "query_id": query_id,
        "attempt_status": "succeeded",
        "actions": [
            {
                "action_id": action_id,
                "action_type": "text_search",
                "payload": {"query_text": normalized_query},
            }
        ],
        "generation_provenance": provenance,
    }
    retrieval = {
        "query_id": query_id,
        "attempt_status": "succeeded",
        "results": [
            {
                "action_id": action_id,
                "errors": [],
                "hits": [
                    paper.model_dump(exclude_none=True, exclude_computed_fields=True)
                    for paper in papers
                ],
                "infrastructure_failure": False,
            }
        ],
        "usage": {
            "llm_calls": 0,
            "search_api_calls": 0,
        },
        "retrieval_provenance": provenance,
    }
    _write_immutable(
        generation_path,
        _canonical_json(generation),
        label="PASA receipt",
    )
    _write_immutable(
        retrieval_path,
        _canonical_json(retrieval),
        label="PASA receipt",
    )
    return {"generation": generation_path, "retrieval": retrieval_path}


def write_pasa_augmented_handoff(
    *,
    base_handoff_path: Path,
    supplement_root: Path,
    output_path: Path,
    supplement_query_count: int,
    index_sha256: str,
) -> dict[str, Any]:
    """Append one immutable offline candidate root to a sealed training handoff."""

    payload = json.loads(base_handoff_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("base training handoff must be a JSON object")
    if payload.get("schema_version") != "openalex-ranking-training-handoff-v1":
        raise ValueError("unsupported base training handoff schema")
    if payload.get("online_llm_requests") != 0:
        raise ValueError("base training handoff used online LLM requests")
    if payload.get("test_partition_touched") is not False:
        raise ValueError("base training handoff touched the test partition")
    if supplement_query_count <= 0 or not supplement_root.is_dir():
        raise ValueError("PASA supplement root and query count are invalid")
    roots = payload.get("ordered_receipt_roots")
    if not isinstance(roots, list) or any(not isinstance(root, str) for root in roots):
        raise ValueError("base training handoff receipt roots are invalid")
    supplement = str(supplement_root.resolve())
    if supplement not in roots:
        roots.append(supplement)
    payload["ordered_receipt_roots"] = roots
    payload["pasa_offline_supplement"] = {
        "index_sha256": index_sha256,
        "llm_request_count": 0,
        "network_request_count": 0,
        "query_count": supplement_query_count,
        "receipt_root": supplement,
        "test_partition_touched": False,
    }
    output = _canonical_json(payload)
    _write_immutable(output_path, output, label="PASA augmented handoff")
    return payload


__all__ = [
    "write_pasa_augmented_handoff",
    "write_pasa_supplement_receipt",
]
