"""Streaming, local trainability audit for saved OpenAlex receipts.

The audit deliberately stores only normalized file contributions in its state.  Raw
receipt payloads remain the source of truth and are never emitted by the CLI.
Unchanged receipt files are not parsed again on later runs.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from paper_search.domain.models import Paper
from paper_search.evaluation.predictions import (
    paper_evaluation_aliases,
    paper_id_aliases,
)
from paper_search.learning.openalex_daily_schedule import search_action_identity


SCHEMA_VERSION = "openalex-receipt-trainability-audit-v2"
_PREVIOUS_SCHEMA_VERSION = "openalex-receipt-trainability-audit-v1"
_STATUS_PRIORITY = {
    "usable": 5,
    "partial": 4,
    "no_results": 3,
    "failed": 2,
    "invalid_hit": 1,
    "malformed_result": 0,
    "missing_result": 0,
}


def _sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _partition(path: Path) -> tuple[dict[str, dict[str, Any]], str]:
    rows: dict[str, dict[str, Any]] = {}
    digest = hashlib.sha256()
    for line in path.read_bytes().splitlines(keepends=True):
        digest.update(line)
        if not line.strip():
            continue
        row = json.loads(line)
        query_id = str(row.get("query_id", ""))
        if not query_id or query_id in rows:
            raise ValueError("training partition contains an invalid or duplicate query_id")
        rows[query_id] = row
    if not rows:
        raise ValueError("training partition is empty")
    return rows, "sha256:" + digest.hexdigest()


def _generation_path(retrieval_path: Path) -> Path:
    parts = list(retrieval_path.parts)
    try:
        index = parts.index("retrieval")
    except ValueError:
        return retrieval_path
    parts[index] = "generation"
    return Path(*parts)


def _error_codes(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    output: list[str] = []
    for item in value:
        if isinstance(item, dict) and isinstance(item.get("code"), str):
            output.append(item["code"])
        elif isinstance(item, str):
            output.append(item)
        else:
            output.append("unknown")
    return output


def _action_identity(action: object) -> tuple[str, str, str] | None:
    if not isinstance(action, dict):
        return None
    try:
        identity = search_action_identity(action)
    except (TypeError, ValueError):
        return None
    if identity is None:
        return None
    return (identity.action_type, identity.search_mode, identity.normalized_text)


def _parse_retrieval(path: Path) -> dict[str, Any]:
    """Parse one retrieval file into a compact contribution."""

    payload = _read_json(path)
    if not isinstance(payload, dict):
        raise ValueError("retrieval receipt must be an object")
    query_id = str(payload.get("query_id", ""))
    if not query_id:
        raise ValueError("retrieval receipt has no query_id")
    generation_file = _generation_path(path)
    generation_actions: dict[str, tuple[str, str, str] | None] = {}
    generation_status = "missing"
    if generation_file.is_file():
        generation = _read_json(generation_file)
        if not isinstance(generation, dict):
            generation_status = "invalid"
        else:
            generation_status = str(generation.get("attempt_status", "unknown"))
            actions = generation.get("actions", [])
            if isinstance(actions, list):
                for action in actions:
                    if isinstance(action, dict) and isinstance(action.get("action_id"), str):
                        generation_actions[action["action_id"]] = _action_identity(action)
            else:
                generation_status = "invalid"

    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError("retrieval receipt results must be a list")
    records: list[dict[str, Any]] = []
    error_codes: Counter[str] = Counter()
    invalid_identity_count = 0
    invalid_hit_count = 0
    valid_hit_count = 0
    for result in results:
        if not isinstance(result, dict):
            records.append({"action_key": f"invalid-result-{len(records)}", "status": "malformed_result", "paper_ids": []})
            continue
        action_id = str(result.get("action_id", ""))
        identity = generation_actions.get(action_id)
        if identity is None:
            invalid_identity_count += 1
            action_key = f"unpaired:{action_id or len(records)}"
        else:
            action_key = json.dumps(identity, ensure_ascii=False, separators=(",", ":"))
        errors = _error_codes(result.get("errors"))
        error_codes.update(errors)
        hits = result.get("hits")
        if not isinstance(hits, list):
            records.append({"action_key": action_key, "status": "malformed_result", "paper_ids": []})
            continue
        paper_ids: list[str] = []
        paper_aliases: set[str] = set()
        invalid_for_result = 0
        for hit in hits:
            try:
                paper = Paper.model_validate(hit)
            except Exception:
                invalid_hit_count += 1
                invalid_for_result += 1
                continue
            valid_hit_count += 1
            paper_aliases.update(paper_evaluation_aliases(paper))
            if paper.canonical_id not in paper_ids:
                paper_ids.append(paper.canonical_id)
        infrastructure_failure = result.get("infrastructure_failure") is True
        if invalid_for_result:
            status = "invalid_hit"
        elif infrastructure_failure or errors:
            status = "partial" if paper_ids else "failed"
        elif paper_ids:
            status = "usable"
        else:
            status = "no_results"
        records.append(
            {
                "action_key": action_key,
                "status": status,
                "paper_ids": paper_ids,
                "paper_aliases": sorted(paper_aliases),
            }
        )

    return {
        "query_id": query_id,
        "generation_status": generation_status,
        "generated_action_count": len(generation_actions),
        "result_count": len(results),
        "invalid_identity_count": invalid_identity_count,
        "valid_hit_count": valid_hit_count,
        "invalid_hit_count": invalid_hit_count,
        "error_codes": dict(sorted(error_codes.items())),
        "actions": records,
    }


def _file_key(path: Path) -> str:
    return str(path.resolve())


def _is_under(path: str, root: Path) -> bool:
    try:
        Path(path).resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _load_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"schema_version": SCHEMA_VERSION, "files": {}}
    state = _read_json(path)
    if not isinstance(state, dict):
        raise ValueError("receipt audit state schema is incompatible")
    if state.get("schema_version") == _PREVIOUS_SCHEMA_VERSION:
        return {
            "schema_version": SCHEMA_VERSION,
            "files": {},
            "receipt_roots": state.get("receipt_roots", []),
        }
    if state.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("receipt audit state schema is incompatible")
    if not isinstance(state.get("files", {}), dict):
        raise ValueError("receipt audit state files are invalid")
    return state


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def audit_openalex_receipts(
    *,
    receipt_roots: list[Path],
    partition_path: Path,
    state_path: Path,
    include_query_rows: bool = False,
) -> dict[str, Any]:
    """Audit receipt files and update an incremental compact state file."""

    if not receipt_roots:
        raise ValueError("at least one receipt root is required")
    partition, partition_sha256 = _partition(partition_path)
    state = _load_state(state_path)
    old_files: dict[str, Any] = state.get("files", {})
    prior_roots = [
        Path(value)
        for value in state.get("receipt_roots", [])
        if isinstance(value, str)
    ]
    effective_roots = list(
        dict.fromkeys([*(root.resolve() for root in prior_roots), *(root.resolve() for root in receipt_roots)])
    )
    # Preserve prior daily windows when the caller supplies only today's root.
    # Files under a supplied root are reconciled, so deleted/changed files there
    # do not silently remain in the aggregate.
    current_files: dict[str, Any] = {
        key: value
        for key, value in old_files.items()
        if not any(_is_under(key, root) for root in effective_roots)
    }
    old_in_scope = {
        key for key in old_files if any(_is_under(key, root) for root in effective_roots)
    }
    current_scope_keys: set[str] = set()
    files_new = files_reused = 0
    for root in effective_roots:
        if not root.is_dir():
            raise ValueError(f"receipt root is unavailable: {root}")
        for path in sorted(root.rglob("retrieval/attempt-01/*.json")):
            key = _file_key(path)
            current_scope_keys.add(key)
            digest = _sha256_bytes(path.read_bytes())
            previous = old_files.get(key)
            if isinstance(previous, dict) and previous.get("sha256") == digest:
                current_files[key] = previous
                files_reused += 1
                continue
            files_new += 1
            try:
                contribution = _parse_retrieval(path)
                parse_error = None
            except Exception as exc:
                contribution = {"query_id": "", "actions": [], "generated_action_count": 0, "result_count": 0, "error_codes": {}}
                parse_error = type(exc).__name__
            current_files[key] = {"sha256": digest, "parse_error": parse_error, "contribution": contribution}

    state = {
        "schema_version": SCHEMA_VERSION,
        "partition_sha256": partition_sha256,
        "receipt_roots": sorted(
            set(state.get("receipt_roots", []))
            | {str(root.resolve()) for root in effective_roots}
        ),
        "files": current_files,
    }
    _write_json(state_path, state)

    all_contributions = [
        entry.get("contribution", {})
        for entry in current_files.values()
        if isinstance(entry, dict) and isinstance(entry.get("contribution"), dict)
    ]
    action_observations: list[dict[str, Any]] = []
    queries_observed: set[str] = set()
    generated_actions = result_observations = valid_hits = invalid_hits = 0
    invalid_identities = 0
    generation_missing = generation_invalid = generated_without_result = 0
    error_counts: Counter[str] = Counter()
    for contribution in all_contributions:
        query_id = str(contribution.get("query_id", ""))
        if query_id:
            queries_observed.add(query_id)
        generated_actions += int(contribution.get("generated_action_count", 0))
        result_observations += int(contribution.get("result_count", 0))
        generation_status = contribution.get("generation_status")
        generation_missing += generation_status == "missing"
        generation_invalid += generation_status == "invalid"
        generated_without_result += max(
            0,
            int(contribution.get("generated_action_count", 0))
            - int(contribution.get("result_count", 0)),
        )
        valid_hits += int(contribution.get("valid_hit_count", 0))
        invalid_hits += int(contribution.get("invalid_hit_count", 0))
        invalid_identities += int(contribution.get("invalid_identity_count", 0))
        error_counts.update(contribution.get("error_codes", {}))
        for action in contribution.get("actions", []):
            if isinstance(action, dict):
                action_observations.append({"query_id": query_id, **action})

    selected: dict[tuple[str, str], dict[str, Any]] = {}
    for record in action_observations:
        key = (str(record.get("query_id", "")), str(record.get("action_key", "")))
        current = selected.get(key)
        if current is None or (_STATUS_PRIORITY.get(str(record.get("status")), -1), str(record.get("paper_ids", []))) > (
            _STATUS_PRIORITY.get(str(current.get("status")), -1), str(current.get("paper_ids", []))
        ):
            selected[key] = record

    candidates_by_query: dict[str, set[str]] = {}
    candidate_aliases_by_query: dict[str, set[str]] = {}
    status_counts: Counter[str] = Counter()
    for record in selected.values():
        status = str(record.get("status", "unknown"))
        status_counts[status] += 1
        if status in {"usable", "partial"}:
            query_id = str(record.get("query_id", ""))
            candidates_by_query.setdefault(query_id, set()).update(
                str(value) for value in record.get("paper_ids", [])
            )
            candidate_aliases_by_query.setdefault(query_id, set()).update(
                str(value) for value in record.get("paper_aliases", [])
            )
    candidate_ready = {query_id for query_id, ids in candidates_by_query.items() if ids}
    aligned = set(partition) & queries_observed
    auto_train = {
        query_id
        for query_id, row in partition.items()
        if row.get("role") == "training" and row.get("split") == "auto_train"
    }
    non_auto_train = len(partition) - len(auto_train)
    gold_matches_by_query: dict[str, set[str]] = {}
    for query_id in candidate_ready & auto_train:
        candidate_aliases = candidate_aliases_by_query.get(query_id, set()) | {
            alias
            for candidate_id in candidates_by_query[query_id]
            for alias in paper_id_aliases(candidate_id)
        }
        gold_matches_by_query[query_id] = {
            gold_id
            for gold_id in partition[query_id].get("gold_paper_ids", [])
            if candidate_aliases & paper_id_aliases(gold_id)
        }
    trainable = {
        query_id for query_id, matches in gold_matches_by_query.items() if matches
    }
    digest_rows = [
        {
            "query_id": query_id,
            "action_key": action_key,
            "status": record.get("status"),
            "paper_ids": sorted(record.get("paper_ids", [])),
            "paper_aliases": sorted(record.get("paper_aliases", [])),
        }
        for (query_id, action_key), record in sorted(selected.items())
    ]
    deterministic_digest = _sha256_bytes(json.dumps(digest_rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "partition_sha256": partition_sha256,
        "receipt_roots": list(state["receipt_roots"]),
        "files_scanned": len(current_files),
        "files_new": files_new,
        "files_reused": files_reused,
        "files_removed": len(old_in_scope - current_scope_keys),
        "parse_error_count": sum(1 for entry in current_files.values() if entry.get("parse_error")),
        "queries_observed": len(queries_observed),
        "queries_aligned_to_partition": len(aligned),
        "unknown_query_count": len(queries_observed - set(partition)),
        "partition_row_count": len(partition),
        "non_auto_train_row_count": non_auto_train,
        "generated_action_observations": generated_actions,
        "retrieval_result_observations": result_observations,
        "generation_missing_count": generation_missing,
        "generation_invalid_count": generation_invalid,
        "generated_action_without_result_count": generated_without_result,
        "unique_action_count": len(selected),
        "duplicate_action_observation_count": max(0, len(action_observations) - len(selected)),
        "invalid_action_identity_count": invalid_identities,
        "action_status_counts": dict(sorted(status_counts.items())),
        "valid_paper_hit_count": valid_hits,
        "invalid_paper_hit_count": invalid_hits,
        "unique_candidate_count": sum(len(ids) for ids in candidates_by_query.values()),
        "candidate_ready_query_count": len(candidate_ready),
        "gold_metadata_query_count": len(auto_train & queries_observed),
        "gold_hit_query_count": len(trainable),
        "gold_hit_count": sum(len(matches) for matches in gold_matches_by_query.values()),
        "trainable_query_count": len(trainable),
        "error_code_counts": dict(sorted(error_counts.items())),
        "deterministic_digest": deterministic_digest,
        "network_calls": 0,
        "llm_calls": 0,
        "test_partition_touched": False,
    }
    if include_query_rows:
        result["query_rows"] = [
            {
                "query_id": query_id,
                "candidate_count": len(candidates_by_query.get(query_id, set())),
                "gold_paper_count": len(
                    partition[query_id].get("gold_paper_ids", [])
                ),
                "gold_hit_count": len(gold_matches_by_query.get(query_id, set())),
                "hard_negative_candidate_count": max(
                    0,
                    len(candidates_by_query.get(query_id, set()))
                    - len(gold_matches_by_query.get(query_id, set())),
                ),
                "positive_and_hard_negative": bool(
                    gold_matches_by_query.get(query_id)
                    and len(candidates_by_query.get(query_id, set()))
                    > len(gold_matches_by_query.get(query_id, set()))
                ),
            }
            for query_id in sorted(auto_train & queries_observed)
        ]
    return result


__all__ = ["SCHEMA_VERSION", "audit_openalex_receipts"]
