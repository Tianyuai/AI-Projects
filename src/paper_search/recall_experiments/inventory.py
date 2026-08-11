"""Offline, fail-closed inventory of historical candidate-recall evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml


STATUS_VALUES = frozenset({"exact", "aggregate_only", "insufficient", "not_comparable"})
EVIDENCE_KEYS = frozenset(
    {
        "actions",
        "provider_responses",
        "candidate_pool",
        "per_query_gold_hits",
        "aggregate_metrics",
        "prompt_model_visibility",
        "gold_denominator",
    }
)


class InventoryError(ValueError):
    """Raised when a historical source binding cannot be verified exactly."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative_path(workspace_root: Path, value: object) -> tuple[str, Path]:
    if not isinstance(value, str) or not value:
        raise InventoryError("source path must be a non-empty relative string")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise InventoryError(f"source path escapes workspace: {value}")
    resolved_root = workspace_root.resolve()
    resolved = (resolved_root / relative).resolve()
    if resolved_root not in resolved.parents and resolved != resolved_root:
        raise InventoryError(f"source path escapes workspace: {value}")
    return relative.as_posix(), resolved


def _require_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise InventoryError(f"{label} must be a mapping")
    return value


def _require_list(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise InventoryError(f"{label} must be a list")
    return value


def _load_yaml(path: Path) -> Mapping[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    return _require_mapping(loaded, str(path))


def _bound_source_records(
    binding: Mapping[str, Any], workspace_root: Path
) -> tuple[list[dict[str, str]], set[str]]:
    paths = _require_list(binding.get("source_paths"), "source_paths")
    hashes = _require_list(binding.get("source_sha256"), "source_sha256")
    if not paths or len(paths) != len(hashes):
        raise InventoryError("source_paths and source_sha256 must be non-empty and aligned")

    records: list[dict[str, str]] = []
    bound_paths: set[str] = set()
    for raw_path, expected_hash in zip(paths, hashes, strict=True):
        path_text, path = _relative_path(workspace_root, raw_path)
        if path_text in bound_paths:
            raise InventoryError(f"duplicate source path in binding: {path_text}")
        if not path.is_file():
            raise InventoryError(f"bound source is missing: {path_text}")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise InventoryError(f"invalid SHA-256 for {path_text}")
        actual_hash = _sha256(path)
        if actual_hash != expected_hash.lower().removeprefix("sha256:"):
            raise InventoryError(f"bound source hash changed: {path_text}")
        bound_paths.add(path_text)
        records.append({"path": path_text, "sha256": actual_hash})
    return records, bound_paths


def _validate_evidence(
    binding: Mapping[str, Any], bound_paths: set[str]
) -> dict[str, dict[str, Any]]:
    evidence = _require_mapping(binding.get("evidence"), "evidence")
    if set(evidence) != EVIDENCE_KEYS:
        raise InventoryError("evidence must declare every evidence class exactly once")
    result: dict[str, dict[str, Any]] = {}
    for kind in sorted(EVIDENCE_KEYS):
        item = _require_mapping(evidence[kind], f"evidence.{kind}")
        if not isinstance(item.get("available"), bool):
            raise InventoryError(f"evidence.{kind}.available must be boolean")
        item_path = item.get("path")
        if item_path is not None:
            if not isinstance(item_path, str) or item_path not in bound_paths:
                raise InventoryError(f"evidence.{kind} references an unbound source")
        if item["available"] and item_path is None:
            raise InventoryError(f"available evidence.{kind} must name a bound source")
        result[kind] = {"available": item["available"], "path": item_path}
    return result


def _catalog_binding(
    binding: Mapping[str, Any], workspace_root: Path
) -> tuple[dict[str, str], list[dict[str, str]]]:
    catalog = _require_mapping(binding.get("gold_catalog"), "gold_catalog")
    association = _require_mapping(catalog.get("association_source"), "association_source")
    association_path, association_file = _relative_path(workspace_root, association.get("path"))
    expected_association_hash = association.get("sha256")
    if not association_file.is_file() or not isinstance(expected_association_hash, str):
        raise InventoryError("gold association source is not bound")
    association_hash = _sha256(association_file)
    if association_hash != expected_association_hash.lower().removeprefix("sha256:"):
        raise InventoryError("gold association source hash changed")

    documents: list[dict[str, str]] = []
    seen_documents: set[str] = set()
    for document in _require_list(catalog.get("document_sources"), "document_sources"):
        item = _require_mapping(document, "document source")
        path_text, source_file = _relative_path(workspace_root, item.get("path"))
        expected_hash = item.get("sha256")
        kind = item.get("kind")
        if path_text in seen_documents or not source_file.is_file():
            raise InventoryError("gold document source is missing or duplicated")
        if not isinstance(expected_hash, str) or not isinstance(kind, str) or not kind:
            raise InventoryError("gold document source must include hash and kind")
        actual_hash = _sha256(source_file)
        if actual_hash != expected_hash.lower().removeprefix("sha256:"):
            raise InventoryError("gold document source hash changed")
        seen_documents.add(path_text)
        documents.append({"path": path_text, "sha256": actual_hash, "kind": kind})
    return (
        {"path": association_path, "sha256": association_hash, "kind": "association_only"},
        documents,
    )


def _read_json_lines(path: Path) -> Iterable[Mapping[str, Any]]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield _require_mapping(json.loads(line), str(path))


def _collect_query_ids(value: object, query_ids: set[str]) -> None:
    if isinstance(value, Mapping):
        query_id = value.get("query_id")
        if isinstance(query_id, str) and query_id:
            query_ids.add(query_id)
        declared_ids = value.get("query_ids")
        if isinstance(declared_ids, list):
            query_ids.update(
                query_id for query_id in declared_ids if isinstance(query_id, str) and query_id
            )
        for child in value.values():
            _collect_query_ids(child, query_ids)
    elif isinstance(value, list):
        for child in value:
            _collect_query_ids(child, query_ids)


def _source_query_ids(sources: list[dict[str, str]], workspace_root: Path) -> list[str]:
    query_ids: set[str] = set()
    for source in sources:
        _, path = _relative_path(workspace_root, source["path"])
        if path.suffix == ".json":
            _collect_query_ids(json.loads(path.read_text(encoding="utf-8")), query_ids)
        elif path.suffix == ".jsonl":
            for row in _read_json_lines(path):
                _collect_query_ids(row, query_ids)
    return sorted(query_ids)


def _snapshot_response_path(manifest_path: Path, workspace_root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise InventoryError("snapshot response path must be a non-empty string")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise InventoryError("snapshot response path escapes its manifest")
    response_path = (manifest_path.parent / relative).resolve()
    resolved_root = workspace_root.resolve()
    if resolved_root not in response_path.parents:
        raise InventoryError("snapshot response path escapes workspace")
    return response_path


def frozen_request_identities(manifest_path: Path, *, workspace_root: Path) -> list[str]:
    """Verify every response named by a snapshot manifest and return its request identities."""
    manifest = _require_mapping(
        json.loads(manifest_path.read_text(encoding="utf-8")), str(manifest_path)
    )
    entries = _require_list(manifest.get("entries"), "snapshot manifest entries")
    request_identities: set[str] = set()
    for entry in entries:
        item = _require_mapping(entry, "snapshot manifest entry")
        request = _require_mapping(item.get("request"), "snapshot request")
        request_identity = request.get("canonical_request_sha256")
        expected_hash = item.get("response_sha256")
        if not isinstance(request_identity, str) or not request_identity:
            raise InventoryError("snapshot entry lacks canonical request identity")
        if not isinstance(expected_hash, str):
            raise InventoryError("snapshot entry lacks response hash")
        response_path = _snapshot_response_path(manifest_path, workspace_root, item.get("response_path"))
        if not response_path.is_file():
            raise InventoryError("snapshot response is missing")
        if _sha256(response_path) != expected_hash.lower().removeprefix("sha256:"):
            raise InventoryError("snapshot response hash changed")
        request_identities.add(request_identity)
    if not request_identities:
        raise InventoryError("snapshot manifest contains no request identities")
    return sorted(request_identities)


def _document_index(
    document_sources: list[dict[str, str]], workspace_root: Path
) -> dict[str, dict[str, bool]]:
    documents: dict[str, dict[str, bool]] = {}
    for source in document_sources:
        if source["kind"] != "normalized_paper_records":
            continue
        _, path = _relative_path(workspace_root, source["path"])
        for outcome in _read_json_lines(path):
            for search in _require_list(outcome.get("searches", []), "searches"):
                data = _require_mapping(search, "search").get("data", [])
                for paper in _require_list(data, "paper data"):
                    record = _require_mapping(paper, "paper")
                    identifiers = [
                        record.get("canonical_id"),
                        record.get("arxiv_id"),
                        record.get("doi"),
                        record.get("openalex_id"),
                    ]
                    fields = {
                        "title": bool(str(record.get("title") or "").strip()),
                        "abstract": bool(str(record.get("abstract") or "").strip()),
                        "authors": bool(record.get("authors")),
                        "year": record.get("publication_year") is not None,
                    }
                    for identifier in identifiers:
                        if isinstance(identifier, str) and identifier:
                            documents[identifier.lower()] = fields
                            normalized = identifier.lower()
                            arxiv_doi_prefix = "doi:10.48550/arxiv."
                            if normalized.startswith(arxiv_doi_prefix):
                                documents[f"arxiv:{normalized.removeprefix(arxiv_doi_prefix)}"] = fields
    return documents


def _catalog_coverage(
    association: dict[str, str], document_sources: list[dict[str, str]], workspace_root: Path
) -> list[dict[str, Any]]:
    _, association_path = _relative_path(workspace_root, association["path"])
    documents = _document_index(document_sources, workspace_root)
    coverage: list[dict[str, Any]] = []
    for row in _read_json_lines(association_path):
        query_id = row.get("query_id")
        gold_ids = _require_list(row.get("relevant_paper_ids"), "relevant_paper_ids")
        if not isinstance(query_id, str):
            raise InventoryError("gold association row lacks query_id")
        fields = [documents.get(str(gold_id).lower(), {}) for gold_id in gold_ids]
        coverage.append(
            {
                "query_id": query_id,
                "gold_count": len(gold_ids),
                "all_gold_have_titles": bool(fields)
                and all(field.get("title", False) for field in fields),
                "title_count": sum(field.get("title", False) for field in fields),
                "abstract_count": sum(field.get("abstract", False) for field in fields),
                "author_count": sum(field.get("authors", False) for field in fields),
                "year_count": sum(field.get("year", False) for field in fields),
            }
        )
    return coverage


def classify_request(request_identity: str, frozen_request_identities: Iterable[str]) -> str:
    """Classify snapshot replay without querying a provider or local index."""
    if request_identity in set(frozen_request_identities):
        return "exact_snapshot_replay"
    return "requires_live_provider_or_local_index"


def build_inventory(config_root: Path, *, workspace_root: Path | None = None) -> dict[str, Any]:
    """Verify bindings and return a JSON-compatible report without mutable side effects."""
    config_root = config_root.resolve()
    root = (workspace_root or Path.cwd()).resolve()
    config_files = sorted(config_root.glob("*.yaml"))
    if not config_files:
        raise InventoryError("no historical bindings found")

    methods: list[dict[str, Any]] = []
    method_ids: set[str] = set()
    source_owners: dict[str, str] = {}
    association_source: dict[str, str] | None = None
    document_sources: list[dict[str, str]] = []
    frozen_requests: set[str] = set()
    for config_file in config_files:
        binding = _load_yaml(config_file)
        method_id = binding.get("method_id")
        status = binding.get("status")
        action_family = binding.get("action_family")
        policy = binding.get("candidate_pool_policy_version")
        if not isinstance(method_id, str) or not method_id:
            raise InventoryError(f"{config_file} lacks method identity")
        if not isinstance(action_family, str) or not action_family:
            raise InventoryError(f"{config_file} lacks action family")
        if not isinstance(policy, str) or not policy:
            raise InventoryError(f"{config_file} lacks method identity, family, or pool policy")
        if method_id in method_ids:
            raise InventoryError(f"duplicate method_id: {method_id}")
        if status not in STATUS_VALUES:
            raise InventoryError(f"invalid status: {status}")
        sources, bound_paths = _bound_source_records(binding, root)
        for source in sources:
            if source["path"] in source_owners:
                raise InventoryError(f"source bound by multiple methods: {source['path']}")
            source_owners[source["path"]] = method_id
        evidence = _validate_evidence(binding, bound_paths)
        provider_evidence = evidence["provider_responses"]
        if provider_evidence["available"]:
            provider_path = provider_evidence["path"]
            if not isinstance(provider_path, str):
                raise InventoryError("available provider response evidence lacks a source path")
            _, manifest_path = _relative_path(root, provider_path)
            frozen_requests.update(frozen_request_identities(manifest_path, workspace_root=root))
        catalog_association, catalog_documents = _catalog_binding(binding, root)
        if association_source is None:
            association_source = catalog_association
        elif association_source != catalog_association:
            raise InventoryError("historical bindings disagree on gold association source")
        for document in catalog_documents:
            if document not in document_sources:
                document_sources.append(document)
        declared_query_ids = _require_list(binding.get("query_ids"), "query_ids")
        if not all(isinstance(query_id, str) for query_id in declared_query_ids):
            raise InventoryError("query_ids must contain strings")
        source_query_ids = _source_query_ids(sources, root)
        if declared_query_ids and set(declared_query_ids) != set(source_query_ids):
            raise InventoryError("declared query_ids do not match the bound sources")
        missing = _require_list(binding.get("missing_or_misaligned"), "missing_or_misaligned")
        if not all(isinstance(message, str) for message in missing):
            raise InventoryError("missing_or_misaligned must contain strings")
        methods.append(
            {
                "method_id": method_id,
                "source_paths": [source["path"] for source in sources],
                "source_sha256": [source["sha256"] for source in sources],
                "query_ids": source_query_ids,
                "action_family": action_family,
                "actions_evidence": evidence["actions"],
                "provider_response_evidence": evidence["provider_responses"],
                "candidate_pool_evidence": evidence["candidate_pool"],
                "gold_denominator_evidence": evidence["gold_denominator"],
                "metric_evidence": evidence["aggregate_metrics"],
                "generation_recipe_reconstructability": binding.get(
                    "generation_recipe_reconstructability"
                ),
                "candidate_pool_policy_version": policy,
                "status": status,
                "missing_or_misaligned": missing,
                "evidence": evidence,
            }
        )
        method_ids.add(method_id)

    if association_source is None:
        raise InventoryError("no gold association source bound")
    coverage = _catalog_coverage(association_source, document_sources, root)
    exact_methods = [
        method
        for method in methods
        if method["status"] == "exact"
        and method["actions_evidence"]["available"]
        and method["provider_response_evidence"]["available"]
    ]
    decisions: list[str] = []
    if exact_methods:
        decisions.append("core_framework_ready")
    else:
        decisions.append("core_framework_blocked")
    if len(exact_methods) >= 2 and {method["action_family"] for method in exact_methods} >= {
        "text_search",
        "title_search",
    }:
        decisions.append("overall_compatibility_ready")
    oracle_catalog_ready = all(item["all_gold_have_titles"] for item in coverage)
    if not oracle_catalog_ready:
        decisions.append("oracle_catalog_blocked")
    return {
        "schema_version": "candidate-recall-history-inventory-v1",
        "source_statuses": sorted(STATUS_VALUES),
        "methods": methods,
        "gold_catalog": {
            "association_source": association_source,
            "document_sources": document_sources,
            "per_query_coverage": coverage,
            "oracle_catalog_ready": oracle_catalog_ready,
        },
        "backend_classifications": {
            "frozen_request": "exact_snapshot_replay",
            "novel_request": "requires_live_provider_or_local_index",
            "frozen_requests": [
                {"request_identity": request, "classification": "exact_snapshot_replay"}
                for request in sorted(frozen_requests)
            ],
        },
        "continuation_decisions": decisions,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    report = build_inventory(args.config_root)
    output_directory = args.out.resolve()
    expected_output = Path.cwd().resolve() / "runs" / "_recall_history_inventory"
    if output_directory != expected_output:
        raise InventoryError(
            "inventory output must be runs/_recall_history_inventory/source-inventory.json"
        )
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "source-inventory.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
