"""Offline normalization of the five hash-bound historical recall sources.

This adapter deliberately separates replayable actions and frozen responses from
aggregate reports.  It never infers an action, candidate ID, or Gold hit that is
not preserved by the bound evidence.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Literal, cast

import yaml

from paper_search.domain.models import DomainModel, Paper
from paper_search.evaluation.dataset import EvaluationQuery, read_jsonl
from paper_search.evaluation.query_evolution_probe import offline_provider_result
from paper_search.recall_experiments.candidate_pool import CandidatePoolBuilder
from paper_search.recall_experiments.contracts import RetrievalActionResult
from paper_search.recall_experiments.inputs.base import (
    FrozenRecallDataset,
    HistoricalRecallBaseline,
    OpaqueEvaluationMaterials,
)


HistoricalTerminalState = Literal[
    "exact_replay_passed",
    "aggregate_only",
    "not_comparable",
    "insufficient_historical_evidence",
]
PerQueryEquality = Literal["passed", "failed", "not_provable", "not_comparable"]


class HistoricalReplayError(ValueError):
    """Raised when a historical binding or its Task 0 inventory has drifted."""


class HistoricalMethodReplay(DomainModel):
    """One normalized historical method with only its proven evidence exposed."""

    method_id: str
    source_run_id: str
    source_hashes: dict[str, str]
    query_ids_available: list[str]
    evidence_level: Literal["exact", "aggregate_only", "insufficient", "not_comparable"]
    candidate_pool_policy_version: str
    fixed_actions: dict[str, dict[str, object]] | None = None
    frozen_dataset: FrozenRecallDataset | None = None
    historical_baseline: HistoricalRecallBaseline | None = None
    candidate_pool_ids_by_query: dict[str, tuple[str, ...]] = {}
    gold_hit_ids_by_query: dict[str, tuple[str, ...]] | None = None
    aggregate_metrics: dict[str, object] | None = None
    terminal_state: HistoricalTerminalState
    per_query_equality: PerQueryEquality
    semantic_mismatch: str | None = None
    unprovable_fields: list[str]


class HistoricalReplaySet(DomainModel):
    """All required historical methods and the evidence-bounded Scheme B result."""

    methods: dict[str, HistoricalMethodReplay]
    scheme_b_terminal_state: HistoricalTerminalState


_METHOD_IDS = frozenset(
    {
        "query-rewrite",
        "llm-query-variants",
        "query-evolution",
        "title-candidates",
        "citation-expansion",
    }
)
_EMPTY_IDENTIFIER_MAP = b"{}"
_EMPTY_IDENTIFIER_MAP_SHA256 = "sha256:" + hashlib.sha256(_EMPTY_IDENTIFIER_MAP).hexdigest()


def load_historical_replays(
    *, inventory_path: Path, config_root: Path, workspace_root: Path | None = None
) -> HistoricalReplaySet:
    """Verify Task 0 bindings and normalize only the preserved historical facts."""
    root = (workspace_root or Path.cwd()).resolve()
    inventory = _load_inventory(inventory_path)
    inventory_methods = _inventory_methods(inventory)
    bindings = _load_bindings(config_root, root)
    if set(inventory_methods) != _METHOD_IDS or set(bindings) != _METHOD_IDS:
        raise HistoricalReplayError("historical inventory must bind exactly the five candidate methods")

    normalized: dict[str, HistoricalMethodReplay] = {}
    for method_id in sorted(_METHOD_IDS):
        record = inventory_methods[method_id]
        binding = bindings[method_id]
        sources = _verified_sources(record, binding, root)
        query_ids = _string_list(record.get("query_ids"), "inventory query_ids")
        dataset = _frozen_dataset(query_ids, sources, binding, root)
        evidence_level = _evidence_level(record)
        if method_id == "query-evolution":
            normalized[method_id] = _query_evolution_replay(
                record=record,
                sources=sources,
                dataset=dataset,
                evidence_level=evidence_level,
            )
        else:
            normalized[method_id] = _aggregate_replay(
                method_id=method_id,
                record=record,
                sources=sources,
                dataset=dataset,
                evidence_level=evidence_level,
            )
    return HistoricalReplaySet(
        methods=normalized,
        scheme_b_terminal_state=_scheme_b_terminal_state(normalized.values()),
    )


def _load_inventory(path: Path) -> Mapping[str, object]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HistoricalReplayError("Task 0 source inventory is unreadable") from error
    if not isinstance(loaded, Mapping):
        raise HistoricalReplayError("Task 0 source inventory must be an object")
    return loaded


def _inventory_methods(inventory: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    methods = inventory.get("methods")
    if not isinstance(methods, list):
        raise HistoricalReplayError("Task 0 source inventory lacks methods")
    result: dict[str, Mapping[str, object]] = {}
    for item in methods:
        if not isinstance(item, Mapping) or not isinstance(item.get("method_id"), str):
            raise HistoricalReplayError("Task 0 source inventory contains an invalid method")
        method_id = item["method_id"]
        if method_id in result:
            raise HistoricalReplayError("Task 0 source inventory has duplicate methods")
        result[method_id] = item
    return result


def _load_bindings(config_root: Path, root: Path) -> dict[str, Mapping[str, object]]:
    if not config_root.is_dir():
        raise HistoricalReplayError("historical config root is missing")
    bindings: dict[str, Mapping[str, object]] = {}
    for path in sorted(config_root.glob("*.yaml")):
        try:
            value = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as error:
            raise HistoricalReplayError("historical binding is unreadable") from error
        if not isinstance(value, Mapping) or not isinstance(value.get("method_id"), str):
            raise HistoricalReplayError("historical binding lacks method_id")
        method_id = value["method_id"]
        if method_id in bindings:
            raise HistoricalReplayError("historical bindings have duplicate methods")
        bindings[method_id] = value
    return bindings


def _verified_sources(
    record: Mapping[str, object], binding: Mapping[str, object], root: Path
) -> dict[str, Path]:
    inventory_paths = _string_list(record.get("source_paths"), "inventory source_paths")
    inventory_hashes = _string_list(record.get("source_sha256"), "inventory source_sha256")
    binding_paths = _string_list(binding.get("source_paths"), "binding source_paths")
    binding_hashes = _string_list(binding.get("source_sha256"), "binding source_sha256")
    if inventory_paths != binding_paths or inventory_hashes != binding_hashes:
        raise HistoricalReplayError("inventory hash mismatch: binding source record")
    if not inventory_paths or len(inventory_paths) != len(inventory_hashes):
        raise HistoricalReplayError("historical source paths and hashes are not aligned")
    result: dict[str, Path] = {}
    for source_path, expected_hash in zip(inventory_paths, inventory_hashes, strict=True):
        path = _workspace_path(root, source_path)
        if not path.is_file():
            raise HistoricalReplayError("bound historical source is missing")
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_hash != _hash_value(expected_hash):
            raise HistoricalReplayError(f"inventory hash mismatch: {source_path}")
        result[source_path] = path
    return result


def _frozen_dataset(
    query_ids: list[str],
    sources: Mapping[str, Path],
    binding: Mapping[str, object],
    root: Path,
) -> FrozenRecallDataset:
    catalog = binding.get("gold_catalog")
    if not isinstance(catalog, Mapping) or not isinstance(catalog.get("association_source"), Mapping):
        raise HistoricalReplayError("historical binding lacks a Gold association source")
    association = catalog["association_source"]
    path_value = association.get("path")
    hash_value = association.get("sha256")
    if not isinstance(path_value, str) or not isinstance(hash_value, str):
        raise HistoricalReplayError("historical Gold association source is malformed")
    association_path = _workspace_path(root, path_value)
    association_bytes = association_path.read_bytes()
    if hashlib.sha256(association_bytes).hexdigest() != _hash_value(hash_value):
        raise HistoricalReplayError("historical Gold association hash mismatch")
    all_queries = read_jsonl(association_path, EvaluationQuery)
    by_id = {query.query_id: query for query in all_queries}
    if any(query_id not in by_id for query_id in query_ids):
        raise HistoricalReplayError("historical source includes a query absent from Gold associations")
    selected = [by_id[query_id] for query_id in query_ids]
    source_hashes = {
        **{
            f"historical_source_{index}": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
            for index, path in enumerate(sources.values())
        },
        "gold_associations": "sha256:" + hashlib.sha256(association_bytes).hexdigest(),
    }
    return FrozenRecallDataset(
        queries=selected,
        source_hashes=source_hashes,
        evaluation_materials=OpaqueEvaluationMaterials(
            gold_records=selected,
            identifier_map_bytes=_EMPTY_IDENTIFIER_MAP,
            identifier_map_sha256=_EMPTY_IDENTIFIER_MAP_SHA256,
        ),
        seed_candidates=[],
    )


def _query_evolution_replay(
    *,
    record: Mapping[str, object],
    sources: Mapping[str, Path],
    dataset: FrozenRecallDataset,
    evidence_level: Literal["exact", "aggregate_only", "insufficient", "not_comparable"],
) -> HistoricalMethodReplay:
    outcomes_path = _source_with_name(sources, "outcomes.jsonl")
    outcomes = [_mapping(json.loads(line), "query-evolution outcome") for line in outcomes_path.read_text(encoding="utf-8").splitlines() if line]
    outcome_by_id = {_required_string(item, "query_id"): item for item in outcomes}
    query_ids = [query.query_id for query in dataset.queries]
    if set(outcome_by_id) != set(query_ids):
        raise HistoricalReplayError("query-evolution outcomes do not match the normalized query slice")
    actions: dict[str, dict[str, object]] = {}
    pools: dict[str, tuple[str, ...]] = {}
    for query_id in query_ids:
        outcome = outcome_by_id[query_id]
        proposal = _mapping(outcome.get("proposal"), "query-evolution proposal")
        subqueries = _list_of_mappings(proposal.get("subqueries"), "query-evolution subqueries")
        searches = _list_of_mappings(outcome.get("searches"), "query-evolution searches")
        if len(subqueries) != len(searches):
            raise HistoricalReplayError("query-evolution actions do not align with frozen responses")
        action_rows: list[dict[str, object]] = []
        results: list[RetrievalActionResult] = []
        for index, (subquery, search) in enumerate(zip(subqueries, searches, strict=True), start=1):
            action_id = f"query-evolution-{index:02d}"
            strategy = _required_string(subquery, "strategy")
            query_text = _required_string(subquery, "text")
            action_rows.append(
                {
                    "action_id": action_id,
                    "action_type": "text_search",
                    "strategy": strategy,
                    "payload": {"query_text": query_text},
                }
            )
            papers = [Paper.model_validate(item) for item in _list_of_mappings(search.get("data"), "query-evolution response data")]
            response = offline_provider_result(papers)
            results.append(
                RetrievalActionResult(
                    action_id=action_id,
                    action_type="text_search",
                    hits=response.data,
                    usage=response.usage,
                    provenance={str(key): str(value) for key, value in response.provenance.items()},
                    errors=response.errors,
                )
            )
        actions[query_id] = {"actions": action_rows}
        pool = CandidatePoolBuilder("canonical-id-first-v1").build(query_id, results)
        pools[query_id] = tuple(entry.paper.canonical_id for entry in pool.entries)
    lock = _load_json(_source_with_name(sources, "probe.lock.json"))
    return HistoricalMethodReplay(
        method_id="query-evolution",
        source_run_id=_required_string(lock, "source_run_id"),
        source_hashes=_source_hashes(sources),
        query_ids_available=query_ids,
        evidence_level=evidence_level,
        candidate_pool_policy_version="canonical-id-first-v1",
        fixed_actions=actions,
        frozen_dataset=dataset,
        candidate_pool_ids_by_query=pools,
        aggregate_metrics=_query_evolution_metrics(_load_json(_source_with_name(sources, "result.json"))),
        terminal_state="not_comparable",
        per_query_equality="not_comparable",
        semantic_mismatch=(
            "The bound source lacks an identifier-map artifact and per-query Gold-hit records, "
            "so canonical response IDs cannot be compared with Gold associations."
        ),
        unprovable_fields=[
            "per_query_gold_hits",
            "total_gold_associations",
            "macro_candidate_recall",
        ],
    )


def _aggregate_replay(
    *,
    method_id: str,
    record: Mapping[str, object],
    sources: Mapping[str, Path],
    dataset: FrozenRecallDataset,
    evidence_level: Literal["exact", "aggregate_only", "insufficient", "not_comparable"],
) -> HistoricalMethodReplay:
    report = _load_json(next(iter(sources.values())))
    terminal: HistoricalTerminalState = (
        "insufficient_historical_evidence"
        if evidence_level == "insufficient"
        else "not_comparable"
        if evidence_level == "not_comparable"
        else "aggregate_only"
    )
    unavailable = ["fixed_actions", "provider_responses", "per_query_candidate_ids"]
    if terminal == "aggregate_only":
        unavailable.extend(["per_query_candidate_equality", "per_query_gold_hit_equality"])
    else:
        unavailable.append("aggregate_comparison")
    return HistoricalMethodReplay(
        method_id=method_id,
        source_run_id=_required_string(report, "run_id"),
        source_hashes=_source_hashes(sources),
        query_ids_available=[query.query_id for query in dataset.queries],
        evidence_level=evidence_level,
        candidate_pool_policy_version=_required_string(record, "candidate_pool_policy_version"),
        frozen_dataset=dataset,
        aggregate_metrics=_aggregate_metrics(report),
        terminal_state=terminal,
        per_query_equality="not_provable",
        unprovable_fields=unavailable,
    )


def _aggregate_metrics(report: Mapping[str, object]) -> dict[str, object]:
    variants = report.get("variants")
    if isinstance(variants, (dict, list)):
        return {"variants": variants}
    summary = report.get("summary")
    if isinstance(summary, Mapping):
        return {"summary": dict(summary)}
    raise HistoricalReplayError("aggregate historical report lacks stored aggregate metrics")


def _query_evolution_metrics(report: Mapping[str, object]) -> dict[str, object]:
    public_report = report.get("public_report")
    if not isinstance(public_report, Mapping):
        raise HistoricalReplayError("query-evolution result lacks a public report")
    return {"public_report": dict(public_report)}


def _scheme_b_terminal_state(methods: Iterable[HistoricalMethodReplay]) -> HistoricalTerminalState:
    exact = [method for method in methods if isinstance(method, HistoricalMethodReplay) and method.terminal_state == "exact_replay_passed"]
    families = {method.method_id for method in exact}
    if len(exact) >= 2 and "query-evolution" in families and any(
        method.method_id in {"title-candidates", "citation-expansion"} for method in exact
    ):
        return "exact_replay_passed"
    return "insufficient_historical_evidence"


def _evidence_level(record: Mapping[str, object]) -> Literal["exact", "aggregate_only", "insufficient", "not_comparable"]:
    value = record.get("evidence_level", record.get("status"))
    if value not in {"exact", "aggregate_only", "insufficient", "not_comparable"}:
        raise HistoricalReplayError("inventory evidence level is invalid")
    return cast(Literal["exact", "aggregate_only", "insufficient", "not_comparable"], value)


def _source_hashes(sources: Mapping[str, Path]) -> dict[str, str]:
    return {path: "sha256:" + hashlib.sha256(source.read_bytes()).hexdigest() for path, source in sources.items()}


def _source_with_name(sources: Mapping[str, Path], name: str) -> Path:
    matches = [path for path_text, path in sources.items() if path_text.endswith(name)]
    if len(matches) != 1:
        raise HistoricalReplayError(f"historical source does not bind exactly one {name}")
    return matches[0]


def _load_json(path: Path) -> Mapping[str, object]:
    try:
        return _mapping(json.loads(path.read_text(encoding="utf-8")), str(path))
    except (OSError, json.JSONDecodeError) as error:
        raise HistoricalReplayError("historical JSON source is unreadable") from error


def _workspace_path(root: Path, value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise HistoricalReplayError("historical source path escapes the workspace")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise HistoricalReplayError("historical source path escapes the workspace")
    return path


def _hash_value(value: str) -> str:
    normalized = value.lower().removeprefix("sha256:")
    if len(normalized) != 64:
        raise HistoricalReplayError("historical source hash is invalid")
    return normalized


def _string_list(value: object, label: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise HistoricalReplayError(f"{label} must be a non-empty string list")
    return list(value)


def _required_string(value: Mapping[str, object], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise HistoricalReplayError(f"historical source lacks {key}")
    return item


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise HistoricalReplayError(f"{label} must be an object")
    return value


def _list_of_mappings(value: object, label: str) -> list[Mapping[str, object]]:
    if not isinstance(value, list) or not all(isinstance(item, Mapping) for item in value):
        raise HistoricalReplayError(f"{label} must be a list of objects")
    return list(value)


__all__ = [
    "HistoricalMethodReplay",
    "HistoricalReplayError",
    "HistoricalReplaySet",
    "HistoricalTerminalState",
    "PerQueryEquality",
    "load_historical_replays",
]
