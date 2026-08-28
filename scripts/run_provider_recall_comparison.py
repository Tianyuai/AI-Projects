"""Run identical CPU-generated recall actions against OpenAlex and Semantic Scholar."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

from paper_search.application.composition import load_locked_identifier_map
from paper_search.application.locks import load_verified_input_lock, lock_sha256
from paper_search.learning.adapters import RecallQueryPolicyGenerator
from paper_search.learning.candidate_ceiling import (
    Core4SemanticBooleanQueryGenerator,
    FullCandidatePoolQueryGenerator,
    select_ceiling_batch,
)
from paper_search.learning.deployment import (
    load_cpu_action_policy,
    load_cpu_pairwise_action_policy,
)
from paper_search.learning.exploration import DeterministicExplorationQueryGenerator
from paper_search.learning.gold_retrievability_audit import FrozenAuditManifest
from paper_search.learning.lexical_bridge_deployment import (
    load_lexical_bridge_model,
)
from paper_search.learning.lexical_bridge_generator import (
    LexicalBridgeCandidateGenerator,
)
from paper_search.learning.structured_graph_candidates import (
    FixedBudgetOpenAlexQueryGenerator,
    StructuredGraphCandidateGenerator,
)
from paper_search.learning.semantic_backfill import SemanticBackfillQueryGenerator
from paper_search.recall_experiments.canary_inputs import (
    LoadedCanaryInput,
    RecallCase,
)
from paper_search.recall_experiments.canary_runtime import (
    build_live_runtime_bundle,
    load_runtime_profile,
    resolve_runtime_secrets,
)
from paper_search.recall_experiments.canary_reporting import CanaryReport
from paper_search.recall_experiments.canary_service import RecallCanaryService
from paper_search.recall_experiments.contracts import (
    RecallActionBatch,
    assert_no_forbidden_identifier_keys_or_patterns,
)
from paper_search.recall_experiments.generation.base import QueryGenerator
from paper_search.recall_experiments.generation.fixed import FixedActionGenerator
from paper_search.recall_experiments.recipes import load_recall_recipe


Provider = Literal["openalex", "semantic_scholar"]
CollectionMode = Literal[
    "policy",
    "exploration",
    "candidate_ceiling",
    "structured_graph",
    "semantic_backfill",
    "fixed_budget_openalex",
    "core4_semantic_boolean",
    "production_lexical",
    "lexical_bridge",
    "frozen_actions",
]


@dataclass(frozen=True)
class ProductionIdentifierContext:
    """Exact evaluator-only identifier context derived from the production lock."""

    identifier_map_bytes: bytes
    evidence: dict[str, object]


def _sha256(content: bytes) -> str:
    return "sha256:" + hashlib.sha256(content).hexdigest()


def _checked_inter_batch_delay(value: float) -> float:
    delay = float(value)
    if not 0.0 <= delay <= 60.0:
        raise ValueError("inter-batch delay must be between 0 and 60 seconds")
    return delay


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _load_infrastructure_failure_marker(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_bytes())
    if not isinstance(payload, dict) or payload.get("schema_version") != (
        "provider-recall-infrastructure-failure-v1"
    ):
        raise ValueError("provider infrastructure failure marker is invalid")
    query_ids = payload.get("query_ids")
    if (
        not isinstance(query_ids, list)
        or not query_ids
        or not all(isinstance(item, str) and item for item in query_ids)
    ):
        raise ValueError("provider infrastructure failure query IDs are invalid")
    return cast(dict[str, object], payload)


def _write_infrastructure_failure_marker(
    *,
    run_path: Path,
    provider: Provider,
    batch_stem: str,
    query_ids: tuple[str, ...],
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "provider-recall-infrastructure-failure-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "provider": provider,
        "batch_stem": batch_stem,
        "run_id": run_path.name,
        "query_ids": list(query_ids),
        "reason": "no_valid_repeat",
        "retryable": True,
    }
    run_path.mkdir(parents=True, exist_ok=True)
    (run_path / "infrastructure-failure.json").write_bytes(
        _canonical_bytes(payload) + b"\n"
    )
    return payload


def _load_selected_query_ids(path: Path) -> tuple[frozenset[str], str]:
    content = path.read_bytes()
    payload = json.loads(content)
    if not isinstance(payload, dict) or payload.get("schema_version") != (
        "provider-recall-query-selection-v1"
    ):
        raise ValueError("query selection schema is invalid")
    values = payload.get("query_ids")
    if (
        not isinstance(values, list)
        or not values
        or not all(isinstance(value, str) and value.strip() for value in values)
    ):
        raise ValueError("query selection IDs are invalid")
    normalized = [value.strip() for value in values]
    if len(normalized) != len(set(normalized)):
        raise ValueError("query selection contains duplicate IDs")
    return frozenset(normalized), _sha256(content)


def _load_fixed_actions(
    path: Path,
    *,
    query_ids: tuple[str, ...] | list[str],
) -> tuple[dict[str, dict[str, object]], str]:
    content = path.read_bytes()
    payload = json.loads(content)
    expected = tuple(query_ids)
    if len(expected) != len(set(expected)):
        raise ValueError("fixed action query IDs must be unique")
    if not isinstance(payload, dict) or set(payload) != set(expected):
        raise ValueError("fixed action query coverage does not match partition")
    actions: dict[str, dict[str, object]] = {}
    for query_id in expected:
        batch = RecallActionBatch.model_validate(payload[query_id])
        dumped = batch.model_dump(mode="json")
        assert_no_forbidden_identifier_keys_or_patterns(dumped)
        if (
            len(batch.actions) != 1
            or batch.actions[0].action_type != "text_search"
            or batch.actions[0].payload.search_mode != "lexical"
        ):
            raise ValueError("fixed action package must contain one lexical text action")
        actions[query_id] = cast(dict[str, object], dumped)
    return actions, _sha256(content)


def _load_rows(
    path: Path,
    *,
    limit: int | None,
    sample_batch_index: int = 0,
    sample_batch_count: int = 1,
    selected_query_ids: frozenset[str] | None = None,
) -> tuple[list[dict[str, object]], tuple[str, str, str]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError("dataset partition is empty")
    identities = {
        (str(row.get("dataset")), str(row.get("split")), str(row.get("role")))
        for row in rows
    }
    if len(identities) != 1:
        raise ValueError("dataset partition must have one dataset/split/role identity")
    identity = next(iter(identities))
    if identity[2] == "final_test":
        raise ValueError("final_test partition cannot be used for provider comparison")
    if identity[2] not in {"training", "development"}:
        raise ValueError("dataset partition role must be training or development")
    if selected_query_ids is not None:
        if limit is not None or sample_batch_count != 1 or sample_batch_index != 0:
            raise ValueError("frozen query selection cannot be combined with resampling")
        rows = [row for row in rows if str(row.get("query_id")) in selected_query_ids]
        found = {str(row.get("query_id")) for row in rows}
        missing = selected_query_ids.difference(found)
        if missing:
            raise ValueError("frozen query IDs are missing from partition")
    if sample_batch_count != 1:
        if limit is None:
            raise ValueError("multi-batch sampling requires a limit")
        rows = select_ceiling_batch(
            rows,
            batch_size=limit,
            batch_index=sample_batch_index,
            batch_count=sample_batch_count,
        )
    elif limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        if limit < len(rows):
            source_count = len(rows)
            rows = [rows[index * source_count // limit] for index in range(limit)]
    return rows, identity


def _parse_providers(values: list[str]) -> tuple[Provider, ...]:
    allowed = {"openalex", "semantic_scholar"}
    if not values or any(value not in allowed for value in values):
        raise ValueError("providers must be openalex and/or semantic_scholar")
    if len(values) != len(set(values)):
        raise ValueError("duplicate providers are not allowed")
    return cast(tuple[Provider, ...], tuple(values))


def _runtime_profile_name(collection_mode: CollectionMode) -> str:
    if collection_mode in {"fixed_budget_openalex", "core4_semantic_boolean"}:
        return "fixed-budget-openalex-live.yaml"
    if collection_mode in {"candidate_ceiling", "structured_graph"}:
        return "candidate-ceiling-live.yaml"
    if collection_mode == "semantic_backfill":
        return "semantic-backfill-live.yaml"
    if collection_mode in {"production_lexical", "frozen_actions"}:
        return "production-lexical-live.yaml"
    if collection_mode == "lexical_bridge":
        return "lexical-bridge-live.yaml"
    return "default-live.yaml"


def _load_production_identifier_context(
    *,
    workspace_root: Path,
    lock_path: Path,
) -> ProductionIdentifierContext:
    """Load the same base-plus-PASA identifier aliases used by production."""

    root = workspace_root.resolve()
    resolved_lock_path = lock_path
    if not resolved_lock_path.is_absolute():
        resolved_lock_path = root / resolved_lock_path
    resolved_lock_path = resolved_lock_path.resolve()
    verified = load_verified_input_lock(resolved_lock_path, artifact_root=root)
    identifier_map, alias_count = load_locked_identifier_map(
        verified.lock,
        verified.artifact_bytes,
    )
    if identifier_map is None:
        raise ValueError("production lock does not bind combined identifier aliases")
    identifier_map_bytes = _canonical_bytes(dict(identifier_map.resolved_pairs()))
    lock = verified.lock
    pasa_binding = lock.baseline.pasa_identity_aliases
    if pasa_binding is None:  # pragma: no cover - guarded by the loader above
        raise ValueError("production lock does not bind PASA identity aliases")
    try:
        display_lock_path = resolved_lock_path.relative_to(root).as_posix()
    except ValueError:
        display_lock_path = str(resolved_lock_path)
    evidence: dict[str, object] = {
        "binding": "production_combined_identifier_map",
        "production_lock_path": display_lock_path,
        "production_lock_sha256": str(lock_sha256(lock)),
        "production_lock_file_sha256": _sha256(resolved_lock_path.read_bytes()),
        "base_identifier_map": {
            "path": str(lock.frozen_data.identifier_map.path),
            "sha256": str(lock.frozen_data.identifier_map.sha256),
        },
        "pasa_identity_aliases": {
            "path": str(pasa_binding.alias_map.path),
            "sha256": str(pasa_binding.alias_map.sha256),
            "alias_count": pasa_binding.alias_count,
        },
        "combined_identifier_map_sha256": _sha256(identifier_map_bytes),
        "combined_alias_count": alias_count,
    }
    return ProductionIdentifierContext(
        identifier_map_bytes=identifier_map_bytes,
        evidence=evidence,
    )


def _validate_report_identifier_context(
    report: CanaryReport,
    *,
    expected_sha256: str,
) -> None:
    actual_sha256 = report.execution_identity.identifier_map_sha256
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "resumed canary report identifier context mismatch: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )


def _loaded_input(
    rows: list[dict[str, object]], identifier_map_bytes: bytes
) -> LoadedCanaryInput:
    cases = tuple(
        RecallCase(
            query_id=str(row["query_id"]),
            query=str(row["query"]),
            gold_paper_ids=tuple(
                str(item) for item in cast(list[object], row["gold_paper_ids"])
            ),
        )
        for row in rows
    )
    input_bytes = _canonical_bytes([case.model_dump(mode="json") for case in cases])
    return LoadedCanaryInput(
        input_kind="jsonl",
        cases=cases,
        evaluation_status="available",
        input_sha256=_sha256(input_bytes),
        identifier_map_bytes=identifier_map_bytes,
        identifier_map_sha256=_sha256(identifier_map_bytes),
    )


def _build_generator_override(
    *,
    collection_mode: CollectionMode,
    role: str,
    max_actions: int,
    workspace_root: Path | None,
) -> QueryGenerator:
    if collection_mode == "exploration":
        if role != "training":
            raise ValueError("exploration collection requires the training role")
        if max_actions != 3:
            raise ValueError("exploration collection requires exactly three actions")
        return DeterministicExplorationQueryGenerator()
    if collection_mode == "candidate_ceiling":
        if role == "final_test":
            raise ValueError("candidate ceiling collection forbids final_test")
        if role not in {"training", "development"}:
            raise ValueError(
                "candidate ceiling collection requires training or development"
            )
        if max_actions != 12:
            raise ValueError("candidate ceiling collection requires exactly twelve actions")
        return FullCandidatePoolQueryGenerator(max_candidates=max_actions)
    if collection_mode == "structured_graph":
        if role == "final_test":
            raise ValueError("structured graph collection forbids final_test")
        if role not in {"training", "development"}:
            raise ValueError(
                "structured graph collection requires training or development"
            )
        if max_actions != 12:
            raise ValueError("structured graph collection requires exactly twelve actions")
        return StructuredGraphCandidateGenerator(max_actions=max_actions)
    if collection_mode == "semantic_backfill":
        if role == "final_test":
            raise ValueError("semantic backfill collection forbids final_test")
        if role not in {"training", "development"}:
            raise ValueError(
                "semantic backfill collection requires training or development"
            )
        if max_actions != 1:
            raise ValueError("semantic backfill collection requires exactly one action")
        return SemanticBackfillQueryGenerator()
    if collection_mode == "fixed_budget_openalex":
        if role == "final_test":
            raise ValueError("fixed budget collection forbids final_test")
        if role not in {"training", "development"}:
            raise ValueError(
                "fixed budget collection requires training or development"
            )
        if max_actions != 6:
            raise ValueError("fixed budget collection requires exactly six actions")
        return FixedBudgetOpenAlexQueryGenerator(max_openalex_actions=max_actions)
    if collection_mode == "core4_semantic_boolean":
        if role == "final_test":
            raise ValueError("A-prime collection forbids final_test")
        if role not in {"training", "development"}:
            raise ValueError("A-prime collection requires training or development")
        if max_actions != 6:
            raise ValueError("A-prime collection requires exactly six actions")
        return Core4SemanticBooleanQueryGenerator()
    if workspace_root is None:
        raise ValueError("policy collection requires a workspace root")
    if collection_mode == "lexical_bridge":
        if role == "final_test":
            raise ValueError("lexical bridge collection forbids final_test")
        if role not in {"training", "development"}:
            raise ValueError(
                "lexical bridge collection requires training or development"
            )
        if max_actions != 4:
            raise ValueError("lexical bridge collection requires exactly four actions")
        base = _production_lexical_generator(
            workspace_root=workspace_root,
            max_actions=3,
        )
        bridge = load_lexical_bridge_model(
            model_path=workspace_root
            / "data/training_private/models/"
            "supervised-lexical-bridge-openalex-v2.joblib",
            manifest_path=workspace_root
            / "data/training/supervised-lexical-bridge-openalex-v2.json",
        )
        return LexicalBridgeCandidateGenerator(
            base,
            bridge=bridge,
            max_actions=max_actions,
        )
    if collection_mode == "production_lexical":
        if role == "final_test":
            raise ValueError("production lexical collection forbids final_test")
        if role not in {"training", "development"}:
            raise ValueError(
                "production lexical collection requires training or development"
            )
        if max_actions != 3:
            raise ValueError("production lexical collection requires exactly three actions")
        return _production_lexical_generator(
            workspace_root=workspace_root,
            max_actions=max_actions,
        )
    result_manifest = json.loads(
        (workspace_root / "data/training/cpu-action-ranker.json").read_text(
            encoding="utf-8"
        )
    )
    policy = load_cpu_action_policy(
        model_path=workspace_root
        / "data/training_private/models/cpu-action-ranker-v1.f64",
        result_path=workspace_root / "data/training/cpu-action-ranker.json",
    )
    return RecallQueryPolicyGenerator(
        policy,
        max_actions=max_actions,
        source_sha256=str(result_manifest["model_sha256"]),
    )


def _production_lexical_generator(
    *, workspace_root: Path, max_actions: int
) -> RecallQueryPolicyGenerator:
    manifest_path = (
        workspace_root
        / "data/training/cpu-pairwise-action-ranker-openalex-v1.json"
    )
    result_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    policy = load_cpu_pairwise_action_policy(
        model_path=workspace_root
        / "data/training_private/models/"
        "cpu-pairwise-action-ranker-openalex-v1.f64",
        manifest_path=manifest_path,
    )
    return RecallQueryPolicyGenerator(
        policy,
        max_actions=max_actions,
        source_sha256=str(result_manifest["model_sha256"]),
        force_lexical_unique=True,
    )


async def _run_provider(
    *,
    provider: Provider,
    rows: list[dict[str, object]],
    chunk_size: int,
    output_root: Path,
    workspace_root: Path,
    resume: bool,
    role: str,
    collection_mode: CollectionMode,
    fixed_actions: dict[str, dict[str, object]] | None = None,
    fixed_actions_sha256: str | None = None,
    inter_batch_delay_seconds: float = 0.0,
    continue_on_no_valid_repeat: bool = False,
    infrastructure_failures: list[dict[str, object]] | None = None,
    identifier_map_bytes: bytes,
) -> list[CanaryReport]:
    profile = load_runtime_profile(
        workspace_root
        / "configs/recall_experiments/runtime"
        / _runtime_profile_name(collection_mode)
    )
    secrets = resolve_runtime_secrets(profile)
    recipe_name = {
        "candidate_ceiling": "scheme-b-candidate-ceiling-live.yaml",
        "structured_graph": "scheme-b-structured-graph-live.yaml",
        "semantic_backfill": "scheme-b-semantic-backfill-live.yaml",
        "fixed_budget_openalex": "fixed-budget-openalex-live.yaml",
        "core4_semantic_boolean": "core4-semantic-boolean-live.yaml",
        "lexical_bridge": "scheme-b-lexical-bridge-live.yaml",
    }.get(collection_mode, "scheme-b-blind-live.yaml")
    recipe = load_recall_recipe(
        workspace_root / "configs/recall_experiments/methods" / recipe_name
    )
    generator_override = (
        None
        if collection_mode == "frozen_actions"
        else _build_generator_override(
            collection_mode=collection_mode,
            role=role,
            max_actions=recipe.recipe.retrieval.max_total_actions,
            workspace_root=workspace_root,
        )
    )
    if collection_mode == "frozen_actions" and (
        fixed_actions is None or fixed_actions_sha256 is None
    ):
        raise ValueError("frozen action collection requires hash-bound actions")
    identifier_map_sha256 = _sha256(identifier_map_bytes)
    reports: list[CanaryReport] = []
    for offset in range(0, len(rows), chunk_size):
        batch = rows[offset : offset + chunk_size]
        batch_number = offset // chunk_size + 1
        batch_stem = f"batch-{batch_number:04d}"
        provider_root = output_root / provider
        completed_paths = sorted(provider_root.glob(f"{batch_stem}*/canary-report.json"))
        if resume and completed_paths:
            completed_report = CanaryReport.model_validate_json(
                completed_paths[-1].read_bytes()
            )
            _validate_report_identifier_context(
                completed_report,
                expected_sha256=identifier_map_sha256,
            )
            reports.append(completed_report)
            continue
        failure_paths = sorted(
            provider_root.glob(f"{batch_stem}*/infrastructure-failure.json")
        )
        if resume and continue_on_no_valid_repeat and failure_paths:
            failure = _load_infrastructure_failure_marker(failure_paths[-1])
            if infrastructure_failures is not None:
                infrastructure_failures.append(failure)
            continue
        run_path = provider_root / batch_stem
        capture_path = output_root / "captures" / provider / batch_stem
        if resume and (run_path.exists() or capture_path.exists()):
            retry_number = 1
            while True:
                suffix = f"-retry-{retry_number:03d}"
                candidate_run = provider_root / f"{batch_stem}{suffix}"
                candidate_capture = (
                    output_root / "captures" / provider / f"{batch_stem}{suffix}"
                )
                if not candidate_run.exists() and not candidate_capture.exists():
                    run_path = candidate_run
                    capture_path = candidate_capture
                    break
                retry_number += 1
        if inter_batch_delay_seconds > 0:
            await asyncio.sleep(inter_batch_delay_seconds)
        bundle = await build_live_runtime_bundle(
            profile=profile,
            secrets=secrets,
            loaded_recipe=recipe,
            capture_root=capture_path,
            search_dependency=provider,
        )
        report: CanaryReport | None = None
        no_valid_repeat = False
        try:
            batch_generator = generator_override
            if fixed_actions is not None:
                query_ids = [str(row["query_id"]) for row in batch]
                batch_generator = FixedActionGenerator(
                    {query_id: fixed_actions[query_id] for query_id in query_ids},
                    expected_query_ids=query_ids,
                    allowed_actions=["text_search"],
                    max_actions=1,
                    source_sha256=fixed_actions_sha256,
                )
            assert batch_generator is not None
            report = await RecallCanaryService(workspace_root=workspace_root).run(
                loaded_recipe=recipe,
                loaded_input=_loaded_input(batch, identifier_map_bytes),
                runtime_bundle=bundle,
                output_path=run_path,
                generator_override=batch_generator,
            )
        except RuntimeError as error:
            if not (
                continue_on_no_valid_repeat
                and str(error) == "canary produced no valid repeat"
            ):
                raise
            no_valid_repeat = True
        finally:
            await bundle.aclose()
        if no_valid_repeat:
            failure = _write_infrastructure_failure_marker(
                run_path=run_path,
                provider=provider,
                batch_stem=batch_stem,
                query_ids=tuple(str(row["query_id"]) for row in batch),
            )
            if infrastructure_failures is not None:
                infrastructure_failures.append(failure)
            continue
        if report is None:  # pragma: no cover - guarded by the branches above
            raise RuntimeError("provider batch ended without a report")
        reports.append(report)
    return reports


def _aggregate(provider: Provider, reports: list[CanaryReport]) -> dict[str, object]:
    rows = [row for report in reports for row in report.result.per_query]
    usage = [report.usage for report in reports]
    hits = sum(int(row.gold_hit_count or 0) for row in rows)
    associations = sum(int(row.gold_association_count or 0) for row in rows)
    return {
        "provider": provider,
        "query_count": len(rows),
        "gold_association_count": associations,
        "gold_hit_count": hits,
        "macro_candidate_recall": (
            sum(float(row.candidate_recall or 0) for row in rows) / len(rows)
            if rows
            else 0.0
        ),
        "micro_candidate_recall": hits / associations if associations else 0.0,
        "mean_candidate_count": (
            sum(row.candidate_count for row in rows) / len(rows) if rows else 0.0
        ),
        "search_api_calls": sum(item.search_api_calls for item in usage),
        "cost_cny": str(sum((item.cost_cny or 0 for item in usage))),
    }


def _provider_overlap(
    provider_reports: dict[Provider, list[CanaryReport]],
) -> dict[str, int]:
    hits_by_provider: dict[Provider, dict[str, frozenset[str]]] = {}
    for provider in ("openalex", "semantic_scholar"):
        reports = provider_reports.get(provider)
        if reports is None:
            raise ValueError("provider overlap requires OpenAlex and Semantic Scholar")
        hits: dict[str, frozenset[str]] = {}
        for report in reports:
            for row in report.result.per_query:
                query_id = str(row.query_id)
                if query_id in hits:
                    raise ValueError("provider report contains duplicate query IDs")
                hits[query_id] = frozenset(str(item) for item in row.gold_hit_ids)
        hits_by_provider[provider] = hits

    openalex_hits = hits_by_provider["openalex"]
    semantic_hits = hits_by_provider["semantic_scholar"]
    if set(openalex_hits) != set(semantic_hits):
        raise ValueError("provider report query identities differ")

    intersection_hits = 0
    openalex_only_hits = 0
    semantic_only_hits = 0
    intersection_queries = 0
    openalex_only_queries = 0
    semantic_only_queries = 0
    union_queries = 0
    for query_id in openalex_hits:
        left = openalex_hits[query_id]
        right = semantic_hits[query_id]
        intersection_hits += len(left.intersection(right))
        openalex_only_hits += len(left.difference(right))
        semantic_only_hits += len(right.difference(left))
        intersection_queries += bool(left and right)
        openalex_only_queries += bool(left and not right)
        semantic_only_queries += bool(right and not left)
        union_queries += bool(left or right)
    return {
        "intersection_gold_hits": intersection_hits,
        "openalex_only_gold_hits": openalex_only_hits,
        "semantic_scholar_only_gold_hits": semantic_only_hits,
        "union_gold_hits": (
            intersection_hits + openalex_only_hits + semantic_only_hits
        ),
        "intersection_gold_hit_queries": intersection_queries,
        "openalex_only_gold_hit_queries": openalex_only_queries,
        "semantic_scholar_only_gold_hit_queries": semantic_only_queries,
        "union_gold_hit_queries": union_queries,
    }


async def _main(args: argparse.Namespace) -> None:
    workspace_root = Path(args.workspace_root).resolve()
    identifier_context = _load_production_identifier_context(
        workspace_root=workspace_root,
        lock_path=Path(args.production_lock),
    )
    inter_batch_delay_seconds = _checked_inter_batch_delay(
        args.inter_batch_delay_seconds
    )
    partition_path = Path(args.partition)
    if not partition_path.is_absolute():
        partition_path = workspace_root / partition_path
    frozen_manifest: FrozenAuditManifest | None = None
    frozen_manifest_sha256: str | None = None
    query_selection_sha256: str | None = None
    selected_query_ids: frozenset[str] | None = None
    if args.query_id_file is not None:
        if args.frozen_manifest is not None:
            raise ValueError("query ID selection cannot be combined with a frozen manifest")
        query_id_path = Path(args.query_id_file)
        if not query_id_path.is_absolute():
            query_id_path = workspace_root / query_id_path
        selected_query_ids, query_selection_sha256 = _load_selected_query_ids(
            query_id_path
        )
    if args.frozen_manifest is not None:
        frozen_manifest_path = Path(args.frozen_manifest)
        if not frozen_manifest_path.is_absolute():
            frozen_manifest_path = workspace_root / frozen_manifest_path
        frozen_manifest = FrozenAuditManifest.model_validate_json(
            frozen_manifest_path.read_text(encoding="utf-8")
        )
        frozen_manifest_sha256 = _sha256(frozen_manifest_path.read_bytes())
        if _sha256(partition_path.read_bytes()) != frozen_manifest.source_sha256:
            raise ValueError("partition does not match frozen audit manifest")
        selected_query_ids = frozenset(item.query_id for item in frozen_manifest.sample)
    rows, (dataset, split, role) = _load_rows(
        partition_path,
        limit=args.limit,
        sample_batch_index=args.sample_batch_index,
        sample_batch_count=args.sample_batch_count,
        selected_query_ids=selected_query_ids,
    )
    fixed_actions: dict[str, dict[str, object]] | None = None
    fixed_actions_sha256: str | None = None
    if args.collection_mode == "frozen_actions":
        if args.fixed_actions_file is None:
            raise ValueError("frozen action collection requires --fixed-actions-file")
        fixed_actions_path = Path(args.fixed_actions_file)
        if not fixed_actions_path.is_absolute():
            fixed_actions_path = workspace_root / fixed_actions_path
        fixed_actions, fixed_actions_sha256 = _load_fixed_actions(
            fixed_actions_path,
            query_ids=[str(row["query_id"]) for row in rows],
        )
    elif args.fixed_actions_file is not None:
        raise ValueError("--fixed-actions-file requires frozen_actions mode")
    output_root = Path(args.output).resolve()
    if output_root.exists() and not args.resume:
        raise FileExistsError(output_root)
    provider_reports: dict[Provider, list[CanaryReport]] = {}
    provider_failures: dict[Provider, list[dict[str, object]]] = {}
    providers_to_run = _parse_providers(args.providers)
    if args.continue_on_no_valid_repeat and len(providers_to_run) != 1:
        raise ValueError(
            "--continue-on-no-valid-repeat requires exactly one provider"
        )
    for provider in providers_to_run:
        provider_failures[provider] = []
        provider_reports[provider] = await _run_provider(
            provider=provider,
            rows=rows,
            chunk_size=args.chunk_size,
            output_root=output_root,
            workspace_root=workspace_root,
            resume=args.resume,
            role=role,
            collection_mode=args.collection_mode,
            fixed_actions=fixed_actions,
            fixed_actions_sha256=fixed_actions_sha256,
            inter_batch_delay_seconds=inter_batch_delay_seconds,
            continue_on_no_valid_repeat=args.continue_on_no_valid_repeat,
            infrastructure_failures=provider_failures[provider],
            identifier_map_bytes=identifier_context.identifier_map_bytes,
        )
    if len(providers_to_run) > 1:
        left_reports = provider_reports[providers_to_run[0]]
        right_reports = provider_reports[providers_to_run[1]]
        for batch_index in range(len(left_reports)):
            if (
                left_reports[batch_index].actions_by_query
                != right_reports[batch_index].actions_by_query
            ):
                raise RuntimeError("provider comparison did not execute identical actions")
    providers = [
        _aggregate(provider, provider_reports[provider])
        for provider in providers_to_run
    ]
    summary = {
        "schema_version": "provider-recall-comparison-v2",
        "created_at": datetime.now(UTC).isoformat(),
        "dataset": f"{dataset}_{split}",
        "partition_path": str(partition_path.relative_to(workspace_root)),
        "partition_role": role,
        "collection_mode": args.collection_mode,
        "exploration_policy": (
            "anchor-compress-rotate-v1"
            if args.collection_mode == "exploration"
            else None
        ),
        "candidate_ceiling_policy": (
            "full-controlled-candidate-pool-v2"
            if args.collection_mode == "candidate_ceiling"
            else None
        ),
        "structured_graph_policy": (
            "structured-graph-candidate-pool-v1"
            if args.collection_mode == "structured_graph"
            else None
        ),
        "semantic_backfill_policy": (
            "openalex-semantic-backfill-v1"
            if args.collection_mode == "semantic_backfill"
            else None
        ),
        "fixed_budget_openalex_policy": (
            "fixed-budget-openalex-v1"
            if args.collection_mode == "fixed_budget_openalex"
            else None
        ),
        "core4_semantic_boolean_policy": (
            "core4-semantic-boolean-v1"
            if args.collection_mode == "core4_semantic_boolean"
            else None
        ),
        "lexical_bridge_policy": (
            "supervised-lexical-bridge-openalex-v2"
            if args.collection_mode == "lexical_bridge"
            else None
        ),
        "frozen_actions_policy": (
            "externally-frozen-single-lexical-action-v1"
            if args.collection_mode == "frozen_actions"
            else None
        ),
        "sample_batch_index": args.sample_batch_index,
        "sample_batch_count": args.sample_batch_count,
        "query_count": len(rows),
        "selection": (
            "frozen_stratified_manifest"
            if frozen_manifest is not None
            else "explicit_frozen_query_ids"
            if query_selection_sha256 is not None
            else "deterministic_even_stride"
            if args.limit is not None
            else "complete_partition"
        ),
        "frozen_manifest_sha256": frozen_manifest_sha256,
        "query_selection_sha256": query_selection_sha256,
        "fixed_actions_sha256": fixed_actions_sha256,
        "identifier_context": identifier_context.evidence,
        "inter_batch_delay_seconds": inter_batch_delay_seconds,
        "actions_identical_across_providers": (
            True if len(providers_to_run) > 1 else None
        ),
        "max_actions_per_query": (
            12
            if args.collection_mode in {"candidate_ceiling", "structured_graph"}
            else 1
            if args.collection_mode in {"semantic_backfill", "frozen_actions"}
            else 6
            if args.collection_mode in {"fixed_budget_openalex", "core4_semantic_boolean"}
            else 3
            if args.collection_mode == "production_lexical"
            else 4
            if args.collection_mode == "lexical_bridge"
            else 3
        ),
        "max_results_per_action": 50,
        "providers": providers,
        "infrastructure_failures": {
            provider: provider_failures[provider] for provider in providers_to_run
        },
        "provider_overlap": (
            _provider_overlap(provider_reports)
            if set(providers_to_run) == {"openalex", "semantic_scholar"}
            else None
        ),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_bytes(_canonical_bytes(summary) + b"\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace-root", default=".")
    parser.add_argument(
        "--production-lock",
        default="deliverables/evaluator/live-evaluator.lock.yaml",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--partition",
        default="data/training_private/partitions/pasa_auto_dev.jsonl",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--frozen-manifest")
    parser.add_argument("--query-id-file")
    parser.add_argument("--fixed-actions-file")
    parser.add_argument(
        "--providers",
        nargs="+",
        choices=("openalex", "semantic_scholar"),
        default=["openalex", "semantic_scholar"],
    )
    parser.add_argument("--sample-batch-index", type=int, default=0)
    parser.add_argument("--sample-batch-count", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=3, choices=(1, 2, 3, 4))
    parser.add_argument("--inter-batch-delay-seconds", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continue-on-no-valid-repeat", action="store_true")
    parser.add_argument(
        "--collection-mode",
        choices=(
            "policy",
            "exploration",
            "candidate_ceiling",
            "structured_graph",
            "semantic_backfill",
            "fixed_budget_openalex",
            "core4_semantic_boolean",
            "production_lexical",
            "lexical_bridge",
            "frozen_actions",
        ),
        default="policy",
    )
    asyncio.run(_main(parser.parse_args()))


if __name__ == "__main__":
    main()
