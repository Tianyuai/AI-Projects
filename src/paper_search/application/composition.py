"""Single lock- and mode-bound application composition root."""

from __future__ import annotations

import os
import stat
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol, cast
from uuid import uuid4

import httpx
from pydantic import SecretStr

from paper_search.application.artifacts import ArtifactFactory
from paper_search.application.experiments import (
    ExperimentDefinition,
    ExperimentFlags,
    ExperimentName,
)
from paper_search.application.contracts import (
    ReadyHealthResponse,
    SearchExecutionResult,
    SearchFailure,
    SearchRequest,
    SearchSuccess,
)
from paper_search.application.locks import (
    InputLock,
    ReplayLock,
    load_verified_input_lock,
    load_verified_input_lock_bytes,
    lock_sha256,
)
from paper_search.application.modes import ModeBinding
from paper_search.application.readiness import (
    build_live_readiness,
    load_live_readiness,
)
from paper_search.application.service import SearchApplicationService, SearchOrchestrator
from paper_search.api.routing import SearchServiceRouter
from paper_search.config import (
    RuntimeConfig,
    parse_budget_bytes,
    validate_mode_authorization,
)
from paper_search.control.budget import HardBudgetController
from paper_search.control.pricing import ActualCostPricer, parse_pricing_policy_bytes
from paper_search.domain.models import (
    BudgetReservation,
    DependencyName,
    DependencyStatus,
    FusedPaper,
    ProviderResult,
    QuerySpec,
    SearchMode,
    SearchBudget,
    SafeRelativePath,
    Sha256,
    UsageActual,
    UsageEstimate,
)
from paper_search.llm.client import OpenAICompatibleLLMClient
from paper_search.llm.prompt_artifacts import render_prompt_system_message
from paper_search.llm.snapshot_adapters import (
    LiveCaptureLLMAnalyzer,
    ReplayLLMAnalyzer,
)
from paper_search.evaluation.dataset import IdentifierMap
from paper_search.learning.cross_vocabulary_bridge import (
    select_production_cross_vocabulary_supplement,
)
from paper_search.learning.lexical_bridge_deployment import (
    load_lexical_bridge_model_bytes,
)
from paper_search.learning.production_query_expansion import (
    SupervisedLexicalBridgePlanEnricher,
)
from paper_search.pipeline.orchestrator import (
    MockSearchOrchestrator,
    OrchestratorResult,
)
from paper_search.ranking.cpu_document import (
    DocumentRankingStage,
    load_cpu_document_ranking_stage_bytes,
)
from paper_search.retrieval.snapshot_adapters import (
    LiveCaptureSearchProvider,
    ReplaySearchProvider,
)
from paper_search.storage.dependency_snapshot import (
    DependencyCaptureStore,
    DependencySnapshotManifestV2,
    DependencySnapshotReader,
)


QueryAnalyzer = Callable[
    [str, BudgetReservation], Awaitable[ProviderResult[dict[str, Any]]]
]
AnalyzerDecorator = Callable[[QueryAnalyzer], QueryAnalyzer]


_LLM_BASE_URL = "https://api.deepseek.com/v1"
_DEPENDENCIES: tuple[DependencyName, ...] = (
    "llm",
    "openalex",
    "semantic_scholar",
)


def _read_confined_source_capture_file(source_root: Path, name: str) -> tuple[Path, bytes]:
    """Return one stable, regular source-capture file snapshot and its provenance."""
    unavailable = "source live capture lock is unavailable"

    def identity(metadata: os.stat_result) -> tuple[int, int, int, int]:
        return (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
            metadata.st_mtime_ns,
        )

    try:
        root = source_root.resolve(strict=True)
        candidate = root / name
        resolved_before = candidate.resolve(strict=True)
        before = os.lstat(candidate)
        if resolved_before.parent != root or not stat.S_ISREG(before.st_mode):
            raise ValueError(unavailable)

        flags = os.O_RDONLY
        if os.name != "nt":
            flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(candidate, flags)
        try:
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode) or identity(opened) != identity(before):
                raise ValueError(unavailable)
            with os.fdopen(descriptor, "rb", closefd=False) as handle:
                payload = handle.read()
            after_read = os.fstat(descriptor)
        finally:
            os.close(descriptor)

        resolved_after = candidate.resolve(strict=True)
        after = os.lstat(candidate)
        if (
            resolved_after != resolved_before
            or resolved_after.parent != root
            or not stat.S_ISREG(after.st_mode)
            or identity(after_read) != identity(before)
            or identity(after) != identity(before)
        ):
            raise ValueError(unavailable)
        return resolved_before, payload
    except OSError as error:
        raise ValueError(unavailable) from error


class _AnalyzerAdapter(Protocol):
    async def generate_json(
        self,
        *,
        prompt_name: str,
        payload: dict[str, object],
        reservation: BudgetReservation,
    ) -> ProviderResult[dict[str, Any]]: ...


@dataclass(frozen=True)
class _AnalyzerBridge:
    adapter: _AnalyzerAdapter

    async def __call__(
        self,
        query: str,
        reservation: BudgetReservation,
    ) -> ProviderResult[dict[str, Any]]:
        return await self.adapter.generate_json(
            prompt_name="query_analyze",
            payload={"query": query},
            reservation=reservation,
        )

    async def repair(
        self,
        query: str,
        invalid_analysis: str,
        reservation: BudgetReservation,
    ) -> ProviderResult[dict[str, Any]]:
        return await self.adapter.generate_json(
            prompt_name="query_analyze",
            payload={
                "query": query,
                "invalid_analysis": invalid_analysis,
            },
            reservation=reservation,
        )


@dataclass(frozen=True)
class ApplicationBundle:
    service: SearchApplicationService
    readiness_probe: Callable[[], ReadyHealthResponse]
    config_hash: Sha256
    artifact_factory: ArtifactFactory
    experiment_id: ExperimentName
    optional_modules: dict[str, bool]
    experiment_config: None
    source_git_sha: str
    prompt_version: Literal[
        "query-analyze-v1",
        "query-analyze-semantic-actions-v2",
        "query-analyze-protected-actions-v3",
    ]
    mode_binding: ModeBinding
    _owns_artifact_factory: bool = field(default=True, repr=False, compare=False)

    async def aclose(self) -> None:
        if self._owns_artifact_factory:
            await self.artifact_factory.aclose()


@dataclass(frozen=True)
class ServerApplicationBundle:
    """Process-bound replay service plus isolated live request composition."""

    replay: ApplicationBundle
    live_factory: Callable[[], ApplicationBundle] | None
    capture_artifact_factory: ArtifactFactory
    service_router: SearchServiceRouter
    _close: Callable[[], Awaitable[None]] = field(repr=False, compare=False)

    async def aclose(self) -> None:
        await self._close()


class _RequestLiveCaptureService:
    """Publish one validated live capture before exposing a success response."""

    def __init__(
        self,
        *,
        bundle: ApplicationBundle,
        input_lock_bytes: bytes,
        release_bundle: Callable[[ApplicationBundle], None],
        run_id_factory: Callable[[], str],
    ) -> None:
        self._bundle = bundle
        self._input_lock_bytes = input_lock_bytes
        self._release_bundle = release_bundle
        self._run_id_factory = run_id_factory

    async def execute_and_publish(self, request: SearchRequest) -> SearchExecutionResult:
        run_id = self._run_id_factory()
        session = None
        try:
            session = self._bundle.artifact_factory.start_capture(
                run_id=run_id,
                input_lock_bytes=self._input_lock_bytes,
                expected_config_hash=getattr(self._bundle, "config_hash", None),
            )
            execution = await self._bundle.service.execute(request, run_id=run_id)
            session.record_execution(execution)
            if isinstance(execution.outcome, SearchFailure):
                session.fail(execution.outcome.error.code)
                return execution
            if not isinstance(execution.outcome, SearchSuccess):
                raise RuntimeError("live execution has an invalid outcome")
            manifest, _ = session.seal()
            response = execution.outcome.response.model_copy(
                update={
                    "snapshot_set_id": manifest.snapshot_set_id,
                    "snapshot_captured_at": manifest.sealed_at,
                }
            )
            execution = execution.model_copy(
                update={"outcome": execution.outcome.model_copy(update={"response": response})}
            )
            session.publish()
            return execution
        except BaseException:
            if session is not None and session.work_dir.exists():
                try:
                    session.fail("internal_error")
                except (OSError, RuntimeError, ValueError):
                    pass
            raise
        finally:
            try:
                await self._bundle.aclose()
            finally:
                self._release_bundle(self._bundle)


def _replay_estimates(
    controller: HardBudgetController,
    *,
    lock: ReplayLock,
    pricer: ActualCostPricer,
) -> tuple[UsageEstimate, dict[str, UsageEstimate]]:
    """Mirror the live estimates so replay budget decisions stay identical."""
    return _live_estimates(controller, lock=lock, pricer=pricer)


def _live_estimates(
    controller: HardBudgetController,
    *,
    lock: InputLock,
    pricer: ActualCostPricer,
) -> tuple[UsageEstimate, dict[str, UsageEstimate]]:
    budget = controller.budget
    output_tokens = budget.max_total_tokens // 6
    input_tokens = budget.max_total_tokens - output_tokens
    elapsed_ms = min(
        budget.max_elapsed_seconds * 1_000,
        lock.baseline.retry.max_attempts * lock.baseline.timeout.read_seconds * 1_000,
    )
    llm_usage = pricer.value_actual_peak(
        dependency="llm",
        model_or_adapter=lock.baseline.primary_model,
        usage=UsageActual(
            llm_calls=lock.baseline.retry.max_attempts,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        ),
    )
    analysis = UsageEstimate.model_validate(
        {**llm_usage.model_dump(mode="python"), "elapsed_ms": elapsed_ms}
    )
    adapter_names = {
        "openalex": "openalex-works-v1",
        "semantic_scholar": "semantic-graph-v1",
    }
    provider_estimates = {}
    for dependency, adapter_name in adapter_names.items():
        valued = pricer.value_actual(
            dependency=dependency,  # type: ignore[arg-type]
            model_or_adapter=adapter_name,
            usage=UsageActual(
                search_api_calls=lock.baseline.retry.max_attempts,
            ),
        )
        provider_estimates[dependency] = UsageEstimate.model_validate(
            {**valued.model_dump(mode="python"), "elapsed_ms": elapsed_ms}
        )
    return analysis, provider_estimates


def _low_confidence_analysis_estimate(
    controller: HardBudgetController,
    *,
    lock: InputLock,
    pricer: ActualCostPricer,
) -> UsageEstimate | None:
    binding = lock.baseline.low_confidence_llm_supplement
    if binding is None:
        return None
    elapsed_ms = min(
        controller.budget.max_elapsed_seconds * 1_000,
        binding.max_llm_attempts * lock.baseline.timeout.read_seconds * 1_000,
    )
    usage = pricer.value_actual_peak(
        dependency="llm",
        model_or_adapter=lock.baseline.primary_model,
        usage=UsageActual(
            llm_calls=binding.max_llm_attempts,
            input_tokens=binding.max_input_tokens * binding.max_llm_attempts,
            output_tokens=binding.max_output_tokens * binding.max_llm_attempts,
        ),
    )
    return UsageEstimate.model_validate(
        {**usage.model_dump(mode="python"), "elapsed_ms": elapsed_ms}
    )


def _confined_manifest_path(path: Path, *, artifact_root: Path) -> Path:
    root = artifact_root.resolve(strict=True)
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ValueError("snapshot manifest is unavailable") from error
    if not resolved.is_file() or not resolved.is_relative_to(root):
        raise ValueError("snapshot manifest must be a file beneath artifact root")
    return resolved


def _locked_budget_profile(path: str) -> Literal["low", "balanced"]:
    name = Path(path).name
    if name == "budget_low.yaml":
        return "low"
    if name == "budget_balanced.yaml":
        return "balanced"
    raise ValueError("lock budget path must identify the low or balanced profile")


def _prompt_system_message(prompt_bytes: bytes) -> str:
    """Build the deterministic system message from the bound prompt artifact."""
    return render_prompt_system_message(prompt_bytes)


class _DeploymentDocumentRankingStage:
    """Expose immutable deployment selection without changing rank semantics."""

    def __init__(
        self,
        stage: DocumentRankingStage,
        *,
        deployment_role: str,
        failover_receipt: list[dict[str, str]],
    ) -> None:
        self._stage = stage
        self.model_id = stage.model_id
        self.deployment_role = deployment_role
        self.failover_receipt = [dict(item) for item in failover_receipt]

    def rank(
        self, query: str, candidates: Sequence[FusedPaper]
    ) -> list[FusedPaper]:
        return self._stage.rank(query, candidates)

    def rank_with_context(
        self,
        query: str,
        candidates: Sequence[FusedPaper],
        *,
        query_spec: QuerySpec,
    ) -> list[FusedPaper]:
        contextual_rank = getattr(self._stage, "rank_with_context", None)
        if callable(contextual_rank):
            return cast(
                list[FusedPaper],
                contextual_rank(query, candidates, query_spec=query_spec),
            )
        return self._stage.rank(query, candidates)

    def context_receipt(
        self, query: str, *, query_spec: QuerySpec
    ) -> dict[str, object] | None:
        receipt = getattr(self._stage, "context_receipt", None)
        if not callable(receipt):
            return None
        return cast(dict[str, object] | None, receipt(query, query_spec=query_spec))


def _locked_document_ranker(
    lock: InputLock,
    artifact_bytes: Mapping[SafeRelativePath, bytes],
    artifact_failures: Mapping[SafeRelativePath, str],
) -> DocumentRankingStage | None:
    binding = lock.baseline.document_ranker
    if binding is None:
        return None
    candidates = [
        ("F5-gated-fusion", binding.manifest, binding.weights),
    ]
    if binding.fallback_manifest is not None and binding.fallback_weights is not None:
        candidates.append(
            ("F4-reliability", binding.fallback_manifest, binding.fallback_weights)
        )
    if binding.emergency_manifest is not None and binding.emergency_weights is not None:
        candidates.append(
            ("B0", binding.emergency_manifest, binding.emergency_weights)
        )
    failover_receipt: list[dict[str, str]] = []
    initialization_errors: list[str] = []
    for role, manifest_binding, weights_binding in candidates:
        missing = next(
            (
                item
                for item in (manifest_binding, weights_binding)
                if item.path not in artifact_bytes
            ),
            None,
        )
        if missing is not None:
            failover_receipt.append(
                {
                    "role": role,
                    "reason": artifact_failures.get(missing.path, "unavailable"),
                }
            )
            continue
        try:
            stage = load_cpu_document_ranking_stage_bytes(
                artifact_bytes[manifest_binding.path],
                artifact_bytes[weights_binding.path],
            )
        except (TypeError, ValueError) as error:
            initialization_errors.append(f"{role}: {error}")
            failover_receipt.append(
                {"role": role, "reason": "initialization_failure"}
            )
            continue
        return _DeploymentDocumentRankingStage(
            stage,
            deployment_role=role,
            failover_receipt=failover_receipt,
        )
    raise ValueError(
        "no deployable document ranker in verified chain: "
        + " | ".join(initialization_errors or ["artifacts unavailable"])
    )


def _locked_query_plan_enricher(
    lock: InputLock,
    artifact_bytes: Mapping[SafeRelativePath, bytes],
) -> SupervisedLexicalBridgePlanEnricher | None:
    binding = lock.baseline.supervised_lexical_bridge
    if binding is None:
        return None
    loaded = load_lexical_bridge_model_bytes(
        model_bytes=artifact_bytes[binding.model.path],
        manifest_bytes=artifact_bytes[binding.manifest.path],
    )
    return SupervisedLexicalBridgePlanEnricher(
        loaded,
        max_total_subqueries=lock.baseline.planner.configured_subqueries_max,
    )


def load_locked_identifier_map(
    lock: InputLock,
    artifact_bytes: Mapping[SafeRelativePath, bytes],
) -> tuple[IdentifierMap | None, int]:
    """Load the exact combined identifier aliases bound by an input lock."""

    binding = lock.baseline.pasa_identity_aliases
    if binding is None:
        return None, 0
    base_map = IdentifierMap.from_bytes(
        artifact_bytes[lock.frozen_data.identifier_map.path],
        source="frozen identifier map",
    )
    pasa_map = IdentifierMap.from_bytes(
        artifact_bytes[binding.alias_map.path],
        source="frozen PASA identity aliases",
    )
    if len(pasa_map.resolved_pairs()) != binding.alias_count:
        raise ValueError("frozen PASA identity alias count mismatch")
    maps = [base_map, pasa_map]
    combined: dict[str, str] = {}
    for identifier_map in maps:
        for alias, target in identifier_map.resolved_pairs():
            existing = combined.get(alias)
            if existing is not None and existing != target:
                raise ValueError(f"conflicting frozen identifier alias: {alias}")
            combined[alias] = target
    payload = json.dumps(combined, sort_keys=True, separators=(",", ":")).encode("utf-8")
    resolved = IdentifierMap.from_bytes(payload, source="combined frozen identifier aliases")
    return resolved, len(resolved.resolved_pairs())


def _replay_readiness(binding: ModeBinding) -> Callable[[], ReadyHealthResponse]:
    response = ReadyHealthResponse(
        status="ready",
        execution_mode="replay",
        snapshot_set_id=binding.snapshot_set_id,
        dependencies=[
            DependencyStatus(
                dependency=dependency,
                state="replayed",
                cache_hit=True,
                error_codes=[],
            )
            for dependency in _DEPENDENCIES
        ],
        last_authorized_probe_at=None,
    )
    return lambda: response


def _live_readiness(
    binding: ModeBinding,
    artifact_root: Path,
    *,
    required_dependencies: tuple[str, ...],
) -> Callable[[], ReadyHealthResponse]:
    evidence = load_live_readiness(artifact_root)

    def probe() -> ReadyHealthResponse:
        if evidence is None:
            return ReadyHealthResponse(
                status="degraded",
                execution_mode="live",
                snapshot_set_id=binding.snapshot_set_id,
                dependencies=[
                    DependencyStatus(
                        dependency=dependency,
                        state="degraded",
                        cache_hit=False,
                        error_codes=[],
                    )
                    for dependency in _DEPENDENCIES
                ],
                last_authorized_probe_at=None,
            )
        return build_live_readiness(
            evidence,
            datetime.now(UTC),
            required_dependencies=required_dependencies,
        )

    return probe


def _snapshot_manifest_time(path: Path) -> datetime:
    try:
        manifest = DependencySnapshotManifestV2.model_validate_json(path.read_bytes())
    except (OSError, ValueError) as error:
        raise ValueError("invalid snapshot manifest") from error
    return manifest.sealed_at


def _replay_factory(
    *,
    reader: DependencySnapshotReader,
    lock: ReplayLock,
    pricer: ActualCostPricer,
    config_hash: Sha256,
    analyzer_decorator: AnalyzerDecorator | None,
    document_ranker: DocumentRankingStage | None,
    query_plan_enricher: SupervisedLexicalBridgePlanEnricher | None,
    identifier_map: IdentifierMap | None,
    identifier_alias_count: int,
) -> Callable[[HardBudgetController, str], SearchOrchestrator]:
    def create(
        controller: HardBudgetController,
        run_id: str,
    ) -> SearchOrchestrator:
        del run_id
        analysis_estimate, provider_estimates = _replay_estimates(
            controller,
            lock=lock,
            pricer=pricer,
        )
        analyzer = ReplayLLMAnalyzer(
            reader=reader,
            model_id=lock.baseline.primary_model,
            prompt_artifact_sha256=lock.baseline.planner.prompt_config.sha256,
            prompt_version=lock.baseline.prompt_version,
        )
        low_confidence_binding = lock.baseline.low_confidence_llm_supplement
        low_confidence_analyzer: QueryAnalyzer | None = None
        if low_confidence_binding is not None:
            low_confidence_analyzer = _AnalyzerBridge(
                ReplayLLMAnalyzer(
                    reader=reader,
                    model_id=lock.baseline.primary_model,
                    prompt_artifact_sha256=(
                        low_confidence_binding.prompt_config.sha256
                    ),
                    prompt_version=low_confidence_binding.prompt_version,
                )
            )
        providers = {
            "openalex": ReplaySearchProvider(dependency="openalex", reader=reader),
            "semantic_scholar": ReplaySearchProvider(
                dependency="semantic_scholar",
                reader=reader,
            ),
        }
        baseline_analyzer: QueryAnalyzer = _AnalyzerBridge(analyzer)
        selected_analyzer = (
            analyzer_decorator(baseline_analyzer)
            if analyzer_decorator is not None
            else baseline_analyzer
        )
        orchestrator = MockSearchOrchestrator(
            controller=controller,
            analyzer=selected_analyzer,
            providers=providers,
            config_hash=config_hash,
            prompt_version=lock.baseline.prompt_version,
            analysis_estimate=analysis_estimate,
            provider_estimate=provider_estimates["openalex"],
            provider_estimates=provider_estimates,
            routing_limits=(
                lock.baseline.retrieval.openalex_calls_min,
                lock.baseline.retrieval.openalex_calls_max,
                lock.baseline.retrieval.semantic_scholar_calls_max,
            ),
            execution_mode="replay",
            document_ranker=document_ranker,
            max_output_papers=lock.baseline.retrieval.max_output_papers,
            openalex_supplement_selector=(
                select_production_cross_vocabulary_supplement
                if lock.baseline.cross_vocabulary_supplement is not None
                else None
            ),
            max_total_openalex_actions=(
                lock.baseline.cross_vocabulary_supplement.max_total_openalex_actions
                if lock.baseline.cross_vocabulary_supplement is not None
                else None
            ),
            query_plan_enricher=query_plan_enricher,
            identifier_map=identifier_map,
            identifier_alias_count=identifier_alias_count,
            max_raw_candidates=lock.baseline.retrieval.max_raw_candidates,
            max_deduplicated_candidates=(
                lock.baseline.retrieval.max_deduplicated_candidates
            ),
            max_additional_raw_candidates=(
                lock.baseline.cross_vocabulary_supplement.max_additional_raw_candidates
                if lock.baseline.cross_vocabulary_supplement is not None
                else None
            ),
            max_total_raw_candidates=(
                lock.baseline.cross_vocabulary_supplement.max_total_raw_candidates
                if lock.baseline.cross_vocabulary_supplement is not None
                else None
            ),
            low_confidence_analyzer=low_confidence_analyzer,
            low_confidence_prompt_version=(
                low_confidence_binding.prompt_version
                if low_confidence_binding is not None
                else None
            ),
            low_confidence_analysis_estimate=_low_confidence_analysis_estimate(
                controller,
                lock=lock,
                pricer=pricer,
            ),
            max_low_confidence_raw_candidates=(
                low_confidence_binding.max_additional_raw_candidates
                if low_confidence_binding is not None
                else None
            ),
            max_low_confidence_deduplicated_candidates=(
                low_confidence_binding.max_additional_deduplicated_candidates
                if low_confidence_binding is not None
                else None
            ),
        )
        return orchestrator

    return create


@dataclass(frozen=True, repr=False)
class _LiveCredentials:
    llm: SecretStr
    semantic_scholar: SecretStr | None
    openalex_keys: tuple[SecretStr, ...] = ()
    openalex_mailto: str | None = None

    def __repr__(self) -> str:
        return (
            "_LiveCredentials(llm=**********, "
            f"openalex_keys={len(self.openalex_keys)}x**********, "
            "semantic_scholar=**********)"
        )


def _openalex_api_keys(environ: Mapping[str, str]) -> tuple[SecretStr, ...]:
    """Collect OPENALEX_API_KEY, OPENALEX_API_KEY_2, ... in order."""
    keys: list[str] = []
    index = 0
    while True:
        name = "OPENALEX_API_KEY" if index == 0 else f"OPENALEX_API_KEY_{index + 1}"
        value = environ.get(name)
        if value is None:
            break
        value = value.strip()
        if value and value not in keys:
            keys.append(value)
        index += 1
    return tuple(SecretStr(key) for key in keys)


class _LiveRunOrchestrator:
    def __init__(
        self,
        *,
        orchestrator: SearchOrchestrator,
        capture_store: DependencyCaptureStore,
        client: httpx.AsyncClient,
        artifact_factory: ArtifactFactory,
        seal_on_completion: bool,
    ) -> None:
        self._orchestrator = orchestrator
        self._capture_store = capture_store
        self._client = client
        self._artifact_factory = artifact_factory
        self._seal_on_completion = seal_on_completion

    async def run(
        self,
        query: str,
        *,
        max_provider_results: int,
    ) -> OrchestratorResult:
        try:
            result = await self._orchestrator.run(
                query,
                max_provider_results=max_provider_results,
            )
            if self._seal_on_completion:
                manifest = self._capture_store.seal()
                return result.model_copy(
                    update={
                        "snapshot_set_id": manifest.snapshot_set_id,
                        "snapshot_captured_at": manifest.sealed_at,
                    }
                )
            return result
        finally:
            await self._client.aclose()
            self._artifact_factory.release_client(self._client)


class _LiveOrchestratorFactory:
    def __init__(
        self,
        *,
        lock: InputLock,
        config_hash: Sha256,
        pricer: ActualCostPricer,
        credentials: _LiveCredentials,
        artifact_factory: ArtifactFactory,
        prompt_instructions: str | None = None,
        low_confidence_prompt_instructions: str | None = None,
        analyzer_decorator: AnalyzerDecorator | None = None,
        document_ranker: DocumentRankingStage | None = None,
        query_plan_enricher: SupervisedLexicalBridgePlanEnricher | None = None,
        identifier_map: IdentifierMap | None = None,
        identifier_alias_count: int = 0,
    ) -> None:
        self._lock = lock
        self._config_hash = config_hash
        self._pricer = pricer
        self._credentials = credentials
        self._artifact_factory = artifact_factory
        self._prompt_instructions = prompt_instructions
        self._low_confidence_prompt_instructions = (
            low_confidence_prompt_instructions
        )
        self._analyzer_decorator = analyzer_decorator
        self._document_ranker = document_ranker
        self._query_plan_enricher = query_plan_enricher
        self._identifier_map = identifier_map
        self._identifier_alias_count = identifier_alias_count

    def __repr__(self) -> str:
        return "_LiveOrchestratorFactory(credentials=**********)"

    def __call__(
        self,
        controller: HardBudgetController,
        run_id: str,
    ) -> _LiveRunOrchestrator:
        lock = self._lock
        seal_on_completion = not self._artifact_factory.has_capture_session(run_id=run_id)
        capture_store = self._artifact_factory.start_dependency_capture(run_id=run_id)
        timeout = httpx.Timeout(
            connect=lock.baseline.timeout.connect_seconds,
            read=lock.baseline.timeout.read_seconds,
            write=lock.baseline.timeout.write_seconds,
            pool=lock.baseline.timeout.pool_seconds,
        )
        client = httpx.AsyncClient(timeout=timeout)
        self._artifact_factory.register_client(client)
        llm_client = OpenAICompatibleLLMClient(
            client=client,
            base_url=_LLM_BASE_URL,
            model=lock.baseline.primary_model,
            api_key=self._credentials.llm.get_secret_value(),
            prompt_version=lock.baseline.prompt_version,
        )
        analysis_estimate, provider_estimates = _live_estimates(
            controller,
            lock=lock,
            pricer=self._pricer,
        )
        analyzer = LiveCaptureLLMAnalyzer(
            client=llm_client,
            capture_store=capture_store,
            pricer=self._pricer,
            controller=controller,
            prompt_artifact_sha256=(lock.baseline.planner.prompt_config.sha256),
            prompt_instructions=self._prompt_instructions,
        )
        low_confidence_binding = lock.baseline.low_confidence_llm_supplement
        low_confidence_analyzer: QueryAnalyzer | None = None
        if low_confidence_binding is not None:
            low_confidence_client = OpenAICompatibleLLMClient(
                client=client,
                base_url=_LLM_BASE_URL,
                model=lock.baseline.primary_model,
                api_key=self._credentials.llm.get_secret_value(),
                prompt_version=low_confidence_binding.prompt_version,
            )
            low_confidence_analyzer = _AnalyzerBridge(
                LiveCaptureLLMAnalyzer(
                    client=low_confidence_client,
                    capture_store=capture_store,
                    pricer=self._pricer,
                    controller=controller,
                    prompt_artifact_sha256=(
                        low_confidence_binding.prompt_config.sha256
                    ),
                    prompt_instructions=(
                        self._low_confidence_prompt_instructions
                    ),
                )
            )
        providers = {
            "openalex": LiveCaptureSearchProvider(
                dependency="openalex",
                client=client,
                capture_store=capture_store,
                pricer=self._pricer,
                controller=controller,
                api_key=(
                    self._credentials.openalex_keys[0].get_secret_value()
                    if self._credentials.openalex_keys
                    else None
                ),
                additional_api_keys=(
                    [key.get_secret_value() for key in self._credentials.openalex_keys[1:]]
                    if len(self._credentials.openalex_keys) > 1
                    else ()
                ),
                mailto=self._credentials.openalex_mailto,
            ),
            "semantic_scholar": LiveCaptureSearchProvider(
                dependency="semantic_scholar",
                client=client,
                capture_store=capture_store,
                pricer=self._pricer,
                controller=controller,
                api_key=(
                    self._credentials.semantic_scholar.get_secret_value()
                    if self._credentials.semantic_scholar is not None
                    else None
                ),
            ),
        }
        baseline_analyzer: QueryAnalyzer = _AnalyzerBridge(analyzer)
        selected_analyzer = (
            self._analyzer_decorator(baseline_analyzer)
            if self._analyzer_decorator is not None
            else baseline_analyzer
        )
        orchestrator = MockSearchOrchestrator(
            controller=controller,
            analyzer=selected_analyzer,
            providers=providers,
            config_hash=self._config_hash,
            prompt_version=lock.baseline.prompt_version,
            analysis_estimate=analysis_estimate,
            provider_estimate=provider_estimates["openalex"],
            provider_estimates=provider_estimates,
            pricer=self._pricer,
            provider_adapter_names={
                "openalex": "openalex-works-v1",
                "semantic_scholar": "semantic-graph-v1",
            },
            routing_limits=(
                lock.baseline.retrieval.openalex_calls_min,
                lock.baseline.retrieval.openalex_calls_max,
                lock.baseline.retrieval.semantic_scholar_calls_max,
            ),
            execution_mode="live",
            document_ranker=self._document_ranker,
            max_output_papers=lock.baseline.retrieval.max_output_papers,
            openalex_supplement_selector=(
                select_production_cross_vocabulary_supplement
                if lock.baseline.cross_vocabulary_supplement is not None
                else None
            ),
            max_total_openalex_actions=(
                lock.baseline.cross_vocabulary_supplement.max_total_openalex_actions
                if lock.baseline.cross_vocabulary_supplement is not None
                else None
            ),
            query_plan_enricher=self._query_plan_enricher,
            identifier_map=self._identifier_map,
            identifier_alias_count=self._identifier_alias_count,
            max_raw_candidates=lock.baseline.retrieval.max_raw_candidates,
            max_deduplicated_candidates=(
                lock.baseline.retrieval.max_deduplicated_candidates
            ),
            max_additional_raw_candidates=(
                lock.baseline.cross_vocabulary_supplement.max_additional_raw_candidates
                if lock.baseline.cross_vocabulary_supplement is not None
                else None
            ),
            max_total_raw_candidates=(
                lock.baseline.cross_vocabulary_supplement.max_total_raw_candidates
                if lock.baseline.cross_vocabulary_supplement is not None
                else None
            ),
            low_confidence_analyzer=low_confidence_analyzer,
            low_confidence_prompt_version=(
                low_confidence_binding.prompt_version
                if low_confidence_binding is not None
                else None
            ),
            low_confidence_analysis_estimate=_low_confidence_analysis_estimate(
                controller,
                lock=lock,
                pricer=self._pricer,
            ),
            max_low_confidence_raw_candidates=(
                low_confidence_binding.max_additional_raw_candidates
                if low_confidence_binding is not None
                else None
            ),
            max_low_confidence_deduplicated_candidates=(
                low_confidence_binding.max_additional_deduplicated_candidates
                if low_confidence_binding is not None
                else None
            ),
        )
        return _LiveRunOrchestrator(
            orchestrator=orchestrator,
            capture_store=capture_store,
            client=client,
            artifact_factory=self._artifact_factory,
            seal_on_completion=seal_on_completion,
        )


class CompositionRoot:
    """Build the only production search boundary from verified immutable inputs."""

    @classmethod
    def compose_server(
        cls,
        *,
        replay_lock_path: Path,
        snapshot_manifest_path: Path,
        artifact_root: Path,
        capture_output_root: Path,
        live_authorized: bool,
        environ: Mapping[str, str] | None = None,
        runtime_config: RuntimeConfig | None = None,
        ablation_config: Path | None = None,
        experiment_dependencies: object | None = None,
        analyzer_decorator: AnalyzerDecorator | None = None,
    ) -> ServerApplicationBundle:
        """Bind one replay service and optionally authorize isolated live captures."""
        try:
            replay_lock_bytes = replay_lock_path.read_bytes()
        except OSError as error:
            raise ValueError("replay lock is unavailable") from error
        verified_replay = load_verified_input_lock_bytes(
            replay_lock_bytes,
            artifact_root=artifact_root,
        )
        replay_lock = verified_replay.lock
        if not isinstance(replay_lock, ReplayLock):
            raise ValueError("server requires a replay lock")
        if live_authorized and not replay_lock.runtime_allow_live:
            raise ValueError("replay lock does not allow live execution")

        source_lock_path: Path | None = None
        source_lock_bytes: bytes | None = None
        if live_authorized:
            capture_root = capture_output_root.resolve()
            source_id_path = Path(replay_lock.source_capture_run_id)
            if (
                source_id_path.name != replay_lock.source_capture_run_id
                or source_id_path.parent != Path(".")
            ):
                raise ValueError("source capture run id escapes capture output root")
            try:
                source_root = (capture_root / replay_lock.source_capture_run_id).resolve(
                    strict=True
                )
            except OSError as error:
                raise ValueError("source live capture lock is unavailable") from error
            if (
                source_root.parent != capture_root
                or source_root.name != replay_lock.source_capture_run_id
            ):
                raise ValueError("source capture run id escapes capture output root")
            source_lock_path, source_lock_bytes = _read_confined_source_capture_file(
                source_root, "config.lock.yaml"
            )
            _, source_replay_bytes = _read_confined_source_capture_file(
                source_root, "replay.lock.yaml"
            )
            if source_replay_bytes != replay_lock_bytes:
                raise ValueError("source capture lineage does not match replay lock")
            source_verified = load_verified_input_lock_bytes(
                source_lock_bytes,
                artifact_root=artifact_root,
            )
            if isinstance(source_verified.lock, ReplayLock):
                raise ValueError("source capture must use a live input lock")
            for field_name in (
                "schema_version",
                "source_git_sha",
                "runtime_allow_live",
                "frozen_data",
                "baseline",
                "budget_config",
                "pricing_policy",
                "quality_gates",
                "capture_policy",
                "project_ledger",
            ):
                if getattr(source_verified.lock, field_name) != getattr(replay_lock, field_name):
                    raise ValueError("source capture lock does not match replay lineage")

        replay = cls.compose(
            lock_path=replay_lock_path,
            mode="replay",
            artifact_root=artifact_root,
            output_root=capture_output_root,
            snapshot_manifest_path=snapshot_manifest_path,
            environ=environ,
            lock_bytes=replay_lock_bytes,
            runtime_config=runtime_config,
            ablation_config=ablation_config,
            experiment_dependencies=experiment_dependencies,
            analyzer_decorator=analyzer_decorator,
        )
        capture_artifact_factory = ArtifactFactory(output_root=capture_output_root.resolve())
        active_live: dict[int, ApplicationBundle] = {}
        closed = False

        def release_live(bundle: ApplicationBundle) -> None:
            active_live.pop(id(bundle), None)

        live_factory: Callable[[], ApplicationBundle] | None = None
        live_service_factory = None
        if live_authorized:
            assert source_lock_path is not None
            assert source_lock_bytes is not None

            def create_live() -> ApplicationBundle:
                if closed:
                    raise RuntimeError("server is shutting down")
                bundle = cls.compose(
                    lock_path=source_lock_path,
                    mode="live",
                    artifact_root=artifact_root,
                    output_root=capture_output_root,
                    network_authorized=True,
                    environ=environ,
                    lock_bytes=source_lock_bytes,
                    artifact_factory=capture_artifact_factory,
                    runtime_config=runtime_config,
                    ablation_config=ablation_config,
                    experiment_dependencies=experiment_dependencies,
                    analyzer_decorator=analyzer_decorator,
                )
                active_live[id(bundle)] = bundle
                return bundle

            def create_live_service() -> _RequestLiveCaptureService:
                bundle = create_live()
                return _RequestLiveCaptureService(
                    bundle=bundle,
                    input_lock_bytes=source_lock_bytes,
                    release_bundle=release_live,
                    run_id_factory=lambda: f"serve-{uuid4()}",
                )

            live_factory = create_live
            live_service_factory = create_live_service

        router = SearchServiceRouter(
            replay_service=replay.service,
            readiness=replay.readiness_probe(),
            live_service_factory=live_service_factory,
            runtime_allow_live=replay_lock.runtime_allow_live,
            server_live_authorized=live_authorized,
        )

        async def close_server() -> None:
            nonlocal closed
            if closed:
                return
            closed = True
            bundles = list(active_live.values())
            active_live.clear()
            try:
                for bundle in bundles:
                    await bundle.aclose()
            finally:
                try:
                    await capture_artifact_factory.aclose()
                finally:
                    await replay.aclose()

        return ServerApplicationBundle(
            replay=replay,
            live_factory=live_factory,
            capture_artifact_factory=capture_artifact_factory,
            service_router=router,
            _close=close_server,
        )

    @classmethod
    def compose(
        cls,
        *,
        lock_path: Path,
        mode: SearchMode,
        artifact_root: Path,
        output_root: Path,
        snapshot_manifest_path: Path | None = None,
        network_authorized: bool = False,
        environ: Mapping[str, str] | None = None,
        lock_bytes: bytes | None = None,
        artifact_factory: ArtifactFactory | None = None,
        runtime_config: RuntimeConfig | None = None,
        ablation_config: Path | None = None,
        experiment_dependencies: object | None = None,
        analyzer_decorator: AnalyzerDecorator | None = None,
    ) -> ApplicationBundle:
        verified = (
            load_verified_input_lock(lock_path, artifact_root=artifact_root)
            if lock_bytes is None
            else load_verified_input_lock_bytes(
                lock_bytes,
                artifact_root=artifact_root,
            )
        )
        lock = verified.lock
        del ablation_config, experiment_dependencies
        experiment_definition = ExperimentDefinition(
            name="main-baseline",
            flags=ExperimentFlags(),
            strategy="fixed-one-round",
        )
        if runtime_config is not None and runtime_config.experiment != "main-baseline":
            raise ValueError("unknown experiment name")
        validate_mode_authorization(
            mode=mode,
            runtime_allow_live=lock.runtime_allow_live,
            network_authorized=network_authorized,
        )
        locked_config_hash = lock_sha256(lock)
        experiment_config = None
        config_hash = locked_config_hash
        document_ranker = _locked_document_ranker(
            lock,
            verified.artifact_bytes,
            verified.ranker_artifact_failures,
        )
        query_plan_enricher = _locked_query_plan_enricher(
            lock,
            verified.artifact_bytes,
        )
        identifier_map, identifier_alias_count = load_locked_identifier_map(
            lock,
            verified.artifact_bytes,
        )
        budget_profile = _locked_budget_profile(lock.budget_config.path)
        budgets: dict[str, SearchBudget] = {
            budget_profile: parse_budget_bytes(verified.artifact_bytes[lock.budget_config.path]),
        }
        resolved_artifact_factory = artifact_factory or ArtifactFactory(
            output_root=output_root.resolve()
        )
        binding: ModeBinding
        orchestrator_factory: Callable[[HardBudgetController, str], SearchOrchestrator]
        snapshot_captured_at: datetime | None = None

        if mode == "replay":
            if not isinstance(lock, ReplayLock):
                raise ValueError("replay mode requires a replay lock")
            if snapshot_manifest_path is None:
                raise ValueError("snapshot manifest is required for replay")
            manifest_path = _confined_manifest_path(
                snapshot_manifest_path,
                artifact_root=artifact_root,
            )
            reader = DependencySnapshotReader(
                manifest_path,
                snapshot_manifest_sha256=lock.snapshot_manifest_sha256,
                snapshot_set_id=lock.snapshot_set_id,
            )
            binding = ModeBinding(
                mode="replay",
                network_authorized=False,
                snapshot_set_id=reader.snapshot_set_id,
                snapshot_manifest_sha256=lock.snapshot_manifest_sha256,
            )
            orchestrator_factory = _replay_factory(
                reader=reader,
                lock=lock,
                pricer=ActualCostPricer(
                    parse_pricing_policy_bytes(verified.artifact_bytes[lock.pricing_policy.path])
                ),
                config_hash=config_hash,
                analyzer_decorator=analyzer_decorator,
                document_ranker=document_ranker,
                query_plan_enricher=query_plan_enricher,
                identifier_map=identifier_map,
                identifier_alias_count=identifier_alias_count,
            )
            readiness_probe = _replay_readiness(binding)
            snapshot_captured_at = _snapshot_manifest_time(manifest_path)
        else:
            resolved_environ = dict(os.environ if environ is None else environ)
            llm_api_key = resolved_environ.get("LLM_API_KEY")
            if not llm_api_key:
                raise ValueError("LLM_API_KEY is required for live execution")
            pricing_policy = parse_pricing_policy_bytes(
                verified.artifact_bytes[lock.pricing_policy.path]
            )
            pricer = ActualCostPricer(pricing_policy)
            credentials = _LiveCredentials(
                llm=SecretStr(llm_api_key),
                openalex_keys=_openalex_api_keys(resolved_environ),
                semantic_scholar=(
                    SecretStr(resolved_environ["SEMANTIC_SCHOLAR_API_KEY"])
                    if resolved_environ.get("SEMANTIC_SCHOLAR_API_KEY")
                    else None
                ),
                openalex_mailto=resolved_environ.get("OPENALEX_MAILTO"),
            )
            orchestrator_factory = _LiveOrchestratorFactory(
                lock=lock,
                config_hash=config_hash,
                pricer=pricer,
                credentials=credentials,
                artifact_factory=resolved_artifact_factory,
                prompt_instructions=_prompt_system_message(
                    verified.artifact_bytes[lock.baseline.planner.prompt_config.path]
                ),
                low_confidence_prompt_instructions=(
                    _prompt_system_message(
                        verified.artifact_bytes[
                            lock.baseline.low_confidence_llm_supplement.prompt_config.path
                        ]
                    )
                    if lock.baseline.low_confidence_llm_supplement is not None
                    else None
                ),
                analyzer_decorator=analyzer_decorator,
                document_ranker=document_ranker,
                query_plan_enricher=query_plan_enricher,
                identifier_map=identifier_map,
                identifier_alias_count=identifier_alias_count,
            )

            binding = ModeBinding(
                mode="live",
                network_authorized=True,
                snapshot_set_id=None,
                snapshot_manifest_sha256=None,
            )
            readiness_probe = _live_readiness(
                binding,
                artifact_root,
                required_dependencies=(
                    "llm",
                    "openalex",
                    *(
                        ("semantic_scholar",)
                        if lock.baseline.retrieval.semantic_scholar_calls_max > 0
                        else ()
                    ),
                ),
            )

        service = SearchApplicationService(
            orchestrator_factory=orchestrator_factory,
            budgets=budgets,
            mode=mode,
            snapshot_set_id=binding.snapshot_set_id or "live-capture-pending",
            snapshot_captured_at=snapshot_captured_at,
            git_sha=lock.source_git_sha,
            max_provider_results=lock.baseline.retrieval.max_results_per_subquery,
        )
        return ApplicationBundle(
            service=service,
            readiness_probe=readiness_probe,
            config_hash=config_hash,
            artifact_factory=resolved_artifact_factory,
            experiment_id=experiment_definition.name,
            optional_modules=experiment_definition.flags.model_dump(mode="python"),
            experiment_config=experiment_config,
            source_git_sha=lock.source_git_sha,
            prompt_version=lock.baseline.prompt_version,
            mode_binding=binding,
            _owns_artifact_factory=artifact_factory is None,
        )
