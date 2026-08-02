"""Single lock- and mode-bound application composition root."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Protocol

import httpx
from pydantic import SecretStr

from paper_search.application.artifacts import ArtifactFactory
from paper_search.application.contracts import ReadyHealthResponse
from paper_search.application.locks import (
    InputLock,
    ReplayLock,
    load_input_lock,
    lock_sha256,
)
from paper_search.application.modes import ModeBinding
from paper_search.application.service import SearchApplicationService, SearchOrchestrator
from paper_search.config import load_budget, validate_mode_authorization
from paper_search.control.budget import HardBudgetController
from paper_search.control.pricing import ActualCostPricer, load_pricing_policy
from paper_search.domain.models import (
    BudgetReservation,
    DependencyName,
    DependencyStatus,
    ProviderResult,
    SearchMode,
    SearchBudget,
    Sha256,
    UsageActual,
    UsageEstimate,
)
from paper_search.llm.client import OpenAICompatibleLLMClient
from paper_search.llm.snapshot_adapters import (
    LiveCaptureLLMAnalyzer,
    ReplayLLMAnalyzer,
)
from paper_search.pipeline.orchestrator import MockSearchOrchestrator, OrchestratorResult
from paper_search.retrieval.snapshot_adapters import (
    LiveCaptureSearchProvider,
    ReplaySearchProvider,
)
from paper_search.storage.dependency_snapshot import (
    DependencyCaptureStore,
    DependencySnapshotManifestV2,
    DependencySnapshotReader,
)


_LLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
_DEPENDENCIES: tuple[DependencyName, ...] = (
    "llm",
    "openalex",
    "semantic_scholar",
)


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


@dataclass(frozen=True)
class ApplicationBundle:
    service: SearchApplicationService
    readiness_probe: Callable[[], ReadyHealthResponse]
    config_hash: Sha256
    artifact_factory: ArtifactFactory
    experiment_id: Literal["main-baseline"]
    source_git_sha: str
    prompt_version: Literal["query-analyze-v1"]
    mode_binding: ModeBinding

    async def aclose(self) -> None:
        await self.artifact_factory.aclose()


def _replay_estimates(
    controller: HardBudgetController,
) -> tuple[UsageEstimate, dict[str, UsageEstimate]]:
    budget = controller.budget
    output_tokens = budget.max_total_tokens // 6
    input_tokens = budget.max_total_tokens - output_tokens
    elapsed_ms = min(
        budget.max_elapsed_seconds * 1_000,
        3 * 20 * 1_000,
    )
    return (
        UsageEstimate(
            llm_calls=3,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_cny=0,
            elapsed_ms=elapsed_ms,
        ),
        {
            dependency: UsageEstimate(
                search_api_calls=3,
                cost_cny=0,
                elapsed_ms=elapsed_ms,
            )
            for dependency in ("openalex", "semantic_scholar")
        },
    )


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
    llm_usage = pricer.value_actual(
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


def _live_readiness(binding: ModeBinding) -> Callable[[], ReadyHealthResponse]:
    response = ReadyHealthResponse(
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
    return lambda: response


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
    config_hash: Sha256,
) -> Callable[[HardBudgetController, str], MockSearchOrchestrator]:
    def create(
        controller: HardBudgetController,
        run_id: str,
    ) -> MockSearchOrchestrator:
        del run_id
        analysis_estimate, provider_estimates = _replay_estimates(controller)
        analyzer = ReplayLLMAnalyzer(
            reader=reader,
            model_id=lock.baseline.primary_model,
            prompt_version=lock.baseline.prompt_version,
        )
        providers = {
            "openalex": ReplaySearchProvider(dependency="openalex", reader=reader),
            "semantic_scholar": ReplaySearchProvider(
                dependency="semantic_scholar",
                reader=reader,
            ),
        }
        return MockSearchOrchestrator(
            controller=controller,
            analyzer=_AnalyzerBridge(analyzer),
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
            embedding_ranker=None,
            citation_expander=None,
            constraint_reranker=None,
        )

    return create


@dataclass(frozen=True, repr=False)
class _LiveCredentials:
    llm: SecretStr
    openalex: SecretStr | None
    semantic_scholar: SecretStr | None

    def __repr__(self) -> str:
        return "_LiveCredentials(llm=**********, openalex=**********, semantic_scholar=**********)"


class _LiveRunOrchestrator:
    def __init__(
        self,
        *,
        orchestrator: MockSearchOrchestrator,
        capture_store: DependencyCaptureStore,
        client: httpx.AsyncClient,
        artifact_factory: ArtifactFactory,
    ) -> None:
        self._orchestrator = orchestrator
        self._capture_store = capture_store
        self._client = client
        self._artifact_factory = artifact_factory

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
            manifest = self._capture_store.seal()
            return result.model_copy(
                update={
                    "snapshot_set_id": manifest.snapshot_set_id,
                    "snapshot_captured_at": manifest.sealed_at,
                }
            )
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
    ) -> None:
        self._lock = lock
        self._config_hash = config_hash
        self._pricer = pricer
        self._credentials = credentials
        self._artifact_factory = artifact_factory

    def __repr__(self) -> str:
        return "_LiveOrchestratorFactory(credentials=**********)"

    def __call__(
        self,
        controller: HardBudgetController,
        run_id: str,
    ) -> _LiveRunOrchestrator:
        lock = self._lock
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
        )
        providers = {
            "openalex": LiveCaptureSearchProvider(
                dependency="openalex",
                client=client,
                capture_store=capture_store,
                pricer=self._pricer,
                controller=controller,
                api_key=(
                    self._credentials.openalex.get_secret_value()
                    if self._credentials.openalex is not None
                    else None
                ),
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
        orchestrator = MockSearchOrchestrator(
            controller=controller,
            analyzer=_AnalyzerBridge(analyzer),
            providers=providers,
            config_hash=self._config_hash,
            prompt_version=lock.baseline.prompt_version,
            analysis_estimate=analysis_estimate,
            provider_estimate=provider_estimates["openalex"],
            provider_estimates=provider_estimates,
            routing_limits=(
                lock.baseline.retrieval.openalex_calls_min,
                lock.baseline.retrieval.openalex_calls_max,
                lock.baseline.retrieval.semantic_scholar_calls_max,
            ),
            execution_mode="live",
            embedding_ranker=None,
            citation_expander=None,
            constraint_reranker=None,
        )
        return _LiveRunOrchestrator(
            orchestrator=orchestrator,
            capture_store=capture_store,
            client=client,
            artifact_factory=self._artifact_factory,
        )


class CompositionRoot:
    """Build the only production search boundary from verified immutable inputs."""

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
    ) -> ApplicationBundle:
        lock = load_input_lock(lock_path, artifact_root=artifact_root)
        validate_mode_authorization(
            mode=mode,
            runtime_allow_live=lock.runtime_allow_live,
            network_authorized=network_authorized,
        )
        config_hash = lock_sha256(lock)
        budget_profile = _locked_budget_profile(lock.budget_config.path)
        budgets: dict[str, SearchBudget] = {
            budget_profile: load_budget(artifact_root / lock.budget_config.path),
        }
        artifact_factory = ArtifactFactory(output_root=output_root.resolve())
        binding: ModeBinding
        orchestrator_factory: Callable[
            [HardBudgetController, str], SearchOrchestrator
        ]
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
                config_hash=config_hash,
            )
            readiness_probe = _replay_readiness(binding)
            snapshot_captured_at = _snapshot_manifest_time(manifest_path)
        else:
            resolved_environ = dict(os.environ if environ is None else environ)
            llm_api_key = resolved_environ.get("LLM_API_KEY")
            if not llm_api_key:
                raise ValueError("LLM_API_KEY is required for live execution")
            pricing_policy = load_pricing_policy(
                artifact_root / lock.pricing_policy.path
            )
            pricer = ActualCostPricer(pricing_policy)
            credentials = _LiveCredentials(
                llm=SecretStr(llm_api_key),
                openalex=(
                    SecretStr(resolved_environ["OPENALEX_API_KEY"])
                    if resolved_environ.get("OPENALEX_API_KEY")
                    else None
                ),
                semantic_scholar=(
                    SecretStr(resolved_environ["SEMANTIC_SCHOLAR_API_KEY"])
                    if resolved_environ.get("SEMANTIC_SCHOLAR_API_KEY")
                    else None
                ),
            )
            orchestrator_factory = _LiveOrchestratorFactory(
                lock=lock,
                config_hash=config_hash,
                pricer=pricer,
                credentials=credentials,
                artifact_factory=artifact_factory,
            )

            binding = ModeBinding(
                mode="live",
                network_authorized=True,
                snapshot_set_id=None,
                snapshot_manifest_sha256=None,
            )
            readiness_probe = _live_readiness(binding)

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
            artifact_factory=artifact_factory,
            experiment_id="main-baseline",
            source_git_sha=lock.source_git_sha,
            prompt_version="query-analyze-v1",
            mode_binding=binding,
        )
