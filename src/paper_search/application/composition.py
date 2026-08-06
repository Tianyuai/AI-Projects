"""Single lock- and mode-bound application composition root."""

from __future__ import annotations

import os
import stat
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, NoReturn, Protocol
from uuid import uuid4

import httpx
import yaml
from pydantic import SecretStr

from paper_search.application.artifacts import ArtifactFactory
from paper_search.application.experiments import (
    ExperimentComponents,
    ExperimentDependencyFactory,
    ExperimentDefinition,
    ExperimentFlags,
    ExperimentName,
    build_experiment_components,
    load_experiment_definition,
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
    ExperimentConfigEvidence,
    RuntimeConfig,
    experiment_config_hash,
    parse_budget_bytes,
    validate_mode_authorization,
)
from paper_search.control.budget import HardBudgetController
from paper_search.control.pricing import ActualCostPricer, parse_pricing_policy_bytes
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
from paper_search.pipeline.orchestrator import (
    EvolutionSearchOrchestrator,
    MockSearchOrchestrator,
    OrchestratorResult,
)
from paper_search.retrieval.snapshot_adapters import (
    LiveCaptureSearchProvider,
    ReplaySearchProvider,
)
from paper_search.retrieval.base import SearchProvider
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


class _BaselineDependencyTrap:
    def build_embedding_ranker(self) -> NoReturn:
        raise AssertionError("main baseline cannot construct embedding")

    def build_citation_expander(self) -> NoReturn:
        raise AssertionError("main baseline cannot construct citation expansion")

    def build_constraint_reranker(self) -> NoReturn:
        raise AssertionError("main baseline cannot construct LLM reranking")

    def build_title_candidate_stage(self) -> NoReturn:
        raise AssertionError("main baseline cannot construct title candidates")


_MAIN_BASELINE_COMPONENTS: ExperimentComponents = build_experiment_components(
    ExperimentDefinition(
        name="main-baseline",
        flags=ExperimentFlags(),
        strategy="fixed-one-round",
    ),
    dependencies=_BaselineDependencyTrap(),
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
        if (
            resolved_before.parent != root
            or not stat.S_ISREG(before.st_mode)
        ):
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


@dataclass(frozen=True)
class _RequestExperimentDependencies:
    analyzer: _AnalyzerAdapter
    providers: Mapping[str, SearchProvider]
    analysis_estimate: UsageEstimate
    provider_estimates: Mapping[str, UsageEstimate]
    runtime_config: RuntimeConfig

    def build_embedding_ranker(self) -> Any:
        from paper_search.ranking.embedding import EmbeddingRanker
        from paper_search.ranking.sentence_transformer import (
            sentence_transformer_factory,
        )

        config = self.runtime_config.embedding
        return EmbeddingRanker(
            encoder_factory=sentence_transformer_factory(config.model_id),
            model_id=config.model_id,
            preferred_device=config.device,
            batch_size=config.batch_size,
            fallback_to_cpu=config.fallback_to_cpu,
        )

    def build_citation_expander(self) -> Any:
        from paper_search.graph.provider_stage import ProviderCitationExpansionStage

        provider = self.providers.get("semantic_scholar")
        if provider is None:
            raise ValueError("citation expansion requires semantic_scholar")
        return ProviderCitationExpansionStage(
            provider=provider,
            call_estimate=self.provider_estimates["semantic_scholar"],
        )

    def build_constraint_reranker(self) -> Any:
        from paper_search.ranking.llm_stage import LLMConstraintRerankingStage

        return LLMConstraintRerankingStage(
            analyzer=self.analyzer,
            call_estimate=self.analysis_estimate,
        )

    def build_title_candidate_stage(self) -> Any:
        from paper_search.retrieval.title_candidates import LLMTitleCandidateStage

        provider = self.providers.get("openalex")
        if provider is None:
            raise ValueError("title candidates require openalex")
        return LLMTitleCandidateStage(
            analyzer=self.analyzer,
            provider=provider,
            max_titles=20,
            llm_estimate=self.analysis_estimate.model_copy(
                update={
                    "llm_calls": 1,
                    "input_tokens": 2000,
                    "output_tokens": 1500,
                    "elapsed_ms": 30000,
                }
            ),
            search_estimate=self.provider_estimates["openalex"].model_copy(
                update={"search_api_calls": 1, "elapsed_ms": 8000}
            ),
        )


def _request_components(
    *,
    definition: ExperimentDefinition,
    dependencies: ExperimentDependencyFactory | None,
    analyzer: _AnalyzerAdapter,
    providers: Mapping[str, SearchProvider],
    analysis_estimate: UsageEstimate,
    provider_estimates: Mapping[str, UsageEstimate],
    runtime_config: RuntimeConfig | None,
) -> ExperimentComponents:
    if definition.name == "main-baseline":
        return _MAIN_BASELINE_COMPONENTS
    resolved_dependencies = dependencies
    if resolved_dependencies is None:
        if runtime_config is None:
            raise ValueError("optional experiments require validated runtime config")
        resolved_dependencies = _RequestExperimentDependencies(
            analyzer=analyzer,
            providers=providers,
            analysis_estimate=analysis_estimate,
            provider_estimates=provider_estimates,
            runtime_config=runtime_config,
        )
    return build_experiment_components(
        definition,
        dependencies=resolved_dependencies,
    )


def _with_evolution(
    *,
    orchestrator: MockSearchOrchestrator,
    controller: HardBudgetController,
    components: ExperimentComponents,
) -> SearchOrchestrator:
    if components.evolution_strategy == "fixed_one_round":
        return orchestrator
    return EvolutionSearchOrchestrator(
        single_round=orchestrator,
        controller=controller,
        strategy=components.evolution_strategy,
    )


@dataclass(frozen=True)
class ApplicationBundle:
    service: SearchApplicationService
    readiness_probe: Callable[[], ReadyHealthResponse]
    config_hash: Sha256
    artifact_factory: ArtifactFactory
    experiment_id: ExperimentName
    optional_modules: dict[str, bool]
    experiment_config: ExperimentConfigEvidence | None
    source_git_sha: str
    prompt_version: Literal["query-analyze-v1"]
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
                update={
                    "outcome": execution.outcome.model_copy(
                        update={"response": response}
                    )
                }
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


def _prompt_system_message(prompt_bytes: bytes) -> str:
    """Build the deterministic system message from the bound prompt artifact."""
    try:
        raw = yaml.safe_load(prompt_bytes)
    except yaml.YAMLError as error:
        raise ValueError("invalid prompt artifact") from error
    if not isinstance(raw, dict):
        raise ValueError("invalid prompt artifact")
    instructions = raw.get("instructions", [])
    if not isinstance(instructions, list) or not all(
        isinstance(item, str) for item in instructions
    ):
        raise ValueError("prompt instructions must be a list of strings")
    lines = ["Respond with a JSON object."]
    response_model = raw.get("response_model")
    if isinstance(response_model, str) and response_model:
        lines.append(f"The JSON object must match the {response_model} contract.")
    lines.extend(f"- {item}" for item in instructions)
    return "\n".join(lines)


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
    config_hash: Sha256,
    experiment_definition: ExperimentDefinition,
    experiment_dependencies: ExperimentDependencyFactory | None,
    runtime_config: RuntimeConfig | None,
) -> Callable[[HardBudgetController, str], SearchOrchestrator]:
    def create(
        controller: HardBudgetController,
        run_id: str,
    ) -> SearchOrchestrator:
        del run_id
        analysis_estimate, provider_estimates = _replay_estimates(controller)
        analyzer = ReplayLLMAnalyzer(
            reader=reader,
            model_id=lock.baseline.primary_model,
            prompt_artifact_sha256=lock.baseline.planner.prompt_config.sha256,
            prompt_version=lock.baseline.prompt_version,
        )
        providers = {
            "openalex": ReplaySearchProvider(dependency="openalex", reader=reader),
            "semantic_scholar": ReplaySearchProvider(
                dependency="semantic_scholar",
                reader=reader,
            ),
        }
        components = _request_components(
            definition=experiment_definition,
            dependencies=experiment_dependencies,
            analyzer=analyzer,
            providers=providers,
            analysis_estimate=analysis_estimate,
            provider_estimates=provider_estimates,
            runtime_config=runtime_config,
        )
        orchestrator = MockSearchOrchestrator(
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
            embedding_ranker=components.embedding_ranker,
            citation_expander=components.citation_expander,
            constraint_reranker=components.constraint_reranker,
            title_candidate_stage=components.title_candidate_stage,
            max_output_papers=lock.baseline.retrieval.max_output_papers,
        )
        return _with_evolution(
            orchestrator=orchestrator,
            controller=controller,
            components=components,
        )

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
        experiment_definition: ExperimentDefinition,
        experiment_dependencies: ExperimentDependencyFactory | None,
        runtime_config: RuntimeConfig | None,
        prompt_instructions: str | None = None,
    ) -> None:
        self._lock = lock
        self._config_hash = config_hash
        self._pricer = pricer
        self._credentials = credentials
        self._artifact_factory = artifact_factory
        self._experiment_definition = experiment_definition
        self._experiment_dependencies = experiment_dependencies
        self._runtime_config = runtime_config
        self._prompt_instructions = prompt_instructions

    def __repr__(self) -> str:
        return "_LiveOrchestratorFactory(credentials=**********)"

    def __call__(
        self,
        controller: HardBudgetController,
        run_id: str,
    ) -> _LiveRunOrchestrator:
        lock = self._lock
        seal_on_completion = not self._artifact_factory.has_capture_session(
            run_id=run_id
        )
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
            prompt_artifact_sha256=(
                lock.baseline.planner.prompt_config.sha256
            ),
            prompt_instructions=self._prompt_instructions,
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
        components = _request_components(
            definition=self._experiment_definition,
            dependencies=self._experiment_dependencies,
            analyzer=analyzer,
            providers=providers,
            analysis_estimate=analysis_estimate,
            provider_estimates=provider_estimates,
            runtime_config=self._runtime_config,
        )
        orchestrator = MockSearchOrchestrator(
            controller=controller,
            analyzer=_AnalyzerBridge(analyzer),
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
            embedding_ranker=components.embedding_ranker,
            citation_expander=components.citation_expander,
            constraint_reranker=components.constraint_reranker,
            title_candidate_stage=components.title_candidate_stage,
            max_output_papers=lock.baseline.retrieval.max_output_papers,
        )
        return _LiveRunOrchestrator(
            orchestrator=_with_evolution(
                orchestrator=orchestrator,
                controller=controller,
                components=components,
            ),
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
        ablation_config: Path = Path("configs/ablations.yaml"),
        experiment_dependencies: ExperimentDependencyFactory | None = None,
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
            if source_id_path.name != replay_lock.source_capture_run_id or source_id_path.parent != Path("."):
                raise ValueError("source capture run id escapes capture output root")
            try:
                source_root = (capture_root / replay_lock.source_capture_run_id).resolve(
                    strict=True
                )
            except OSError as error:
                raise ValueError("source live capture lock is unavailable") from error
            if source_root.parent != capture_root or source_root.name != replay_lock.source_capture_run_id:
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
                if getattr(source_verified.lock, field_name) != getattr(
                    replay_lock, field_name
                ):
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
        ablation_config: Path = Path("configs/ablations.yaml"),
        experiment_dependencies: ExperimentDependencyFactory | None = None,
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
        experiment_definition = (
            ExperimentDefinition(
                name="main-baseline",
                flags=ExperimentFlags(),
                strategy="fixed-one-round",
            )
            if runtime_config is None
            else load_experiment_definition(
                runtime_config.experiment,
                ablation_config=ablation_config,
            )
        )
        validate_mode_authorization(
            mode=mode,
            runtime_allow_live=lock.runtime_allow_live,
            network_authorized=network_authorized,
        )
        locked_config_hash = lock_sha256(lock)
        experiment_config = (
            None
            if experiment_definition.name == "main-baseline"
            else ExperimentConfigEvidence(
                experiment=experiment_definition,
                embedding=(
                    runtime_config.embedding
                    if runtime_config is not None
                    and experiment_definition.flags.embedding
                    else None
                ),
            )
        )
        config_hash = experiment_config_hash(
            input_lock_sha256=locked_config_hash,
            evidence=experiment_config,
        )
        budget_profile = _locked_budget_profile(lock.budget_config.path)
        budgets: dict[str, SearchBudget] = {
            budget_profile: parse_budget_bytes(
                verified.artifact_bytes[lock.budget_config.path]
            ),
        }
        resolved_artifact_factory = artifact_factory or ArtifactFactory(
            output_root=output_root.resolve()
        )
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
                experiment_definition=experiment_definition,
                experiment_dependencies=experiment_dependencies,
                runtime_config=runtime_config,
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
                experiment_definition=experiment_definition,
                experiment_dependencies=experiment_dependencies,
                runtime_config=runtime_config,
                prompt_instructions=_prompt_system_message(
                    verified.artifact_bytes[lock.baseline.planner.prompt_config.path]
                ),
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
            prompt_version="query-analyze-v1",
            mode_binding=binding,
            _owns_artifact_factory=artifact_factory is None,
        )
