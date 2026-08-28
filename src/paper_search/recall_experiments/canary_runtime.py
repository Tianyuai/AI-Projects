"""Fixed, secret-free runtime configuration for candidate-recall canaries."""

from __future__ import annotations

import os
from dataclasses import dataclass
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import yaml
import httpx
from dotenv import dotenv_values
from pydantic import SecretStr
from pydantic import Field

from paper_search.domain.models import DomainModel
from paper_search.config import load_budget
from paper_search.control.budget import HardBudgetController
from paper_search.control.pricing import ActualCostPricer, parse_pricing_policy_bytes
from paper_search.domain.models import UsageActual, UsageEstimate
from paper_search.llm.client import OpenAICompatibleLLMClient
from paper_search.llm.snapshot_adapters import LiveCaptureLLMAnalyzer
from paper_search.recall_experiments.composition import RecallRuntime, build_live_runtime
from paper_search.recall_experiments.generation.backends import BudgetedLLMBackend
from paper_search.recall_experiments.recipes import (
    DeepSeekPromptGeneratorRecipe,
    LoadedRecallRecipe,
)
from paper_search.recall_experiments.retrieval.backends import (
    BudgetedCitationBackend,
    BudgetedSearchBackend,
)
from paper_search.retrieval.snapshot_adapters import (
    LiveCaptureSearchProvider,
    SearchAttemptGate,
)
from paper_search.storage.dependency_snapshot import DependencyCaptureStore
from paper_search.domain.models import Sha256


_SEMANTIC_SCHOLAR_MIN_REQUEST_INTERVAL_SECONDS = 1.1


class RecallRuntimeProfile(DomainModel):
    schema_version: Literal["recall-runtime-profile-v1"]
    env_file: Path
    pricing_policy: Path
    budget: Path
    capture_responses: bool
    llm_model: str = Field(min_length=1)
    llm_reservation_input_tokens: int = Field(strict=True, gt=0)
    llm_reservation_output_tokens: int = Field(strict=True, gt=0)
    openalex_minimum_request_interval_seconds: float = Field(default=0.0, ge=0)


class RecallRuntimeSecrets(DomainModel):
    llm_api_key: SecretStr
    openalex_api_key: SecretStr
    semantic_scholar_api_key: SecretStr
    additional_openalex_api_keys: tuple[SecretStr, ...] = ()


_BUNDLE_CAPABILITY = object()


def _search_reservation_calls(
    dependency: Literal["openalex", "semantic_scholar"],
    *,
    configured_key_count: int,
) -> int:
    if configured_key_count <= 0:
        raise ValueError("configured key count must be positive")
    if dependency == "openalex":
        return max(3, configured_key_count)
    return 3


@dataclass(frozen=True, init=False)
class RecallLiveRuntimeBundle:
    runtime: RecallRuntime
    client: httpx.AsyncClient
    capture_store: DependencyCaptureStore
    _sealed: bool = False
    _capability: object = None

    def __init__(
        self,
        *,
        runtime: RecallRuntime,
        client: httpx.AsyncClient,
        capture_store: DependencyCaptureStore,
        _capability: object,
    ) -> None:
        if _capability is not _BUNDLE_CAPABILITY:
            raise ValueError("live runtime bundle must be built by the fixed factory")
        object.__setattr__(self, "runtime", runtime)
        object.__setattr__(self, "client", client)
        object.__setattr__(self, "capture_store", capture_store)
        object.__setattr__(self, "_sealed", False)
        object.__setattr__(self, "_capability", _capability)

    def validate_capability(self) -> None:
        if self._capability is not _BUNDLE_CAPABILITY:
            raise ValueError("live runtime bundle capability is invalid")
        backends = (
            self.runtime.search_backend,
            self.runtime.citation_backend,
            self.runtime.llm_backend,
        )
        if not all(
            getattr(backend, "owns_live_resources", lambda **_: False)(
                client=self.client, capture_store=self.capture_store
            )
            for backend in backends
        ):
            raise ValueError("live runtime bundle resources do not match dispatch resources")

    async def seal(self) -> tuple[Sha256, Sha256]:
        if self._sealed:
            raise RuntimeError("canary runtime bundle is already sealed")
        manifest = self.capture_store.seal()
        object.__setattr__(self, "_sealed", True)
        return self.capture_store.manifest_sha256, manifest.snapshot_set_id

    async def aclose(self) -> None:
        try:
            if not self._sealed:
                self.capture_store.seal()
                object.__setattr__(self, "_sealed", True)
        finally:
            await self.client.aclose()


def load_runtime_profile(
    path: str | Path,
    *,
    env_file: str | Path | None = None,
    pricing_policy: str | Path | None = None,
    budget: str | Path | None = None,
) -> RecallRuntimeProfile:
    profile_path = Path(path).resolve()
    raw = yaml.safe_load(profile_path.read_bytes())
    if not isinstance(raw, dict):
        raise ValueError("runtime profile must contain a mapping")
    secret_fields = {"llm_api_key", "openalex_api_key", "semantic_scholar_api_key"}
    if secret_fields.intersection(raw):
        raise ValueError("secret fields are not allowed in runtime profiles")
    base = profile_path.parent
    for name, override in (
        ("env_file", env_file),
        ("pricing_policy", pricing_policy),
        ("budget", budget),
    ):
        value = override if override is not None else raw.get(name)
        if not isinstance(value, (str, Path)):
            raise ValueError(f"runtime profile {name} is required")
        candidate = Path(value)
        raw[name] = (candidate if candidate.is_absolute() else base / candidate).resolve()
    return RecallRuntimeProfile.model_validate(raw)


def resolve_runtime_secrets(
    profile: RecallRuntimeProfile,
    *,
    environ: Mapping[str, str] | None = None,
    openalex_key_slot: int | None = None,
) -> RecallRuntimeSecrets:
    process = os.environ if environ is None else environ
    dotenv = dotenv_values(profile.env_file)

    def value(name: str) -> str | None:
        candidate = process.get(name) or dotenv.get(name)
        return candidate if isinstance(candidate, str) and candidate else None

    required = {
        "llm_api_key": value("LLM_API_KEY"),
        "openalex_api_key": value("OPENALEX_API_KEY"),
        "semantic_scholar_api_key": value("SEMANTIC_SCHOLAR_API_KEY"),
    }
    missing = [name for name, item in required.items() if item is None]
    if missing:
        raise ValueError("missing required runtime secrets: " + ", ".join(sorted(missing)))
    additional: list[SecretStr] = []
    for index in range(2, 100):
        item = value(f"OPENALEX_API_KEY_{index}")
        if item is None:
            break
        additional.append(SecretStr(item))
    openalex_api_key = SecretStr(required["openalex_api_key"] or "")
    if openalex_key_slot is not None:
        configured = [openalex_api_key, *additional]
        if (
            type(openalex_key_slot) is not int
            or not 1 <= openalex_key_slot <= len(configured)
        ):
            raise ValueError("OpenAlex key slot is unavailable")
        openalex_api_key = configured[openalex_key_slot - 1]
        additional = []
    return RecallRuntimeSecrets(
        llm_api_key=SecretStr(required["llm_api_key"] or ""),
        openalex_api_key=openalex_api_key,
        semantic_scholar_api_key=SecretStr(required["semantic_scholar_api_key"] or ""),
        additional_openalex_api_keys=tuple(additional),
    )


async def build_live_runtime_bundle(
    *,
    profile: RecallRuntimeProfile,
    secrets: RecallRuntimeSecrets,
    loaded_recipe: LoadedRecallRecipe,
    capture_root: Path,
    client: httpx.AsyncClient | None = None,
    search_dependency: Literal["openalex", "semantic_scholar"] = "openalex",
    openalex_attempt_gate: SearchAttemptGate | None = None,
) -> RecallLiveRuntimeBundle:
    """Build the only approved live runtime from verified, secret-free profile inputs."""
    if profile.capture_responses is not True:
        raise ValueError("live canary runtime requires response capture")
    generator = loaded_recipe.recipe.generator
    if isinstance(generator, DeepSeekPromptGeneratorRecipe):
        if loaded_recipe.prompt_sha256 is None:
            raise ValueError("DeepSeek recipe lacks a bound prompt hash")
        if generator.model != profile.llm_model:
            raise ValueError("recipe and runtime LLM models do not match")
    prompt_sha256 = loaded_recipe.prompt_sha256 or ("sha256:" + "0" * 63 + "1")
    policy_bytes = profile.pricing_policy.read_bytes()
    pricer = ActualCostPricer(parse_pricing_policy_bytes(policy_bytes))
    controller = HardBudgetController(load_budget(profile.budget), formal_live=True)
    owned_client = client or httpx.AsyncClient(
        timeout=httpx.Timeout(connect=5, read=20, write=20, pool=5)
    )
    capture = DependencyCaptureStore(capture_root)
    try:
        search_provider = LiveCaptureSearchProvider(
            dependency=search_dependency,
            client=owned_client,
            capture_store=capture,
            pricer=pricer,
            controller=controller,
            api_key=(
                secrets.openalex_api_key.get_secret_value()
                if search_dependency == "openalex"
                else secrets.semantic_scholar_api_key.get_secret_value()
            ),
            additional_api_keys=(
                tuple(
                    key.get_secret_value()
                    for key in secrets.additional_openalex_api_keys
                )
                if search_dependency == "openalex"
                else ()
            ),
            minimum_request_interval_seconds=(
                _SEMANTIC_SCHOLAR_MIN_REQUEST_INTERVAL_SECONDS
                if search_dependency == "semantic_scholar"
                else profile.openalex_minimum_request_interval_seconds
            ),
            attempt_gate=openalex_attempt_gate,
        )
        citation_dependency = search_dependency
        citation_provider = search_provider
        llm_client = OpenAICompatibleLLMClient(
            client=owned_client,
            base_url="https://api.deepseek.com/v1",
            model=profile.llm_model,
            api_key=secrets.llm_api_key.get_secret_value(),
        )
        analyzer = LiveCaptureLLMAnalyzer(
            client=llm_client,
            capture_store=capture,
            pricer=pricer,
            controller=controller,
            prompt_artifact_sha256=prompt_sha256,
        )
        llm_usage = UsageActual(
            llm_calls=1,
            input_tokens=profile.llm_reservation_input_tokens,
            output_tokens=profile.llm_reservation_output_tokens,
        )
        llm_cost = pricer.value_actual(
            dependency="llm", model_or_adapter=profile.llm_model, usage=llm_usage
        ).cost_cny
        search_reservation_calls = _search_reservation_calls(
            search_dependency,
            configured_key_count=(
                1 + len(secrets.additional_openalex_api_keys)
                if search_dependency == "openalex"
                else 1
            ),
        )
        search_cost = pricer.value_actual(
            dependency=search_dependency,
            model_or_adapter=(
                "openalex-works-v1"
                if search_dependency == "openalex"
                else "semantic-graph-v1"
            ),
            usage=UsageActual(search_api_calls=search_reservation_calls),
        ).cost_cny
        citation_reservation_calls = search_reservation_calls * (
            2 if citation_dependency == "openalex" else 1
        )
        citation_cost = pricer.value_actual(
            dependency=citation_dependency,
            model_or_adapter=(
                "openalex-works-v1"
                if citation_dependency == "openalex"
                else "semantic-graph-v1"
            ),
            usage=UsageActual(search_api_calls=citation_reservation_calls),
        ).cost_cny
        llm_estimate = UsageEstimate(
            llm_calls=1,
            input_tokens=profile.llm_reservation_input_tokens,
            output_tokens=profile.llm_reservation_output_tokens,
            cost_cny=llm_cost,
        )
        runtime = build_live_runtime(
            search_backend=BudgetedSearchBackend(
                provider=search_provider,
                controller=controller,
                call_estimate=UsageEstimate(
                    search_api_calls=search_reservation_calls,
                    cost_cny=search_cost,
                ),
                dependency=search_dependency,
            ),
            citation_backend=BudgetedCitationBackend(
                provider=citation_provider,
                controller=controller,
                call_estimate=UsageEstimate(
                    search_api_calls=citation_reservation_calls,
                    cost_cny=citation_cost,
                ),
                dependency=citation_dependency,
            ),
            llm_backend=BudgetedLLMBackend(
                analyzer=analyzer,
                controller=controller,
                initial_estimate=llm_estimate,
                repair_estimate=llm_estimate,
            ),
        )
    except Exception:
        if client is None:
            await owned_client.aclose()
        raise
    return RecallLiveRuntimeBundle(
        runtime=runtime,
        client=owned_client,
        capture_store=capture,
        _capability=_BUNDLE_CAPABILITY,
    )


__all__ = [
    "RecallRuntimeProfile",
    "RecallRuntimeSecrets",
    "RecallLiveRuntimeBundle",
    "build_live_runtime_bundle",
    "load_runtime_profile",
    "resolve_runtime_secrets",
]
