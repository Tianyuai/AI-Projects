"""Bounded, explicitly authorized live dependency probe and readiness mapping."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import tempfile
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

import httpx
from pydantic import ConfigDict, ValidationError, field_validator, model_validator

from paper_search.application.contracts import (
    DependencyStatus,
    ReadyHealthResponse,
)
from paper_search.domain.models import DomainModel


FRESHNESS = timedelta(minutes=15)
_EXPECTED_DEPENDENCIES = ("llm", "openalex", "semantic_scholar")
_DEFAULT_LLM_BASE_URL = "https://api.deepseek.com/v1"
_DEFAULT_LLM_MODEL = "deepseek-v4-flash"
_OPENALEX_PROBE_URL = "https://api.openalex.org/works"
_SEMANTIC_SCHOLAR_PROBE_URL = (
    "https://api.semanticscholar.org/graph/v1/paper/search"
)


class AuthorizedCapability(DomainModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    name: Literal["llm", "openalex", "semantic_scholar"]
    state: Literal["ready", "degraded", "failed"]
    observed_at: datetime

    @field_validator("observed_at")
    @classmethod
    def require_utc_observation(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
            raise ValueError("readiness observation must be UTC")
        return value.astimezone(UTC)


class AuthorizedReadinessEvidence(DomainModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["gate0-readiness-v1"]
    generated_at: datetime
    capabilities: list[AuthorizedCapability]

    @field_validator("generated_at")
    @classmethod
    def require_utc_generation(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
            raise ValueError("readiness generation must be UTC")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_complete_safe_catalog(self) -> AuthorizedReadinessEvidence:
        names = [capability.name for capability in self.capabilities]
        if names != list(_EXPECTED_DEPENDENCIES):
            raise ValueError("readiness capability catalog is invalid")
        if any(
            capability.observed_at > self.generated_at
            for capability in self.capabilities
        ):
            raise ValueError("readiness observation follows report generation")
        return self


def _probe_state(response: httpx.Response, *, valid: bool) -> str:
    if response.status_code == 200 and valid:
        return "ready"
    if response.status_code == 429 or 500 <= response.status_code <= 599:
        return "degraded"
    return "failed"


async def _probe_llm(
    client: httpx.AsyncClient,
    *,
    api_key: str,
    base_url: str,
    model: str,
) -> str:
    endpoint = f"{base_url.rstrip('/')}/chat/completions"
    payload: dict[str, object] = {
        "model": model,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
        "temperature": 0,
    }
    try:
        response = await client.post(
            endpoint,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            follow_redirects=False,
        )
    except (httpx.TimeoutException, httpx.RequestError):
        return "failed"
    return _probe_state(response, valid=True)


async def _probe_openalex(
    client: httpx.AsyncClient,
    *,
    api_key: str | None,
) -> str:
    params: dict[str, object] = {"per_page": 1, "select": "id"}
    if api_key:
        params["api_key"] = api_key
    try:
        response = await client.get(
            _OPENALEX_PROBE_URL,
            params=params,
            follow_redirects=False,
        )
    except (httpx.TimeoutException, httpx.RequestError):
        return "failed"
    valid = False
    if response.status_code == 200:
        try:
            payload = response.json()
        except ValueError:
            payload = None
        valid = (
            isinstance(payload, dict)
            and isinstance(payload.get("results"), list)
        )
    return _probe_state(response, valid=valid)


async def _probe_semantic_scholar(
    client: httpx.AsyncClient,
    *,
    api_key: str | None,
) -> str:
    params: dict[str, object] = {
        "query": "readiness probe",
        "limit": 1,
        "fields": "paperId",
    }
    headers = {"Accept": "application/json"}
    if api_key:
        headers["x-api-key"] = api_key
    try:
        response = await client.get(
            _SEMANTIC_SCHOLAR_PROBE_URL,
            params=params,
            headers=headers,
            follow_redirects=False,
        )
    except (httpx.TimeoutException, httpx.RequestError):
        return "failed"
    valid = False
    if response.status_code == 200:
        try:
            payload = response.json()
        except ValueError:
            payload = None
        valid = isinstance(payload, dict) and isinstance(payload.get("data"), list)
    return _probe_state(response, valid=valid)


async def probe_live_readiness(
    *,
    llm_api_key: str,
    llm_base_url: str = _DEFAULT_LLM_BASE_URL,
    llm_model: str = _DEFAULT_LLM_MODEL,
    openalex_api_key: str | None = None,
    semantic_scholar_api_key: str | None = None,
    client: httpx.AsyncClient | None = None,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> AuthorizedReadinessEvidence:
    """Run one bounded probe per dependency and return safe evidence."""

    owns_client = client is None
    if client is None:
        client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=5,
                read=20,
                write=20,
                pool=5,
            )
        )
    observed_at: list[tuple[str, str, datetime]] = []
    try:
        llm_state = await _probe_llm(
            client,
            api_key=llm_api_key,
            base_url=llm_base_url,
            model=llm_model,
        )
        observed_at.append(("llm", llm_state, clock()))
        openalex_state = await _probe_openalex(
            client,
            api_key=openalex_api_key,
        )
        observed_at.append(("openalex", openalex_state, clock()))
        scholar_state = await _probe_semantic_scholar(
            client,
            api_key=semantic_scholar_api_key,
        )
        observed_at.append(("semantic_scholar", scholar_state, clock()))
        generated_at = clock()
    finally:
        if owns_client:
            await client.aclose()
    return AuthorizedReadinessEvidence(
        schema_version="gate0-readiness-v1",
        generated_at=generated_at,
        capabilities=[
            AuthorizedCapability(
                name=name,
                state=state,
                observed_at=observed,
            )
            for name, state, observed in observed_at
        ],
    )


def build_live_readiness(
    evidence: AuthorizedReadinessEvidence,
    now: datetime,
    *,
    required_dependencies: Sequence[str] = _EXPECTED_DEPENDENCIES,
) -> ReadyHealthResponse:
    """Map the latest authorized probe evidence to a non-billable health state."""

    states = {capability.name: capability.state for capability in evidence.capabilities}
    required = tuple(required_dependencies)
    unknown = set(required).difference(_EXPECTED_DEPENDENCIES)
    if unknown:
        raise ValueError(
            f"unknown readiness dependency: {sorted(unknown)}"
        )
    elapsed = now - evidence.generated_at
    fresh = timedelta(0) <= elapsed < FRESHNESS
    ready = fresh and all(
        states[requirement] == "ready" for requirement in required
    )
    return ReadyHealthResponse(
        status="ready" if ready else "degraded",
        execution_mode="live",
        snapshot_set_id=None,
        dependencies=[
            DependencyStatus(
                dependency=name,
                state=states[name],
                cache_hit=False,
                error_codes=[],
            )
            for name in _EXPECTED_DEPENDENCIES
        ],
        last_authorized_probe_at=evidence.generated_at,
    )


def load_live_readiness(artifact_root: Path) -> AuthorizedReadinessEvidence | None:
    """Load the latest operator evidence without leaking private paths."""

    candidates = (
        artifact_root / "data" / "annotation_work" / "provider_readiness.live.json",
        artifact_root / "data" / "provider_readiness.json",
    )
    for path in candidates:
        try:
            content = path.read_bytes()
        except OSError:
            continue
        try:
            return AuthorizedReadinessEvidence.model_validate_json(
                content,
                strict=True,
            )
        except ValidationError:
            return None
    return None


def write_authorized_readiness(
    path: Path,
    evidence: AuthorizedReadinessEvidence,
) -> None:
    """Write canonical readiness evidence through a sibling temporary file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    content = (
        json.dumps(
            evidence.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    fd, temporary = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one bounded authorized live dependency probe."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/annotation_work/provider_readiness.live.json"),
    )
    args = parser.parse_args()
    llm_api_key = os.environ.get("LLM_API_KEY", "")
    if not llm_api_key:
        print("readiness failed: LLM_API_KEY is required", file=__import__("sys").stderr)
        return 2
    try:
        evidence = asyncio.run(
            probe_live_readiness(
                llm_api_key=llm_api_key,
                llm_base_url=os.environ.get(
                    "LLM_BASE_URL",
                    _DEFAULT_LLM_BASE_URL,
                ),
                llm_model=os.environ.get("LLM_MODEL_PRIMARY", _DEFAULT_LLM_MODEL),
                openalex_api_key=os.environ.get("OPENALEX_API_KEY") or None,
                semantic_scholar_api_key=(
                    os.environ.get("SEMANTIC_SCHOLAR_API_KEY") or None
                ),
            )
        )
    except (OSError, ValueError, ValidationError):
        print("readiness failed", file=__import__("sys").stderr)
        return 2
    write_authorized_readiness(args.output, evidence)
    states = " ".join(
        f"{capability.name}={capability.state}"
        for capability in evidence.capabilities
    )
    print(f"readiness status={states} path={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
