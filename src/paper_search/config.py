"""Validated runtime configuration and reproducible configuration hashing."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import date
from pathlib import Path
from typing import Any, Literal, Mapping

import yaml
from dotenv import dotenv_values
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
)

from paper_search.domain.models import SearchBudget


BudgetConfig = SearchBudget


class RuntimeConfig(BaseModel):
    """Validated non-secret project config plus secrets loaded from the environment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    budget_profile: Literal["low", "balanced"]
    budget: SearchBudget
    llm_base_url: str = Field(min_length=1)
    llm_model_primary: str = Field(min_length=1)
    llm_model_fallback: str = Field(min_length=1)
    openalex_api_key: SecretStr | None = None
    semantic_scholar_api_key: SecretStr | None = None
    llm_api_key: SecretStr | None = None

    def config_hash(self) -> str:
        """Hash reproducibility settings without including API secrets."""

        public_config = self.model_dump(
            mode="json",
            exclude={
                "openalex_api_key",
                "semantic_scholar_api_key",
                "llm_api_key",
            },
        )
        return canonical_config_hash(public_config)


def canonical_json_bytes(config: Mapping[str, Any]) -> bytes:
    """Serialize a config as sorted-key compact UTF-8 JSON."""

    return json.dumps(
        config,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_config_hash(config: Mapping[str, Any]) -> str:
    """Return the PRD configuration fingerprint."""

    digest = hashlib.sha256(canonical_json_bytes(config)).hexdigest()
    return f"sha256:{digest}"


def validate_year_range(
    start_year: int,
    end_year: int,
    *,
    current_year: int | None = None,
) -> tuple[int, int]:
    """Validate the inclusive PRD year range: 1900..current year + 1."""

    maximum_year = (current_year if current_year is not None else date.today().year) + 1
    if not 1900 <= start_year <= maximum_year:
        raise ValueError(f"start_year must be between 1900 and {maximum_year}")
    if not 1900 <= end_year <= maximum_year:
        raise ValueError(f"end_year must be between 1900 and {maximum_year}")
    if start_year > end_year:
        raise ValueError("start_year must not exceed end_year")
    return start_year, end_year


def load_budget(path: str | Path) -> SearchBudget:
    """Load and validate one YAML budget profile."""

    budget_path = Path(path)
    with budget_path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)
    if not isinstance(raw, dict):
        raise ValueError(f"budget file must contain a mapping: {budget_path}")
    return SearchBudget.model_validate(raw)


def load_runtime_config(
    path: str | Path,
    *,
    env_file: str | Path | None = ".env",
) -> RuntimeConfig:
    """Load base YAML and budget YAML, then apply dotenv and process env overrides."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)
    if not isinstance(raw, dict):
        raise ValueError(f"config file must contain a mapping: {config_path}")

    secret_fields = {"openalex_api_key", "semantic_scholar_api_key", "llm_api_key"}
    forbidden_secrets = secret_fields.intersection(raw)
    if forbidden_secrets:
        names = ", ".join(sorted(forbidden_secrets))
        raise ValueError(f"secret fields are not allowed in YAML config: {names}")

    profile = raw.get("budget_profile")
    if profile not in {"low", "balanced"}:
        raise ValueError("budget_profile must be 'low' or 'balanced'")
    data = dict(raw)
    data["budget"] = load_budget(config_path.parent / f"budget_{profile}.yaml")

    dotenv = dotenv_values(env_file) if env_file is not None else {}
    env_mapping = {
        "OPENALEX_API_KEY": "openalex_api_key",
        "SEMANTIC_SCHOLAR_API_KEY": "semantic_scholar_api_key",
        "LLM_API_KEY": "llm_api_key",
        "LLM_BASE_URL": "llm_base_url",
        "LLM_MODEL_PRIMARY": "llm_model_primary",
        "LLM_MODEL_FALLBACK": "llm_model_fallback",
    }
    for env_name, field_name in env_mapping.items():
        value = os.environ.get(env_name) or dotenv.get(env_name)
        if value:
            data[field_name] = value

    return RuntimeConfig.model_validate(data)
