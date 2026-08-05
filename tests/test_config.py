import hashlib
import json
from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from paper_search.config import (
    BudgetConfig,
    RuntimeConfig,
    RuntimeSettings,
    canonical_config_hash,
    load_budget,
    validate_year_range,
)


def test_runtime_settings_rejects_non_reproducible_timeout_or_retry_values() -> None:
    assert RuntimeSettings.model_validate({"artifact_root": "artifacts"}).allow_live is False

    with pytest.raises(ValidationError):
        RuntimeSettings.model_validate(
            {"artifact_root": "artifacts", "connect_timeout_seconds": 6}
        )


def test_runtime_config_exposes_required_reproducible_sections() -> None:
    assert {"runtime", "policy_bindings", "capture_policy", "routing", "retry"} <= set(
        RuntimeConfig.model_fields
    )


def test_config_hash_is_stable_for_key_order_and_utf8() -> None:
    query = "\u68c0\u7d22\u589e\u5f3a\u751f\u6210"
    left = {"query": query, "limits": {"calls": 5, "tokens": 10_000}}
    right = {"limits": {"tokens": 10_000, "calls": 5}, "query": query}

    left_hash = canonical_config_hash(left)

    assert left_hash == canonical_config_hash(right)
    assert left_hash.startswith("sha256:")
    assert len(left_hash) == len("sha256:") + 64


def test_config_hash_matches_sorted_utf8_json_sha256() -> None:
    config = {"\u6a21\u578b": "\u5343\u95ee", "year": 2026}
    canonical = json.dumps(
        config,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")

    assert canonical_config_hash(config) == f"sha256:{hashlib.sha256(canonical).hexdigest()}"


@pytest.mark.parametrize("start_year,end_year", [(1900, 1900), (1900, 2027), (2027, 2027)])
def test_year_range_accepts_prd_boundaries(start_year: int, end_year: int) -> None:
    assert validate_year_range(start_year, end_year, current_year=2026) == (
        start_year,
        end_year,
    )


@pytest.mark.parametrize("start_year,end_year", [(1899, 2020), (2020, 2028), (2021, 2020)])
def test_year_range_rejects_invalid_values(start_year: int, end_year: int) -> None:
    with pytest.raises(ValueError):
        validate_year_range(start_year, end_year, current_year=2026)


def test_year_range_uses_current_year_by_default() -> None:
    assert validate_year_range(1900, date.today().year + 1) == (1900, date.today().year + 1)


@pytest.mark.parametrize(
    ("filename", "tokens", "cost", "rerank_candidates", "search_calls"),
    [
        ("budget_low.yaml", 10_000, 0.10, 12, 12),
        ("budget_balanced.yaml", 24_000, 0.30, 30, 18),
    ],
)
def test_budget_profiles_match_prd(
    filename: str,
    tokens: int,
    cost: float,
    rerank_candidates: int,
    search_calls: int,
) -> None:
    budget = load_budget(f"configs/{filename}")

    assert budget.max_search_api_calls == search_calls
    assert budget.target_search_api_calls == 8
    assert budget.max_llm_calls == 5
    assert budget.target_llm_calls == 3
    assert budget.max_iterations == 2
    assert budget.max_subqueries == 6
    assert budget.max_rerank_candidates == rerank_candidates
    assert budget.max_output_papers == 50
    assert budget.max_citation_seeds == 2
    assert budget.target_citation_seeds == 1
    assert budget.max_elapsed_seconds == 90
    assert budget.soft_deadline_seconds == 80
    assert budget.max_total_tokens == tokens
    assert budget.max_cost_cny == pytest.approx(cost)


def test_budget_rejects_unknown_or_invalid_limits() -> None:
    valid = {
        "max_search_api_calls": 12,
        "target_search_api_calls": 8,
        "max_llm_calls": 5,
        "target_llm_calls": 3,
        "max_iterations": 2,
        "max_subqueries": 6,
        "max_rerank_candidates": 12,
        "max_output_papers": 50,
        "max_citation_seeds": 2,
        "target_citation_seeds": 1,
        "max_elapsed_seconds": 90,
        "soft_deadline_seconds": 80,
        "max_total_tokens": 10_000,
        "max_cost_cny": 0.10,
    }

    with pytest.raises(ValidationError):
        BudgetConfig.model_validate({**valid, "max_total_tokens": -1})
    with pytest.raises(ValidationError):
        BudgetConfig.model_validate({**valid, "unexpected": 1})
    with pytest.raises(ValidationError):
        BudgetConfig.model_validate({**valid, "target_llm_calls": 6})
    with pytest.raises(ValidationError):
        BudgetConfig.model_validate({**valid, "soft_deadline_seconds": 90})


def test_provider_readiness_hash_matches_embedded_config() -> None:
    readiness = json.loads(Path("data/provider_readiness.json").read_text(encoding="utf-8"))

    assert readiness["config_hash"] == canonical_config_hash(readiness["config"])
    assert readiness["security"]["secrets_recorded"] is False


def test_env_example_has_only_documented_variables() -> None:
    expected = {
        "HF_TOKEN",
        "OPENALEX_API_KEY",
        "SEMANTIC_SCHOLAR_API_KEY",
        "LLM_API_KEY",
        "LLM_BASE_URL",
        "LLM_MODEL_PRIMARY",
        "LLM_MODEL_FALLBACK",
    }
    lines = Path(".env.example").read_text(encoding="utf-8").splitlines()
    variables = {line.split("=", 1)[0] for line in lines if line and not line.startswith("#")}

    assert variables == expected
    assert all(line.endswith("=") for line in lines if line and not line.startswith("#"))
