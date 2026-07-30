import copy
import importlib
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from paper_search.domain.models import UsageActual


FIXTURE_PATH = Path(__file__).parents[1] / "fixtures" / "pricing" / "pricing-policy-test-v1.yaml"
QUALITY_POLICY_PATH = Path(__file__).parents[2] / "configs" / "quality_gates_v1.yaml"


def pricing_api() -> tuple[type[Any], type[Any], type[Exception], Any, Any, Any]:
    try:
        module = importlib.import_module("paper_search.control.pricing")
    except ModuleNotFoundError:
        pytest.fail("paper_search.control.pricing must be implemented")
    for name in (
        "ActualCostPricer",
        "PricingPolicy",
        "PricingPolicyError",
        "load_pricing_policy",
        "load_quality_gate_policy",
        "pricing_policy_sha256",
    ):
        assert hasattr(module, name), f"{name} must be implemented"
    return (
        module.ActualCostPricer,
        module.PricingPolicy,
        module.PricingPolicyError,
        module.load_pricing_policy,
        module.load_quality_gate_policy,
        module.pricing_policy_sha256,
    )


def load_fixture_policy() -> Any:
    _, _, _, load_pricing_policy, _, _ = pricing_api()
    return load_pricing_policy(FIXTURE_PATH)


def test_fixture_policy_values_llm_usage_with_exact_decimal_cost() -> None:
    actual_cost_pricer, _, _, _, _, _ = pricing_api()
    pricer = actual_cost_pricer(
        load_fixture_policy(),
        valued_at=datetime(2026, 7, 30, tzinfo=UTC),
    )

    valued = pricer.value_actual(
        dependency="llm",
        model_or_adapter="qwen-test-v1",
        usage=UsageActual(llm_calls=1, input_tokens=5, output_tokens=3),
    )

    assert valued.cost_cny == Decimal("0.000119")


def test_fixture_policy_values_provider_request_with_exact_decimal_cost() -> None:
    actual_cost_pricer, _, _, _, _, _ = pricing_api()
    pricer = actual_cost_pricer(
        load_fixture_policy(),
        valued_at=datetime(2026, 7, 30, tzinfo=UTC),
    )

    valued = pricer.value_actual(
        dependency="openalex",
        model_or_adapter="openalex-works-v1",
        usage=UsageActual(search_api_calls=2),
    )

    assert valued.cost_cny == Decimal("0.000100")


def test_policy_identity_is_canonical_and_deterministic(tmp_path: Path) -> None:
    _, _, _, load_pricing_policy, _, pricing_policy_sha256 = pricing_api()
    copied_path = tmp_path / "copied.yaml"
    copied_path.write_bytes(FIXTURE_PATH.read_bytes())

    assert pricing_policy_sha256(load_pricing_policy(FIXTURE_PATH)) == pricing_policy_sha256(
        load_pricing_policy(copied_path)
    )


def test_unknown_model_or_missing_priced_unit_fails_closed() -> None:
    actual_cost_pricer, pricing_policy, pricing_error, _, _, _ = pricing_api()
    policy = load_fixture_policy()
    pricer = actual_cost_pricer(policy, valued_at=datetime(2026, 7, 30, tzinfo=UTC))

    with pytest.raises(pricing_error, match="rate"):
        pricer.value_actual(
            dependency="llm",
            model_or_adapter="unknown-model",
            usage=UsageActual(llm_calls=1),
        )

    missing_output = pricing_policy.model_validate(
        {
            **policy.model_dump(mode="python"),
            "rates": [rate.model_dump(mode="python") for rate in policy.rates if rate.unit != "output_token"],
        }
    )
    missing_unit_pricer = actual_cost_pricer(
        missing_output,
        valued_at=datetime(2026, 7, 30, tzinfo=UTC),
    )
    with pytest.raises(pricing_error, match="rate"):
        missing_unit_pricer.value_actual(
            dependency="llm",
            model_or_adapter="qwen-test-v1",
            usage=UsageActual(llm_calls=1, output_tokens=1),
        )


def test_duplicate_rate_ineffective_policy_missing_usage_and_billed_mismatch_fail_closed() -> None:
    actual_cost_pricer, pricing_policy, pricing_error, _, _, _ = pricing_api()
    policy = load_fixture_policy()
    duplicate = policy.model_dump(mode="python")
    duplicate["rates"] = [*duplicate["rates"], duplicate["rates"][0]]
    with pytest.raises(ValidationError, match="duplicate"):
        pricing_policy.model_validate(duplicate)

    with pytest.raises(pricing_error, match="not yet effective"):
        actual_cost_pricer(
            policy,
            valued_at=datetime(2026, 6, 30, 23, 59, 59, tzinfo=UTC),
        )

    pricer = actual_cost_pricer(policy, valued_at=datetime(2026, 7, 30, tzinfo=UTC))
    with pytest.raises(pricing_error, match="usage"):
        pricer.value_actual(
            dependency="llm",
            model_or_adapter="qwen-test-v1",
            usage=UsageActual(),
        )
    with pytest.raises(pricing_error, match="billed"):
        pricer.value_actual(
            dependency="llm",
            model_or_adapter="qwen-test-v1",
            usage=UsageActual(llm_calls=1, input_tokens=5, output_tokens=3, cost_cny=Decimal("0.1")),
        )


def test_quality_policy_encodes_every_approved_rule_and_requires_resolution(tmp_path: Path) -> None:
    _, _, _, _, load_quality_gate_policy, _ = pricing_api()
    policy = load_quality_gate_policy(QUALITY_POLICY_PATH)
    by_id = {rule.rule_id: rule for rule in policy.rules}
    expected = {
        "prediction-cardinality",
        "hard-failure-cardinality",
        "integrity-failures",
        "provenance-failures",
        "sanitization-failures",
        "unaccounted-usage-failures",
        "budget-ledgers-within-cap",
        "model-produced-analysis-rate",
        "strong-constraint-recall",
        "retrieval-response-rate",
        "fuzzy-merge-accuracy",
        "fuzzy-merge-audit-denominator",
        "hard-filter-recall-loss",
        "macro-recall-positive",
        "micro-recall-positive",
        "promotion-median-macro-f1-delta",
        "promotion-bootstrap-lower-bound",
        "promotion-validation-macro-f1-drop",
        "promotion-bootstrap-samples",
    }
    assert expected <= set(by_id)
    assert by_id["model-produced-analysis-rate"].threshold == Decimal("0.99")
    assert by_id["hard-filter-recall-loss"].threshold == Decimal("0.02")
    assert by_id["promotion-bootstrap-samples"].threshold == 1000
    assert all(rule.source_refs and rule.resolution for rule in policy.rules)

    raw = yaml.safe_load(QUALITY_POLICY_PATH.read_bytes())
    assert isinstance(raw, dict)
    rules = raw["rules"]
    assert isinstance(rules, list)
    invalid_rule = copy.deepcopy(rules[0])
    invalid_rule.pop("resolution")
    raw["rules"] = [invalid_rule, *rules[1:]]
    invalid_path = tmp_path / "unresolved.yaml"
    invalid_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="resolution"):
        load_quality_gate_policy(invalid_path)
