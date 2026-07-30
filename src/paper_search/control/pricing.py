"""Versioned pricing and quality-Gate policy contracts."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from decimal import Decimal, ROUND_HALF_EVEN
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import ValidationError, model_validator

from paper_search.domain.models import (
    DependencyName,
    DomainModel,
    MoneyCny,
    NonEmptyStr,
    Sha256,
    UsageActual,
)


class PricingPolicyError(ValueError):
    """Raised when actual live usage cannot be valued exactly and safely."""


class PricingRate(DomainModel):
    """One approved price for a dependency model or adapter usage unit."""

    dependency: DependencyName
    model_or_adapter: NonEmptyStr
    unit: Literal["input_token", "output_token", "request"]
    price_cny_per_unit: MoneyCny


class PricingPolicy(DomainModel):
    """An operator-approved, versioned, deterministic pricing policy."""

    schema_version: Literal["pricing-policy-v1"]
    currency: Literal["CNY"]
    effective_at: datetime
    source_identity: NonEmptyStr
    rounding_quantum_cny: Decimal
    rates: list[PricingRate]

    @model_validator(mode="after")
    def validate_policy(self) -> Self:
        if self.effective_at.tzinfo is None:
            raise ValueError("pricing policy effective_at must include a timezone")
        if self.rounding_quantum_cny != Decimal("0.000001"):
            raise ValueError("rounding_quantum_cny must equal 0.000001")
        keys = [(rate.dependency, rate.model_or_adapter, rate.unit) for rate in self.rates]
        if len(keys) != len(set(keys)):
            raise ValueError("pricing policy contains duplicate rates")
        return self


class QualityGateRule(DomainModel):
    """One authoritative quality, formal-validity, reporting, or promotion rule."""

    rule_id: NonEmptyStr
    classification: Literal[
        "formal_validity", "baseline_quality", "reporting_only", "promotion"
    ]
    measure: NonEmptyStr
    operator: Literal["eq", "gt", "gte", "lte"]
    threshold: Decimal | int
    applies_to: list[Literal["dev", "validation", "frozen_audit", "optional"]]
    source_refs: list[NonEmptyStr]
    resolution: NonEmptyStr

    @model_validator(mode="after")
    def validate_authoritative_resolution(self) -> Self:
        if not self.source_refs:
            raise ValueError("authoritative quality rule requires source_refs")
        return self


class QualityGatePolicy(DomainModel):
    """The complete immutable rule table consumed by Gate evaluation."""

    schema_version: Literal["quality-gates-v1"]
    rules: list[QualityGateRule]

    @model_validator(mode="after")
    def validate_rule_ids(self) -> Self:
        rule_ids = [rule.rule_id for rule in self.rules]
        if len(rule_ids) != len(set(rule_ids)):
            raise ValueError("quality gate policy contains duplicate rule IDs")
        return self


def _load_yaml_mapping(path: Path, *, policy_name: str) -> dict[str, object]:
    try:
        raw = yaml.safe_load(path.read_bytes())
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"invalid {policy_name}: {path}") from error
    if not isinstance(raw, dict):
        raise ValueError(f"{policy_name} must contain a mapping: {path}")
    return raw


def load_pricing_policy(path: Path) -> PricingPolicy:
    """Load one exact, versioned pricing policy without guessing production rates."""

    raw = _load_yaml_mapping(path, policy_name="pricing policy")
    try:
        return PricingPolicy.model_validate(raw)
    except ValidationError as error:
        raise ValueError(f"invalid pricing policy: {path}") from error


def load_quality_gate_policy(path: Path) -> QualityGatePolicy:
    """Load the complete Gate policy, rejecting unresolved authoritative rows."""

    raw = _load_yaml_mapping(path, policy_name="quality gate policy")
    try:
        return QualityGatePolicy.model_validate(raw)
    except ValidationError as error:
        raise ValueError(f"invalid quality gate policy (missing source_refs or resolution): {path}") from error


def canonical_pricing_policy_bytes(policy: PricingPolicy) -> bytes:
    """Return stable UTF-8 JSON bytes for the immutable policy identity."""

    return json.dumps(
        policy.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def pricing_policy_sha256(policy: PricingPolicy) -> Sha256:
    """Return the content identity of an approved pricing policy."""

    return f"sha256:{hashlib.sha256(canonical_pricing_policy_bytes(policy)).hexdigest()}"


class ActualCostPricer:
    """Value actual dependency usage with one approved policy and one rounding boundary."""

    def __init__(self, policy: PricingPolicy, *, valued_at: datetime | None = None) -> None:
        self._policy = PricingPolicy.model_validate(policy.model_dump(mode="python"))
        instant = valued_at or datetime.now(UTC)
        if instant.tzinfo is None:
            raise PricingPolicyError("valued_at must include a timezone")
        if self._policy.effective_at > instant:
            raise PricingPolicyError("pricing policy is not yet effective")
        self._rates = {
            (rate.dependency, rate.model_or_adapter, rate.unit): rate.price_cny_per_unit
            for rate in self._policy.rates
        }

    @property
    def policy_sha256(self) -> Sha256:
        return pricing_policy_sha256(self._policy)

    def value_actual(
        self,
        *,
        dependency: DependencyName,
        model_or_adapter: str,
        usage: UsageActual,
    ) -> UsageActual:
        """Return usage with an exact price or fail closed before settlement."""

        units = self._billable_units(dependency, usage)
        if not any(quantity > 0 for _, quantity in units):
            raise PricingPolicyError("actual usage has no billable units")
        total = Decimal("0")
        for unit, quantity in units:
            if quantity == 0:
                continue
            rate = self._rates.get((dependency, model_or_adapter, unit))
            if rate is None:
                raise PricingPolicyError(
                    f"missing pricing rate for {dependency}/{model_or_adapter}/{unit}"
                )
            total += rate * quantity
        valued_cost = total.quantize(self._policy.rounding_quantum_cny, rounding=ROUND_HALF_EVEN)
        if usage.cost_cny is not None and usage.cost_cny != valued_cost:
            raise PricingPolicyError("authoritative billed amount does not match pricing policy")
        return UsageActual.model_validate({**usage.model_dump(mode="python"), "cost_cny": valued_cost})

    @staticmethod
    def _billable_units(
        dependency: DependencyName, usage: UsageActual
    ) -> tuple[tuple[Literal["input_token", "output_token", "request"], int], ...]:
        if dependency == "llm":
            if usage.search_api_calls:
                raise PricingPolicyError("LLM usage cannot include search API calls")
            return (
                ("input_token", usage.input_tokens),
                ("output_token", usage.output_tokens),
                ("request", usage.llm_calls),
            )
        if usage.llm_calls or usage.input_tokens or usage.output_tokens:
            raise PricingPolicyError("provider usage cannot include LLM usage")
        return (("request", usage.search_api_calls),)
