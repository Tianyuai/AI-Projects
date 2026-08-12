"""Versioned pricing and quality-Gate policy contracts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, ROUND_HALF_EVEN
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import ConfigDict, Field, ValidationError, model_validator

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


class _PolicyModel(DomainModel):
    """Strict serialized-policy model base without changing global domain contracts."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class PricingRate(_PolicyModel):
    """One approved price for a dependency model or adapter usage unit."""

    dependency: DependencyName
    model_or_adapter: NonEmptyStr
    unit: Literal["input_token", "output_token", "request"]
    price_cny_per_unit: MoneyCny


class PricingPolicy(_PolicyModel):
    """An operator-approved, versioned, deterministic pricing policy."""

    schema_version: Literal["pricing-policy-v1"]
    currency: Literal["CNY"]
    effective_at: datetime
    source_identity: NonEmptyStr
    rounding_quantum_cny: Decimal
    rates: list[PricingRate] = Field(min_length=1)

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


class QualityGateRule(_PolicyModel):
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
        if self.resolution not in {
            "enforced",
            "reporting-only",
            "optional-promotion-only",
        } and not self.resolution.startswith("superseded-by-"):
            raise ValueError("quality rule resolution is not an approved disposition")
        return self


@dataclass(frozen=True)
class AuthoritativeQualityRow:
    """One source-controlled historical or integrated quality-Gate target."""

    rule_id: str
    classification: str
    measure: str
    operator: str
    threshold: Decimal | int
    applies_to: tuple[str, ...]
    source_refs: tuple[str, ...]
    resolution: str


def _quality_row(
    rule_id: str,
    classification: str,
    measure: str,
    operator: str,
    threshold: Decimal | int,
    applies_to: tuple[str, ...],
    source_refs: tuple[str, ...],
    resolution: str,
) -> AuthoritativeQualityRow:
    return AuthoritativeQualityRow(
        rule_id,
        classification,
        measure,
        operator,
        threshold,
        applies_to,
        source_refs,
        resolution,
    )


AUTHORITATIVE_QUALITY_CATALOG: tuple[AuthoritativeQualityRow, ...] = (
    _quality_row("prediction-cardinality", "formal_validity", "predictions_per_frozen_query_in_order", "eq", 1, ("dev", "validation"), ("integrated-design:authoritative-quality-gates:prediction-cardinality",), "enforced"),
    _quality_row("hard-failure-cardinality", "formal_validity", "supplemental_failure_records_per_hard_failed_query", "eq", 1, ("dev", "validation"), ("integrated-design:authoritative-quality-gates:hard-failure-cardinality",), "enforced"),
    _quality_row("integrity-failures", "formal_validity", "integrity_failures", "eq", 0, ("dev", "validation", "frozen_audit"), ("integrated-design:authoritative-quality-gates:integrity",), "enforced"),
    _quality_row("provenance-failures", "formal_validity", "provenance_failures", "eq", 0, ("dev", "validation", "frozen_audit"), ("integrated-design:authoritative-quality-gates:provenance",), "enforced"),
    _quality_row("sanitization-failures", "formal_validity", "sanitization_failures", "eq", 0, ("dev", "validation", "frozen_audit"), ("integrated-design:authoritative-quality-gates:sanitization",), "enforced"),
    _quality_row("unaccounted-usage-failures", "formal_validity", "unaccounted_usage_failures", "eq", 0, ("dev", "validation", "frozen_audit"), ("integrated-design:authoritative-quality-gates:unaccounted-usage",), "enforced"),
    _quality_row("budget-ledgers-within-cap", "formal_validity", "budget_ledgers_over_hard_cap", "eq", 0, ("dev", "validation"), ("integrated-design:authoritative-quality-gates:budget-ledgers",), "enforced"),
    _quality_row("model-produced-analysis-rate", "baseline_quality", "valid_model_produced_query_analysis_rate", "gte", Decimal("0.99"), ("dev", "validation"), ("PRD:FR-01", "integrated-design:authoritative-quality-gates:model-analysis"), "enforced"),
    _quality_row("strong-constraint-recall", "baseline_quality", "audited_strong_constraint_recall", "gte", Decimal("0.90"), ("frozen_audit",), ("PRD:FR-01", "integrated-design:authoritative-quality-gates:strong-constraint"), "enforced"),
    _quality_row("retrieval-response-rate", "baseline_quality", "parseable_configured_retrieval_response_rate", "gte", Decimal("0.95"), ("dev", "validation"), ("PRD:FR-03", "integrated-design:authoritative-quality-gates:retrieval-response"), "enforced"),
    _quality_row("fuzzy-merge-accuracy", "baseline_quality", "audited_fuzzy_merge_accuracy", "gte", Decimal("0.98"), ("frozen_audit",), ("PRD:FR-04", "integrated-design:authoritative-quality-gates:fuzzy-merge"), "enforced"),
    _quality_row("fuzzy-merge-audit-denominator", "formal_validity", "audited_fuzzy_merge_decision_count", "gt", 0, ("frozen_audit",), ("integrated-design:authoritative-quality-gates:fuzzy-merge",), "enforced"),
    _quality_row("hard-filter-recall-loss", "baseline_quality", "hard_filter_absolute_recall_loss", "lte", Decimal("0.02"), ("dev", "validation"), ("PRD:FR-05", "integrated-design:authoritative-quality-gates:hard-filter"), "enforced"),
    _quality_row("macro-recall-positive", "baseline_quality", "macro_identifier_map_recall", "gt", 0, ("dev", "validation"), ("R3:identifier-map-recall", "integrated-design:authoritative-quality-gates:macro-recall"), "enforced"),
    _quality_row("micro-recall-positive", "baseline_quality", "micro_identifier_map_recall", "gt", 0, ("dev", "validation"), ("R3:identifier-map-recall", "integrated-design:authoritative-quality-gates:micro-recall"), "enforced"),
    _quality_row("report-macro-precision", "reporting_only", "macro_precision", "gte", 0, ("dev", "validation"), ("PRD:FR-12", "integrated-design:authoritative-quality-gates:reporting"), "reporting-only"),
    _quality_row("report-micro-precision", "reporting_only", "micro_precision", "gte", 0, ("dev", "validation"), ("PRD:FR-12", "integrated-design:authoritative-quality-gates:reporting"), "reporting-only"),
    _quality_row("report-macro-recall", "reporting_only", "macro_recall", "gte", 0, ("dev", "validation"), ("PRD:FR-12", "integrated-design:authoritative-quality-gates:reporting"), "superseded-by-macro-recall-positive"),
    _quality_row("report-micro-recall", "reporting_only", "micro_recall", "gte", 0, ("dev", "validation"), ("PRD:FR-12", "integrated-design:authoritative-quality-gates:reporting"), "superseded-by-micro-recall-positive"),
    _quality_row("report-macro-f1", "reporting_only", "macro_f1", "gte", 0, ("dev", "validation"), ("PRD:14.0", "PRD:FR-12", "integrated-design:authoritative-quality-gates:reporting"), "reporting-only"),
    _quality_row("report-micro-f1", "reporting_only", "micro_f1", "gte", 0, ("dev", "validation"), ("PRD:FR-12", "integrated-design:authoritative-quality-gates:reporting"), "reporting-only"),
    _quality_row("report-recall-at-5", "reporting_only", "recall_at_5", "gte", 0, ("dev", "validation"), ("PRD:FR-12", "integrated-design:authoritative-quality-gates:reporting"), "reporting-only"),
    _quality_row("report-recall-at-10", "reporting_only", "recall_at_10", "gte", 0, ("dev", "validation"), ("PRD:FR-12", "integrated-design:authoritative-quality-gates:reporting"), "reporting-only"),
    _quality_row("report-recall-at-20", "reporting_only", "recall_at_20", "gte", 0, ("dev", "validation"), ("PRD:FR-12", "integrated-design:authoritative-quality-gates:reporting"), "reporting-only"),
    _quality_row("report-hard-failure-rate", "reporting_only", "hard_failure_rate", "gte", 0, ("dev", "validation"), ("PRD:15.3", "integrated-design:authoritative-quality-gates:reporting"), "reporting-only"),
    _quality_row("report-partial-result-rate", "reporting_only", "partial_result_rate", "gte", 0, ("dev", "validation"), ("PRD:15.3", "integrated-design:authoritative-quality-gates:reporting"), "reporting-only"),
    _quality_row("report-planner-fallback-rate", "reporting_only", "planner_fallback_rate", "gte", 0, ("dev", "validation"), ("integrated-design:baseline-composition", "integrated-design:authoritative-quality-gates:reporting"), "reporting-only"),
    _quality_row("report-latency-p50", "reporting_only", "latency_p50_ms", "gte", 0, ("dev", "validation"), ("PRD:NFR-03", "integrated-design:authoritative-quality-gates:reporting"), "reporting-only"),
    _quality_row("report-latency-p95", "reporting_only", "latency_p95_ms", "gte", 0, ("dev", "validation"), ("PRD:NFR-03", "integrated-design:authoritative-quality-gates:reporting"), "reporting-only"),
    _quality_row("report-external-calls", "reporting_only", "external_calls", "gte", 0, ("dev", "validation"), ("PRD:FR-12", "integrated-design:authoritative-quality-gates:reporting"), "reporting-only"),
    _quality_row("report-tokens", "reporting_only", "actual_tokens", "gte", 0, ("dev", "validation"), ("PRD:FR-12", "integrated-design:authoritative-quality-gates:reporting"), "reporting-only"),
    _quality_row("report-valued-cost", "reporting_only", "valued_cost_cny", "gte", 0, ("dev", "validation"), ("PRD:NFR-03", "integrated-design:authoritative-quality-gates:reporting"), "reporting-only"),
    _quality_row("report-cache-hit-rate", "reporting_only", "cache_hit_rate", "gte", 0, ("dev", "validation"), ("PRD:FR-03", "integrated-design:authoritative-quality-gates:reporting"), "reporting-only"),
    _quality_row("promotion-median-macro-f1-delta", "promotion", "median_macro_f1_delta_over_three_runs", "gte", Decimal("0.01"), ("optional",), ("PRD:FR-02", "PRD:model-promotion-gate", "integrated-design:promotion-gate"), "optional-promotion-only"),
    _quality_row("promotion-bootstrap-lower-bound", "promotion", "bootstrap_95_ci_lower_bound", "gte", Decimal("-0.005"), ("optional",), ("PRD:FR-02", "integrated-design:promotion-gate"), "optional-promotion-only"),
    _quality_row("promotion-validation-macro-f1-drop", "promotion", "validation_macro_f1_drop", "lte", Decimal("0.01"), ("optional",), ("PRD:FR-02", "PRD:model-promotion-gate", "integrated-design:promotion-gate"), "optional-promotion-only"),
    _quality_row("promotion-bootstrap-samples", "promotion", "bootstrap_samples", "eq", 1000, ("optional",), ("PRD:FR-02", "integrated-design:promotion-gate"), "optional-promotion-only"),
    _quality_row("dev-macro-f1-min", "reporting_only", "dev_macro_f1", "gte", Decimal("0.30"), ("dev",), ("PRD:14.0",), "reporting-only"),
    _quality_row("dev-macro-f1-delta", "reporting_only", "dev_macro_f1_delta_vs_raw_openalex", "gte", Decimal("0.03"), ("dev",), ("PRD:14.0",), "reporting-only"),
    _quality_row("validation-macro-f1-delta", "reporting_only", "validation_macro_f1_delta_vs_raw_openalex", "gte", Decimal("0.02"), ("validation",), ("PRD:14.0",), "reporting-only"),
    _quality_row("structured-schema-valid-rate", "reporting_only", "schema_valid_rate", "eq", 1, ("dev", "validation"), ("PRD:14.2",), "reporting-only"),
    _quality_row("structured-valid-paper-link-rate", "reporting_only", "valid_paper_link_rate", "gte", Decimal("0.99"), ("dev", "validation"), ("PRD:14.2",), "reporting-only"),
    _quality_row("structured-reason-complete-rate", "reporting_only", "reason_complete_rate", "gte", Decimal("0.95"), ("dev", "validation"), ("PRD:14.2",), "reporting-only"),
    _quality_row("structured-verifiable-citation-edge-rate", "reporting_only", "verifiable_citation_edge_rate", "eq", 1, ("dev", "validation"), ("PRD:14.2",), "reporting-only"),
    _quality_row("structured-fabrication-count", "reporting_only", "fabricated_paper_or_relation_count", "eq", 0, ("dev", "validation"), ("PRD:14.2", "PRD:15.2"), "reporting-only"),
    _quality_row("latency-p50-target-ms", "reporting_only", "latency_p50_ms", "lte", 30000, ("dev", "validation"), ("PRD:NFR-03", "PRD:15.3"), "reporting-only"),
    _quality_row("latency-p95-target-ms", "reporting_only", "latency_p95_ms", "lte", 80000, ("dev", "validation"), ("PRD:NFR-03", "PRD:15.3"), "reporting-only"),
    _quality_row("batch-hard-failure-rate", "reporting_only", "hard_failure_rate", "lte", Decimal("0.02"), ("dev", "validation"), ("PRD:15.3",), "reporting-only"),
    _quality_row("batch-partial-result-rate", "reporting_only", "partial_result_rate", "lte", Decimal("0.05"), ("dev", "validation"), ("PRD:15.3",), "reporting-only"),
    _quality_row("unseen-domain-macro-f1-drop", "reporting_only", "unseen_domain_macro_f1_drop", "lte", Decimal("0.05"), ("frozen_audit",), ("PRD:14.3",), "reporting-only"),
    _quality_row("query-type-macro-f1-drop", "reporting_only", "query_type_macro_f1_drop", "lte", Decimal("0.02"), ("frozen_audit",), ("PRD:14.3", "PRD:15.2"), "reporting-only"),
    _quality_row("doi-exact-merge-rate", "reporting_only", "doi_exact_merge_rate", "eq", 1, ("frozen_audit",), ("PRD:FR-04", "PRD:15.2"), "reporting-only"),
    _quality_row("rerank-relevance-judgement-accuracy", "promotion", "rerank_relevance_judgement_accuracy", "gte", Decimal("0.85"), ("optional",), ("PRD:FR-07", "PRD:15.2"), "optional-promotion-only"),
    _quality_row("cached-repeat-latency-p50-ms", "reporting_only", "cached_repeat_latency_p50_ms", "lte", 8000, ("dev", "validation"), ("PRD:NFR-03",), "reporting-only"),
    _quality_row("adaptive-macro-f1-delta", "promotion", "adaptive_macro_f1_delta", "gte", Decimal("0.02"), ("optional",), ("PRD:Task-12",), "optional-promotion-only"),
    _quality_row("adaptive-internal-score-delta", "promotion", "adaptive_internal_score_delta", "gte", Decimal("0.02"), ("optional",), ("PRD:Task-12",), "optional-promotion-only"),
    _quality_row("adaptive-f1-decline", "promotion", "adaptive_f1_decline", "lte", Decimal("0.005"), ("optional",), ("PRD:Task-12",), "optional-promotion-only"),
    _quality_row("adaptive-failure-rate-increase", "promotion", "adaptive_failure_rate_increase", "lte", Decimal("0.01"), ("optional",), ("PRD:Task-12",), "optional-promotion-only"),
    _quality_row("bilingual-query-f1-drop", "reporting_only", "bilingual_query_f1_drop", "lte", Decimal("0.05"), ("frozen_audit",), ("PRD:14.3",), "reporting-only"),
)


class QualityGatePolicy(_PolicyModel):
    """The complete immutable rule table consumed by Gate evaluation."""

    schema_version: Literal["quality-gates-v1"]
    rules: list[QualityGateRule]

    @model_validator(mode="after")
    def validate_catalog_coverage(self) -> Self:
        by_id = {rule.rule_id: rule for rule in self.rules}
        if len(by_id) != len(self.rules):
            raise ValueError("quality gate policy contains duplicate rule IDs")
        for rule in self.rules:
            if rule.resolution.startswith("superseded-by-"):
                target_rule_id = rule.resolution.removeprefix("superseded-by-")
                if target_rule_id not in by_id:
                    raise ValueError(
                        f"superseded quality rule target is missing: {target_rule_id}"
                    )
        catalog = {row.rule_id: row for row in AUTHORITATIVE_QUALITY_CATALOG}
        missing = sorted(set(catalog) - set(by_id))
        if missing:
            raise ValueError(f"missing authoritative quality rule: {missing[0]}")
        unexpected = sorted(set(by_id) - set(catalog))
        if unexpected:
            raise ValueError(f"unknown authoritative quality rule: {unexpected[0]}")
        for rule_id, target in catalog.items():
            rule = by_id[rule_id]
            actual = (
                rule.classification,
                rule.measure,
                rule.operator,
                rule.threshold,
                tuple(rule.applies_to),
                tuple(rule.source_refs),
                rule.resolution,
            )
            expected = (
                target.classification,
                target.measure,
                target.operator,
                target.threshold,
                target.applies_to,
                target.source_refs,
                target.resolution,
            )
            if actual != expected:
                raise ValueError(f"authoritative quality rule does not match catalog: {rule_id}")
        return self


def _parse_yaml_mapping(content: bytes, *, policy_name: str) -> dict[str, object]:
    try:
        raw = yaml.safe_load(content)
    except (UnicodeError, yaml.YAMLError) as error:
        raise ValueError(f"invalid {policy_name}") from error
    if not isinstance(raw, dict):
        raise ValueError(f"{policy_name} must contain a mapping")
    return raw


def _decimal_from_yaml_string(value: object, *, field: str) -> Decimal:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a Decimal string, not a Python float")
    try:
        parsed = Decimal(value)
    except ArithmeticError as error:
        raise ValueError(f"{field} must be a valid Decimal string") from error
    if not parsed.is_finite():
        raise ValueError(f"{field} must be a finite Decimal string")
    return parsed


def _datetime_from_yaml_string(value: object, *, field: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{field} must be an ISO-8601 datetime") from error
    if parsed.tzinfo is None:
        raise ValueError(f"{field} must include a timezone")
    return parsed


def _normalize_pricing_policy(raw: dict[str, object]) -> dict[str, object]:
    normalized = dict(raw)
    normalized["effective_at"] = _datetime_from_yaml_string(
        raw.get("effective_at"), field="effective_at"
    )
    normalized["rounding_quantum_cny"] = _decimal_from_yaml_string(
        raw.get("rounding_quantum_cny"), field="rounding_quantum_cny"
    )
    rates = raw.get("rates")
    if not isinstance(rates, list):
        raise ValueError("rates must be a list")
    normalized_rates: list[dict[str, object]] = []
    for index, rate in enumerate(rates):
        if not isinstance(rate, dict):
            raise ValueError(f"rates[{index}] must be a mapping")
        normalized_rate = dict(rate)
        normalized_rate["price_cny_per_unit"] = _decimal_from_yaml_string(
            rate.get("price_cny_per_unit"), field=f"rates[{index}].price_cny_per_unit"
        )
        normalized_rates.append(normalized_rate)
    normalized["rates"] = normalized_rates
    return normalized


def _normalize_quality_gate_policy(raw: dict[str, object]) -> dict[str, object]:
    normalized = dict(raw)
    rules = raw.get("rules")
    if not isinstance(rules, list):
        raise ValueError("rules must be a list")
    normalized_rules: list[dict[str, object]] = []
    for index, rule in enumerate(rules):
        if not isinstance(rule, dict):
            raise ValueError(f"rules[{index}] must be a mapping")
        normalized_rule = dict(rule)
        threshold = rule.get("threshold")
        if isinstance(threshold, bool):
            raise ValueError(f"rules[{index}].threshold must be an integer or Decimal string")
        if isinstance(threshold, str):
            normalized_rule["threshold"] = _decimal_from_yaml_string(
                threshold, field=f"rules[{index}].threshold"
            )
        elif isinstance(threshold, int):
            normalized_rule["threshold"] = threshold
        else:
            raise ValueError(f"rules[{index}].threshold must be an integer or Decimal string")
        normalized_rules.append(normalized_rule)
    normalized["rules"] = normalized_rules
    return normalized


def parse_pricing_policy_bytes(content: bytes) -> PricingPolicy:
    """Parse one exact pricing-policy byte snapshot without reopening a path."""
    try:
        raw = _normalize_pricing_policy(
            _parse_yaml_mapping(content, policy_name="pricing policy")
        )
        return PricingPolicy.model_validate(raw)
    except (OSError, RuntimeError, ValidationError, ValueError):
        raise ValueError("invalid pricing policy") from None


def parse_quality_gate_policy_bytes(content: bytes) -> QualityGatePolicy:
    """Parse the complete authoritative Gate policy from one byte snapshot."""
    try:
        raw = _normalize_quality_gate_policy(
            _parse_yaml_mapping(content, policy_name="quality gate policy")
        )
        return QualityGatePolicy.model_validate(raw)
    except (OSError, RuntimeError, ValidationError, ValueError):
        raise ValueError("invalid quality gate policy") from None


def load_pricing_policy(path: Path) -> PricingPolicy:
    """Load one exact, versioned pricing policy without guessing production rates."""
    try:
        content = path.read_bytes()
    except (OSError, RuntimeError, ValueError):
        raise ValueError(f"invalid pricing policy: {path}") from None
    try:
        return parse_pricing_policy_bytes(content)
    except (OSError, RuntimeError, ValueError):
        raise ValueError(f"invalid pricing policy: {path}") from None


def load_quality_gate_policy(path: Path) -> QualityGatePolicy:
    """Load the complete Gate policy, rejecting unresolved authoritative rows."""
    try:
        content = path.read_bytes()
    except (OSError, RuntimeError, ValueError):
        raise ValueError(f"invalid quality gate policy: {path}") from None
    try:
        return parse_quality_gate_policy_bytes(content)
    except (OSError, RuntimeError, ValueError):
        raise ValueError(f"invalid quality gate policy: {path}") from None


def canonical_pricing_policy_bytes(policy: PricingPolicy) -> bytes:
    """Return stable UTF-8 JSON bytes for the immutable policy identity."""

    def canonical_money(value: Decimal) -> str:
        return format(value.quantize(Decimal("0.000001")), "f")

    payload: dict[str, Any] = {
        "schema_version": policy.schema_version,
        "currency": policy.currency,
        "effective_at": policy.effective_at.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        "source_identity": policy.source_identity,
        "rounding_quantum_cny": canonical_money(policy.rounding_quantum_cny),
        "rates": [
            {
                "dependency": rate.dependency,
                "model_or_adapter": rate.model_or_adapter,
                "unit": rate.unit,
                "price_cny_per_unit": canonical_money(rate.price_cny_per_unit),
            }
            for rate in policy.rates
        ],
    }
    rates = payload["rates"]
    assert isinstance(rates, list)
    rates.sort(
        key=lambda rate: (
            rate["dependency"],
            rate["model_or_adapter"],
            rate["unit"],
        )
    )
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def pricing_policy_sha256(policy: PricingPolicy) -> Sha256:
    """Return the content identity of an approved pricing policy."""

    return f"sha256:{hashlib.sha256(canonical_pricing_policy_bytes(policy)).hexdigest()}"


def quantize_cost_cny(value: Decimal, quantum: Decimal) -> Decimal:
    """Quantize a complete exact cost once at the policy boundary."""

    return value.quantize(quantum, rounding=ROUND_HALF_EVEN)


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

    @property
    def canonical_policy_bytes(self) -> bytes:
        return canonical_pricing_policy_bytes(self._policy)

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
        valued_cost = quantize_cost_cny(total, self._policy.rounding_quantum_cny)
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
