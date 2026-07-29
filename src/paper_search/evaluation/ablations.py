"""Offline injected ablation reporting for public-safe experiment selection."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from paper_search.evaluation.experiments import ExperimentAggregate
from paper_search.evolution import EvolutionStrategy


PUBLIC_MODULES: tuple[str, ...] = (
    "query_planning",
    "multi_source",
    "embedding",
    "citation_expansion",
    "llm_rerank",
    "fixed_two_round",
    "adaptive_evolution",
    "low_budget",
    "balanced",
)

_SAFE_CASE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9-]*\Z")
_SAFE_MODULE_NAME = re.compile(r"[a-z][a-z0-9_]*\Z")
_PUBLIC_MODULE_SET = frozenset(PUBLIC_MODULES)


def evolution_strategy_for_modules(
    modules: Mapping[str, bool],
) -> EvolutionStrategy:
    """Map public evolution flags to an offline evolution strategy."""

    fixed_two = modules.get("fixed_two_round")
    adaptive = modules.get("adaptive_evolution")
    if not isinstance(fixed_two, bool) or not isinstance(adaptive, bool):
        raise ValueError("evolution flags must be booleans")
    if fixed_two and adaptive:
        raise ValueError("evolution flags are mutually exclusive")
    if adaptive:
        return "adaptive"
    if fixed_two:
        return "fixed_two_round"
    return "fixed_one_round"


def _validate_split_phase(
    split: str,
    phase: Literal["tuning", "selection_only"],
) -> None:
    if not split.strip():
        raise ValueError("split must not be blank")
    if split == "validation" and phase != "selection_only":
        raise ValueError("validation ablations must use phase 'selection_only'")
    if phase == "tuning" and split != "dev":
        raise ValueError("tuning ablations must use split 'dev'")


def _validate_unique_case_names(case_names: Iterable[str]) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for case_name in case_names:
        if case_name in seen:
            duplicates.add(case_name)
        seen.add(case_name)
    if duplicates:
        raise ValueError("duplicate ablation case names are not allowed")


def _normalize_case_payload(data: Any) -> Any:
    if not isinstance(data, Mapping):
        return data

    payload = dict(data)
    modules = payload.get("modules")
    if isinstance(modules, Mapping):
        normalized_modules = dict(modules)
        for key, value in normalized_modules.items():
            if not isinstance(key, str) or not _SAFE_MODULE_NAME.fullmatch(key):
                raise ValueError("module names must be safe public identifiers")
            if key not in _PUBLIC_MODULE_SET:
                raise ValueError("modules must use only public boolean flags")
            if not isinstance(value, bool):
                raise ValueError("modules must contain booleans only")
        missing = _PUBLIC_MODULE_SET.difference(normalized_modules)
        if missing:
            raise ValueError("modules must define every public boolean flag")
        payload["modules"] = normalized_modules
    return payload


def _validate_case_name(name: str) -> str:
    if not _SAFE_CASE_NAME.fullmatch(name):
        raise ValueError("ablation case names must be safe public identifiers")
    return name


class AblationCase(BaseModel):
    """One offline public ablation case defined only by safe boolean modules."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    name: str = Field(min_length=1)
    modules: dict[str, bool]

    @model_validator(mode="before")
    @classmethod
    def _validate_module_payload(cls, data: Any) -> Any:
        return _normalize_case_payload(data)

    @model_validator(mode="after")
    def _validate_name(self) -> "AblationCase":
        _validate_case_name(self.name)
        return self


class _AblationResult(BaseModel):
    """Public report row for one ablation case."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    name: str = Field(min_length=1)
    modules: dict[str, bool]
    aggregate: ExperimentAggregate

    @model_validator(mode="before")
    @classmethod
    def _validate_module_payload(cls, data: Any) -> Any:
        return _normalize_case_payload(data)

    @model_validator(mode="after")
    def _validate_name(self) -> "_AblationResult":
        _validate_case_name(self.name)
        return self


class AblationReport(BaseModel):
    """Aggregate-only report for one offline ablation run."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    split: str = Field(min_length=1)
    phase: Literal["tuning", "selection_only"]
    cases: tuple[_AblationResult, ...]

    @model_validator(mode="after")
    def _validate_policy(self) -> "AblationReport":
        _validate_split_phase(self.split, self.phase)
        _validate_unique_case_names(case.name for case in self.cases)
        return self


def run_ablations(
    *,
    cases: Sequence[AblationCase],
    split: str,
    evaluator: Callable[[AblationCase], ExperimentAggregate],
    phase: Literal["tuning", "selection_only"] = "tuning",
) -> AblationReport:
    """Evaluate supplied cases offline using an injected aggregate evaluator."""

    _validate_split_phase(split, phase)
    _validate_unique_case_names(case.name for case in cases)

    return AblationReport(
        split=split,
        phase=phase,
        cases=tuple(
            _AblationResult(
                name=case.name,
                modules=dict(case.modules),
                aggregate=evaluator(case),
            )
            for case in cases
        ),
    )
