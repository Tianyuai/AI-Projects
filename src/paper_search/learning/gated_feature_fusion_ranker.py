"""Research-only gated late fusion over independent document feature families."""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from collections.abc import Iterable, Mapping, Sequence, Set
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar, cast

import numpy as np

from paper_search.domain.models import QuerySpec
from paper_search.evaluation.predictions import (
    paper_evaluation_id,
    paper_matches_evaluation_ids,
)
from paper_search.learning.cpu_document_ranker import (
    DocumentCandidateEvidence,
    DocumentRankingQuery,
)
from paper_search.learning.method_usage_evidence import (
    LEGACY_METHOD_TEXT_MATCH_SCHEMA_VERSION,
    METHOD_USAGE_EVIDENCE_SCHEMA_VERSION,
    method_usage_evidence_fraction,
)
from paper_search.learning.negation_evidence import (
    NEGATION_EVIDENCE_SCHEMA_VERSION,
    negation_evidence_fractions,
    negation_topic_relevant,
)
from paper_search.learning.query_constraint_annotations import (
    FrozenConstraintProfileStore,
)
from paper_search.learning.query_constraint_profile import QueryConstraintProfile
from paper_search.learning.query_constraint_profile import profile_query_constraints
from paper_search.learning.task_slot_document_ranker import (
    BaselineDocumentRanker,
    FrozenTaskSlotLabel,
    FrozenTaskSlotLabelStore,
    FrozenTaskValue,
    task_slot_candidate_features,
)
from paper_search.query.parser import extract_explicit_year_bounds, rule_fallback
from paper_search.retrieval.pasa_paper_database import (
    ARXIV_MISSING_YEAR_EVIDENCE_POLICY,
    PASA_TRAINING_CONSTRAINT_INJECTED_SOURCE,
    PASA_TRAINING_GOLD_INJECTED_SOURCE,
    effective_publication_year,
)


FusionFamily = str
FeatureKey = TypeVar("FeatureKey")
FUSION_FAMILIES = frozenset(
    {"task_provenance", "entity", "hard_constraint", "reliability"}
)
TASK_PROVENANCE_ALLOWED_STATUSES = frozenset(
    {"reviewed", "base_accepted", "runtime_deterministic"}
)
_INVALID_ENTITY_TERMS = frozenset({"how", "what", "when", "where", "which", "who"})
DECLARED_YEAR_EVIDENCE_POLICY = "declared-publication-year-only-v1"
_YEAR_EVIDENCE_POLICIES = frozenset(
    {DECLARED_YEAR_EVIDENCE_POLICY, ARXIV_MISSING_YEAR_EVIDENCE_POLICY}
)
_METHOD_EVIDENCE_SCHEMAS = frozenset(
    {
        LEGACY_METHOD_TEXT_MATCH_SCHEMA_VERSION,
        METHOD_USAGE_EVIDENCE_SCHEMA_VERSION,
    }
)


@dataclass(frozen=True)
class FusionQueryContext:
    task_label: FrozenTaskSlotLabel | None = None
    constraint_profile: QueryConstraintProfile | None = None
    task_runtime_inferred: bool = False
    constraint_runtime_inferred: bool = False


class FusionContextStore(Protocol):
    def for_scoring_query(
        self, query: str, *, query_spec: QuerySpec | None = None
    ) -> FusionQueryContext: ...

    def for_training_query(self, query: str) -> FusionQueryContext: ...


class FrozenFusionContextStore:
    """Join frozen task and constraint labels without parsing live query text."""

    def __init__(
        self,
        *,
        task_store: FrozenTaskSlotLabelStore,
        constraint_store: FrozenConstraintProfileStore,
    ) -> None:
        self.task_store = task_store
        self.constraint_store = constraint_store

    def for_scoring_query(
        self, query: str, *, query_spec: QuerySpec | None = None
    ) -> FusionQueryContext:
        return FusionQueryContext(
            task_label=self.task_store.for_scoring_query(query),
            constraint_profile=self.constraint_store.for_scoring_query(query),
        )

    def for_training_query(self, query: str) -> FusionQueryContext:
        return FusionQueryContext(
            task_label=self.task_store.for_training_query(query),
            constraint_profile=self.constraint_store.for_training_query(query),
        )


def _normalized_terms(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(
        sorted(
            {" ".join(value.casefold().split()) for value in values if value.strip()},
            key=lambda value: (-len(value), value),
        )
    )


def _text_contains(text: str, phrase: str) -> bool:
    normalized_text = " ".join(text.casefold().split())
    normalized_phrase = " ".join(phrase.casefold().split())
    if not normalized_phrase:
        return False
    return bool(
        re.search(
            rf"(?<![\w]){re.escape(normalized_phrase)}(?![\w])",
            normalized_text,
        )
    )


def _term_index(terms: Sequence[str]) -> dict[str, tuple[str, ...]]:
    indexed: dict[str, list[str]] = {}
    for term in terms:
        tokens = re.findall(r"[\w+_.-]+", term.casefold())
        if tokens:
            indexed.setdefault(tokens[0], []).append(term)
    return {key: tuple(values) for key, values in indexed.items()}


def _matching_terms(query: str, index: Mapping[str, Sequence[str]]) -> list[str]:
    tokens = set(re.findall(r"[\w+_.-]+", query.casefold()))
    candidates = {
        term for token in tokens for term in index.get(token, ())
    }
    return [term for term in candidates if _text_contains(query, term)]


def _first_explicit_group(query: str, pattern: str) -> str | None:
    match = re.search(pattern, query, flags=re.IGNORECASE)
    if match is None:
        return None
    value = " ".join(match.group(1).strip(" ,.;:?").casefold().split())
    return value or None


def _explicit_runtime_slots(query: str) -> dict[str, object]:
    """Extract only syntax-delimited slots; ambiguous free text stays unlabelled."""

    method = _first_explicit_group(
        query,
        r"\b(?:using|employing|employs?|based\s+on)\s+(.+?)"
        r"(?=\s+(?:for|on|in|from|after|before|without|excluding)\b|[,.;?]|$)",
    )
    task = _first_explicit_group(
        query,
        r"\bfor\s+(.+?)"
        r"(?=\s+(?:using|with|on|in|from|after|before|without|excluding)\b|[,.;?]|$)",
    )
    dataset_matches = re.findall(
        r"\b(?:the\s+)?([A-Z][A-Za-z0-9+_.-]{1,}"
        r"(?:\s+(?:19|20)\d{2})?)\s+"
        r"(?:dataset|benchmark|corpus|challenge)\b",
        query,
    )
    dataset_matches.extend(
        re.findall(
            r"\b(?:datasets?|benchmarks?|corpora|challenges?)\s+"
            r"(?:such\s+as|like|including)\s+"
            r"([A-Z][A-Za-z0-9+_.-]{1,}(?:\s+(?:19|20)\d{2})?)\b",
            query,
        )
    )
    datasets = [
        " ".join(value.casefold().split())
        for value in dataset_matches
    ]
    year_from, year_to = extract_explicit_year_bounds(query)
    return {
        "methods": [method] if method else [],
        "tasks": [task] if task else [],
        "datasets": list(_normalized_terms(datasets)),
        "year_from": year_from,
        "year_to": year_to,
    }


class UnifiedFusionContextResolver:
    """Frozen-first, deterministic query context used by evaluation and live ranking."""

    resolver_id = "unified-local-fusion-context-v3"

    def __init__(
        self,
        *,
        task_store: FrozenTaskSlotLabelStore,
        constraint_store: FrozenConstraintProfileStore,
        task_terms: Iterable[str] = (),
        method_terms: Iterable[str] = (),
        dataset_terms: Iterable[str] = (),
    ) -> None:
        self.task_store = task_store
        self.constraint_store = constraint_store
        self.task_terms = _normalized_terms(task_terms)
        self.method_terms = _normalized_terms(method_terms)
        self.dataset_terms = _normalized_terms(dataset_terms)
        self._task_term_index = _term_index(self.task_terms)
        self._method_term_index = _term_index(self.method_terms)
        self._dataset_term_index = _term_index(self.dataset_terms)
        identity = json.dumps(
            {
                "resolver_id": self.resolver_id,
                "raw_slot_policy": (
                    "frozen-lexicon-exact-plus-explicit-method-task-"
                    "dataset-year-negation"
                ),
                "task_terms": self.task_terms,
                "method_terms": self.method_terms,
                "dataset_terms": self.dataset_terms,
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        self.resolver_sha256 = "sha256:" + hashlib.sha256(identity).hexdigest()

    @staticmethod
    def _runtime_task_label(query: str, tasks: Sequence[str]) -> FrozenTaskSlotLabel | None:
        normalized = _normalized_terms(tasks)
        if not normalized:
            return None
        digest = "sha256:" + hashlib.sha256(query.encode("utf-8")).hexdigest()
        return FrozenTaskSlotLabel(
            query_id=f"runtime:{digest.removeprefix('sha256:')[:16]}",
            query_sha256=digest,
            role="runtime",
            split="runtime",
            tasks=tuple(FrozenTaskValue(value, 0.9) for value in normalized),
            ambiguous_fields=(),
            task_label_status="runtime_deterministic",
        )

    def _local_spec(self, query: str, query_spec: QuerySpec | None) -> QuerySpec:
        if query_spec is not None:
            return query_spec
        base = rule_fallback(query)
        explicit = _explicit_runtime_slots(query)
        tasks = [
            value
            for value in _normalized_terms(
                [
                    *_matching_terms(query, self._task_term_index),
                    *cast(list[str], explicit["tasks"]),
                ]
            )
            if value not in _INVALID_ENTITY_TERMS
        ]
        methods = [
            value
            for value in _normalized_terms(
                [
                    *_matching_terms(query, self._method_term_index),
                    *cast(list[str], explicit["methods"]),
                ]
            )
            if value not in _INVALID_ENTITY_TERMS
        ]
        datasets = [
            value
            for value in _normalized_terms(
                [
                    *_matching_terms(query, self._dataset_term_index),
                    *cast(list[str], explicit["datasets"]),
                ]
            )
            if value not in _INVALID_ENTITY_TERMS
        ]
        return base.model_copy(
            update={
                "tasks": tasks,
                "methods": methods,
                "datasets": datasets,
                "year_from": (
                    explicit["year_from"]
                    if explicit["year_from"] is not None
                    else None
                    if explicit["year_to"] is not None
                    else base.year_from
                ),
                "year_to": (
                    explicit["year_to"]
                    if explicit["year_to"] is not None
                    else None
                    if explicit["year_from"] is not None
                    else base.year_to
                ),
            }
        )

    @staticmethod
    def _local_profile(spec: QuerySpec, *, explicit: bool) -> QueryConstraintProfile:
        profile = profile_query_constraints(spec)
        structured = set(profile.labels).intersection(
            {"method", "dataset", "task", "year", "negation"}
        )
        if not structured:
            return profile
        confidence = 0.9 if explicit else 1.0
        return profile.model_copy(update={"confidence": confidence})

    def _resolve(
        self, query: str, *, query_spec: QuerySpec | None
    ) -> tuple[FusionQueryContext, dict[str, str]]:
        frozen_task = self.task_store.for_scoring_query(query)
        frozen_profile = self.constraint_store.for_scoring_query(query)
        local_spec = self._local_spec(query, query_spec)
        task = frozen_task or self._runtime_task_label(query, local_spec.tasks)
        profile = frozen_profile or self._local_profile(
            local_spec, explicit=query_spec is not None
        )
        return (
            FusionQueryContext(
                task_label=task,
                constraint_profile=profile,
                task_runtime_inferred=frozen_task is None and task is not None,
                constraint_runtime_inferred=(
                    frozen_profile is None and profile is not None
                ),
            ),
            {
                "task": (
                    "frozen"
                    if frozen_task is not None
                    else "local_query_spec"
                    if query_spec is not None
                    else "local_rules"
                ),
                "constraint": (
                    "frozen"
                    if frozen_profile is not None
                    else "local_query_spec"
                    if query_spec is not None
                    else "local_rules"
                ),
            },
        )

    def for_scoring_query(
        self, query: str, *, query_spec: QuerySpec | None = None
    ) -> FusionQueryContext:
        return self._resolve(query, query_spec=query_spec)[0]

    def for_local_query(
        self, query: str, *, query_spec: QuerySpec | None = None
    ) -> FusionQueryContext:
        """Infer context without consulting frozen rows for directed backfill."""

        spec = self._local_spec(query, query_spec)
        return FusionQueryContext(
            task_label=self._runtime_task_label(query, spec.tasks),
            constraint_profile=self._local_profile(
                spec, explicit=query_spec is not None
            ),
            task_runtime_inferred=bool(spec.tasks),
            constraint_runtime_inferred=True,
        )

    def for_training_query(self, query: str) -> FusionQueryContext:
        # Fitting remains frozen-only; runtime inference must not create train leakage.
        return FusionQueryContext(
            task_label=self.task_store.for_training_query(query),
            constraint_profile=self.constraint_store.for_training_query(query),
        )

    def context_receipt(
        self, query: str, *, query_spec: QuerySpec | None = None
    ) -> dict[str, object]:
        context, source = self._resolve(query, query_spec=query_spec)
        task = context.task_label
        profile = context.constraint_profile
        payload: dict[str, object] = {
            "schema_version": "fusion-query-context-receipt-v1",
            "query_sha256": "sha256:" + hashlib.sha256(query.encode()).hexdigest(),
            "resolver_id": self.resolver_id,
            "resolver_sha256": self.resolver_sha256,
            "context_source": dict(sorted(source.items())),
            "tasks": [value.normalized_value for value in task.tasks] if task else [],
            "methods": list(profile.methods) if profile else [],
            "datasets": list(profile.datasets) if profile else [],
            "year_from": profile.year_from if profile else None,
            "year_to": profile.year_to if profile else None,
            "exclusions": list(profile.exclusions) if profile else [],
            "activated_families": sorted(
                family
                for family in FUSION_FAMILIES
                if gated_family_eligibility(context, family, gated=True)
            ),
        }
        canonical = json.dumps(
            payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        ).encode()
        payload["context_sha256"] = "sha256:" + hashlib.sha256(canonical).hexdigest()
        return payload


def gated_family_eligibility(
    context: FusionQueryContext,
    family: FusionFamily,
    *,
    gated: bool,
    minimum_confidence: float = 0.8,
) -> bool:
    """Return a query-level gate; ungated remains label-presence constrained."""

    if family not in FUSION_FAMILIES:
        raise ValueError(f"unsupported fusion family: {family}")
    if family == "task_provenance":
        label = context.task_label
        if label is None or label.reliability_weight <= 0.0:
            return False
        return not gated or label.task_label_status in TASK_PROVENANCE_ALLOWED_STATUSES
    if family == "reliability":
        return True
    profile = context.constraint_profile
    if profile is None:
        return False
    if family == "entity":
        present = bool({"method", "dataset"}.intersection(profile.labels))
    else:
        present = (
            "year" in profile.labels
            and (profile.year_from is not None or profile.year_to is not None)
        ) or ("negation" in profile.labels and bool(profile.exclusions))
    return present and (not gated or profile.confidence >= minimum_confidence)


def _source_features(
    candidate: DocumentCandidateEvidence,
    *,
    prefix: str,
    scale: float,
) -> dict[str, float]:
    values: dict[str, float] = {}
    for source, rank in sorted(candidate.source_ranks.items()):
        values[f"{prefix}-source={source}"] = scale / rank
    return values


def _training_gold_injected_only(candidate: DocumentCandidateEvidence) -> bool:
    if PASA_TRAINING_GOLD_INJECTED_SOURCE not in candidate.paper.sources:
        return False
    return bool(candidate.source_ranks) and all(
        "pasa" in source.casefold() for source in candidate.source_ranks
    )


def _pasa_only_candidate(candidate: DocumentCandidateEvidence) -> bool:
    return bool(candidate.source_ranks) and all(
        "pasa" in source.casefold() for source in candidate.source_ranks
    )


def training_candidate_source_features_suppressed(
    candidate: DocumentCandidateEvidence, *, is_gold: bool
) -> bool:
    return _training_gold_injected_only(candidate) or (
        is_gold and _pasa_only_candidate(candidate)
    )


def runtime_candidate_eligible_for_family(
    candidate: DocumentCandidateEvidence,
    family: FusionFamily,
) -> bool:
    """Keep training-only PASA evidence out of source-sensitive runtime scores."""

    if family not in FUSION_FAMILIES:
        raise ValueError(f"unsupported fusion family: {family}")
    if PASA_TRAINING_CONSTRAINT_INJECTED_SOURCE in candidate.paper.sources:
        return family == "hard_constraint"
    if family in {"reliability", "task_provenance"}:
        return not _training_gold_injected_only(candidate)
    return True


def training_candidate_eligible_for_family(
    candidate: DocumentCandidateEvidence,
    family: FusionFamily,
    *,
    is_gold: bool = False,
) -> bool:
    if not runtime_candidate_eligible_for_family(candidate, family):
        return False
    if family in {"reliability", "task_provenance"}:
        return not training_candidate_source_features_suppressed(
            candidate, is_gold=is_gold
        )
    return True


def bounded_preference_pairs(
    positives: Sequence[dict[FeatureKey, float]],
    negatives: Sequence[dict[FeatureKey, float]],
    limit: int | None,
) -> list[tuple[dict[FeatureKey, float], dict[FeatureKey, float]]]:
    pairs: list[tuple[dict[FeatureKey, float], dict[FeatureKey, float]]] = []
    for negative in negatives:
        for positive in positives:
            if positive == negative:
                continue
            pairs.append((positive, negative))
            if limit is not None and len(pairs) >= limit:
                return pairs
    return pairs


def bounded_hard_constraint_preference_pairs(
    context: FusionQueryContext,
    positives: Sequence[dict[str, float]],
    negatives: Sequence[dict[str, float]],
    *,
    hard_negative_limit: int,
    pair_limit: int | None,
) -> list[tuple[dict[str, float], dict[str, float]]]:
    """Keep only directionally valid hard-constraint supervision pairs."""

    profile = context.constraint_profile
    if profile is None:
        return []

    def priority(values: Mapping[str, float]) -> tuple[float, float, float]:
        return (
            -float(values.get("hard-negation-conflict", 0.0)),
            -float(values.get("hard-year-conflict", 0.0)),
            -float(values.get("hard-year-missing", 0.0)),
        )

    ordered_negatives = [
        values
        for _index_value, values in sorted(
            enumerate(negatives),
            key=lambda item: (*priority(item[1]), item[0]),
        )[:hard_negative_limit]
    ]
    pairs: list[tuple[dict[str, float], dict[str, float]]] = []
    for negative in ordered_negatives:
        for positive in positives:
            if not hard_constraint_pair_signals(context, positive, negative):
                continue
            pairs.append((positive, negative))
    return _balanced_directional_pairs(
        pairs,
        signal_order=("year", "negation"),
        signal_getter=lambda pair: hard_constraint_pair_signals(context, *pair),
        limit=pair_limit,
    )


def hard_constraint_pair_signals(
    context: FusionQueryContext,
    positive: Mapping[str, float],
    negative: Mapping[str, float],
) -> frozenset[str]:
    """Return hard signals whose Gold-to-negative direction is trustworthy."""

    profile = context.constraint_profile
    if profile is None:
        return frozenset()
    signals: set[str] = set()
    if "year" in profile.labels and (
        profile.year_from is not None or profile.year_to is not None
    ):
        if (
            positive.get("hard-year-compliant", 0.0)
            > negative.get("hard-year-compliant", 0.0)
            or positive.get("hard-year-conflict", 0.0)
            < negative.get("hard-year-conflict", 0.0)
            or positive.get("hard-year-missing", 0.0)
            < negative.get("hard-year-missing", 0.0)
        ):
            signals.add("year")
    if "negation" in profile.labels and profile.exclusions:
        if (
            positive.get("hard-negation-conflict", 0.0)
            < negative.get("hard-negation-conflict", 0.0)
        ):
            signals.add("negation")
    return frozenset(signals)


def entity_pair_signals(
    context: FusionQueryContext,
    positive: Mapping[str, float],
    negative: Mapping[str, float],
) -> frozenset[str]:
    """Return entity signals whose Gold has stronger direct text evidence."""

    profile = context.constraint_profile
    if profile is None:
        return frozenset()
    signals: set[str] = set()
    for signal in ("method", "dataset"):
        if signal not in profile.labels:
            continue
        key = f"entity-{signal}-text-match"
        if positive.get(key, 0.0) > negative.get(key, 0.0):
            signals.add(signal)
    return frozenset(signals)


def bounded_entity_preference_pairs(
    context: FusionQueryContext,
    positives: Sequence[dict[str, float]],
    negatives: Sequence[dict[str, float]],
    *,
    hard_negative_limit: int,
    pair_limit: int | None,
) -> list[tuple[dict[str, float], dict[str, float]]]:
    """Build entity pairs only when Gold has stronger method/dataset evidence."""

    pairs: list[tuple[dict[str, float], dict[str, float]]] = []
    for negative in negatives[:hard_negative_limit]:
        for positive in positives:
            if not entity_pair_signals(context, positive, negative):
                continue
            pairs.append((positive, negative))
    return _balanced_directional_pairs(
        pairs,
        signal_order=("method", "dataset"),
        signal_getter=lambda pair: entity_pair_signals(context, *pair),
        limit=pair_limit,
    )


def _balanced_directional_pairs(
    pairs: Sequence[tuple[dict[str, float], dict[str, float]]],
    *,
    signal_order: Sequence[str],
    signal_getter: Any,
    limit: int | None,
) -> list[tuple[dict[str, float], dict[str, float]]]:
    """Apply a query cap without letting one gated signal hide another."""

    if limit is None or len(pairs) <= limit:
        return list(pairs)
    selected_indices: list[int] = []
    covered: set[str] = set()
    for signal in signal_order:
        if signal in covered or len(selected_indices) >= limit:
            continue
        for index, pair in enumerate(pairs):
            if index in selected_indices:
                continue
            signals = set(signal_getter(pair))
            if signal not in signals:
                continue
            selected_indices.append(index)
            covered.update(signals)
            break
    for index in range(len(pairs)):
        if len(selected_indices) >= limit:
            break
        if index not in selected_indices:
            selected_indices.append(index)
    return [pairs[index] for index in selected_indices]


def _candidate_text(candidate: DocumentCandidateEvidence) -> str:
    return ". ".join(
        value
        for value in (candidate.paper.title, candidate.paper.abstract or "")
        if value
    )


def _match_fraction(text: str, values: Sequence[str]) -> float:
    normalized = _normalized_terms(values)
    if not normalized:
        return 0.0
    return sum(_text_contains(text, value) for value in normalized) / len(normalized)


def _has_negation_candidate_contrast(
    context: FusionQueryContext,
    candidates: Sequence[DocumentCandidateEvidence],
    *,
    query: str,
) -> bool:
    profile = context.constraint_profile
    if profile is None or "negation" not in profile.labels or not profile.exclusions:
        return True
    conflicts = []
    for candidate in candidates:
        text = _candidate_text(candidate)
        conflict = negation_evidence_fractions(text, profile.exclusions)[0] > 0.0
        conflicts.append(
            conflict
            and negation_topic_relevant(query, text, profile.exclusions)
        )
    return any(conflicts) and not all(conflicts)


def gated_family_candidate_features(
    context: FusionQueryContext,
    candidate: DocumentCandidateEvidence,
    *,
    baseline_rank: int,
    family: FusionFamily,
    constraint_text_evidence: bool = False,
    query: str | None = None,
    suppress_source_features: bool = False,
    publication_year_evidence_policy: str = DECLARED_YEAR_EVIDENCE_POLICY,
    method_usage_evidence_schema_version: str = (
        METHOD_USAGE_EVIDENCE_SCHEMA_VERSION
    ),
) -> dict[str, float]:
    """Build field-separated features; title and abstract are intentionally unread."""

    if baseline_rank <= 0:
        raise ValueError("fusion baseline rank must be positive")
    if family not in FUSION_FAMILIES:
        raise ValueError(f"unsupported fusion family: {family}")
    if family == "task_provenance":
        return task_slot_candidate_features(
            context.task_label,
            candidate,
            baseline_rank=baseline_rank,
            families={"task_slot_reliability"},
            reliability_components={"provenance_status"},
        )
    if family == "reliability":
        ranks = tuple(candidate.source_ranks.values())
        return {
            "reliability-source-support": min(candidate.support_count, 6) / 6.0,
            "reliability-best-source-reciprocal": 1.0 / min(ranks),
            "reliability-mean-source-reciprocal": sum(1.0 / rank for rank in ranks)
            / len(ranks),
            "reliability-baseline-reciprocal": 1.0 / baseline_rank,
            "reliability-year-present": float(
                candidate.paper.publication_year is not None
            ),
        }
    profile = context.constraint_profile
    if profile is None:
        return {}
    if family == "entity":
        values: dict[str, float] = {}
        text = _candidate_text(candidate)
        allow_source_features = (
            not _training_gold_injected_only(candidate)
            and not suppress_source_features
        )
        if "method" in profile.labels:
            if allow_source_features:
                values.update(
                    _source_features(
                        candidate,
                        prefix="entity-method",
                        scale=profile.confidence,
                    )
                )
            if constraint_text_evidence:
                if (
                    method_usage_evidence_schema_version
                    == METHOD_USAGE_EVIDENCE_SCHEMA_VERSION
                ):
                    method_match = method_usage_evidence_fraction(
                        candidate.paper.title,
                        candidate.paper.abstract or "",
                        profile.methods,
                    )
                elif (
                    method_usage_evidence_schema_version
                    == LEGACY_METHOD_TEXT_MATCH_SCHEMA_VERSION
                ):
                    method_match = _match_fraction(text, profile.methods)
                else:
                    raise ValueError("unsupported method usage evidence schema")
                values["entity-method-text-match"] = method_match
        if "dataset" in profile.labels:
            if allow_source_features:
                values.update(
                    _source_features(
                        candidate,
                        prefix="entity-dataset",
                        scale=profile.confidence,
                    )
                )
            if constraint_text_evidence:
                values["entity-dataset-text-match"] = _match_fraction(
                    text, profile.datasets
                )
        return values

    if publication_year_evidence_policy not in _YEAR_EVIDENCE_POLICIES:
        raise ValueError("unsupported publication year evidence policy")
    values = {}
    year = candidate.paper.publication_year
    if publication_year_evidence_policy == ARXIV_MISSING_YEAR_EVIDENCE_POLICY:
        year = effective_publication_year(candidate.paper)
    if "year" in profile.labels:
        if year is None:
            values["hard-year-missing"] = 1.0
        else:
            compliant = (profile.year_from is None or year >= profile.year_from) and (
                profile.year_to is None or year <= profile.year_to
            )
            values["hard-year-compliant"] = float(compliant)
            values["hard-year-conflict"] = float(not compliant)
    if "negation" in profile.labels:
        if constraint_text_evidence:
            text = _candidate_text(candidate)
            conflict, clean = negation_evidence_fractions(text, profile.exclusions)
            if query is not None and not negation_topic_relevant(
                query, text, profile.exclusions
            ):
                conflict = clean = 0.0
            values["hard-negation-conflict"] = conflict
            values["hard-negation-clean"] = clean
    return values


def _index(name: str, dimension: int) -> int:
    digest = hashlib.blake2b(name.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % dimension


def _hashed(values: Mapping[str, float], dimension: int) -> dict[int, float]:
    output: dict[int, float] = {}
    for name, value in values.items():
        index = _index(name, dimension)
        output[index] = output.get(index, 0.0) + value
    return output


def _score(weights: np.ndarray, values: Mapping[int, float]) -> float:
    return float(sum(weights[index] * value for index, value in values.items()))


def anchored_family_weights(
    candidate: Mapping[str, np.ndarray],
    production: Mapping[str, np.ndarray],
    *,
    families: Set[str],
) -> dict[str, np.ndarray]:
    """Replace selected candidate families with exact production vectors."""

    selected = frozenset(families)
    if set(candidate) != set(production):
        raise ValueError("candidate and production families must match")
    if not selected or not selected.issubset(candidate):
        raise ValueError("anchored families must be a non-empty enabled subset")
    output: dict[str, np.ndarray] = {}
    for family, candidate_vector in candidate.items():
        production_vector = production[family]
        if candidate_vector.shape != production_vector.shape:
            raise ValueError(f"family weight shape mismatch: {family}")
        source = production_vector if family in selected else candidate_vector
        output[family] = source.copy()
    return output


class GatedFeatureFusionRanker:
    """Train each family independently, then combine only bounded residuals."""

    model_id = "gated-feature-fusion-document-ranker-research-v1"

    def __init__(
        self,
        *,
        baseline_ranker: BaselineDocumentRanker,
        context_store: FusionContextStore,
        feature_families: Set[str],
        gated: bool = True,
        dimension: int = 2048,
        epochs: int = 12,
        learning_rate: float = 0.05,
        l2: float = 1e-6,
        family_caps: Mapping[str, float] | None = None,
        maximum_total_residual: float = 0.35,
        hard_negative_limit: int = 100,
        constraint_text_evidence: bool = False,
        runtime_context_scoring: bool = False,
        max_pairs_per_query_family: int | None = None,
        pair_budget_by_family: Mapping[str, int] | None = None,
        publication_year_evidence_policy: str = DECLARED_YEAR_EVIDENCE_POLICY,
        method_usage_evidence_schema_version: str = (
            METHOD_USAGE_EVIDENCE_SCHEMA_VERSION
        ),
    ) -> None:
        families = frozenset(feature_families)
        unsupported = families - FUSION_FAMILIES
        if not families or unsupported:
            raise ValueError(f"unsupported fusion families: {sorted(unsupported)}")
        caps = dict(family_caps or {family: 0.1 for family in families})
        if set(caps) != set(families) or any(
            not 0.0 < value <= 1.0 for value in caps.values()
        ):
            raise ValueError("fusion family caps must cover enabled families")
        if dimension <= 0 or epochs <= 0 or hard_negative_limit <= 0:
            raise ValueError("fusion ranker sizes must be positive")
        if learning_rate <= 0 or l2 < 0 or not 0 < maximum_total_residual <= 1:
            raise ValueError("invalid fusion optimization settings")
        if (
            max_pairs_per_query_family is not None
            and max_pairs_per_query_family <= 0
        ):
            raise ValueError("fusion pair limit must be positive")
        if max_pairs_per_query_family is not None and pair_budget_by_family is not None:
            raise ValueError("uniform and family-specific pair budgets cannot be mixed")
        budgets = (
            {str(family): int(value) for family, value in pair_budget_by_family.items()}
            if pair_budget_by_family is not None
            else None
        )
        if budgets is not None and (
            set(budgets) != set(families)
            or any(isinstance(value, bool) or value <= 0 for value in budgets.values())
        ):
            raise ValueError("fusion pair budgets must cover enabled families")
        if publication_year_evidence_policy not in _YEAR_EVIDENCE_POLICIES:
            raise ValueError("unsupported publication year evidence policy")
        if method_usage_evidence_schema_version not in _METHOD_EVIDENCE_SCHEMAS:
            raise ValueError("unsupported method usage evidence schema")
        self.baseline_ranker = baseline_ranker
        self.context_store = context_store
        self.feature_families = families
        self.gated = gated
        self.dimension = dimension
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.l2 = l2
        self.family_caps = caps
        self.maximum_total_residual = maximum_total_residual
        self.hard_negative_limit = hard_negative_limit
        self.constraint_text_evidence = constraint_text_evidence
        self.runtime_context_scoring = runtime_context_scoring
        self.max_pairs_per_query_family = max_pairs_per_query_family
        self.pair_budget_by_family = budgets
        self.publication_year_evidence_policy = publication_year_evidence_policy
        self.method_usage_evidence_schema_version = (
            method_usage_evidence_schema_version
        )
        self.weights = {
            family: np.zeros(dimension, dtype=np.float64) for family in families
        }
        self.last_fit_query_count = {family: 0 for family in families}

    def _pair_budget(self, family: str) -> int | None:
        if self.pair_budget_by_family is not None:
            return self.pair_budget_by_family[family]
        return self.max_pairs_per_query_family

    def _b0_rows(
        self, query: str, candidates: Sequence[DocumentCandidateEvidence]
    ) -> list[DocumentCandidateEvidence]:
        input_rows = list(candidates)
        rows = list(self.baseline_ranker.rank(query, input_rows))
        input_ids = [paper_evaluation_id(row.paper) for row in input_rows]
        output_ids = [paper_evaluation_id(row.paper) for row in rows]
        if len(input_ids) != len(output_ids) or set(input_ids) != set(output_ids):
            raise ValueError("B0 ranker changed fusion candidate identity")
        return rows

    def fit(self, queries: Sequence[DocumentRankingQuery]) -> dict[str, int]:
        validated_queries = [
            DocumentRankingQuery.model_validate(query) for query in queries
        ]
        query_ids = [query.query_id for query in validated_queries]
        if len(query_ids) != len(set(query_ids)):
            raise ValueError("fusion training query ids must be unique")
        pairs: dict[str, list[tuple[dict[int, float], dict[int, float]]]] = {
            family: [] for family in self.feature_families
        }
        used = {family: 0 for family in self.feature_families}
        for query in validated_queries:
            context = self.context_store.for_training_query(query.query)
            rows = self._b0_rows(query.query, query.candidates)
            gold = set(query.gold_paper_ids)
            for family in self.feature_families:
                if not gated_family_eligibility(context, family, gated=self.gated):
                    continue
                positives: list[dict[int, float]] = []
                negatives: list[dict[int, float]] = []
                named_positives: list[dict[str, float]] = []
                named_negatives: list[dict[str, float]] = []
                for rank, candidate in enumerate(rows, start=1):
                    is_gold = paper_matches_evaluation_ids(candidate.paper, gold)
                    if not training_candidate_eligible_for_family(
                        candidate, family, is_gold=is_gold
                    ):
                        continue
                    named_values = gated_family_candidate_features(
                        context,
                        candidate,
                        baseline_rank=rank,
                        family=family,
                        constraint_text_evidence=self.constraint_text_evidence,
                        query=query.query,
                        suppress_source_features=(
                            family == "entity"
                            and training_candidate_source_features_suppressed(
                                candidate, is_gold=is_gold
                            )
                        ),
                        publication_year_evidence_policy=(
                            self.publication_year_evidence_policy
                        ),
                        method_usage_evidence_schema_version=(
                            self.method_usage_evidence_schema_version
                        ),
                    )
                    values = _hashed(named_values, self.dimension)
                    directional_entity = (
                        family == "entity" and self.constraint_text_evidence
                    )
                    if is_gold:
                        if family == "hard_constraint" or directional_entity:
                            named_positives.append(named_values)
                        else:
                            positives.append(values)
                    elif family == "hard_constraint" or directional_entity:
                        named_negatives.append(named_values)
                    elif len(negatives) < self.hard_negative_limit:
                        negatives.append(values)
                if family == "hard_constraint":
                    named_pairs = bounded_hard_constraint_preference_pairs(
                        context,
                        named_positives,
                        named_negatives,
                        hard_negative_limit=self.hard_negative_limit,
                        pair_limit=self._pair_budget(family),
                    )
                    query_pairs = [
                        (_hashed(positive, self.dimension), _hashed(negative, self.dimension))
                        for positive, negative in named_pairs
                    ]
                elif family == "entity" and self.constraint_text_evidence:
                    named_pairs = bounded_entity_preference_pairs(
                        context,
                        named_positives,
                        named_negatives,
                        hard_negative_limit=self.hard_negative_limit,
                        pair_limit=self._pair_budget(family),
                    )
                    query_pairs = [
                        (_hashed(positive, self.dimension), _hashed(negative, self.dimension))
                        for positive, negative in named_pairs
                    ]
                else:
                    query_pairs = bounded_preference_pairs(
                        positives,
                        negatives,
                        self._pair_budget(family),
                    )
                if query_pairs:
                    used[family] += 1
                    pairs[family].extend(query_pairs)
        for family in self.feature_families:
            weights = self.weights[family]
            for _epoch in range(self.epochs):
                for positive, negative in pairs[family]:
                    difference = dict(positive)
                    for index, value in negative.items():
                        difference[index] = difference.get(index, 0.0) - value
                    margin = _score(weights, difference)
                    gradient_scale = 1.0 / (
                        1.0 + math.exp(min(60.0, margin))
                    )
                    for index, value in difference.items():
                        weights[index] += self.learning_rate * (
                            gradient_scale * value - self.l2 * weights[index]
                        )
        self.last_fit_query_count = used
        return {family: len(rows) for family, rows in pairs.items()}

    def fit_batches(
        self, batches: Iterable[Sequence[DocumentRankingQuery]]
    ) -> dict[str, int]:
        """Incrementally fit bounded batches and accumulate audit counts."""

        pair_counts = {family: 0 for family in self.feature_families}
        query_counts = {family: 0 for family in self.feature_families}
        seen_query_ids: set[str] = set()
        for batch in batches:
            batch_ids = {
                DocumentRankingQuery.model_validate(query).query_id
                for query in batch
            }
            if len(batch_ids) != len(batch) or seen_query_ids & batch_ids:
                raise ValueError("fusion training query ids must be unique")
            seen_query_ids.update(batch_ids)
            batch_pairs = self.fit(batch)
            for family in self.feature_families:
                pair_counts[family] += batch_pairs[family]
                query_counts[family] += self.last_fit_query_count[family]
        self.last_fit_query_count = query_counts
        return pair_counts

    def rank(
        self,
        query: str,
        candidates: Sequence[DocumentCandidateEvidence],
    ) -> list[DocumentCandidateEvidence]:
        return self.rank_variant(
            query,
            candidates,
            families=self.feature_families,
            gated=self.gated,
        )

    def rank_with_context(
        self,
        query: str,
        candidates: Sequence[DocumentCandidateEvidence],
        *,
        query_spec: QuerySpec,
    ) -> list[DocumentCandidateEvidence]:
        return self.rank_variant(
            query,
            candidates,
            families=self.feature_families,
            gated=self.gated,
            query_spec=query_spec,
        )

    def context_receipt(
        self, query: str, *, query_spec: QuerySpec
    ) -> dict[str, object] | None:
        receipt = getattr(self.context_store, "context_receipt", None)
        if not callable(receipt):
            return None
        output = dict(cast(dict[str, object], receipt(query, query_spec=query_spec)))
        context = self.context_store.for_scoring_query(query, query_spec=query_spec)
        effective = self._effective_scoring_context(context)
        output["runtime_context_scoring"] = self.runtime_context_scoring
        output["effective_activated_families"] = sorted(
            family
            for family in self.feature_families
            if gated_family_eligibility(effective, family, gated=self.gated)
        )
        identity = json.dumps(
            output, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
        output["ranking_context_sha256"] = (
            "sha256:" + hashlib.sha256(identity).hexdigest()
        )
        return output

    def _effective_scoring_context(
        self, context: FusionQueryContext
    ) -> FusionQueryContext:
        if self.runtime_context_scoring:
            return context
        return FusionQueryContext(
            task_label=(
                None if context.task_runtime_inferred else context.task_label
            ),
            constraint_profile=(
                None
                if context.constraint_runtime_inferred
                else context.constraint_profile
            ),
        )

    def rank_variant(
        self,
        query: str,
        candidates: Sequence[DocumentCandidateEvidence],
        *,
        families: Set[str],
        gated: bool,
        query_spec: QuerySpec | None = None,
    ) -> list[DocumentCandidateEvidence]:
        selected = frozenset(families)
        if not selected or not selected.issubset(self.feature_families):
            raise ValueError("fusion rank variant must select trained families")
        rows = self._b0_rows(query, candidates)
        if not rows:
            return rows
        context = self.context_store.for_scoring_query(query, query_spec=query_spec)
        context = self._effective_scoring_context(context)
        if (
            self.constraint_text_evidence
            and "hard_constraint" in selected
            and not _has_negation_candidate_contrast(context, rows, query=query)
        ):
            return rows
        divisor = max(1, len(rows) - 1)
        combined: list[float] = []
        for rank, candidate in enumerate(rows, start=1):
            residual = 0.0
            for family in selected:
                if not gated_family_eligibility(context, family, gated=gated):
                    continue
                if not runtime_candidate_eligible_for_family(candidate, family):
                    continue
                values = _hashed(
                    gated_family_candidate_features(
                        context,
                        candidate,
                        baseline_rank=rank,
                        family=family,
                        constraint_text_evidence=self.constraint_text_evidence,
                        query=query,
                        publication_year_evidence_policy=(
                            self.publication_year_evidence_policy
                        ),
                        method_usage_evidence_schema_version=(
                            self.method_usage_evidence_schema_version
                        ),
                    ),
                    self.dimension,
                )
                residual += self.family_caps[family] * math.tanh(
                    _score(self.weights[family], values)
                )
            residual = max(
                -self.maximum_total_residual,
                min(self.maximum_total_residual, residual),
            )
            combined.append(1.0 - (rank - 1) / divisor + residual)
        return [
            rows[index]
            for index in sorted(
                range(len(rows)), key=lambda index: (-combined[index], index)
            )
        ]

    def manifest_fields(self) -> dict[str, object]:
        return {
            "schema_version": "gated-feature-fusion-research-manifest-v1",
            "model_id": self.model_id,
            "feature_families": sorted(self.feature_families),
            "gated": self.gated,
            "dimension_per_family": self.dimension,
            "family_caps": dict(sorted(self.family_caps.items())),
            "maximum_total_residual": self.maximum_total_residual,
            "task_provenance_allowed_statuses": sorted(
                TASK_PROVENANCE_ALLOWED_STATUSES
            ),
            "title_used_for_scoring": False,
            "abstract_used_for_scoring": False,
            "title_used_for_constraint_evidence": self.constraint_text_evidence,
            "abstract_used_for_constraint_evidence": self.constraint_text_evidence,
            "constraint_text_evidence": self.constraint_text_evidence,
            "method_usage_evidence_schema_version": (
                self.method_usage_evidence_schema_version
            ),
            "negation_evidence_schema_version": NEGATION_EVIDENCE_SCHEMA_VERSION,
            "publication_year_evidence_policy": (
                self.publication_year_evidence_policy
            ),
            "runtime_context_scoring": self.runtime_context_scoring,
            "max_pairs_per_query_family": self.max_pairs_per_query_family,
            "pair_budget_by_family": (
                dict(sorted(self.pair_budget_by_family.items()))
                if self.pair_budget_by_family is not None
                else None
            ),
            "development_labels_used_for_training": False,
            "training_query_count_by_family": dict(
                sorted(self.last_fit_query_count.items())
            ),
        }


def _manifest_int(manifest: Mapping[str, object], name: str) -> int:
    value = manifest.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"F5 manifest {name} must be a positive integer")
    return value


def _manifest_float(manifest: Mapping[str, object], name: str) -> float:
    value = manifest.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"F5 manifest {name} must be numeric")
    output = float(value)
    if not math.isfinite(output):
        raise ValueError(f"F5 manifest {name} must be finite")
    return output


def load_gated_feature_fusion_ranker_bytes(
    manifest_bytes: bytes,
    weights_bytes: bytes,
    *,
    baseline_ranker: BaselineDocumentRanker,
    context_store: FusionContextStore,
) -> GatedFeatureFusionRanker:
    """Load a content-addressed F5 inference artifact without fitting it again."""

    raw_manifest = json.loads(manifest_bytes)
    if not isinstance(raw_manifest, dict):
        raise ValueError("F5 manifest must be an object")
    manifest = cast(dict[str, object], raw_manifest)
    if manifest.get("schema_version") != "gated-feature-fusion-research-manifest-v1":
        raise ValueError("unsupported F5 manifest schema")
    if manifest.get("model_id") != GatedFeatureFusionRanker.model_id:
        raise ValueError("unsupported F5 model id")
    actual_weights_hash = "sha256:" + hashlib.sha256(weights_bytes).hexdigest()
    if manifest.get("weights_sha256") != actual_weights_hash:
        raise ValueError("F5 weights hash mismatch")

    raw_families = manifest.get("feature_families")
    if not isinstance(raw_families, list) or any(
        not isinstance(family, str) for family in raw_families
    ):
        raise ValueError("F5 manifest feature_families must be a string array")
    families = frozenset(raw_families)
    if len(families) != len(raw_families) or not families:
        raise ValueError("F5 manifest feature families must be unique and non-empty")
    unsupported = families - FUSION_FAMILIES
    if unsupported:
        raise ValueError(f"unsupported fusion families: {sorted(unsupported)}")

    raw_caps = manifest.get("family_caps")
    if not isinstance(raw_caps, dict):
        raise ValueError("F5 manifest family_caps must be an object")
    caps = {str(name): float(value) for name, value in raw_caps.items()}
    if set(caps) != set(families):
        raise ValueError("F5 manifest family caps must cover enabled families")
    dimension = _manifest_int(manifest, "dimension_per_family")
    maximum_total_residual = _manifest_float(
        manifest, "maximum_total_residual"
    )
    if manifest.get("gated") is not True:
        raise ValueError("F5 inference artifact must keep gated fusion enabled")
    raw_pair_limit = manifest.get("max_pairs_per_query_family")
    if raw_pair_limit is not None and (
        isinstance(raw_pair_limit, bool)
        or not isinstance(raw_pair_limit, int)
        or raw_pair_limit <= 0
    ):
        raise ValueError("F5 manifest pair limit must be a positive integer")
    raw_pair_budgets = manifest.get("pair_budget_by_family")
    pair_budgets: dict[str, int] | None = None
    if raw_pair_budgets is not None:
        if not isinstance(raw_pair_budgets, dict):
            raise ValueError("F5 manifest family pair budgets must be an object")
        pair_budgets = {}
        for family, value in raw_pair_budgets.items():
            if (
                not isinstance(family, str)
                or isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ValueError("F5 manifest family pair budgets are invalid")
            pair_budgets[family] = value
        if set(pair_budgets) != set(families):
            raise ValueError("F5 manifest family pair budgets must cover enabled families")
    if raw_pair_limit is not None and pair_budgets is not None:
        raise ValueError("F5 manifest mixes uniform and family pair budgets")

    raw_family_hashes = manifest.get("family_weight_sha256")
    if not isinstance(raw_family_hashes, dict) or set(raw_family_hashes) != set(
        families
    ):
        raise ValueError("F5 family weight hashes must cover enabled families")
    expected_family_hashes = {
        str(name): str(value) for name, value in raw_family_hashes.items()
    }

    decoded: dict[str, np.ndarray] = {}
    offset = 0
    vector_size = dimension * 8
    while offset < len(weights_bytes):
        if len(weights_bytes) - offset < 4:
            raise ValueError("truncated F5 family name length")
        name_size = struct.unpack_from("<I", weights_bytes, offset)[0]
        offset += 4
        if name_size <= 0 or len(weights_bytes) - offset < name_size + vector_size:
            raise ValueError("truncated F5 family weight record")
        try:
            family = weights_bytes[offset : offset + name_size].decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("invalid F5 family name encoding") from error
        offset += name_size
        raw_vector = weights_bytes[offset : offset + vector_size]
        offset += vector_size
        if family in decoded:
            raise ValueError("duplicate F5 family weight record")
        if family not in families:
            raise ValueError(f"unexpected F5 family weight record: {family}")
        actual_family_hash = "sha256:" + hashlib.sha256(raw_vector).hexdigest()
        if expected_family_hashes[family] != actual_family_hash:
            raise ValueError(f"F5 family weight hash mismatch: {family}")
        decoded[family] = np.frombuffer(raw_vector, dtype="<f8").copy()
    if set(decoded) != set(families):
        raise ValueError("F5 weights are missing an enabled family")

    ranker = GatedFeatureFusionRanker(
        baseline_ranker=baseline_ranker,
        context_store=context_store,
        feature_families=families,
        gated=True,
        dimension=dimension,
        family_caps=caps,
        maximum_total_residual=maximum_total_residual,
        constraint_text_evidence=bool(
            manifest.get("constraint_text_evidence", False)
        ),
        runtime_context_scoring=bool(manifest.get("runtime_context_scoring", False)),
        max_pairs_per_query_family=raw_pair_limit,
        pair_budget_by_family=pair_budgets,
        publication_year_evidence_policy=str(
            manifest.get(
                "publication_year_evidence_policy",
                DECLARED_YEAR_EVIDENCE_POLICY,
            )
        ),
        method_usage_evidence_schema_version=str(
            manifest.get(
                "method_usage_evidence_schema_version",
                LEGACY_METHOD_TEXT_MATCH_SCHEMA_VERSION,
            )
        ),
    )
    ranker.weights = decoded
    return ranker


__all__ = [
    "FUSION_FAMILIES",
    "DECLARED_YEAR_EVIDENCE_POLICY",
    "TASK_PROVENANCE_ALLOWED_STATUSES",
    "FrozenFusionContextStore",
    "FusionQueryContext",
    "GatedFeatureFusionRanker",
    "anchored_family_weights",
    "gated_family_candidate_features",
    "gated_family_eligibility",
    "bounded_preference_pairs",
    "runtime_candidate_eligible_for_family",
    "training_candidate_eligible_for_family",
    "load_gated_feature_fusion_ranker_bytes",
]
