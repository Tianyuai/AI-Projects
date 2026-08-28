"""Frozen sampling and full-pool generation for candidate-ceiling diagnostics."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

from paper_search.domain.models import QuerySpec
from paper_search.evaluation.dataset import write_frozen_bytes
from paper_search.learning.candidates import (
    DeterministicActionCandidateGenerator,
    query_content_terms,
)
from paper_search.learning.contracts import PolicyActionCandidate
from paper_search.learning.routing import RoutedQuery, RuleQueryRouter
from paper_search.recall_experiments.contracts import (
    RecallActionBatch,
    RecallGenerationContext,
    RetrievalActionResult,
    TextSearchAction,
    TextSearchPayload,
    TitleSearchAction,
    TitleSearchPayload,
)
from paper_search.recall_experiments.generation.base import GenerationResult


_POLICY_VERSION = "full-controlled-candidate-pool-v2"
_POLICY_MATERIAL = json.dumps(
    {
        "candidate_generator": "deterministic-action-candidates-v1-plus-ceiling-v2",
        "deduplication": "canonical-search-text-and-mode-v2",
        "selection": "all_candidates",
        "version": _POLICY_VERSION,
    },
    sort_keys=True,
    separators=(",", ":"),
).encode("utf-8")


def select_ceiling_batch(
    rows: list[dict[str, object]],
    *,
    batch_size: int,
    batch_index: int,
    batch_count: int,
    excluded_query_ids: set[str] | frozenset[str] = frozenset(),
) -> list[dict[str, object]]:
    if batch_size <= 0 or batch_count <= 0:
        raise ValueError("batch size and count must be positive")
    if not 0 <= batch_index < batch_count:
        raise ValueError("batch index must be within batch count")
    eligible = [
        row for row in rows if str(row.get("query_id")) not in excluded_query_ids
    ]
    requested = batch_size * batch_count
    if requested > len(eligible):
        raise ValueError("ceiling batch request exceeds available rows")
    return [
        eligible[
            (batch_index + offset * batch_count) * len(eligible) // requested
        ]
        for offset in range(batch_size)
    ]


def freeze_ceiling_batch(
    rows: list[dict[str, object]],
    path: Path,
) -> str:
    if not rows:
        raise ValueError("ceiling batch is empty")
    content = "".join(
        json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
        for row in rows
    ).encode("utf-8")
    write_frozen_bytes(path, content)
    return "sha256:" + hashlib.sha256(content).hexdigest()


def _to_recall_action(
    candidate: PolicyActionCandidate,
) -> TextSearchAction | TitleSearchAction:
    common = {
        "action_id": f"ceiling-{candidate.action_id}",
        "strategy": (
            "candidate-family:semantic"
            if candidate.search_mode == "semantic"
            else "candidate-family:boolean-phrase"
            if candidate.action_id
            in {"candidate-boolean-relaxed", "candidate-phrase-proximity"}
            else "candidate-family:baseline"
        ),
    }
    if candidate.action_type == "title_search":
        return TitleSearchAction(
            **common,
            action_type="title_search",
            payload=TitleSearchPayload(title_text=candidate.text),
        )
    return TextSearchAction(
        **common,
        action_type="text_search",
        payload=TextSearchPayload(
            query_text=candidate.text,
            search_mode=candidate.search_mode,
        ),
    )


def _deduplicate_candidates(
    candidates: list[PolicyActionCandidate],
) -> list[PolicyActionCandidate]:
    selected: list[PolicyActionCandidate] = []
    seen: set[tuple[str, str]] = set()
    for candidate in candidates:
        text = " ".join(unicodedata.normalize("NFKC", candidate.text).split())
        key = (candidate.search_mode, text.casefold())
        if key in seen:
            continue
        seen.add(key)
        selected.append(candidate.model_copy(update={"text": text}))
    return selected


class FullCandidatePoolQueryGenerator:
    """Execute the complete bounded candidate pool for ceiling measurement."""

    generator_type = "fixed_actions"
    source_sha256 = "sha256:" + hashlib.sha256(_POLICY_MATERIAL).hexdigest()
    candidate_policy = _POLICY_VERSION
    context_source = "rules_fallback"

    def __init__(self, *, max_candidates: int = 12) -> None:
        self.max_candidates = max_candidates
        self._router = RuleQueryRouter()
        self._candidate_generator = DeterministicActionCandidateGenerator(
            max_candidates=max(2, max_candidates - 1)
        )

    def _route(self, context: RecallGenerationContext) -> RoutedQuery:
        return self._router.route(context.original_query)

    async def generate(self, context: RecallGenerationContext) -> GenerationResult:
        routed = self._route(context)
        baseline = self._candidate_generator.generate(
            routed.query_spec,
            query_kind=routed.query_kind,
        )
        content_terms = query_content_terms(context.original_query)
        v2_candidates = [
            PolicyActionCandidate(
                action_id="candidate-semantic-original",
                action_type="text_search",
                text=context.original_query,
                origin="deterministic_rule",
                provider_hint="openalex",
                search_mode="semantic",
            )
        ]
        if len(content_terms) >= 4:
            optional = " OR ".join(content_terms[2:5])
            v2_candidates.extend(
                [
                    PolicyActionCandidate(
                        action_id="candidate-boolean-relaxed",
                        action_type="text_search",
                        text=(
                            f"{content_terms[0]} AND {content_terms[1]} "
                            f"AND ({optional})"
                        ),
                        origin="deterministic_rule",
                        provider_hint="openalex",
                    ),
                    PolicyActionCandidate(
                        action_id="candidate-phrase-proximity",
                        action_type="text_search",
                        text=f'"{" ".join(content_terms[:4])}"~8',
                        origin="deterministic_rule",
                        provider_hint="openalex",
                    ),
                ]
            )
        candidates = _deduplicate_candidates(
            [baseline[0], *v2_candidates, *baseline[1:]]
        )[: max(1, self.max_candidates - 1)]
        actions = [
            _to_recall_action(candidate) for candidate in candidates
        ]
        batch = RecallActionBatch(actions=actions)
        artifact_bytes = json.dumps(
            batch.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return GenerationResult(
            query_id=context.query_id,
            action_batch=batch,
            artifact_bytes=artifact_bytes,
            provenance={
                "candidate_policy": self.candidate_policy,
                "candidate_pool_size": str(len(candidates)),
                "collection_mode": "candidate_ceiling",
                "gold_visibility": "blind",
                "query_kind": routed.query_kind,
                "context_source": self.context_source,
                "selected_candidate_ids": ",".join(
                    candidate.action_id for candidate in candidates
                ),
            },
        )


    async def refine(
        self,
        context: RecallGenerationContext,
        anchor_generation: GenerationResult,
        first_round_results: Sequence[RetrievalActionResult],
    ) -> GenerationResult:
        query_terms = query_content_terms(context.original_query)
        query_set = set(query_terms)
        supported_by_paper: dict[str, set[str]] = defaultdict(set)
        paper_index = 0
        for result in first_round_results:
            for paper in result.hits:
                if paper_index >= 10:
                    break
                paper_index += 1
                title_terms = set(query_content_terms(paper.title))
                if len(title_terms.intersection(query_set)) < 2:
                    continue
                for term in title_terms.difference(query_set):
                    supported_by_paper[term].add(str(paper_index))
            if paper_index >= 10:
                break
        expansion_terms = [
            term
            for term, supports in sorted(
                supported_by_paper.items(),
                key=lambda item: (-len(item[1]), item[0]),
            )
            if len(supports) >= 2
        ][:2]
        if not expansion_terms or len(anchor_generation.action_batch.actions) >= self.max_candidates:
            return anchor_generation.model_copy(
                update={
                    "provenance": {
                        **anchor_generation.provenance,
                        "prf_status": "insufficient_cross_paper_support",
                    }
                }
            )
        query_text = " ".join([*query_terms[:5], *expansion_terms])
        action = TextSearchAction(
            action_id="ceiling-candidate-prf-1",
            strategy="candidate-family:prf",
            action_type="text_search",
            payload=TextSearchPayload(query_text=query_text),
        )
        batch = RecallActionBatch(
            actions=[*anchor_generation.action_batch.actions, action]
        )
        artifact_bytes = json.dumps(
            batch.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return anchor_generation.model_copy(
            update={
                "action_batch": batch,
                "artifact_bytes": artifact_bytes,
                "provenance": {
                    **anchor_generation.provenance,
                    "candidate_pool_size": str(len(batch.actions)),
                    "prf_status": "cross_paper_supported",
                    "prf_terms": ",".join(expansion_terms),
                },
            }
        )


class ContextEnhancedCandidatePoolQueryGenerator(FullCandidatePoolQueryGenerator):
    """Use an already frozen local QuerySpec to build blind retrieval expressions."""

    candidate_policy = "context-enhanced-candidate-pool-v1"
    context_source = "frozen_query_spec"
    source_sha256 = "sha256:" + hashlib.sha256(
        b"context-enhanced-candidate-pool-v1:full-controlled-candidate-pool-v2"
    ).hexdigest()

    def _route(self, context: RecallGenerationContext) -> RoutedQuery:
        spec = QuerySpec.model_validate(context.query_spec)
        if " ".join(spec.original_query.split()) != " ".join(
            context.original_query.split()
        ):
            raise ValueError("frozen query context does not match original query")
        return RoutedQuery(
            query_kind=self._router.route(context.original_query).query_kind,
            query_spec=spec,
        )


class Core4SemanticBooleanQueryGenerator:
    """Frozen A-prime policy: core4 lexical + original semantic + boolean."""

    generator_type = "fixed_actions"
    candidate_policy = "core4-semantic-boolean-v1"
    source_sha256 = "sha256:" + hashlib.sha256(
        b"core4-semantic-boolean-v1:full-controlled-candidate-pool-v2"
    ).hexdigest()

    def __init__(self) -> None:
        self.max_openalex_actions = 6
        self._source = FullCandidatePoolQueryGenerator(max_candidates=12)

    async def generate(self, context: RecallGenerationContext) -> GenerationResult:
        source = await self._source.generate(context)
        core = [
            action
            for action in source.action_batch.actions
            if action.strategy == "candidate-family:baseline"
        ][:4]
        semantic = next(
            action
            for action in source.action_batch.actions
            if action.action_id == "ceiling-candidate-semantic-original"
        )
        boolean = next(
            (
                action
                for action in source.action_batch.actions
                if action.action_id == "ceiling-candidate-boolean-relaxed"
            ),
            None,
        )
        selected = [*core, semantic]
        if boolean is not None:
            selected.append(boolean)
        identities: set[tuple[str, str, str]] = set()
        actions: list[TextSearchAction | TitleSearchAction] = []
        for action in selected:
            if isinstance(action, TextSearchAction):
                text = action.payload.query_text
                mode = action.payload.search_mode
            elif isinstance(action, TitleSearchAction):
                text = action.payload.title_text
                mode = "lexical"
            else:
                raise ValueError("A-prime source emitted a non-search action")
            identity = (
                action.action_type,
                mode,
                " ".join(unicodedata.normalize("NFKC", text).split()).casefold(),
            )
            if identity in identities:
                continue
            identities.add(identity)
            actions.append(action)
        if len(actions) > self.max_openalex_actions:
            raise ValueError("A-prime action composition exceeds six actions")
        batch = RecallActionBatch(actions=actions)
        artifact_bytes = json.dumps(
            batch.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return GenerationResult(
            query_id=context.query_id,
            action_batch=batch,
            artifact_bytes=artifact_bytes,
            provenance={
                "candidate_policy": self.candidate_policy,
                "candidate_pool_size": str(len(actions)),
                "collection_mode": "core4_semantic_boolean",
                "gold_visibility": "blind",
                "graph_status": "disabled",
                "prf_status": "disabled",
                "title_target_status": "disabled",
                "semantic_original_count": "1",
                "max_openalex_actions": "6",
            },
        )


_QUOTED_PHRASE = re.compile(r'["“”]([^"“”]{2,80})["“”]')
_PARENTHETICAL_ALIAS = re.compile(
    r"([A-Za-z][A-Za-z0-9+_. -]{2,60}?)\s*\(([A-Z][A-Za-z0-9+_.-]{1,15})\)"
)
_CAPITALIZED_ENTITY = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9+_.-]{1,}(?:\s+|$)){1,4}"
)


def _normalized_query(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split())


def _query_entities(value: str) -> list[str]:
    selected: list[str] = []
    for match in _PARENTHETICAL_ALIAS.finditer(value):
        selected.extend((match.group(1), match.group(2)))
    selected.extend(match.group(1) for match in _QUOTED_PHRASE.finditer(value))
    selected.extend(match.group(0) for match in _CAPITALIZED_ENTITY.finditer(value))
    result: list[str] = []
    seen: set[str] = set()
    for raw in selected:
        normalized = _normalized_query(raw).strip(" ,.;:?!")
        folded = normalized.casefold()
        if (
            not normalized
            or folded in {"find", "which", "what", "who", "show", "give"}
            or folded in seen
        ):
            continue
        seen.add(folded)
        result.append(normalized)
    return result


def _join_distinct(values: Sequence[str], *, limit: int = 5) -> str:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = _normalized_query(value)
        folded = normalized.casefold()
        if not normalized or folded in seen:
            continue
        seen.add(folded)
        result.append(normalized)
        if len(result) == limit:
            break
    return " ".join(result)


class QueryAdaptiveHighRecallGenerator:
    """Gold-blind OpenAlex actions derived from one frozen local query context."""

    generator_type = "fixed_actions"
    candidate_policy = "query-adaptive-high-recall-v2"
    source_sha256 = "sha256:" + hashlib.sha256(
        b"query-adaptive-high-recall-v2:frozen-context:diversified-actions-v1"
    ).hexdigest()

    def __init__(
        self,
        *,
        frozen_query_specs: Mapping[str, QuerySpec],
        max_openalex_actions: int = 6,
    ) -> None:
        if not 2 <= max_openalex_actions <= 6:
            raise ValueError("max_openalex_actions must be between two and six")
        if not frozen_query_specs:
            raise ValueError("frozen query context must not be empty")
        self.max_openalex_actions = max_openalex_actions
        self._specs = {
            query_id: QuerySpec.model_validate(spec)
            for query_id, spec in frozen_query_specs.items()
        }

    async def generate(self, context: RecallGenerationContext) -> GenerationResult:
        spec = self._specs.get(context.query_id)
        if spec is None:
            raise ValueError(f"frozen query context is missing: {context.query_id}")
        original = _normalized_query(context.original_query)
        if _normalized_query(spec.original_query) != original:
            raise ValueError("frozen query context does not match original query")

        exclusion_terms = {
            term
            for exclusion in spec.exclusions
            for term in query_content_terms(exclusion)
        }
        topic_terms = [
            term
            for term in query_content_terms(original)
            if term not in exclusion_terms
        ]
        topic_query = " ".join(topic_terms)
        context_query = _join_distinct(
            [
                *spec.tasks[:2],
                *spec.methods[:2],
                *spec.datasets[:2],
                *spec.topics[:1],
                *spec.venues[:1],
            ],
            limit=5,
        )
        entity_query = _join_distinct(
            [
                *_query_entities(original)[:3],
                *spec.datasets[:1],
                *spec.methods[:1],
                *spec.tasks[:1],
            ],
            limit=5,
        )
        clean_constraint_query = _join_distinct(
            [
                *spec.must_have[:2],
                *spec.tasks[:1],
                *spec.methods[:1],
                *spec.datasets[:1],
            ],
            limit=5,
        )
        proposals = [
            (
                "high-recall-original-lexical",
                "high-recall:original",
                original,
                "lexical",
            ),
            (
                "high-recall-original-semantic",
                "high-recall:original",
                original,
                "semantic",
            ),
            (
                "high-recall-topic-lexical",
                "high-recall:topic-compression",
                topic_query,
                "lexical",
            ),
            (
                "high-recall-entity-lexical",
                "high-recall:entity-alias",
                entity_query,
                "lexical",
            ),
            (
                "high-recall-context-semantic",
                "high-recall:frozen-context",
                context_query,
                "semantic",
            ),
            (
                "high-recall-topic-semantic",
                "high-recall:topic-compression",
                topic_query,
                "semantic",
            ),
            (
                "high-recall-constraint-clean",
                "high-recall:constraint-clean",
                clean_constraint_query,
                "lexical",
            ),
            (
                "high-recall-context-lexical",
                "high-recall:frozen-context",
                context_query,
                "lexical",
            ),
            (
                "high-recall-topic-head",
                "high-recall:concept-rotation",
                " ".join(topic_terms[:5]),
                "lexical",
            ),
            (
                "high-recall-topic-tail",
                "high-recall:concept-rotation",
                " ".join(topic_terms[-5:]),
                "lexical",
            ),
        ]
        actions: list[TextSearchAction] = []
        identities: set[tuple[str, str]] = set()
        for action_id, strategy, raw_text, mode in proposals:
            text = _normalized_query(raw_text)
            if not text:
                continue
            identity = (mode, text.casefold())
            if identity in identities:
                continue
            identities.add(identity)
            actions.append(
                TextSearchAction(
                    action_id=action_id,
                    strategy=strategy,
                    action_type="text_search",
                    payload=TextSearchPayload(
                        query_text=text,
                        search_mode=mode,
                    ),
                )
            )
            if len(actions) == self.max_openalex_actions:
                break
        batch = RecallActionBatch(actions=actions)
        artifact_bytes = json.dumps(
            batch.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return GenerationResult(
            query_id=context.query_id,
            action_batch=batch,
            artifact_bytes=artifact_bytes,
            provenance={
                "candidate_policy": self.candidate_policy,
                "candidate_pool_size": str(len(actions)),
                "collection_mode": "query_adaptive_high_recall",
                "context_source": "frozen_unified_context",
                "gold_visibility": "blind",
                "max_openalex_actions": str(self.max_openalex_actions),
                "boolean_status": "disabled_low_marginal_yield",
                "llm_request_count": "0",
            },
        )


__all__ = [
    "ContextEnhancedCandidatePoolQueryGenerator",
    "Core4SemanticBooleanQueryGenerator",
    "FullCandidatePoolQueryGenerator",
    "QueryAdaptiveHighRecallGenerator",
    "freeze_ceiling_batch",
    "select_ceiling_batch",
]
