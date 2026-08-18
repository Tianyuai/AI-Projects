"""Evidence-steered generation with deterministic provenance validation."""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Collection, Mapping, Sequence
from typing import Annotated, Literal, cast

from pydantic import Field, ValidationError, field_validator

from paper_search.domain.models import DomainModel, NonEmptyStr
from paper_search.recall_experiments.contracts import (
    GoldVisibility,
    RecallActionBatch,
    RecallGenerationContext,
    RetrievalActionResult,
    assert_no_forbidden_identifier_keys_or_patterns,
)
from paper_search.recall_experiments.generation.backends import (
    LLMBackend,
    LLMBackendResult,
    LLMGenerationRequest,
)
from paper_search.recall_experiments.generation.base import (
    GenerationResult,
    LLMCallReceipt,
)
from paper_search.recall_experiments.generation.deepseek import (
    DeepSeekPromptGenerator,
    RecallGenerationFailure,
    RecallPromptArtifact,
    build_generation_payload,
)
from paper_search.recall_experiments.validation import (
    ActionValidationFailure,
    validate_action_batch,
)


_MAX_CANDIDATE_PAPERS = 50
_MAX_EVIDENCE_PAPERS_V2 = 5
_MAX_EVIDENCE_PAPERS_V3 = 3
_MAX_SNIPPETS_PER_PAPER = 2
_MAX_SNIPPET_CHARS = 400
_MAX_ABSTRACT_SCAN_CHARS = 12_000
_MAX_TITLE_CHARS = 300
_MAX_AUTHOR_CHARS = 120
_MAX_VENUE_CHARS = 200
_MAX_AUTHORS = 6
_EVIDENCE_IDENTIFIER = re.compile(
    r"(?:\b10\.\d{4,9}/\S+|https?://\S+|\bW\d{6,}\b|\bS2:\S+|"
    r"\barxiv\s*:?\s*\d{4}\.\d{4,5}(?:v\d+)?\b|"
    r"\barxiv\s*:?\s*[a-z-]+/\d{7}(?:v\d+)?\b|"
    r"\b(?:PMID|PubMed)\s*:?\s*\d+\b|\b[0-9a-f]{40}\b)",
    flags=re.IGNORECASE,
)
_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9]*|[\u4e00-\u9fff]+")
_ACRONYM = re.compile(r"\b[A-Z][A-Z0-9-]{1,}\b")
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "for",
        "from",
        "how",
        "in",
        "into",
        "is",
        "it",
        "of",
        "on",
        "or",
        "paper",
        "papers",
        "study",
        "that",
        "the",
        "their",
        "this",
        "to",
        "using",
        "what",
        "which",
        "with",
    }
)
_COMPRESSION_STOPWORDS = _STOPWORDS.union(
    {
        "approach",
        "approaches",
        "classic",
        "given",
        "problem",
        "perspective",
        "utilize",
        "utilizes",
    }
)
_NARRATIVE_SHELL = frozenset(
    {
        "contains",
        "covering",
        "create",
        "different",
        "have",
        "numerous",
        "presented",
        "provide",
        "research",
        "studies",
        "used",
        "works",
    }
)
_PARALLEL_METHODS = re.compile(
    r"\b(?:utilize|utilizes|use|uses|using|employ|employs|apply|applies)\s+"
    r"(?P<methods>.+?)\s+for\s+(?:the\s+)?task\s+of\s+(?P<task>.+?)[?.!]*$",
    flags=re.IGNORECASE,
)
_PARENTHETICAL_ACRONYM = re.compile(r"\(([A-Z][A-Z0-9-]{1,})\)")


class EvidenceSupport(DomainModel):
    rank: Annotated[int, Field(strict=True, gt=0)]
    exact_phrase: NonEmptyStr


class SearchProposal(DomainModel):
    action_id: NonEmptyStr
    query_text: NonEmptyStr
    expansion_kind: Literal["paper_expression", "query_complement"]
    query_support: list[NonEmptyStr] = Field(default_factory=list)
    evidence_support: list[EvidenceSupport] = Field(default_factory=list)

    @field_validator("query_support", mode="before")
    @classmethod
    def normalize_single_query_support(cls, value: object) -> object:
        return [value] if isinstance(value, str) else value


class EvidenceSteeredDeepSeekGenerator:
    """Generate one anchor and accept only source-verifiable refinements."""

    def __init__(
        self,
        *,
        backend: LLMBackend,
        prompt: RecallPromptArtifact,
        visibility: GoldVisibility,
        allowed_actions: Collection[str],
        max_actions: int,
    ) -> None:
        if set(allowed_actions) != {"text_search"}:
            raise ValueError("evidence-steered generation supports text_search only")
        if max_actions < 1:
            raise ValueError("max_actions must be positive")
        self._backend = backend
        self._prompt = prompt
        self._visibility = visibility
        self._allowed_actions = frozenset(allowed_actions)
        self._max_actions = max_actions
        folded_version = prompt.version.casefold()
        self._generation_version: Literal["v2", "v3", "v4"] = (
            "v4"
            if folded_version.endswith("-v4")
            else "v3"
            if folded_version.endswith("-v3")
            else "v2"
        )
        self._anchor = DeepSeekPromptGenerator(
            backend=backend,
            prompt=prompt,
            visibility=visibility,
            allowed_actions=allowed_actions,
            max_actions=1,
        )

    @property
    def prompt_sha256(self) -> str:
        return self._anchor.prompt_sha256

    async def generate(self, context: RecallGenerationContext) -> GenerationResult:
        payload = build_generation_payload(
            context,
            self._visibility,
            allowed_actions=self._allowed_actions,
            max_actions=1,
        )
        payload.pop("seed_candidates", None)
        payload["generation_phase"] = "anchor"
        generation = await self._anchor.generate_with_payload(context, payload)
        if (
            len(generation.action_batch.actions) != 1
            or generation.action_batch.actions[0].action_id != "anchor"
            or generation.action_batch.actions[0].strategy != "evidence-steered:anchor"
        ):
            raise RecallGenerationFailure(
                "generation_failure", [], generation.call_receipts
            )
        return generation

    async def refine(
        self,
        context: RecallGenerationContext,
        anchor_generation: GenerationResult,
        first_round_results: Sequence[RetrievalActionResult],
    ) -> GenerationResult:
        base_batch = anchor_generation.action_batch
        if self._generation_version in {"v3", "v4"}:
            base_batch = _append_safe_query_complement(
                context,
                anchor_generation.action_batch,
                max_actions=self._max_actions,
                remove_narrative_shell=self._generation_version == "v4",
            )
        payload = build_refinement_payload(
            context,
            anchor_generation,
            first_round_results,
            allowed_actions=self._allowed_actions,
            max_actions=self._max_actions,
            generation_version=self._generation_version,
        )
        mode = cast(str, payload["refinement_mode"])
        evidence = cast(list[dict[str, object]], payload["first_round_evidence"])
        if (
            mode == "anchor_only"
            or (
                self._generation_version in {"v3", "v4"}
                and mode != "evidence_grounded"
            )
            or len(base_batch.actions) >= self._max_actions
        ):
            return _final_result(
                anchor_generation,
                base_batch,
                mode=mode,
                evidence_count=len(evidence),
                audit=[],
                generation_version=self._generation_version,
            )

        assert_no_forbidden_identifier_keys_or_patterns(payload)
        backend_result = await self._backend.generate(
            LLMGenerationRequest(
                prompt_name=self._prompt.name,
                payload=payload,
                prompt_instructions=_render_refinement_prompt(self._prompt),
                prompt_bytes=self._prompt.source_bytes,
                prompt_artifact_sha256=self._prompt.sha256,
            ),
            "initial",
        )
        receipt = _call_receipt(backend_result)
        if backend_result.errors or backend_result.infrastructure_failure:
            reason = (
                backend_result.errors[0].code
                if backend_result.errors
                else "refinement_call_failure"
            )
            return _final_result(
                anchor_generation,
                base_batch,
                mode=mode,
                evidence_count=len(evidence),
                audit=[{"decision": "rejected", "reason": reason}],
                extra_receipt=receipt,
                backend_provenance=backend_result.provenance,
                generation_version=self._generation_version,
            )

        final_batch, audit = validate_refinement_proposals(
            context,
            base_batch,
            backend_result.data,
            evidence,
            max_actions=self._max_actions,
            max_proposals=1 if self._generation_version in {"v3", "v4"} else 2,
            require_cross_paper_support=self._generation_version in {"v3", "v4"},
            require_cross_title_support=self._generation_version == "v4",
        )
        return _final_result(
            anchor_generation,
            final_batch,
            mode=mode,
            evidence_count=len(evidence),
            audit=audit,
            extra_receipt=receipt,
            backend_provenance=backend_result.provenance,
            generation_version=self._generation_version,
        )


def build_refinement_payload(
    context: RecallGenerationContext,
    anchor_generation: GenerationResult,
    first_round_results: Sequence[RetrievalActionResult],
    *,
    allowed_actions: Collection[str],
    max_actions: int,
    generation_version: Literal["v2", "v3", "v4"] = "v2",
) -> dict[str, object]:
    """Build a Gold-blind, bounded evidence payload for one optional refinement call."""
    base = build_generation_payload(
        context,
        "blind",
        allowed_actions=allowed_actions,
        max_actions=max_actions,
    )
    base.pop("seed_candidates", None)
    anchor_queries = [
        action.payload.query_text
        for action in anchor_generation.action_batch.actions
        if action.action_type == "text_search"
    ]
    evidence = _paper_evidence(
        context.original_query,
        anchor_queries,
        first_round_results,
        max_papers=(
            _MAX_EVIDENCE_PAPERS_V3
            if generation_version in {"v3", "v4"}
            else _MAX_EVIDENCE_PAPERS_V2
        ),
    )
    had_hits = any(result.hits for result in first_round_results)
    concepts = _content_tokens(" ".join([context.original_query, *anchor_queries]))
    safe_complement = build_safe_query_complement(
        context.original_query,
        " ".join(anchor_queries),
        remove_narrative_shell=generation_version == "v4",
    )
    evidence_passes_gate = (
        bool(evidence)
        if generation_version == "v2"
        else _evidence_quality_gate_v4(context.original_query, evidence)
        if generation_version == "v4"
        else _evidence_quality_gate(context.original_query, evidence)
    )
    if evidence_passes_gate:
        mode = "evidence_grounded"
    elif (
        safe_complement is not None
        if generation_version in {"v3", "v4"}
        else len(concepts) >= 3
    ):
        mode = "query_grounded_only"
    else:
        mode = "anchor_only"
    base.pop("allowed_action_schema", None)
    base.update(
        {
            "generation_phase": "refinement",
            "refinement_mode": mode,
            "anchor_actions": anchor_generation.action_batch.model_dump(mode="json")[
                "actions"
            ],
            "first_round_state": (
                "papers_found"
                if evidence
                else "papers_found_no_usable_evidence"
                if had_hits
                else "zero_results"
            ),
            "first_round_evidence": evidence,
            "proposal_schema": {
                "top_level_keys": ["proposals"],
                "max_proposals": (
                    min(1, max(0, max_actions - len(anchor_queries) - int(safe_complement is not None)))
                    if generation_version in {"v3", "v4"}
                    else max(0, max_actions - len(anchor_queries))
                ),
                "expansion_kinds": (
                    ["paper_expression"]
                    if generation_version in {"v3", "v4"}
                    else ["paper_expression", "query_complement"]
                ),
                "required_proposal_keys": [
                    "action_id",
                    "query_text",
                    "expansion_kind",
                    "query_support",
                    "evidence_support",
                ],
                "evidence_support_keys": ["rank", "exact_phrase"],
            },
        }
    )
    assert_no_forbidden_identifier_keys_or_patterns(base)
    return base


def validate_refinement_proposals(
    context: RecallGenerationContext,
    anchor_batch: RecallActionBatch,
    raw: object,
    evidence: Sequence[Mapping[str, object]],
    *,
    max_actions: int,
    max_proposals: int = 2,
    require_cross_paper_support: bool = False,
    require_cross_title_support: bool = False,
) -> tuple[RecallActionBatch, list[dict[str, object]]]:
    """Validate each proposal independently and retain the immutable anchor prefix."""
    if not isinstance(raw, Mapping) or set(raw) != {"proposals"}:
        return anchor_batch, [{"decision": "rejected", "reason": "invalid_proposal_batch"}]
    raw_proposals = raw.get("proposals")
    if not isinstance(raw_proposals, list):
        return anchor_batch, [{"decision": "rejected", "reason": "invalid_proposal_batch"}]

    proposals: list[SearchProposal] = []
    audit: list[dict[str, object]] = []
    for raw_proposal in raw_proposals[:max_proposals]:
        try:
            proposals.append(SearchProposal.model_validate(raw_proposal))
        except ValidationError:
            audit.append(
                {
                    "action_id": (
                        str(raw_proposal.get("action_id", "unknown"))
                        if isinstance(raw_proposal, Mapping)
                        else "unknown"
                    ),
                    "decision": "rejected",
                    "reason": "invalid_proposal",
                }
            )
    if len(raw_proposals) > max_proposals:
        audit.append(
            {
                "decision": "rejected",
                "reason": "proposal_limit_exceeded",
                "rejected_count": len(raw_proposals) - max_proposals,
            }
        )

    accepted = list(anchor_batch.actions)
    original = _normalized(context.original_query)
    original_tokens = _content_tokens(context.original_query)
    original_acronyms = set(_ACRONYM.findall(context.original_query))
    evidence_by_rank = {
        cast(int, item["rank"]): item
        for item in evidence
        if isinstance(item.get("rank"), int)
    }

    for proposal in proposals:
        entry: dict[str, object] = {
            "action_id": proposal.action_id,
            "query_text": proposal.query_text,
            "expansion_kind": proposal.expansion_kind,
            "query_support": list(proposal.query_support),
            "evidence_support": [
                support.model_dump(mode="json") for support in proposal.evidence_support
            ],
        }
        reason = _proposal_rejection_reason(
            proposal,
            original=original,
            original_tokens=original_tokens,
            original_acronyms=original_acronyms,
            evidence_by_rank=evidence_by_rank,
            all_evidence=evidence,
            require_cross_paper_support=require_cross_paper_support,
            require_cross_title_support=require_cross_title_support,
        )
        if reason is not None:
            audit.append({**entry, "decision": "rejected", "reason": reason})
            continue
        if len(accepted) >= max_actions:
            audit.append(
                {**entry, "decision": "rejected", "reason": "action_limit_exceeded"}
            )
            continue
        raw_action = {
            "action_id": proposal.action_id,
            "action_type": "text_search",
            "strategy": (
                "evidence-steered:paper-expression"
                if proposal.expansion_kind == "paper_expression"
                else "evidence-steered:query-complement"
            ),
            "payload": {"query_text": proposal.query_text},
        }
        try:
            candidate = validate_action_batch(
                {"actions": [
                    *(action.model_dump(mode="json") for action in accepted),
                    raw_action,
                ]},
                context,
                allowed_actions={"text_search"},
                max_actions=max_actions,
            )
        except ActionValidationFailure:
            audit.append(
                {
                    **entry,
                    "decision": "rejected",
                    "reason": "duplicate_or_low_marginal_value",
                }
            )
            continue
        accepted = list(candidate.actions)
        audit.append({**entry, "decision": "accepted", "reason": "supported"})

    return RecallActionBatch(actions=accepted), audit


def _proposal_rejection_reason(
    proposal: SearchProposal,
    *,
    original: str,
    original_tokens: set[str],
    original_acronyms: set[str],
    evidence_by_rank: Mapping[int, Mapping[str, object]],
    all_evidence: Sequence[Mapping[str, object]],
    require_cross_paper_support: bool,
    require_cross_title_support: bool,
) -> str | None:
    if _EVIDENCE_IDENTIFIER.search(proposal.query_text):
        return "identifier_or_url"
    if not proposal.query_support:
        return "missing_query_support"
    if any(_normalized(phrase) not in original for phrase in proposal.query_support):
        return "missing_query_support"

    valid_evidence: list[tuple[EvidenceSupport, Mapping[str, object]]] = []
    for support in proposal.evidence_support:
        item = evidence_by_rank.get(support.rank)
        if item is None or not _contains_exact_span(
            _evidence_fields(item), support.exact_phrase
        ):
            return "missing_evidence_support"
        valid_evidence.append((support, item))
    if proposal.expansion_kind == "paper_expression" and not valid_evidence:
        introduced = set(_ACRONYM.findall(proposal.query_text)).difference(original_acronyms)
        if any(_edit_distance_at_most_one(item, anchor) for item in introduced for anchor in original_acronyms):
            return "near_anchor_mutation"
        return "missing_evidence_support"

    proposal_tokens = _content_tokens(proposal.query_text)
    query_support_tokens = _content_tokens(" ".join(proposal.query_support))
    reused_query_tokens = proposal_tokens.intersection(original_tokens)
    if not reused_query_tokens or not reused_query_tokens.issubset(query_support_tokens):
        return "missing_query_support"

    introduced_acronyms = set(_ACRONYM.findall(proposal.query_text)).difference(
        original_acronyms
    )
    for introduced in introduced_acronyms:
        support_text = " ".join(
            support.exact_phrase for support, _ in valid_evidence
        )
        if not _contains_exact_span([support_text], introduced):
            if any(
                _edit_distance_at_most_one(introduced, anchor)
                for anchor in original_acronyms
            ):
                return "near_anchor_mutation"
            return "unsupported_acronym"
        linked = any(
            _contains_exact_span(_evidence_fields(item), introduced)
            and len(
                _content_tokens(_evidence_text(item)).intersection(original_tokens)
            )
            >= 2
            for _, item in valid_evidence
        )
        if not linked:
            return "near_anchor_mutation"

    introduced_tokens = proposal_tokens.difference(original_tokens)
    if proposal.expansion_kind == "query_complement":
        if introduced_tokens:
            return "unsupported_query_expansion"
    else:
        evidence_tokens = _content_tokens(
            " ".join(support.exact_phrase for support, _ in valid_evidence)
        )
        if not introduced_tokens.issubset(evidence_tokens):
            return "missing_evidence_support"
        if require_cross_paper_support and any(
            sum(
                token in _content_tokens(_evidence_text(item))
                for item in all_evidence
            )
            < 2
            for token in introduced_tokens
        ):
            return "unsupported_evidence_drift"
        if require_cross_title_support and any(
            sum(
                token in _content_tokens(str(item.get("title", "")))
                for item in all_evidence
            )
            < 2
            for token in introduced_tokens
        ):
            return "unsupported_title_expression"
    return None


def build_safe_query_complement(
    original_query: str,
    anchor_query: str,
    *,
    remove_narrative_shell: bool = False,
) -> str | None:
    """Compress explicit query wording without introducing model-generated concepts."""
    normalized_original = unicodedata.normalize("NFKC", original_query).strip()
    parallel = _PARALLEL_METHODS.search(normalized_original)
    if parallel is not None:
        facets = [
            part.strip(" ,")
            for part in re.split(r",\s*|\s+and\s+", parallel.group("methods"))
            if part.strip(" ,")
        ]
        if len(facets) >= 3:
            selected = facets[-2:]
            task_tokens = [
                token
                for token in _WORD.findall(parallel.group("task"))
                if token.casefold() not in _COMPRESSION_STOPWORDS
            ]
            candidate = " ".join([*selected, *task_tokens])
            return _validated_complement(candidate, original_query, anchor_query)

    acronym_spans: list[tuple[int, int]] = []
    for match in _PARENTHETICAL_ACRONYM.finditer(normalized_original):
        start = match.start()
        prefix = normalized_original[:start]
        words = list(re.finditer(r"[A-Za-z][A-Za-z-]*", prefix))
        phrase_start = start
        for word in reversed(words):
            if word.end() != phrase_start and prefix[word.end() : phrase_start].strip():
                break
            if not word.group(0)[0].isupper():
                break
            phrase_start = word.start()
        acronym_spans.append((phrase_start, match.end()))

    masked = normalized_original
    for start, end in reversed(acronym_spans):
        acronym = _PARENTHETICAL_ACRONYM.search(masked[start:end])
        if acronym is not None:
            masked = f"{masked[:start]} {acronym.group(1)} {masked[end:]}"
    compression_stopwords = (
        _COMPRESSION_STOPWORDS.union(_NARRATIVE_SHELL)
        if remove_narrative_shell
        else _COMPRESSION_STOPWORDS
    )
    tokens = [
        token
        for token in _WORD.findall(masked)
        if token.casefold() not in compression_stopwords
    ]
    candidate = " ".join(tokens[:12])
    return _validated_complement(candidate, original_query, anchor_query)


def _validated_complement(
    candidate: str, original_query: str, anchor_query: str
) -> str | None:
    candidate = " ".join(candidate.split()).strip()
    if not candidate or len(_content_tokens(candidate)) < 3:
        return None
    if not _content_tokens(candidate).issubset(_content_tokens(original_query)):
        return None
    if _normalized(candidate) == _normalized(anchor_query):
        return None
    if len(_span_tokens(candidate)) >= len(_span_tokens(anchor_query)):
        return None
    original_acronyms = set(_ACRONYM.findall(original_query))
    if not set(_ACRONYM.findall(candidate)).issubset(original_acronyms):
        return None
    return candidate


def _append_safe_query_complement(
    context: RecallGenerationContext,
    anchor_batch: RecallActionBatch,
    *,
    max_actions: int,
    remove_narrative_shell: bool = False,
) -> RecallActionBatch:
    if len(anchor_batch.actions) >= max_actions:
        return anchor_batch
    anchor_query = " ".join(
        action.payload.query_text
        for action in anchor_batch.actions
        if action.action_type == "text_search"
    )
    complement = build_safe_query_complement(
        context.original_query,
        anchor_query,
        remove_narrative_shell=remove_narrative_shell,
    )
    if complement is None:
        return anchor_batch
    try:
        return validate_action_batch(
            {
                "actions": [
                    *(action.model_dump(mode="json") for action in anchor_batch.actions),
                    {
                        "action_id": "query-compression",
                        "action_type": "text_search",
                        "strategy": "evidence-steered:query-compression",
                        "payload": {"query_text": complement},
                    },
                ]
            },
            context,
            allowed_actions={"text_search"},
            max_actions=max_actions,
        )
    except ActionValidationFailure:
        return anchor_batch


def _paper_evidence(
    original_query: str,
    anchor_queries: Sequence[str],
    action_results: Sequence[RetrievalActionResult],
    *,
    max_papers: int,
) -> list[dict[str, object]]:
    query_text = " ".join([original_query, *anchor_queries])
    concepts = _content_tokens(query_text)
    protected = set(_ACRONYM.findall(query_text))
    threshold = 2 if len(concepts) >= 2 else 1
    candidates: list[tuple[tuple[int, int, int], dict[str, object], set[str]]] = []
    seen_titles: set[str] = set()
    provider_rank = 0
    for result in action_results:
        for paper in result.hits:
            provider_rank += 1
            if provider_rank > _MAX_CANDIDATE_PAPERS:
                break
            title = _sanitize_evidence_text(paper.title)[:_MAX_TITLE_CHARS].strip()
            if not title or title.casefold() in seen_titles:
                continue
            seen_titles.add(title.casefold())
            abstract = _sanitize_evidence_text(
                (paper.abstract or "")[:_MAX_ABSTRACT_SCAN_CHARS]
            )
            title_tokens = _content_tokens(title)
            abstract_tokens = _content_tokens(abstract)
            title_overlap = len(title_tokens.intersection(concepts))
            abstract_overlap = len(abstract_tokens.intersection(concepts))
            combined = f"{title} {abstract}".casefold()
            protected_match = any(anchor.casefold() in combined for anchor in protected)
            if protected_match:
                tier = 0
                overlap = max(title_overlap, abstract_overlap)
            elif title_overlap >= threshold:
                tier = 1
                overlap = title_overlap
            elif abstract_overlap >= threshold:
                tier = 2
                overlap = abstract_overlap
            else:
                continue
            item: dict[str, object] = {
                "rank": provider_rank,
                "title": title,
                "snippets": _focused_snippets(abstract, concepts, protected),
                "authors": [
                    sanitized
                    for author in paper.authors[:_MAX_AUTHORS]
                    if (
                        sanitized := _sanitize_evidence_text(author)[
                            :_MAX_AUTHOR_CHARS
                        ].strip()
                    )
                ],
                "publication_year": paper.publication_year,
                "venue": (
                    _sanitize_evidence_text(paper.venue or "")[:_MAX_VENUE_CHARS].strip()
                    or None
                ),
            }
            candidates.append(((tier, -overlap, provider_rank), item, title_tokens))
        if provider_rank >= _MAX_CANDIDATE_PAPERS:
            break

    selected: list[dict[str, object]] = []
    selected_title_tokens: list[set[str]] = []
    for _, item, title_tokens in sorted(candidates, key=lambda candidate: candidate[0]):
        if any(_jaccard(title_tokens, prior) >= 0.85 for prior in selected_title_tokens):
            continue
        selected.append(item)
        selected_title_tokens.append(title_tokens)
        if len(selected) == max_papers:
            break
    return selected


def _evidence_quality_gate(
    original_query: str, evidence: Sequence[Mapping[str, object]]
) -> bool:
    """Require two independent papers to agree on at least two query concepts."""
    if len(evidence) < 2:
        return False
    query_tokens = _content_tokens(original_query)
    coverage = [
        _content_tokens(_evidence_text(item)).intersection(query_tokens)
        for item in evidence
    ]
    return any(
        len(left.intersection(right)) >= 2
        for index, left in enumerate(coverage)
        for right in coverage[index + 1 :]
    )


def _evidence_quality_gate_v4(
    original_query: str, evidence: Sequence[Mapping[str, object]]
) -> bool:
    """Require cross-title agreement on a new paper expression and query concepts."""
    if len(evidence) < 2:
        return False
    query_tokens = _content_tokens(original_query)
    title_tokens = [
        _content_tokens(str(item.get("title", ""))) for item in evidence
    ]
    qualified = [
        tokens for tokens in title_tokens if len(tokens.intersection(query_tokens)) >= 2
    ]
    return any(
        bool(left.intersection(right).difference(query_tokens))
        for index, left in enumerate(qualified)
        for right in qualified[index + 1 :]
    )


def _focused_snippets(
    abstract: str, concepts: set[str], protected: set[str]
) -> list[str]:
    if not abstract:
        return []
    lowered = abstract.casefold()
    positions: list[int] = []
    for term in sorted({*concepts, *(item.casefold() for item in protected)}, key=len, reverse=True):
        position = lowered.find(term)
        if position >= 0:
            positions.append(position)
    snippets: list[str] = []
    spans: list[tuple[int, int]] = []
    for position in sorted(positions):
        start = max(0, position - (_MAX_SNIPPET_CHARS // 2))
        end = min(len(abstract), start + _MAX_SNIPPET_CHARS)
        start = max(0, end - _MAX_SNIPPET_CHARS)
        if any(not (end <= old_start or start >= old_end) for old_start, old_end in spans):
            continue
        snippets.append(abstract[start:end].strip())
        spans.append((start, end))
        if len(snippets) == _MAX_SNIPPETS_PER_PAPER:
            break
    return snippets


def _render_refinement_prompt(prompt: RecallPromptArtifact) -> str:
    return "\n".join(
        [
            "Respond with one strict EvidenceProposalBatch JSON object.",
            f"Prompt version: {prompt.version}.",
            f"Model: {prompt.model}; temperature 0.",
            "The top-level object must contain only proposals.",
            "proposals must be an array with no more than proposal_schema.max_proposals items.",
            "Each proposal must contain exactly action_id, query_text, expansion_kind, query_support, and evidence_support.",
            "expansion_kind must be paper_expression or query_complement.",
            "query_support must quote exact spans from query.original_query.",
            "evidence_support items must contain rank and exact_phrase copied from that evidence item.",
            "paper_expression requires evidence_support; query_complement must be derivable from query_support alone.",
            "Do not copy anchor_actions into proposals and do not request a repair.",
            "Do not return Markdown, URLs, identifiers, or prose outside the JSON object.",
            *(f"- {instruction}" for instruction in prompt.instructions),
        ]
    )


def _final_result(
    anchor: GenerationResult,
    batch: RecallActionBatch,
    *,
    mode: str,
    evidence_count: int,
    audit: list[dict[str, object]],
    extra_receipt: LLMCallReceipt | None = None,
    backend_provenance: Mapping[str, str] | None = None,
    generation_version: Literal["v2", "v3", "v4"] = "v2",
) -> GenerationResult:
    receipts = list(anchor.call_receipts)
    if extra_receipt is not None:
        receipts.append(extra_receipt)
    provenance = {
        **anchor.provenance,
        **dict(backend_provenance or {}),
        "generation_mode": f"evidence-steered-{generation_version}",
        "refinement_mode": mode,
        "first_round_evidence_count": str(evidence_count),
        "proposal_audit_json": json.dumps(
            audit, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ),
    }
    artifact = json.dumps(
        batch.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return GenerationResult(
        query_id=anchor.query_id,
        action_batch=batch,
        artifact_bytes=artifact,
        provenance=provenance,
        call_receipts=receipts,
        repair_count=anchor.repair_count,
    )


def _call_receipt(result: LLMBackendResult) -> LLMCallReceipt:
    if result.infrastructure_failure:
        terminal = "infrastructure_failure"
    elif result.repairable:
        terminal = "repairable_failure"
    else:
        terminal = "succeeded"
    return LLMCallReceipt(
        call_kind="initial",
        usage=result.usage,
        provenance=result.provenance,
        errors=result.errors,
        terminal_state=terminal,
    )


def _evidence_text(item: Mapping[str, object]) -> str:
    return _normalized(" ".join(_evidence_fields(item)))


def _evidence_fields(item: Mapping[str, object]) -> list[str]:
    parts: list[str] = []
    for key in ("title", "venue"):
        value = item.get(key)
        if isinstance(value, str):
            parts.append(value)
    for key in ("snippets", "authors"):
        value = item.get(key)
        if isinstance(value, list):
            parts.extend(str(part) for part in value)
    return parts


def _contains_exact_span(fields: Sequence[str], phrase: str) -> bool:
    phrase_tokens = _span_tokens(phrase)
    if not phrase_tokens:
        return False
    for field in fields:
        tokens = _span_tokens(field)
        width = len(phrase_tokens)
        if any(tokens[index : index + width] == phrase_tokens for index in range(len(tokens) - width + 1)):
            return True
    return False


def _span_tokens(value: str) -> list[str]:
    return [token.casefold() for token in _WORD.findall(unicodedata.normalize("NFKC", value))]


def _content_tokens(value: str) -> set[str]:
    return {
        stemmed
        for token in _WORD.findall(unicodedata.normalize("NFKC", value))
        if (stemmed := _stem(token)) not in _STOPWORDS
        and (len(stemmed) >= 2 or stemmed.isdigit())
    }


def _stem(token: str) -> str:
    value = token.casefold().strip("-")
    for suffix, replacement in (
        ("ies", "y"),
        ("ness", ""),
        ("ments", "ment"),
        ("ing", ""),
        ("ed", ""),
        ("s", ""),
    ):
        if value.endswith(suffix) and len(value) - len(suffix) >= 4:
            return value[: -len(suffix)] + replacement
    return value


def _normalized(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _sanitize_evidence_text(value: str) -> str:
    return " ".join(_EVIDENCE_IDENTIFIER.sub(" ", value).split())


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left.union(right)
    return len(left.intersection(right)) / len(union) if union else 1.0


def _edit_distance_at_most_one(left: str, right: str) -> bool:
    left_folded = left.casefold()
    right_folded = right.casefold()
    if abs(len(left_folded) - len(right_folded)) > 1:
        return False
    if len(left_folded) == len(right_folded):
        return sum(a != b for a, b in zip(left_folded, right_folded, strict=True)) <= 1
    shorter, longer = (
        (left_folded, right_folded)
        if len(left_folded) < len(right_folded)
        else (right_folded, left_folded)
    )
    mismatch = 0
    for index, character in enumerate(shorter):
        if character != longer[index + mismatch]:
            mismatch += 1
            if mismatch > 1 or character != longer[index + mismatch]:
                return False
    return True


__all__ = [
    "EvidenceSteeredDeepSeekGenerator",
    "build_refinement_payload",
    "build_safe_query_complement",
    "validate_refinement_proposals",
]
