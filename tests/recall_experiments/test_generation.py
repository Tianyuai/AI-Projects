from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from paper_search.control.budget import HardBudgetController
from paper_search.domain.models import (
    ErrorDetail,
    Paper,
    ProviderResult,
    QuerySpec,
    SearchBudget,
    UsageActual,
    UsageEstimate,
)
from paper_search.recall_experiments.contracts import (
    GoldDocument,
    RecallActionBatch,
    RecallGenerationContext,
    RetrievalActionResult,
    SeedCandidate,
)
from paper_search.recall_experiments.generation.backends import (
    BudgetedLLMBackend,
    LLMBackendResult,
)
from paper_search.recall_experiments.generation.base import GenerationResult
from paper_search.recall_experiments.generation.deepseek import (
    DeepSeekPromptGenerator,
    RecallGenerationFailure,
    RecallPromptArtifact,
    build_generation_payload,
    build_repair_payload,
    render_recall_prompt,
)
from paper_search.recall_experiments.generation.evidence_steered import (
    EvidenceSteeredDeepSeekGenerator,
    build_refinement_payload,
    build_safe_query_complement,
    validate_refinement_proposals,
)
from paper_search.recall_experiments.generation.fixed import FixedActionGenerator
from paper_search.recall_experiments.generation.manual import ManualActionGenerator
from paper_search.recall_experiments.recipes import load_recall_recipe
from paper_search.recall_experiments.validation import (
    ActionValidationFailure,
    ActionValidationIssue,
)


def _context(query_id: str) -> RecallGenerationContext:
    return RecallGenerationContext(
        query_id=query_id,
        original_query="graph retrieval",
        query_spec=QuerySpec(original_query="graph retrieval", research_goal="graph retrieval"),
    )


def _actions(query_text: str = "graph retrieval") -> dict[str, object]:
    return {
        "actions": [
            {
                "action_id": "search-1",
                "action_type": "text_search",
                "strategy": "fixed",
                "payload": {"query_text": query_text},
            }
        ]
    }


def test_fixed_generation_preserves_the_bound_action_bytes_and_validates_before_return() -> None:
    raw = json.dumps(_actions(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    generator = FixedActionGenerator(
        {"query-1": raw},
        expected_query_ids=["query-1"],
        allowed_actions={"text_search"},
        max_actions=1,
    )

    result = asyncio.run(generator.generate(_context("query-1")))

    assert result.query_id == "query-1"
    assert result.artifact_bytes == raw
    assert result.action_batch.actions[0].action_id == "search-1"


@pytest.mark.parametrize("as_bytes", [True, False])
def test_fixed_generation_preserves_formatted_and_reordered_byte_or_string_sources(
    as_bytes: bool,
) -> None:
    source_text = """{
  "actions": [
    {
      "strategy": "fixed",
      "payload": {"query_text": "graph retrieval"},
      "action_type": "text_search",
      "action_id": "search-1"
    }
  ]
}"""
    source = source_text.encode("utf-8") if as_bytes else source_text
    generator = FixedActionGenerator(
        {"query-1": source},
        expected_query_ids=["query-1"],
        allowed_actions={"text_search"},
        max_actions=1,
    )

    result = asyncio.run(generator.generate(_context("query-1")))

    assert result.artifact_bytes == source_text.encode("utf-8")
    assert result.action_batch.actions[0].action_id == "search-1"


def test_fixed_generation_rejects_unknown_or_missing_query_ids() -> None:
    with pytest.raises(ValueError, match="coverage"):
        FixedActionGenerator(
            {"query-1": _actions()},
            expected_query_ids=["query-1", "query-2"],
            allowed_actions={"text_search"},
            max_actions=1,
        )

    generator = FixedActionGenerator(
        {"query-1": _actions()},
        expected_query_ids=["query-1"],
        allowed_actions={"text_search"},
        max_actions=1,
    )
    with pytest.raises(ValueError, match="unknown query"):
        asyncio.run(generator.generate(_context("query-2")))


def test_fixed_generation_freezes_nested_caller_actions_at_construction() -> None:
    caller_owned = {"query-1": _actions()}
    generator = FixedActionGenerator(
        caller_owned,
        expected_query_ids=["query-1"],
        allowed_actions={"text_search"},
        max_actions=1,
    )
    before = asyncio.run(generator.generate(_context("query-1")))

    actions = caller_owned["query-1"]["actions"]
    assert isinstance(actions, list)
    first = actions[0]
    assert isinstance(first, dict)
    payload = first["payload"]
    assert isinstance(payload, dict)
    payload["query_text"] = "mutated after construction"
    actions.append(
        {
            "action_id": "late-action",
            "action_type": "text_search",
            "strategy": "late",
            "payload": {"query_text": "late mutation"},
        }
    )

    after = asyncio.run(generator.generate(_context("query-1")))

    assert after.artifact_bytes == before.artifact_bytes
    assert after.action_batch == before.action_batch


def test_manual_generation_reads_user_prepared_json_without_an_llm(tmp_path) -> None:
    actions_path = tmp_path / "actions.json"
    actions_path.write_text(json.dumps({"query-1": _actions()}), encoding="utf-8")
    generator = ManualActionGenerator(
        actions_path,
        expected_query_ids=["query-1"],
        allowed_actions={"text_search"},
        max_actions=1,
    )

    result = asyncio.run(generator.generate(_context("query-1")))

    assert result.action_batch.actions[0].payload.query_text == "graph retrieval"


def test_manual_generation_identity_hashes_the_exact_consumed_bytes(tmp_path) -> None:
    actions_path = tmp_path / "actions.json"
    original = json.dumps({"query-1": _actions()}).encode("utf-8")
    actions_path.write_bytes(original)
    generator = ManualActionGenerator(
        actions_path,
        expected_query_ids=["query-1"],
        allowed_actions={"text_search"},
        max_actions=1,
    )

    actions_path.write_text('{"query-1":{"actions":[]}}', encoding="utf-8")

    assert generator.source_sha256 == f"sha256:{hashlib.sha256(original).hexdigest()}"
    result = asyncio.run(generator.generate(_context("query-1")))
    assert result.action_batch.actions[0].payload.query_text == "graph retrieval"


class _RecordingLLMBackend:
    def __init__(self, results: list[LLMBackendResult]) -> None:
        self._results = iter(results)
        self.calls: list[tuple[str, object]] = []

    async def generate(self, request: object, call_kind: str) -> LLMBackendResult:
        self.calls.append((call_kind, request))
        result = next(self._results)
        return result.model_copy(
            update={"provenance": {"backend_call_id": f"call-{len(self.calls)}"}}
        )


def _prompt() -> RecallPromptArtifact:
    return RecallPromptArtifact.from_yaml_bytes(
        b"# source bytes are identity-sensitive\nname: recall_scheme_b\nversion: recall-scheme-b-v1\nmodel: deepseek-v4-flash\ntemperature: 0\ninstructions:\n  - Generate only permitted retrieval actions.\n"
    )


def _evidence_prompt_v3() -> RecallPromptArtifact:
    return RecallPromptArtifact.from_yaml_bytes(
        b"# source bytes are identity-sensitive\nname: recall_evidence_v3\nversion: recall-evidence-steered-open-profile-v3\nmodel: deepseek-v4-flash\ntemperature: 0\ninstructions:\n  - Generate only permitted retrieval actions.\n"
    )


def _evidence_prompt_v4() -> RecallPromptArtifact:
    return RecallPromptArtifact.from_yaml_bytes(
        b"# source bytes are identity-sensitive\nname: recall_evidence_v4\nversion: recall-evidence-steered-open-profile-v4\nmodel: deepseek-v4-flash\ntemperature: 0\ninstructions:\n  - Generate only permitted retrieval actions.\n"
    )


def _oracle_context(*, year_from: int | None = None) -> RecallGenerationContext:
    return RecallGenerationContext(
        query_id="query-1",
        original_query="graph retrieval",
        query_spec=QuerySpec(
            original_query="graph retrieval", research_goal="graph retrieval", year_from=year_from
        ),
        seed_queries=["graph retrieval survey"],
        gold_documents=[
            GoldDocument(
                title="OpenAlex and Semantic Scholar coverage study",
                abstract="This ordinary prose mentions OpenAlex and Semantic Scholar.",
                authors=["A. Researcher"],
                publication_year=2024,
                metadata_coverage={
                    "title": True,
                    "abstract": True,
                    "authors": True,
                    "publication_year": True,
                },
            )
        ],
    )


def _backend_result(data: dict[str, object]) -> LLMBackendResult:
    return LLMBackendResult(data=data)


def test_evidence_steered_generation_uses_sanitized_first_round_paper_evidence() -> None:
    anchor = {
        "actions": [
            {
                "action_id": "anchor",
                "action_type": "text_search",
                "strategy": "evidence-steered:anchor",
                "payload": {"query_text": "graph retrieval"},
            }
        ]
    }
    refined = {
        "proposals": [
            {
                "action_id": "evidence_expansion",
                "query_text": "dense graph retrieval",
                "expansion_kind": "paper_expression",
                "query_support": ["graph retrieval"],
                "evidence_support": [
                    {
                        "rank": 1,
                        "exact_phrase": "Dense retrieval for attributed graphs",
                    }
                ],
            }
        ]
    }
    backend = _RecordingLLMBackend(
        [_backend_result(anchor), _backend_result(refined)]
    )
    generator = EvidenceSteeredDeepSeekGenerator(
        backend=backend,
        prompt=_prompt(),
        visibility="blind",
        allowed_actions={"text_search"},
        max_actions=3,
    )
    context = _context("query-1")

    anchor_result = asyncio.run(generator.generate(context))
    result = asyncio.run(
        generator.refine(
            context,
            anchor_result,
            [
                RetrievalActionResult(
                    action_id="anchor",
                    action_type="text_search",
                    hits=[
                        Paper(
                            canonical_id="doi:10.1000/private-id",
                            title="Dense retrieval for attributed graphs",
                            abstract=(
                                "Graph retrieval is a paper-common expression from the first-round abstract. "
                                "Repository https://example.test/private"
                            ),
                            authors=["A. Researcher"],
                            publication_year=2024,
                            venue="IR Journal",
                            doi="10.1000/private-id",
                            openalex_id="W123456789",
                            url="https://example.test/private",
                            sources=["openalex"],
                        )
                    ],
                )
            ],
        )
    )

    assert [kind for kind, _ in backend.calls] == ["initial", "initial"]
    anchor_payload = getattr(backend.calls[0][1], "payload")
    refinement_payload = getattr(backend.calls[1][1], "payload")
    assert anchor_payload["generation_phase"] == "anchor"
    assert anchor_payload["allowed_action_schema"]["max_actions"] == 1
    assert refinement_payload["generation_phase"] == "refinement"
    assert refinement_payload["anchor_actions"] == anchor["actions"]
    assert refinement_payload["first_round_evidence"] == [
        {
            "rank": 1,
            "title": "Dense retrieval for attributed graphs",
            "snippets": [
                "Graph retrieval is a paper-common expression from the first-round abstract. Repository"
            ],
            "authors": ["A. Researcher"],
            "publication_year": 2024,
            "venue": "IR Journal",
        }
    ]
    serialized = json.dumps(refinement_payload)
    assert "10.1000/private-id" not in serialized
    assert "W123456789" not in serialized
    assert "https://example.test/private" not in serialized
    assert [action.action_id for action in result.action_batch.actions] == [
        "anchor",
        "evidence_expansion",
    ]
    assert len(result.call_receipts) == 2
    assert result.repair_count == 0


def test_refinement_evidence_strips_embedded_arxiv_and_pmid_identifiers() -> None:
    context = _context("query-identifiers")
    batch = RecallActionBatch.model_validate(_actions())
    anchor = GenerationResult(
        query_id=context.query_id,
        action_batch=batch,
        artifact_bytes=batch.model_dump_json().encode("utf-8"),
    )

    payload = build_refinement_payload(
        context,
        anchor,
        [
            RetrievalActionResult(
                action_id="search-1",
                action_type="text_search",
                hits=[
                    Paper(
                        canonical_id="openalex:W1000000",
                        title=(
                            "Graph retrieval arXiv:2107.03374 PMID: 12345678 "
                            "PubMed: 87654321"
                        ),
                        abstract=(
                            "Graph retrieval appears in arXiv 2203.07814 and "
                            "arXiv:hep-th/9901001 with S2 "
                            "0123456789abcdef0123456789abcdef01234567."
                        ),
                        sources=["openalex"],
                    )
                ],
            )
        ],
        allowed_actions={"text_search"},
        max_actions=3,
    )

    serialized = json.dumps(payload["first_round_evidence"])
    assert "2107.03374" not in serialized
    assert "2203.07814" not in serialized
    assert "PMID" not in serialized
    assert "PubMed" not in serialized
    assert "hep-th/9901001" not in serialized
    assert "0123456789abcdef0123456789abcdef01234567" not in serialized


def test_refinement_evidence_is_bounded_for_the_existing_scheme_b_token_reservation() -> None:
    context = _context("query-1")
    batch = RecallActionBatch.model_validate(_actions())
    anchor = GenerationResult(
        query_id=context.query_id,
        action_batch=batch,
        artifact_bytes=batch.model_dump_json().encode("utf-8"),
    )
    distinct_terms = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]
    hits = [
        Paper(
            canonical_id=f"openalex:W{index + 1000000}",
            title=f"Graph retrieval evidence {distinct_terms[index]}",
            abstract="graph retrieval " + ("x" * 900),
            sources=["openalex"],
        )
        for index in range(8)
    ]

    payload = build_refinement_payload(
        context,
        anchor,
        [
            RetrievalActionResult(
                action_id="search-1", action_type="text_search", hits=hits
            )
        ],
        allowed_actions={"text_search"},
        max_actions=3,
    )

    evidence = payload["first_round_evidence"]
    assert isinstance(evidence, list)
    assert len(evidence) == 5
    assert all(
        len(str(snippet)) <= 400
        for item in evidence
        for snippet in item["snippets"]
    )


def test_refinement_selects_query_relevant_late_hit_and_uses_focused_snippets() -> None:
    context = RecallGenerationContext(
        query_id="query-swag",
        original_query="SWAG Gaussian posterior uncertainty calibration",
        query_spec=QuerySpec(
            original_query="SWAG Gaussian posterior uncertainty calibration",
            research_goal="SWAG Gaussian posterior uncertainty calibration",
        ),
    )
    batch = RecallActionBatch.model_validate(_actions("SWAG Gaussian posterior"))
    anchor = GenerationResult(
        query_id=context.query_id,
        action_batch=batch,
        artifact_bytes=batch.model_dump_json().encode("utf-8"),
    )
    noisy = [
        Paper(
            canonical_id=f"openalex:W{index + 1000000}",
            title=f"Unrelated optimization study {index}",
            abstract="Generic neural network training results without the requested concepts.",
            sources=["openalex"],
        )
        for index in range(22)
    ]
    relevant = Paper(
        canonical_id="openalex:W9999999",
        title="Gaussian Stochastic Weight Averaging for Bayesian Deep Learning",
        abstract=(
            "Background text. SWAG approximates a Gaussian posterior from SGD iterates "
            "for uncertainty calibration. Closing text."
        ),
        sources=["openalex"],
    )

    payload = build_refinement_payload(
        context,
        anchor,
        [
            RetrievalActionResult(
                action_id="search-1",
                action_type="text_search",
                hits=[*noisy, relevant],
            )
        ],
        allowed_actions={"text_search"},
        max_actions=3,
    )

    evidence = payload["first_round_evidence"]
    assert isinstance(evidence, list)
    assert evidence[0]["rank"] == 23
    assert evidence[0]["title"] == relevant.title
    assert evidence[0]["snippets"] == [relevant.abstract]
    assert all(item["title"] != noisy[0].title for item in evidence)
    assert payload["refinement_mode"] == "evidence_grounded"


def test_refinement_rejects_unsupported_near_anchor_mutation_without_repair_call() -> None:
    anchor_output = {
        "actions": [
            {
                "action_id": "anchor",
                "action_type": "text_search",
                "strategy": "evidence-steered:anchor",
                "payload": {"query_text": "SWAG Gaussian posterior uncertainty"},
            }
        ]
    }
    proposal_output = {
        "proposals": [
            {
                "action_id": "evidence-expansion",
                "query_text": "SWA Gaussian posterior SGD uncertainty",
                "expansion_kind": "paper_expression",
                "query_support": ["Gaussian posterior", "uncertainty"],
                "evidence_support": [],
            }
        ]
    }
    backend = _RecordingLLMBackend(
        [_backend_result(anchor_output), _backend_result(proposal_output)]
    )
    generator = EvidenceSteeredDeepSeekGenerator(
        backend=backend,
        prompt=_prompt(),
        visibility="blind",
        allowed_actions={"text_search"},
        max_actions=3,
    )
    context = RecallGenerationContext(
        query_id="query-swag",
        original_query="SWAG Gaussian posterior uncertainty",
        query_spec=QuerySpec(
            original_query="SWAG Gaussian posterior uncertainty",
            research_goal="SWAG Gaussian posterior uncertainty",
        ),
    )

    anchor = asyncio.run(generator.generate(context))
    result = asyncio.run(
        generator.refine(
            context,
            anchor,
            [
                RetrievalActionResult(
                    action_id="anchor",
                    action_type="text_search",
                    hits=[
                        Paper(
                            canonical_id="openalex:W1000000",
                            title="Uncertainty estimation for neural networks",
                            abstract="Gaussian posterior calibration for uncertainty estimation.",
                            sources=["openalex"],
                        )
                    ],
                )
            ],
        )
    )

    assert [action.payload.query_text for action in result.action_batch.actions] == [
        "SWAG Gaussian posterior uncertainty"
    ]
    assert [kind for kind, _ in backend.calls] == ["initial", "initial"]
    audit = json.loads(result.provenance["proposal_audit_json"])
    assert audit[0]["decision"] == "rejected"
    assert audit[0]["reason"] == "near_anchor_mutation"


def test_evidence_support_does_not_treat_swa_as_exact_span_inside_swag() -> None:
    context = RecallGenerationContext(
        query_id="query-swag-prefix",
        original_query="SWAG Gaussian posterior uncertainty",
        query_spec=QuerySpec(
            original_query="SWAG Gaussian posterior uncertainty",
            research_goal="SWAG Gaussian posterior uncertainty",
        ),
    )
    anchor = RecallActionBatch.model_validate(
        {
            "actions": [
                {
                    "action_id": "anchor",
                    "action_type": "text_search",
                    "strategy": "evidence-steered:anchor",
                    "payload": {"query_text": "SWAG Gaussian posterior uncertainty"},
                }
            ]
        }
    )

    batch, audit = validate_refinement_proposals(
        context,
        anchor,
        {
            "proposals": [
                {
                    "action_id": "prefix-bypass",
                    "query_text": "SWA Gaussian posterior uncertainty",
                    "expansion_kind": "paper_expression",
                    "query_support": "Gaussian posterior uncertainty",
                    "evidence_support": [{"rank": 1, "exact_phrase": "SWA"}],
                }
            ]
        },
        [
            {
                "rank": 1,
                "title": "SWAG Gaussian posterior uncertainty",
                "snippets": [],
                "authors": [],
                "publication_year": 2019,
                "venue": None,
            }
        ],
        max_actions=3,
    )

    assert batch == anchor
    assert audit[0]["decision"] == "rejected"
    assert audit[0]["reason"] == "missing_evidence_support"


def test_evidence_steered_anchor_payload_omits_text_search_seed_identifiers() -> None:
    backend = _RecordingLLMBackend(
        [
            _backend_result(
                {
                    "actions": [
                        {
                            "action_id": "anchor",
                            "action_type": "text_search",
                            "strategy": "evidence-steered:anchor",
                            "payload": {"query_text": "graph retrieval"},
                        }
                    ]
                }
            )
        ]
    )
    generator = EvidenceSteeredDeepSeekGenerator(
        backend=backend,
        prompt=_prompt(),
        visibility="blind",
        allowed_actions={"text_search"},
        max_actions=3,
    )
    context = RecallGenerationContext(
        query_id="query-seed",
        original_query="graph retrieval",
        query_spec=QuerySpec(
            original_query="graph retrieval", research_goal="graph retrieval"
        ),
        seed_candidates=[
            SeedCandidate(
                paper=Paper(
                    canonical_id="doi:10.1000/private-seed",
                    title="Private seed title",
                    sources=["openalex"],
                )
            )
        ],
    )

    asyncio.run(generator.generate(context))

    payload = getattr(backend.calls[0][1], "payload")
    assert "seed_candidates" not in payload
    assert "10.1000/private-seed" not in json.dumps(payload)


def test_evidence_steered_rejects_empty_anchor_batch() -> None:
    backend = _RecordingLLMBackend([_backend_result({"actions": []})])
    generator = EvidenceSteeredDeepSeekGenerator(
        backend=backend,
        prompt=_prompt(),
        visibility="blind",
        allowed_actions={"text_search"},
        max_actions=3,
    )

    with pytest.raises(RecallGenerationFailure, match="generation_failure"):
        asyncio.run(generator.generate(_context("query-empty-anchor")))


def test_query_grounded_complement_is_allowed_when_first_round_has_no_evidence() -> None:
    anchor_output = {
        "actions": [
            {
                "action_id": "anchor",
                "action_type": "text_search",
                "strategy": "evidence-steered:anchor",
                "payload": {"query_text": "robust graph retrieval"},
            }
        ]
    }
    proposal_output = {
        "proposals": [
            {
                "action_id": "query-complement",
                "query_text": "graph retrieval robustness benchmark",
                "expansion_kind": "query_complement",
                "query_support": ["graph retrieval", "robustness benchmark"],
                "evidence_support": [],
            }
        ]
    }
    backend = _RecordingLLMBackend(
        [_backend_result(anchor_output), _backend_result(proposal_output)]
    )
    generator = EvidenceSteeredDeepSeekGenerator(
        backend=backend,
        prompt=_prompt(),
        visibility="blind",
        allowed_actions={"text_search"},
        max_actions=3,
    )
    context = RecallGenerationContext(
        query_id="query-robust",
        original_query="robust graph retrieval robustness benchmark",
        query_spec=QuerySpec(
            original_query="robust graph retrieval robustness benchmark",
            research_goal="robust graph retrieval robustness benchmark",
        ),
    )

    anchor = asyncio.run(generator.generate(context))
    result = asyncio.run(
        generator.refine(
            context,
            anchor,
            [
                RetrievalActionResult(
                    action_id="anchor", action_type="text_search", hits=[]
                )
            ],
        )
    )

    assert [action.payload.query_text for action in result.action_batch.actions] == [
        "robust graph retrieval",
        "graph retrieval robustness benchmark",
    ]
    refinement_payload = getattr(backend.calls[1][1], "payload")
    assert refinement_payload["refinement_mode"] == "query_grounded_only"


def test_near_anchor_expression_is_allowed_only_with_linked_exact_evidence() -> None:
    context = RecallGenerationContext(
        query_id="query-swag",
        original_query="SWAG Gaussian posterior uncertainty",
        query_spec=QuerySpec(
            original_query="SWAG Gaussian posterior uncertainty",
            research_goal="SWAG Gaussian posterior uncertainty",
        ),
    )
    anchor = RecallActionBatch.model_validate(
        {
            "actions": [
                {
                    "action_id": "anchor",
                    "action_type": "text_search",
                    "strategy": "evidence-steered:anchor",
                    "payload": {"query_text": "SWAG Gaussian posterior uncertainty"},
                }
            ]
        }
    )
    evidence = [
        {
            "rank": 23,
            "title": "SWA for Bayesian deep learning",
            "snippets": [
                "SWA estimates a Gaussian posterior for uncertainty calibration."
            ],
            "authors": [],
            "publication_year": 2020,
            "venue": None,
        }
    ]

    batch, audit = validate_refinement_proposals(
        context,
        anchor,
        {
            "proposals": [
                {
                    "action_id": "evidence-expansion",
                    "query_text": "SWA Gaussian posterior uncertainty",
                    "expansion_kind": "paper_expression",
                    "query_support": "Gaussian posterior uncertainty",
                    "evidence_support": [
                        {
                            "rank": 23,
                            "exact_phrase": "SWA estimates a Gaussian posterior",
                        }
                    ],
                }
            ]
        },
        evidence,
        max_actions=3,
    )

    assert [action.payload.query_text for action in batch.actions] == [
        "SWAG Gaussian posterior uncertainty",
        "SWA Gaussian posterior uncertainty",
    ]
    assert audit[0]["decision"] == "accepted"


def test_malformed_proposal_does_not_discard_valid_sibling() -> None:
    context = _context("query-mixed-proposals")
    anchor = RecallActionBatch.model_validate(_actions("graph retrieval"))
    evidence = [
        {
            "rank": 4,
            "title": "Dense graph retrieval",
            "snippets": ["Dense graph retrieval for attributed networks."],
            "authors": [],
            "publication_year": 2024,
            "venue": None,
        }
    ]

    batch, audit = validate_refinement_proposals(
        context,
        anchor,
        {
            "proposals": [
                {
                    "action_id": "valid",
                    "query_text": "dense graph retrieval",
                    "expansion_kind": "paper_expression",
                    "query_support": "graph retrieval",
                    "evidence_support": [
                        {"rank": 4, "exact_phrase": "Dense graph retrieval"}
                    ],
                },
                {"action_id": "malformed", "query_text": 123},
            ]
        },
        evidence,
        max_actions=3,
    )

    assert [action.action_id for action in batch.actions] == ["search-1", "valid"]
    assert sorted(item["decision"] for item in audit) == ["accepted", "rejected"]
    assert any(item["reason"] == "invalid_proposal" for item in audit)


def test_proposal_must_declare_every_query_concept_it_reuses() -> None:
    context = _context("query-incomplete-source")
    anchor = RecallActionBatch.model_validate(_actions("graph retrieval"))

    batch, audit = validate_refinement_proposals(
        context,
        anchor,
        {
            "proposals": [
                {
                    "action_id": "incomplete",
                    "query_text": "dense graph retrieval",
                    "expansion_kind": "paper_expression",
                    "query_support": "graph",
                    "evidence_support": [
                        {"rank": 1, "exact_phrase": "dense retrieval"}
                    ],
                }
            ]
        },
        [
            {
                "rank": 1,
                "title": "Dense retrieval",
                "snippets": [],
                "authors": [],
                "publication_year": 2024,
                "venue": None,
            }
        ],
        max_actions=3,
    )

    assert batch == anchor
    assert audit[0]["reason"] == "missing_query_support"


def test_query_complement_cannot_add_unsupported_year() -> None:
    context = _context("query-year-bypass")
    anchor = RecallActionBatch.model_validate(_actions("graph retrieval"))

    batch, audit = validate_refinement_proposals(
        context,
        anchor,
        {
            "proposals": [
                {
                    "action_id": "year-bypass",
                    "query_text": "graph retrieval 2020",
                    "expansion_kind": "query_complement",
                    "query_support": "graph retrieval",
                    "evidence_support": [],
                }
            ]
        },
        [],
        max_actions=3,
    )

    assert batch == anchor
    assert audit[0]["reason"] == "unsupported_query_expansion"


def test_query_complement_cannot_mutate_single_digit_version() -> None:
    context = RecallGenerationContext(
        query_id="query-version",
        original_query="graph retrieval version 3",
        query_spec=QuerySpec(
            original_query="graph retrieval version 3",
            research_goal="graph retrieval version 3",
        ),
    )
    anchor = RecallActionBatch.model_validate(_actions("graph retrieval version 3"))

    batch, audit = validate_refinement_proposals(
        context,
        anchor,
        {
            "proposals": [
                {
                    "action_id": "version-bypass",
                    "query_text": "graph retrieval version 4",
                    "expansion_kind": "query_complement",
                    "query_support": "graph retrieval version 3",
                    "evidence_support": [],
                }
            ]
        },
        [],
        max_actions=3,
    )

    assert batch == anchor
    assert audit[0]["reason"] == "unsupported_query_expansion"


def test_refinement_considers_at_most_two_proposals() -> None:
    context = _context("query-proposal-limit")
    anchor = RecallActionBatch.model_validate(_actions("graph retrieval"))
    proposals = [
        {"action_id": "bad-1", "query_text": 1},
        {"action_id": "bad-2", "query_text": 2},
        {
            "action_id": "third-valid",
            "query_text": "dense graph retrieval",
            "expansion_kind": "paper_expression",
            "query_support": "graph retrieval",
            "evidence_support": [
                {"rank": 1, "exact_phrase": "dense graph retrieval"}
            ],
        },
    ]

    batch, audit = validate_refinement_proposals(
        context,
        anchor,
        {"proposals": proposals},
        [
            {
                "rank": 1,
                "title": "Dense graph retrieval",
                "snippets": [],
                "authors": [],
                "publication_year": 2024,
                "venue": None,
            }
        ],
        max_actions=3,
    )

    assert batch == anchor
    assert any(item["reason"] == "proposal_limit_exceeded" for item in audit)


def test_safe_query_complement_compresses_td_ode_query_without_new_terms() -> None:
    original = (
        "Which study approaches the problem of convergence rates of classic TD "
        "from the perspective of Ordinary Differential Equations (ODE) analysis?"
    )

    complement = build_safe_query_complement(
        original,
        "convergence rates temporal difference learning ordinary differential "
        "equations ODE analysis",
    )

    assert complement == "convergence rates TD ODE analysis"


def test_safe_query_complement_facets_long_parallel_method_list() -> None:
    original = (
        "What papers utilize VAEs, normalizing flows, reinforcement learning, "
        "optimal transport and diffusion models for the task of predicting the "
        "3D structure of molecules given a molecular graph?"
    )

    complement = build_safe_query_complement(
        original,
        "molecular graph 3D structure prediction variational autoencoder "
        "normalizing flow reinforcement learning optimal transport diffusion model",
    )

    assert complement == (
        "optimal transport diffusion models predicting 3D structure molecules "
        "molecular graph"
    )


def test_safe_query_complement_removes_narrative_shell_but_keeps_relation_constraints() -> None:
    complement = build_safe_query_complement(
        "What studies have presented connections between RNNs and early versions of GNNs?",
        "connections between recurrent neural networks and early graph neural networks",
        remove_narrative_shell=True,
    )

    assert complement == "connections between RNNs early versions GNNs"


def test_safe_query_complement_removes_retrieval_request_verbs() -> None:
    complement = build_safe_query_complement(
        "Which works used image generation models to create synthetic images for classification tasks?",
        "generative image synthesis models synthetic visual data image classification",
        remove_narrative_shell=True,
    )

    assert complement == "image generation models synthetic images classification tasks"


def test_v3_safe_query_complement_preserves_historical_narrative_behavior() -> None:
    complement = build_safe_query_complement(
        "What studies have presented connections between RNNs and early versions of GNNs?",
        "connections between recurrent neural networks and early graph neural networks",
    )

    assert complement == (
        "studies have presented connections between RNNs early versions GNNs"
    )


def test_low_quality_first_round_skips_second_llm_and_keeps_safe_complement() -> None:
    anchor_output = {
        "actions": [
            {
                "action_id": "anchor",
                "action_type": "text_search",
                "strategy": "evidence-steered:anchor",
                "payload": {
                    "query_text": (
                        "convergence rates temporal difference learning ordinary "
                        "differential equations ODE analysis"
                    )
                },
            }
        ]
    }
    backend = _RecordingLLMBackend([_backend_result(anchor_output)])
    generator = EvidenceSteeredDeepSeekGenerator(
        backend=backend,
        prompt=_evidence_prompt_v3(),
        visibility="blind",
        allowed_actions={"text_search"},
        max_actions=3,
    )
    original = (
        "Which study approaches the problem of convergence rates of classic TD "
        "from the perspective of Ordinary Differential Equations (ODE) analysis?"
    )
    context = RecallGenerationContext(
        query_id="query-td-ode",
        original_query=original,
        query_spec=QuerySpec(original_query=original, research_goal=original),
    )

    anchor = asyncio.run(generator.generate(context))
    result = asyncio.run(
        generator.refine(
            context,
            anchor,
            [
                RetrievalActionResult(
                    action_id="anchor",
                    action_type="text_search",
                    hits=[
                        Paper(
                            canonical_id="openalex:W1000000",
                            title="A broad convergence survey",
                            abstract="Convergence theory across optimization methods.",
                            sources=["openalex"],
                        )
                    ],
                )
            ],
        )
    )

    assert [action.payload.query_text for action in result.action_batch.actions] == [
        anchor_output["actions"][0]["payload"]["query_text"],
        "convergence rates TD ODE analysis",
    ]
    assert [kind for kind, _ in backend.calls] == ["initial"]
    assert result.provenance["refinement_mode"] == "query_grounded_only"


def test_high_quality_evidence_payload_is_capped_at_three_papers_and_one_proposal() -> None:
    context = _context("query-quality-gate")
    batch = RecallActionBatch.model_validate(_actions("graph retrieval"))
    anchor = GenerationResult(
        query_id=context.query_id,
        action_batch=batch,
        artifact_bytes=batch.model_dump_json().encode("utf-8"),
    )
    hits = [
        Paper(
            canonical_id=f"openalex:W{1000000 + index}",
            title=f"Dense graph retrieval benchmark {index}",
            abstract="Dense graph retrieval benchmark for attributed networks.",
            sources=["openalex"],
        )
        for index in range(6)
    ]

    payload = build_refinement_payload(
        context,
        anchor,
        [
            RetrievalActionResult(
                action_id="search-1", action_type="text_search", hits=hits
            )
        ],
        allowed_actions={"text_search"},
        max_actions=3,
        generation_version="v3",
    )

    assert payload["refinement_mode"] == "evidence_grounded"
    assert len(payload["first_round_evidence"]) == 3
    assert payload["proposal_schema"]["max_proposals"] == 1
    assert payload["proposal_schema"]["expansion_kinds"] == ["paper_expression"]


def test_v4_evidence_gate_rejects_abstract_only_generic_overlap() -> None:
    original = (
        "Which research contains over thousands of single-choice questions "
        "covering numerous different ability dimensions?"
    )
    context = RecallGenerationContext(
        query_id="query-noisy-abstract-overlap",
        original_query=original,
        query_spec=QuerySpec(original_query=original, research_goal=original),
    )
    batch = RecallActionBatch.model_validate(
        _actions("thousands of single-choice questions multiple ability dimensions")
    )
    anchor = GenerationResult(
        query_id=context.query_id,
        action_batch=batch,
        artifact_bytes=batch.model_dump_json().encode("utf-8"),
    )
    hits = [
        Paper(
            canonical_id=f"openalex:W{1000000 + index}",
            title=title,
            abstract=(
                "This benchmark contains thousands of questions and measures "
                "multiple ability dimensions."
            ),
            sources=["openalex"],
        )
        for index, title in enumerate(
            [
                "Gradient-based document recognition",
                "A flexible particle simulation tool",
                "Implementation science in health services",
            ]
        )
    ]

    payload = build_refinement_payload(
        context,
        anchor,
        [
            RetrievalActionResult(
                action_id="search-1", action_type="text_search", hits=hits
            )
        ],
        allowed_actions={"text_search"},
        max_actions=3,
        generation_version="v4",
    )

    assert payload["refinement_mode"] != "evidence_grounded"


def test_v4_evidence_gate_accepts_shared_title_level_paper_expression() -> None:
    context = _context("query-title-expression")
    batch = RecallActionBatch.model_validate(_actions("graph retrieval"))
    anchor = GenerationResult(
        query_id=context.query_id,
        action_batch=batch,
        artifact_bytes=batch.model_dump_json().encode("utf-8"),
    )
    hits = [
        Paper(
            canonical_id=f"openalex:W{1000000 + index}",
            title=title,
            abstract="Attributed networks and benchmark evaluation.",
            sources=["openalex"],
        )
        for index, title in enumerate(
            [
                "Dense graph retrieval for attributed networks",
                "Dense graph retrieval benchmark evaluation",
                "Sparse graph retrieval methods",
            ]
        )
    ]

    payload = build_refinement_payload(
        context,
        anchor,
        [
            RetrievalActionResult(
                action_id="search-1", action_type="text_search", hits=hits
            )
        ],
        allowed_actions={"text_search"},
        max_actions=3,
        generation_version="v4",
    )

    assert payload["refinement_mode"] == "evidence_grounded"


def test_evidence_steered_v3_live_recipe_locks_the_optimized_scheme_b_module() -> None:
    loaded = load_recall_recipe(
        Path(
            "configs/recall_experiments/methods/"
            "evidence-steered-open-profile-v3-live.yaml"
        )
    )

    assert loaded.recipe.method_id == "evidence-steered-open-profile-v3"
    assert loaded.recipe.generator.evidence_steered is True
    assert loaded.recipe.generator.gold_visibility == "blind"
    assert loaded.recipe.generator.max_generated_actions == 3
    assert loaded.recipe.retrieval.max_total_actions == 3
    assert loaded.prompt_bytes is not None
    prompt = RecallPromptArtifact.from_yaml_bytes(loaded.prompt_bytes)
    assert prompt.version == "recall-evidence-steered-open-profile-v3"
    rendered = render_recall_prompt(prompt)
    assert "deterministically adds at most one query-only compression" in rendered
    assert "multiple evidence items independently connect" in rendered


def test_evidence_steered_v4_live_recipe_locks_strict_title_consensus() -> None:
    loaded = load_recall_recipe(
        Path(
            "configs/recall_experiments/methods/"
            "evidence-steered-open-profile-v4-live.yaml"
        )
    )

    assert loaded.recipe.method_id == "evidence-steered-open-profile-v4"
    assert loaded.recipe.generator.evidence_steered is True
    assert loaded.recipe.generator.gold_visibility == "blind"
    assert loaded.recipe.generator.max_generated_actions == 3
    assert loaded.prompt_bytes is not None
    prompt = RecallPromptArtifact.from_yaml_bytes(loaded.prompt_bytes)
    assert prompt.version == "recall-evidence-steered-open-profile-v4"
    rendered = render_recall_prompt(prompt)
    assert "shared non-query paper expression in their titles" in rendered


def test_evidence_expansion_rejects_new_domain_term_without_cross_paper_support() -> None:
    context = _context("query-domain-drift")
    anchor = RecallActionBatch.model_validate(_actions("graph retrieval"))
    evidence = [
        {
            "rank": 1,
            "title": "Graph retrieval for fake news classification",
            "snippets": ["Graph retrieval supports fake news classification."],
            "authors": [],
            "publication_year": 2024,
            "venue": None,
        },
        {
            "rank": 2,
            "title": "Dense graph retrieval benchmark",
            "snippets": ["Dense graph retrieval benchmark for attributed networks."],
            "authors": [],
            "publication_year": 2024,
            "venue": None,
        },
    ]

    batch, audit = validate_refinement_proposals(
        context,
        anchor,
        {
            "proposals": [
                {
                    "action_id": "drift",
                    "query_text": "graph retrieval fake news classification",
                    "expansion_kind": "paper_expression",
                    "query_support": "graph retrieval",
                    "evidence_support": [
                        {
                            "rank": 1,
                            "exact_phrase": "fake news classification",
                        }
                    ],
                }
            ]
        },
        evidence,
        max_actions=3,
        max_proposals=1,
        require_cross_paper_support=True,
    )

    assert batch == anchor
    assert audit[0]["reason"] == "unsupported_evidence_drift"


def test_v4_proposal_rejects_term_shared_only_in_abstracts() -> None:
    context = _context("query-v4-abstract-bypass")
    anchor = RecallActionBatch.model_validate(_actions("graph retrieval"))
    evidence = [
        {
            "rank": 1,
            "title": "Graph retrieval for attributed networks",
            "snippets": ["Dense representations improve graph retrieval."],
            "authors": [],
            "publication_year": 2024,
            "venue": None,
        },
        {
            "rank": 2,
            "title": "Graph retrieval benchmark evaluation",
            "snippets": ["Dense representations support graph retrieval."],
            "authors": [],
            "publication_year": 2023,
            "venue": None,
        },
    ]

    batch, audit = validate_refinement_proposals(
        context,
        anchor,
        {
            "proposals": [
                {
                    "action_id": "abstract-bypass",
                    "query_text": "dense graph retrieval",
                    "expansion_kind": "paper_expression",
                    "query_support": "graph retrieval",
                    "evidence_support": [
                        {"rank": 1, "exact_phrase": "Dense representations"},
                        {"rank": 2, "exact_phrase": "Dense representations"},
                    ],
                }
            ]
        },
        evidence,
        max_actions=3,
        max_proposals=1,
        require_cross_paper_support=True,
        require_cross_title_support=True,
    )

    assert batch == anchor
    assert audit[0]["reason"] == "unsupported_title_expression"


def test_short_query_without_evidence_skips_optional_second_llm_call() -> None:
    anchor_output = {
        "actions": [
            {
                "action_id": "anchor",
                "action_type": "text_search",
                "strategy": "evidence-steered:anchor",
                "payload": {"query_text": "SWAG"},
            }
        ]
    }
    backend = _RecordingLLMBackend([_backend_result(anchor_output)])
    generator = EvidenceSteeredDeepSeekGenerator(
        backend=backend,
        prompt=_prompt(),
        visibility="blind",
        allowed_actions={"text_search"},
        max_actions=3,
    )
    context = RecallGenerationContext(
        query_id="query-short",
        original_query="SWAG",
        query_spec=QuerySpec(original_query="SWAG", research_goal="SWAG"),
    )

    anchor = asyncio.run(generator.generate(context))
    result = asyncio.run(
        generator.refine(
            context,
            anchor,
            [
                RetrievalActionResult(
                    action_id="anchor", action_type="text_search", hits=[]
                )
            ],
        )
    )

    assert result.action_batch == anchor.action_batch
    assert [kind for kind, _ in backend.calls] == ["initial"]
    assert result.provenance["refinement_mode"] == "anchor_only"


def test_query_complement_allows_hyphen_to_space_grammatical_variant() -> None:
    context = RecallGenerationContext(
        query_id="query-domain-shift",
        original_query="post-hoc calibration in domain-shift scenarios",
        query_spec=QuerySpec(
            original_query="post-hoc calibration in domain-shift scenarios",
            research_goal="post-hoc calibration in domain-shift scenarios",
        ),
    )
    anchor = RecallActionBatch.model_validate(
        {
            "actions": [
                {
                    "action_id": "anchor",
                    "action_type": "text_search",
                    "strategy": "evidence-steered:anchor",
                    "payload": {"query_text": "post-hoc calibration domain-shift"},
                }
            ]
        }
    )

    batch, audit = validate_refinement_proposals(
        context,
        anchor,
        {
            "proposals": [
                {
                    "action_id": "complement",
                    "query_text": "post hoc calibration domain shift",
                    "expansion_kind": "query_complement",
                    "query_support": "post-hoc calibration in domain-shift scenarios",
                    "evidence_support": [],
                }
            ]
        },
        [],
        max_actions=3,
    )

    assert [action.payload.query_text for action in batch.actions] == [
        "post-hoc calibration domain-shift",
        "post hoc calibration domain shift",
    ]
    assert audit[0]["decision"] == "accepted"


def _error(code: str) -> LLMBackendResult:
    return LLMBackendResult(
        errors=[ErrorDetail(code=code, message="sealed fake failure", retryable=False, provider="fake")],
        infrastructure_failure=code != "invalid_json",
        repairable=code == "invalid_json",
    )


def _assert_no_identifier_material(value: object) -> None:
    forbidden = ("doi", "canonical_id", "openalex_id", "semantic_scholar_id", "request_id", "url")
    if isinstance(value, dict):
        assert not any(any(item in key.casefold() for item in forbidden) for key in value)
        for child in value.values():
            _assert_no_identifier_material(child)
    elif isinstance(value, list):
        for child in value:
            _assert_no_identifier_material(child)
    elif isinstance(value, str):
        assert "10.1234/" not in value
        assert "https://" not in value


def test_oracle_payload_contains_only_safe_generation_context_and_keeps_ordinary_prose() -> None:
    payload = build_generation_payload(
        _oracle_context(), visibility="oracle", allowed_actions={"text_search"}, max_actions=1
    )

    assert set(payload) == {"query", "seed_queries", "seed_candidates", "allowed_action_schema", "gold_documents"}
    gold = payload["gold_documents"]
    assert isinstance(gold, list)
    assert gold[0]["title"] == "OpenAlex and Semantic Scholar coverage study"
    assert "OpenAlex" in gold[0]["abstract"]
    _assert_no_identifier_material(payload)


def test_blind_payload_has_no_gold_documents_at_any_nesting_level() -> None:
    payload = build_generation_payload(
        _oracle_context(), visibility="blind", allowed_actions={"text_search"}, max_actions=1
    )

    def _keys(value: object) -> list[str]:
        if isinstance(value, dict):
            return list(value) + [key for child in value.values() for key in _keys(child)]
        if isinstance(value, list):
            return [key for child in value for key in _keys(child)]
        return []

    assert "gold_documents" not in _keys(payload)


def test_historical_visibility_is_preserved_without_an_oracle_upgrade() -> None:
    payload = build_generation_payload(
        _oracle_context(), visibility="historical", historical_visibility="blind", allowed_actions={"text_search"}, max_actions=1
    )

    assert "gold_documents" not in payload


def test_rendered_recall_prompt_is_deterministic_and_locks_deepseek_settings() -> None:
    message = render_recall_prompt(_prompt())

    assert "RecallActionBatch" in message
    assert "deepseek-v4-flash" in message
    assert "temperature 0" in message
    assert "text_search" in message
    assert "supplied seed_canonical_id" in message


def test_rendered_prompt_declares_exact_action_payload_contracts() -> None:
    message = render_recall_prompt(_prompt())

    assert "action_id must be a unique non-empty JSON string" in message
    assert "actions must be a JSON array of action objects" in message
    assert (
        "action_type must be a JSON string exactly equal to one supplied "
        "allowed_action_types value" in message
    )
    assert "strategy must be a non-empty JSON string" in message
    assert (
        'text_search payload keys: exactly ["query_text"]; query_text must be a '
        "non-empty JSON string of at most 300 characters" in message
    )
    assert (
        'title_search payload keys: exactly ["title_text"]; title_text must be a '
        "non-empty JSON string of at most 300 characters" in message
    )
    assert (
        'citation_expand payload keys: exactly '
        '["seed_canonical_id", "direction", "limit"]' in message
    )
    assert "seed_canonical_id must copy a supplied seed verbatim" in message
    assert 'direction must be exactly "references", "citations", or "both"' in message
    assert "limit must be a positive JSON integer, never a boolean" in message
    assert "Do not add any unlisted keys to an action or payload" in message
    assert "Do not add limit to text_search or title_search payloads" in message


@pytest.mark.parametrize(
    ("invalid", "expected_code", "context"),
    [
        ({"actions": [{**_actions()["actions"][0]}, {**_actions()["actions"][0]}]}, "duplicate_action", _context("query-1")),
        ({"actions": [{**_actions()["actions"][0], "payload": {"query_text": "  "}}]}, "empty_action", _context("query-1")),
        ({"actions": [{**_actions()["actions"][0], "payload": {"query_text": "x" * 301}}]}, "action_too_long", _context("query-1")),
        ({"actions": [{**_actions()["actions"][0], "action_type": "title_search", "payload": {"title_text": "title"}}]}, "disallowed_action_type", _context("query-1")),
        ({"actions": [{**_actions()["actions"][0], "payload": {"query_text": "graph 2018"}}]}, "year_conflict", _oracle_context(year_from=2020)),
        ({"actions": [{"action_id": "search-1", "action_type": "text_search", "strategy": "fixed", "payload": {}}]}, "missing_required_field", _context("query-1")),
        ({"actions": [{"action_id": "cite-1", "action_type": "citation_expand", "strategy": "fixed", "payload": {"seed_canonical_id": "unknown", "direction": "references", "limit": 1}}]}, "unknown_seed_candidate", _context("query-1")),
        ({"actions": [_actions()["actions"][0], {"action_id": "search-2", "action_type": "text_search", "strategy": "fixed", "payload": {"query_text": "second"}}]}, "action_limit_exceeded", _context("query-1")),
    ],
)
def test_structured_validation_failure_triggers_one_repair(
    invalid: dict[str, object], expected_code: str, context: RecallGenerationContext
) -> None:
    backend = _RecordingLLMBackend([_backend_result(invalid), _backend_result(_actions())])
    generator = DeepSeekPromptGenerator(
        backend=backend,
        prompt=_prompt(),
        visibility="blind",
        allowed_actions={"text_search", "citation_expand"},
        max_actions=2 if expected_code == "duplicate_action" else 1,
    )

    result = asyncio.run(generator.generate(context))

    assert result.action_batch.actions[0].action_id == "search-1"
    assert [kind for kind, _ in backend.calls] == ["initial", "repair"]
    repair_request = backend.calls[1][1]
    repair_payload = getattr(repair_request, "payload")
    assert repair_payload["validation_errors"][0]["code"] == expected_code
    assert repair_payload["validation_errors"][0]["message"]
    assert repair_payload["validation_errors"][0]["repair_instruction"]
    assert repair_payload["allowed_change_scope"] == ["actions"]
    assert repair_payload["repair_instruction"] == (
        "Correct every listed validation error; preserve only valid action content; "
        "return the complete corrected RecallActionBatch JSON object."
    )


def test_analyzer_invalid_json_triggers_one_repair_with_previous_output() -> None:
    backend = _RecordingLLMBackend([_error("invalid_json"), _backend_result(_actions())])
    generator = DeepSeekPromptGenerator(
        backend=backend, prompt=_prompt(), visibility="blind", allowed_actions={"text_search"}, max_actions=1
    )

    result = asyncio.run(generator.generate(_context("query-1")))

    assert result.action_batch.actions[0].action_id == "search-1"
    assert [receipt.call_kind for receipt in result.call_receipts] == ["initial", "repair"]
    assert result.repair_count == 1
    assert len({receipt.provenance["backend_call_id"] for receipt in result.call_receipts}) == 2
    assert [kind for kind, _ in backend.calls] == ["initial", "repair"]
    repair_payload = getattr(backend.calls[1][1], "payload")
    assert repair_payload["validation_errors"] == [
        {
            "code": "invalid_json",
            "field_path": "",
            "message": "invalid JSON",
            "repair_instruction": "Return one valid JSON object with only the top-level actions key.",
        }
    ]
    assert repair_payload["previous_output"] == {}


def test_repair_feedback_explains_action_id_type_and_forbidden_payload_fields() -> None:
    invalid = {
        "actions": [
            {
                "action_id": 1,
                "action_type": "text_search",
                "strategy": "synthetic",
                "payload": {"query_text": "dataset distillation", "limit": 10},
            }
        ]
    }
    backend = _RecordingLLMBackend([_backend_result(invalid), _backend_result(_actions())])
    generator = DeepSeekPromptGenerator(
        backend=backend,
        prompt=_prompt(),
        visibility="blind",
        allowed_actions={"text_search"},
        max_actions=1,
    )

    asyncio.run(generator.generate(_context("query-1")))

    repair_payload = getattr(backend.calls[1][1], "payload")
    errors = repair_payload["validation_errors"]
    assert any(
        item["field_path"].endswith("action_id")
        and "valid string" in item["message"]
        and "non-empty JSON string" in item["repair_instruction"]
        for item in errors
    )
    assert any(
        item["field_path"].endswith("payload.limit")
        and "Extra inputs" in item["message"]
        and "Remove this unlisted field" in item["repair_instruction"]
        for item in errors
    )


@pytest.mark.parametrize(
    ("invalid_payload", "instruction_fragment"),
    [
        ({"query_text": ""}, "non-empty JSON string"),
        ({"query_text": "x" * 301}, "300 characters or fewer"),
    ],
)
def test_repair_feedback_gives_field_specific_text_correction(
    invalid_payload: dict[str, object], instruction_fragment: str
) -> None:
    invalid = {
        "actions": [
            {
                "action_id": "a-1",
                "action_type": "text_search",
                "strategy": "synthetic",
                "payload": invalid_payload,
            }
        ]
    }
    backend = _RecordingLLMBackend([_backend_result(invalid), _backend_result(_actions())])
    generator = DeepSeekPromptGenerator(
        backend=backend,
        prompt=_prompt(),
        visibility="blind",
        allowed_actions={"text_search"},
        max_actions=1,
    )

    asyncio.run(generator.generate(_context("query-1")))

    errors = getattr(backend.calls[1][1], "payload")["validation_errors"]
    assert instruction_fragment in errors[0]["repair_instruction"]


def test_initial_and_repair_requests_carry_the_same_rendered_prompt_and_exact_source_identity() -> None:
    source = b"# retained comment\nname: recall_scheme_b\nversion: recall-scheme-b-v1\nmodel: deepseek-v4-flash\ntemperature: 0\ninstructions:\n  - Generate only permitted retrieval actions.\n"
    prompt = RecallPromptArtifact.from_yaml_bytes(source)
    backend = _RecordingLLMBackend([_backend_result({"actions": [{}]}), _backend_result(_actions())])
    generator = DeepSeekPromptGenerator(
        backend=backend, prompt=prompt, visibility="blind", allowed_actions={"text_search"}, max_actions=1
    )

    asyncio.run(generator.generate(_context("query-1")))

    initial_request = backend.calls[0][1]
    repair_request = backend.calls[1][1]
    expected_digest = "sha256:" + hashlib.sha256(source).hexdigest()
    assert getattr(initial_request, "prompt_instructions") == render_recall_prompt(prompt)
    assert getattr(repair_request, "prompt_instructions") == render_recall_prompt(prompt)
    assert getattr(initial_request, "prompt_bytes") == source
    assert getattr(repair_request, "prompt_bytes") == source
    assert getattr(initial_request, "prompt_artifact_sha256") == expected_digest
    assert getattr(repair_request, "prompt_artifact_sha256") == expected_digest


class _ReservationRecordingAnalyzer:
    def __init__(self) -> None:
        self.results = iter([{"actions": [{}]}, _actions()])
        self.calls: list[tuple[object, str | None, str | None]] = []

    async def generate_json(
        self,
        *,
        prompt_name: str,
        payload: dict[str, object],
        reservation: object,
        prompt_instructions: str | None = None,
        prompt_artifact_sha256: str | None = None,
    ) -> ProviderResult[dict[str, object]]:
        assert prompt_name == "recall_scheme_b"
        assert payload
        self.calls.append((reservation, prompt_instructions, prompt_artifact_sha256))
        return ProviderResult(
            data=next(self.results),
            usage=UsageActual(llm_calls=1, cost_cny=Decimal("0.01")),
            provenance={
                "provider": "sealed-fake",
                "endpoint": "/sealed",
                "model_id": "fake-model",
                "requested_at": datetime.now(UTC).isoformat(),
                "response_hash": "sha256:sealed",
            },
            cache_hit=True,
            latency_ms=0,
            errors=[],
        )


def test_initial_and_repair_use_distinct_budget_reservations_with_the_exact_prompt_identity() -> None:
    analyzer = _ReservationRecordingAnalyzer()
    controller = HardBudgetController(
        SearchBudget(
            max_search_api_calls=1,
            target_search_api_calls=1,
            max_llm_calls=2,
            target_llm_calls=2,
            max_total_tokens=10,
            max_cost_cny=Decimal("1"),
            max_elapsed_seconds=60,
            soft_deadline_seconds=10,
        )
    )
    backend = BudgetedLLMBackend(
        analyzer=analyzer,
        controller=controller,
        initial_estimate=UsageEstimate(llm_calls=1, cost_cny=Decimal("0.01")),
        repair_estimate=UsageEstimate(llm_calls=1, cost_cny=Decimal("0.01")),
    )
    prompt = _prompt()
    generator = DeepSeekPromptGenerator(
        backend=backend, prompt=prompt, visibility="blind", allowed_actions={"text_search"}, max_actions=1
    )

    asyncio.run(generator.generate(_context("query-1")))

    assert len(analyzer.calls) == 2
    assert analyzer.calls[0][0].reservation_id != analyzer.calls[1][0].reservation_id
    assert [item[1] for item in analyzer.calls] == [render_recall_prompt(prompt)] * 2
    assert [item[2] for item in analyzer.calls] == [prompt.sha256] * 2
    assert [item["action"] for item in controller.export_state()["terminal_outcomes"]] == [
        "recall.generate.initial",
        "recall.generate.repair",
    ]


def test_prompt_digest_is_the_exact_yaml_bytes_digest_and_matches_the_recipe_loader() -> None:
    loaded = load_recall_recipe(Path("configs/recall_experiments/methods/scheme-b-blind.yaml"))
    assert loaded.prompt_bytes is not None
    prompt = RecallPromptArtifact.from_yaml_bytes(loaded.prompt_bytes)
    changed_whitespace = RecallPromptArtifact.from_yaml_bytes(loaded.prompt_bytes + b"# changed\n")

    assert prompt.sha256 == loaded.prompt_sha256
    assert changed_whitespace.sha256 != prompt.sha256


@pytest.mark.parametrize("forbidden", ["10.1234/seed", "https://example.invalid/paper", "W12345678", "S2:123"])
def test_seed_prose_identifier_patterns_are_rejected_after_identifier_fields_are_removed(forbidden: str) -> None:
    context = _context("query-1").model_copy(
        update={
            "seed_candidates": [
                SeedCandidate(
                    paper=Paper(
                        canonical_id="seed-1",
                        title=f"Ordinary {forbidden}",
                        abstract="OpenAlex remains ordinary prose.",
                        sources=["openalex"],
                    )
                )
            ]
        }
    )

    with pytest.raises(ValueError, match="forbidden identifier pattern"):
        build_generation_payload(
            context, visibility="blind", allowed_actions={"text_search"}, max_actions=1
        )


def test_seed_prose_can_mention_openalex_without_an_identifier() -> None:
    context = _context("query-1").model_copy(
        update={
            "seed_candidates": [
                SeedCandidate(
                    paper=Paper(
                        canonical_id="seed-1",
                        title="OpenAlex coverage study",
                        abstract="Semantic Scholar is discussed as ordinary prose.",
                        sources=["openalex"],
                    )
                )
            ]
        }
    )

    payload = build_generation_payload(
        context, visibility="blind", allowed_actions={"text_search"}, max_actions=1
    )

    assert payload["seed_candidates"][0]["title"] == "OpenAlex coverage study"


def test_citation_seed_id_is_exposed_for_a_valid_generated_action() -> None:
    context = _context("query-1").model_copy(
        update={
            "seed_candidates": [
                SeedCandidate(
                    paper=Paper(
                        canonical_id="seed-1",
                        title="Frozen citation seed",
                        sources=["semantic_scholar"],
                    )
                )
            ]
        }
    )
    payload = build_generation_payload(
        context, visibility="blind", allowed_actions={"citation_expand"}, max_actions=1
    )

    assert payload["seed_candidates"][0]["seed_canonical_id"] == "seed-1"


def test_second_invalid_output_is_a_generation_failure() -> None:
    backend = _RecordingLLMBackend(
        [_backend_result({"actions": [{}]}), _backend_result({"actions": [{}]})]
    )
    generator = DeepSeekPromptGenerator(
        backend=backend, prompt=_prompt(), visibility="blind", allowed_actions={"text_search"}, max_actions=1
    )

    with pytest.raises(RecallGenerationFailure, match="generation_failure") as caught:
        asyncio.run(generator.generate(_context("query-1")))
    assert caught.value.code == "generation_failure"
    assert [kind for kind, _ in backend.calls] == ["initial", "repair"]


@pytest.mark.parametrize(
    "code", ["authentication_error", "rate_limited", "network_error", "snapshot_unavailable", "accounting_failure"]
)
def test_infrastructure_failures_never_consume_a_semantic_repair(code: str) -> None:
    backend = _RecordingLLMBackend([_error(code)])
    generator = DeepSeekPromptGenerator(
        backend=backend, prompt=_prompt(), visibility="blind", allowed_actions={"text_search"}, max_actions=1
    )

    with pytest.raises(RecallGenerationFailure, match="infrastructure_failure") as caught:
        asyncio.run(generator.generate(_context("query-1")))
    assert caught.value.code == "infrastructure_failure"
    assert [kind for kind, _ in backend.calls] == ["initial"]


def test_repair_payload_is_limited_to_local_validation_diagnostics() -> None:
    from paper_search.recall_experiments.validation import ActionValidationFailure, ActionValidationIssue

    payload = build_repair_payload(
        ActionValidationFailure(
            [ActionValidationIssue(code="invalid_json", field_path="actions", message="bad")],
            previous_output={"actions": "bad"},
        )
    )

    assert payload == {
        "previous_output": {"actions": "bad"},
        "validation_errors": [
            {
                "code": "invalid_json",
                "field_path": "actions",
                "message": "bad",
                "repair_instruction": "Replace actions with a JSON array of valid action objects.",
            }
        ],
        "allowed_change_scope": ["actions"],
        "repair_instruction": (
            "Correct every listed validation error; preserve only valid action content; "
            "return the complete corrected RecallActionBatch JSON object."
        ),
    }


def test_repair_payload_maps_multiple_citation_errors_to_their_own_fields() -> None:
    from paper_search.recall_experiments.validation import (
        ActionValidationFailure,
        ActionValidationIssue,
    )

    payload = build_repair_payload(
        ActionValidationFailure(
            [
                ActionValidationIssue(
                    code="invalid_json",
                    field_path="actions.0.0.citation_expand.payload.direction",
                    message="Input should be 'references', 'citations' or 'both'",
                ),
                ActionValidationIssue(
                    code="invalid_json",
                    field_path="actions.0.0.citation_expand.payload.limit",
                    message="Input should be a valid integer",
                ),
            ],
            previous_output={"actions": []},
        )
    )

    errors = payload["validation_errors"]
    assert errors[0]["field_path"].endswith("payload.direction")
    assert '"references", "citations", or "both"' in errors[0]["repair_instruction"]
    assert errors[1]["field_path"].endswith("payload.limit")
    assert "positive JSON integer" in errors[1]["repair_instruction"]


@pytest.mark.parametrize(
    ("issue", "expected_instruction"),
    [
        (
            ActionValidationIssue(
                code="disallowed_action_type",
                field_path="actions.0.action_type",
                message="action type is not allowed",
            ),
            "Delete the entire action containing this disallowed action_type.",
        ),
        (
            ActionValidationIssue(
                code="unknown_seed_candidate",
                field_path="actions.0.payload.seed_canonical_id",
                message="citation seed must be present in context seed_candidates",
            ),
            "Delete the entire citation_expand action containing this unknown seed.",
        ),
        (
            ActionValidationIssue(
                code="year_conflict",
                field_path="actions.0.payload.query_text",
                message="search text conflicts with query year constraints",
            ),
            "Remove every explicit year from this search text.",
        ),
    ],
)
def test_repair_instructions_are_executable_without_initial_request_context(
    issue: ActionValidationIssue, expected_instruction: str
) -> None:
    payload = build_repair_payload(
        ActionValidationFailure([issue], previous_output={"actions": []})
    )

    assert set(payload) == {
        "previous_output",
        "validation_errors",
        "allowed_change_scope",
        "repair_instruction",
    }
    assert payload["validation_errors"][0]["repair_instruction"] == expected_instruction


def test_title_informed_prompt_recipe_is_a_blind_four_action_text_search_method() -> None:
    loaded = load_recall_recipe(
        Path("configs/recall_experiments/methods/title-informed-blind-live.yaml")
    )

    assert loaded.recipe.method_id == "title-informed-blind"
    assert loaded.recipe.generator.type == "deepseek_prompt"
    assert loaded.recipe.generator.gold_visibility == "blind"
    assert loaded.recipe.generator.max_generated_actions == 4
    assert loaded.recipe.retrieval.allowed_actions == ["text_search"]
    assert loaded.recipe.retrieval.backend == "live_provider"
    assert loaded.recipe.retrieval.max_total_actions == 4
    assert loaded.prompt_bytes is not None
    prompt = RecallPromptArtifact.from_yaml_bytes(loaded.prompt_bytes)
    rendered = render_recall_prompt(prompt)
    assert "query.original_query is the fixed content insertion point" in rendered
    assert "anchor_full" in rendered
    assert "subject_task" in rendered
    assert "method_task" in rendered
    assert "dataset_task" in rendered
    assert "Omit a combination slot" in rendered
    assert "Do not use Gold" in rendered


@pytest.mark.parametrize(
    ("recipe_name", "method_id", "prompt_version"),
    [
        (
            "academic-bridge-blind-live.yaml",
            "academic-bridge-blind",
            "recall-academic-bridge-blind-v1",
        ),
        (
            "open-evidence-blind-live.yaml",
            "open-evidence-blind",
            "recall-open-evidence-blind-v1",
        ),
    ],
)
def test_query_expression_canary_recipes_are_blind_three_action_text_search_methods(
    recipe_name: str, method_id: str, prompt_version: str
) -> None:
    loaded = load_recall_recipe(
        Path("configs/recall_experiments/methods") / recipe_name
    )

    assert loaded.recipe.method_id == method_id
    assert loaded.recipe.generator.type == "deepseek_prompt"
    assert loaded.recipe.generator.model == "deepseek-v4-flash"
    assert loaded.recipe.generator.temperature == 0
    assert loaded.recipe.generator.gold_visibility == "blind"
    assert loaded.recipe.generator.max_generated_actions == 3
    assert loaded.recipe.generator.repair_attempts == 1
    assert loaded.recipe.retrieval.allowed_actions == ["text_search"]
    assert loaded.recipe.retrieval.backend == "live_provider"
    assert loaded.recipe.retrieval.max_results_per_action == 50
    assert loaded.recipe.retrieval.max_total_actions == 3
    assert loaded.prompt_bytes is not None
    prompt = RecallPromptArtifact.from_yaml_bytes(loaded.prompt_bytes)
    assert prompt.version == prompt_version


def test_open_evidence_prompt_locks_the_approved_generation_method() -> None:
    loaded = load_recall_recipe(
        Path("configs/recall_experiments/methods/open-evidence-blind-live.yaml")
    )
    assert loaded.prompt_bytes is not None
    rendered = render_recall_prompt(RecallPromptArtifact.from_yaml_bytes(loaded.prompt_bytes))

    for required_rule in (
        "open evidence profile",
        "protected anchors",
        "explicit, normalized, or cautiously inferred",
        "open-evidence:anchor",
        "open-evidence:scholarly-bridge",
        "open-evidence:complement",
        "Do not fill all three slots",
        "concept drift",
        "weak generic terms",
        "Do not use Gold",
    ):
        assert required_rule in rendered


def test_academic_bridge_control_does_not_contain_treatment_only_rules() -> None:
    loaded = load_recall_recipe(
        Path("configs/recall_experiments/methods/academic-bridge-blind-live.yaml")
    )
    assert loaded.prompt_bytes is not None
    rendered = render_recall_prompt(RecallPromptArtifact.from_yaml_bytes(loaded.prompt_bytes))

    assert "preserve exact technical terms" in rendered
    assert "paper-common terminology" in rendered
    assert "open evidence profile" not in rendered
    assert "open-evidence:complement" not in rendered


@pytest.mark.parametrize(
    "method_name", ["academic-bridge-blind", "open-evidence-blind"]
)
def test_query_expression_canary_replay_recipe_changes_only_the_backend(
    method_name: str,
) -> None:
    live = load_recall_recipe(
        Path(f"configs/recall_experiments/methods/{method_name}-live.yaml")
    )
    replay = load_recall_recipe(
        Path(f"configs/recall_experiments/methods/{method_name}.yaml")
    )

    live_payload = live.recipe.model_dump(mode="json")
    replay_payload = replay.recipe.model_dump(mode="json")
    assert live_payload["retrieval"].pop("backend") == "live_provider"
    assert replay_payload["retrieval"].pop("backend") == "snapshot_replay"
    assert replay_payload == live_payload
    assert replay.prompt_bytes == live.prompt_bytes
    assert replay.prompt_sha256 == live.prompt_sha256


@pytest.mark.parametrize("mode", ["oracle", "blind"])
def test_scheme_b_exploration_recipes_lock_the_deepseek_generation_recipe(mode: str) -> None:
    loaded = load_recall_recipe(Path(f"configs/recall_experiments/methods/scheme-b-{mode}.yaml"))

    assert loaded.recipe.method_id == f"scheme-b-{mode}"
    assert loaded.recipe.generator.type == "deepseek_prompt"
    assert loaded.recipe.generator.model == "deepseek-v4-flash"
    assert loaded.recipe.generator.temperature == 0
    assert loaded.recipe.generator.repair_attempts == 1
    assert loaded.recipe.generator.gold_visibility == mode
    assert loaded.prompt_bytes is not None
    assert loaded.prompt_sha256 is not None
