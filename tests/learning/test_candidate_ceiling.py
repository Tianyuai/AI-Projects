from __future__ import annotations

import asyncio

import pytest

from paper_search.learning.candidate_ceiling import (
    Core4SemanticBooleanQueryGenerator,
    FullCandidatePoolQueryGenerator,
    freeze_ceiling_batch,
    select_ceiling_batch,
)
from paper_search.query.parser import rule_fallback
from paper_search.domain.models import Paper
from paper_search.recall_experiments.contracts import (
    RecallGenerationContext,
    RetrievalActionResult,
)
from paper_search.recall_experiments.validation import validate_action_batch


def _rows(count: int = 90) -> list[dict[str, object]]:
    return [
        {
            "dataset": "pasa",
            "split": "auto_train",
            "role": "training",
            "query_id": f"q-{index:03d}",
            "query": f"query {index}",
            "gold_paper_ids": [f"arxiv:2001.{index:05d}"],
        }
        for index in range(count)
    ]


def test_three_ceiling_batches_are_deterministic_and_disjoint() -> None:
    rows = _rows()
    batches = [
        select_ceiling_batch(
            rows,
            batch_size=10,
            batch_index=index,
            batch_count=3,
        )
        for index in range(3)
    ]

    ids = [{str(row["query_id"]) for row in batch} for batch in batches]
    assert all(len(batch) == 10 for batch in batches)
    assert not ids[0].intersection(ids[1])
    assert not ids[0].intersection(ids[2])
    assert not ids[1].intersection(ids[2])
    assert batches[1] == select_ceiling_batch(
        rows,
        batch_size=10,
        batch_index=1,
        batch_count=3,
    )


def test_ceiling_batch_selection_rejects_invalid_or_oversized_requests() -> None:
    with pytest.raises(ValueError, match="batch index"):
        select_ceiling_batch(_rows(), batch_size=10, batch_index=3, batch_count=3)
    with pytest.raises(ValueError, match="exceeds"):
        select_ceiling_batch(_rows(20), batch_size=10, batch_index=0, batch_count=3)


def test_ceiling_batches_exclude_all_pilot_query_ids_before_sampling() -> None:
    rows = _rows()
    excluded = {"q-000", "q-018", "q-036", "q-054", "q-072"}

    batches = [
        select_ceiling_batch(
            rows,
            batch_size=10,
            batch_index=index,
            batch_count=3,
            excluded_query_ids=excluded,
        )
        for index in range(3)
    ]

    selected = {str(row["query_id"]) for batch in batches for row in batch}
    assert not selected.intersection(excluded)
    assert len(selected) == 30


def test_ceiling_batch_freeze_is_immutable_and_hash_bound(tmp_path) -> None:
    batch = select_ceiling_batch(
        _rows(), batch_size=10, batch_index=0, batch_count=3
    )
    path = tmp_path / "ceiling-fold-1.jsonl"

    first = freeze_ceiling_batch(batch, path)
    second = freeze_ceiling_batch(batch, path)

    assert first == second
    assert first.startswith("sha256:")
    assert len(path.read_text(encoding="utf-8").splitlines()) == 10


def test_full_candidate_generator_emits_every_gold_blind_candidate() -> None:
    query = "Which paper proposed graph diffusion networks for retrieval?"
    context = RecallGenerationContext(
        query_id="q-ceiling",
        original_query=query,
        query_spec=rule_fallback(query),
    )
    generator = FullCandidatePoolQueryGenerator(max_candidates=12)

    result = asyncio.run(generator.generate(context))

    assert 3 <= len(result.action_batch.actions) <= 12
    assert result.provenance["collection_mode"] == "candidate_ceiling"
    assert result.provenance["gold_visibility"] == "blind"
    assert int(result.provenance["candidate_pool_size"]) == len(
        result.action_batch.actions
    )
    assert result.action_batch.actions[0].action_id == "ceiling-candidate-anchor"
    assert all(
        action.action_id.startswith("ceiling-candidate-")
        for action in result.action_batch.actions
    )
    assert result.call_receipts == []
    assert generator.source_sha256.startswith("sha256:")


def test_v2_candidate_pool_emits_semantic_and_boolean_phrase_families() -> None:
    query = (
        "Which paper proposed graph diffusion networks for information retrieval "
        "without supervised fine tuning?"
    )
    context = RecallGenerationContext(
        query_id="q-candidate-families",
        original_query=query,
        query_spec=rule_fallback(query),
    )

    result = asyncio.run(
        FullCandidatePoolQueryGenerator(max_candidates=12).generate(context)
    )

    actions = {action.action_id: action for action in result.action_batch.actions}
    semantic = actions["ceiling-candidate-semantic-original"]
    assert semantic.payload.query_text == query
    assert semantic.payload.search_mode == "semantic"
    boolean = actions["ceiling-candidate-boolean-relaxed"]
    assert " AND " in boolean.payload.query_text
    assert " OR " in boolean.payload.query_text
    assert boolean.payload.search_mode == "lexical"
    phrase = actions["ceiling-candidate-phrase-proximity"]
    assert phrase.payload.query_text.startswith('"')
    assert phrase.payload.query_text.endswith("~8")
    assert result.provenance["candidate_policy"] == "full-controlled-candidate-pool-v2"
    validated = validate_action_batch(
        result.artifact_bytes.decode("utf-8"),
        context,
        allowed_actions=("text_search", "title_search"),
        max_actions=12,
    )
    validated_semantic = next(
        action
        for action in validated.actions
        if action.action_id == "ceiling-candidate-semantic-original"
    )
    assert validated_semantic.payload.search_mode == "semantic"


def test_core4_semantic_boolean_generator_freezes_a_prime_action_identity() -> None:
    query = (
        "Which paper proposed graph diffusion networks for information retrieval "
        "without supervised fine tuning?"
    )
    context = RecallGenerationContext(
        query_id="q-a-prime",
        original_query=query,
        query_spec=rule_fallback(query),
    )

    result = asyncio.run(Core4SemanticBooleanQueryGenerator().generate(context))

    actions = result.action_batch.actions
    assert [action.action_id for action in actions] == [
        "ceiling-candidate-anchor",
        "ceiling-candidate-text-1",
        "ceiling-candidate-text-2",
        "ceiling-candidate-text-3",
        "ceiling-candidate-semantic-original",
        "ceiling-candidate-boolean-relaxed",
    ]
    assert len(actions) == 6
    assert sum(action.payload.search_mode == "semantic" for action in actions) == 1
    assert actions[0].payload.query_text == query
    assert actions[4].payload.query_text == query
    assert all("title-target" not in action.action_id for action in actions)
    assert result.provenance["candidate_policy"] == "core4-semantic-boolean-v1"
    assert result.provenance["title_target_status"] == "disabled"


def test_core4_semantic_boolean_leaves_budget_unused_without_valid_boolean() -> None:
    query = "Find graph retrieval"
    context = RecallGenerationContext(
        query_id="q-a-prime-short",
        original_query=query,
        query_spec=rule_fallback(query),
    )

    result = asyncio.run(Core4SemanticBooleanQueryGenerator().generate(context))

    actions = result.action_batch.actions
    assert len(actions) < 6
    assert actions[0].action_id == "ceiling-candidate-anchor"
    assert actions[-1].action_id == "ceiling-candidate-semantic-original"
    assert all(action.action_id != "ceiling-candidate-boolean-relaxed" for action in actions)


def test_v2_prf_adds_only_cross_paper_supported_terms() -> None:
    query = "Find graph diffusion methods for information retrieval"
    context = RecallGenerationContext(
        query_id="q-prf",
        original_query=query,
        query_spec=rule_fallback(query),
    )
    generator = FullCandidatePoolQueryGenerator(max_candidates=12)
    initial = asyncio.run(generator.generate(context))
    first_round = [
        RetrievalActionResult(
            action_id=initial.action_batch.actions[0].action_id,
            action_type="text_search",
            hits=[
                Paper(
                    canonical_id="paper-1",
                    title="Graph diffusion retrieval with spectral propagation",
                ),
                Paper(
                    canonical_id="paper-2",
                    title="Spectral propagation for graph based retrieval",
                ),
                Paper(
                    canonical_id="paper-noise",
                    title="Graph retrieval with isolated protein terminology",
                ),
            ],
        )
    ]

    refined = asyncio.run(generator.refine(context, initial, first_round))

    added = refined.action_batch.actions[len(initial.action_batch.actions) :]
    assert len(added) == 1
    assert added[0].action_id == "ceiling-candidate-prf-1"
    assert added[0].strategy == "candidate-family:prf"
    assert "spectral" in added[0].payload.query_text
    assert "propagation" in added[0].payload.query_text
    assert "protein" not in added[0].payload.query_text
    assert len(refined.action_batch.actions) <= 12


def test_v2_prf_rejects_single_paper_feedback_noise() -> None:
    query = "Find graph diffusion methods for information retrieval"
    context = RecallGenerationContext(
        query_id="q-prf-noise",
        original_query=query,
        query_spec=rule_fallback(query),
    )
    generator = FullCandidatePoolQueryGenerator(max_candidates=12)
    initial = asyncio.run(generator.generate(context))
    first_round = [
        RetrievalActionResult(
            action_id=initial.action_batch.actions[0].action_id,
            action_type="text_search",
            hits=[
                Paper(
                    canonical_id="paper-1",
                    title="Graph diffusion retrieval with isolated protein terminology",
                )
            ],
        )
    ]

    refined = asyncio.run(generator.refine(context, initial, first_round))

    assert refined.action_batch == initial.action_batch
    assert refined.provenance["prf_status"] == "insufficient_cross_paper_support"


def test_full_candidate_generator_deduplicates_canonical_search_text() -> None:
    query = (
        "Which work proposed OTOv1, a method proposed to avoid fine-tuning and "
        "perform end-to-end training and compression of the DNN once?"
    )
    context = RecallGenerationContext(
        query_id="q-duplicate-candidate",
        original_query=query,
        query_spec=rule_fallback(query),
    )

    result = asyncio.run(
        FullCandidatePoolQueryGenerator(max_candidates=12).generate(context)
    )

    search_keys = []
    for action in result.action_batch.actions:
        payload = action.payload
        text = getattr(payload, "query_text", None) or getattr(
            payload, "title_text", None
        )
        search_keys.append(
            (
                getattr(payload, "search_mode", "lexical"),
                " ".join(text.split()).casefold(),
            )
        )
    assert len(search_keys) == len(set(search_keys))
    assert "candidate-title-1" not in result.provenance["selected_candidate_ids"]


def test_full_candidate_generator_emits_validation_canonical_unicode() -> None:
    query = "Which work introduced ﬁne-tuning for retrieval?"
    context = RecallGenerationContext(
        query_id="q-unicode-canonicalization",
        original_query=query,
        query_spec=rule_fallback(query),
    )

    result = asyncio.run(
        FullCandidatePoolQueryGenerator(max_candidates=12).generate(context)
    )
    validated = validate_action_batch(
        result.artifact_bytes.decode("utf-8"),
        context,
        allowed_actions=("text_search", "title_search"),
        max_actions=12,
    )

    assert validated == result.action_batch
