from __future__ import annotations

from paper_search.learning.action_labels import ActionWeakLabel
from paper_search.learning.contracts import PolicyActionCandidate, QueryPolicyInput
from paper_search.learning.cpu_action_ranker import (
    CpuActionRanker,
    run_cpu_action_experiment,
)
from paper_search.query.parser import rule_fallback


def _candidate(action_id: str, text: str, origin: str = "deterministic_rule"):
    return PolicyActionCandidate(
        action_id=action_id,
        action_type="text_search",
        text=text,
        origin=origin,
        provider_hint="either",
    )


def _request() -> QueryPolicyInput:
    query = "Which paper proposed graph diffusion?"
    return QueryPolicyInput(
        query_id="probe",
        original_query=query,
        query_kind="semantic",
        query_spec=rule_fallback(query),
        allowed_action_types=["text_search", "title_search"],
        max_actions=4,
    )


def test_cpu_action_ranker_learns_and_round_trips_weights(tmp_path) -> None:
    positive = _candidate("positive", "graph diffusion")
    negative = _candidate("negative", "which paper proposed")
    rows = [
        *[
            ActionWeakLabel(
                dataset="pasa",
                split="auto_train",
                role="training",
                query_id=f"p{i}",
                query="Which paper proposed graph diffusion?",
                query_kind="semantic",
                action=positive,
                label="positive",
            )
            for i in range(20)
        ],
        *[
            ActionWeakLabel(
                dataset="pasa",
                split="auto_train",
                role="training",
                query_id=f"n{i}",
                query="Which paper proposed graph diffusion?",
                query_kind="semantic",
                action=negative,
                label="hard_negative",
            )
            for i in range(20)
        ],
    ]
    ranker = CpuActionRanker(dimension=256, epochs=12, seed=7)
    ranker.fit(rows)
    before = ranker.score(_request(), [positive, negative])
    ranker.save(tmp_path / "weights.f64")

    loaded = CpuActionRanker.load(
        tmp_path / "weights.f64",
        dimension=256,
        confidence_threshold=0.4,
    )

    assert before[0] > before[1]
    assert loaded.score(_request(), [positive, negative]) == before
    assert loaded.model_id == "cpu-action-ranker-v1"


def test_cpu_action_experiment_selects_dev_threshold_and_freezes_artifacts(
    tmp_path,
) -> None:
    positive = _candidate("positive", "graph diffusion")
    negative = _candidate("negative", "which paper proposed")
    train_rows = [
        ActionWeakLabel(
            dataset="pasa",
            split="auto_train",
            role="training",
            query_id=f"train-{index}",
            query="Which paper proposed graph diffusion?",
            query_kind="semantic",
            action=action,
            label=label,
        )
        for index, (action, label) in enumerate(
            [(positive, "positive"), (negative, "hard_negative")] * 20
        )
    ]
    dev_rows = [
        row.model_copy(
            update={
                "split": "auto_dev",
                "role": "development",
                "query_id": f"dev-{index}",
            }
        )
        for index, row in enumerate(train_rows[:4])
    ]
    train_path = tmp_path / "train.jsonl"
    dev_path = tmp_path / "dev.jsonl"
    for path, rows in ((train_path, train_rows), (dev_path, dev_rows)):
        path.write_text(
            "".join(row.model_dump_json() + "\n" for row in rows),
            encoding="utf-8",
        )

    summary = run_cpu_action_experiment(
        train_path=train_path,
        development_path=dev_path,
        model_path=tmp_path / "weights.f64",
        result_path=tmp_path / "result.json",
        dimension=256,
        epochs=12,
        seed=7,
    )

    assert summary.learned.f1 > summary.anchor_only.f1
    assert summary.learned_non_anchor.f1 > summary.anchor_non_anchor.f1
    assert summary.threshold_selected_on == "development"
    assert (tmp_path / "weights.f64").stat().st_size == 256 * 8
    assert (tmp_path / "result.json").is_file()
