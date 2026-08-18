from __future__ import annotations

from paper_search.learning.cpu_baseline import (
    HashedLogisticTermRanker,
    evaluate_probabilities,
    run_cpu_baseline_experiment,
    select_f1_threshold,
)
from paper_search.learning.weak_labels import QueryTermLabel


def _label(query_id: str, term: str, positive: bool, index: int) -> QueryTermLabel:
    return QueryTermLabel(
        dataset="pasa",
        split="auto_train",
        role="training",
        query_id=query_id,
        query=f"find {term} papers",
        action_text=term,
        label="positive" if positive else "hard_negative",
        query_term_index=index,
    )


def test_cpu_ranker_learns_repeated_positive_terms_deterministically() -> None:
    rows = [
        *[_label(f"p{i}", "diffusion", True, 1) for i in range(20)],
        *[_label(f"n{i}", "introduced", False, 1) for i in range(20)],
    ]
    first = HashedLogisticTermRanker(dimension=256, epochs=12, seed=7)
    second = HashedLogisticTermRanker(dimension=256, epochs=12, seed=7)

    first.fit(rows)
    second.fit(rows)

    probe = [_label("a", "diffusion", True, 1), _label("b", "introduced", False, 1)]
    assert first.predict_proba(probe) == second.predict_proba(probe)
    assert first.predict_proba(probe)[0] > first.predict_proba(probe)[1]


def test_threshold_selection_uses_highest_f1_with_stable_tie_break() -> None:
    labels = [True, True, False, False]
    probabilities = [0.9, 0.6, 0.55, 0.1]

    threshold = select_f1_threshold(labels, probabilities)
    metrics = evaluate_probabilities(labels, probabilities, threshold=threshold)

    assert threshold == 0.6
    assert metrics.f1 == 1.0
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0


def test_experiment_writes_model_and_beats_all_negative_on_signal(tmp_path) -> None:
    train = [
        *[_label(f"p{i}", "diffusion", True, 1) for i in range(20)],
        *[_label(f"n{i}", "introduced", False, 1) for i in range(20)],
    ]
    dev = [
        row.model_copy(update={"role": "development", "split": "auto_dev"})
        for row in (
            _label("dp", "diffusion", True, 1),
            _label("dn", "introduced", False, 1),
        )
    ]
    train_path = tmp_path / "train.jsonl"
    dev_path = tmp_path / "dev.jsonl"
    for path, rows in ((train_path, train), (dev_path, dev)):
        path.write_text(
            "".join(row.model_dump_json() + "\n" for row in rows),
            encoding="utf-8",
        )

    summary = run_cpu_baseline_experiment(
        train_path=train_path,
        development_path=dev_path,
        result_path=tmp_path / "result.json",
        model_path=tmp_path / "weights.f64",
        dimension=256,
        epochs=12,
        seed=7,
    )

    assert summary.learned.f1 > summary.all_negative.f1
    assert summary.development_count == 2
    assert (tmp_path / "result.json").is_file()
    assert (tmp_path / "weights.f64").stat().st_size == 256 * 8
