from __future__ import annotations

import numpy as np
import pytest

from paper_search.learning.anchored_fusion import (
    PRIMARY_METRICS,
    blend_anchored_family_weights,
    new_only_batch_indexes,
    new_only_query_ids,
    scale_anchored_family_weights,
    select_conservative_alpha,
    select_conservative_scale,
)


def test_blend_keeps_frozen_families_and_interpolates_only_trainable_ones() -> None:
    production = {
        "entity": np.array([1.0, 3.0]),
        "hard_constraint": np.array([2.0, 4.0]),
        "reliability": np.array([5.0, 7.0]),
        "task_provenance": np.array([10.0, 14.0]),
    }
    candidate = {
        "entity": np.array([5.0, 7.0]),
        "hard_constraint": np.array([20.0, 40.0]),
        "reliability": np.array([50.0, 70.0]),
        "task_provenance": np.array([18.0, 22.0]),
    }

    blended = blend_anchored_family_weights(
        production,
        candidate,
        alpha=0.25,
        trainable_families={"entity", "task_provenance"},
    )

    np.testing.assert_array_equal(blended["reliability"], production["reliability"])
    np.testing.assert_array_equal(
        blended["hard_constraint"], production["hard_constraint"]
    )
    np.testing.assert_allclose(blended["entity"], np.array([2.0, 4.0]))
    np.testing.assert_allclose(blended["task_provenance"], np.array([12.0, 16.0]))
    assert blended["reliability"] is not production["reliability"]


def test_blend_rejects_incompatible_artifacts() -> None:
    production = {"entity": np.zeros(2), "reliability": np.zeros(2)}
    candidate = {"entity": np.zeros(3), "reliability": np.zeros(2)}

    with pytest.raises(ValueError, match="shape"):
        blend_anchored_family_weights(
            production,
            candidate,
            alpha=0.5,
            trainable_families={"entity"},
        )


def test_scale_changes_only_the_calibrated_production_family() -> None:
    production = {
        "entity": np.array([1.0, 3.0]),
        "reliability": np.array([4.0, -2.0]),
        "task_provenance": np.array([5.0, 7.0]),
    }

    scaled = scale_anchored_family_weights(
        production,
        family="reliability",
        scale=0.25,
    )

    np.testing.assert_allclose(scaled["reliability"], np.array([1.0, -0.5]))
    np.testing.assert_array_equal(scaled["entity"], production["entity"])
    np.testing.assert_array_equal(
        scaled["task_provenance"], production["task_provenance"]
    )
    assert scaled["entity"] is not production["entity"]


def test_new_only_query_ids_requires_production_to_be_a_subset() -> None:
    assert new_only_query_ids({"q1", "q2"}, {"q1", "q2", "q3"}) == frozenset(
        {"q3"}
    )

    with pytest.raises(ValueError, match="not a subset"):
        new_only_query_ids({"q1", "missing"}, {"q1", "q2"})


def test_new_only_batch_indexes_skip_batches_with_only_production_queries() -> None:
    assert new_only_batch_indexes(
        ("old-1", "old-2", "old-3", "new-1", "new-2"),
        (2, 2, 1),
        {"new-1", "new-2"},
    ) == frozenset({1, 2})


def test_selects_smallest_alpha_that_beats_b0_and_does_not_regress_production() -> None:
    production = {metric: 0.50 for metric in PRIMARY_METRICS}
    b0 = {metric: 0.45 for metric in PRIMARY_METRICS}
    candidates = {
        0.25: {**production, "recall_at_10": 0.49, "mrr": 0.51},
        0.50: {**production, "mrr": 0.51},
        0.75: {metric: 0.52 for metric in PRIMARY_METRICS},
    }

    assert select_conservative_alpha(
        production_metrics=production,
        b0_metrics=b0,
        candidate_metrics_by_alpha=candidates,
    ) == 0.50


def test_alpha_selection_requires_a_strict_gain() -> None:
    production = {metric: 0.50 for metric in PRIMARY_METRICS}

    assert (
        select_conservative_alpha(
            production_metrics=production,
            b0_metrics={metric: 0.40 for metric in PRIMARY_METRICS},
            candidate_metrics_by_alpha={0.25: dict(production)},
        )
        is None
    )


def test_selects_largest_reliability_scale_that_clears_the_gate() -> None:
    production = {metric: 0.40 for metric in PRIMARY_METRICS}
    b0 = {metric: 0.50 for metric in PRIMARY_METRICS}
    candidates = {
        0.75: {**b0, "recall_at_50": 0.49},
        0.50: {**b0, "mrr": 0.51},
        0.25: {metric: 0.52 for metric in PRIMARY_METRICS},
        0.00: {metric: 0.53 for metric in PRIMARY_METRICS},
    }

    assert select_conservative_scale(
        production_metrics=production,
        b0_metrics=b0,
        candidate_metrics_by_scale=candidates,
    ) == 0.50
