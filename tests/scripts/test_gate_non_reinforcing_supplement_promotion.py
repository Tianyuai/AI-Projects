from __future__ import annotations

from scripts.gate_non_reinforcing_supplement_promotion import (
    DEFAULT_CONFIRMATION,
    DEFAULT_FIXED,
    DEFAULT_OUTPUT,
    GATE_SCHEMA_VERSION,
    GATE_SCOPE,
    _promotion_recommended,
)


def _directions(*, top50_improved: int) -> dict[str, dict[str, int]]:
    return {
        str(cutoff): {
            "improved_query_count": top50_improved if cutoff == 50 else 0,
            "worsened_query_count": 0,
            "unchanged_query_count": 128 - (top50_improved if cutoff == 50 else 0),
        }
        for cutoff in (5, 10, 20, 50)
    }


def test_gate_uses_fair_initial_order_replays() -> None:
    assert DEFAULT_FIXED == (
        "f5-topk-candidate-ab-identity-nonreinforcing-fairmerge-v1.json"
    )
    assert DEFAULT_CONFIRMATION == (
        "f5-topk-candidate-ab-identity-nonreinforcing-fairmerge-confirmation-v1.json"
    )
    assert DEFAULT_OUTPUT == (
        "nonreinforcing-supplement-fairmerge-promotion-gate-v5.json"
    )
    assert GATE_SCHEMA_VERSION == (
        "nonreinforcing-supplement-fairmerge-promotion-gate-v5"
    )
    assert GATE_SCOPE == "same-provider-supplement-fair-initial-order"


def test_promotion_requires_top50_gain_in_disjoint_current_policy_confirmation() -> None:
    development = _directions(top50_improved=5)
    confirmation = _directions(top50_improved=0)

    assert (
        _promotion_recommended(
            merge_fix_passed=True,
            development_directions=development,
            confirmation_directions=confirmation,
            independent_current_policy_confirmation=True,
        )
        is False
    )


def test_promotion_accepts_positive_disjoint_current_policy_confirmation() -> None:
    assert (
        _promotion_recommended(
            merge_fix_passed=True,
            development_directions=_directions(top50_improved=5),
            confirmation_directions=_directions(top50_improved=1),
            independent_current_policy_confirmation=True,
        )
        is True
    )
