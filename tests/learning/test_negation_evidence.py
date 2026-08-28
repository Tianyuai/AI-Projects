from paper_search.learning.negation_evidence import (
    classify_exclusion_stance,
    negation_topic_terms,
    negation_topic_relevant,
)


def test_exclusion_stance_ignores_terminal_query_punctuation() -> None:
    assert (
        classify_exclusion_stance(
            "We use adversarial training for robust recognition.",
            "adversarial training?",
        )
        == "conflict"
    )


def test_exclusion_stance_rejects_negated_bridge_false_positive() -> None:
    assert (
        classify_exclusion_stance(
            "Calculi equipped with priority are not replacement free.",
            "replacement",
        )
        == "unknown"
    )


def test_exclusion_stance_does_not_treat_background_convention_as_own_use() -> None:
    assert (
        classify_exclusion_stance(
            "Deep stabilization is generally formulated with the help of explicit "
            "motion estimation modules. We propose a blind method.",
            "explicit motion estimation",
        )
        == "unknown"
    )


def test_exclusion_stance_does_not_treat_later_prior_work_as_own_use() -> None:
    assert (
        classify_exclusion_stance(
            "A robust recognition model. Prior work using adversarial training is "
            "reviewed before our method is introduced.",
            "adversarial training",
        )
        == "unknown"
    )


def test_exclusion_stance_accepts_explicit_method_in_title_position() -> None:
    assert (
        classify_exclusion_stance(
            "Using adversarial training for robust recognition.",
            "adversarial training",
        )
        == "conflict"
    )


def test_exclusion_stance_rejects_noun_uses_after_unrelated_verb() -> None:
    assert (
        classify_exclusion_stance(
            "We analyze the two uses of the replacement and compare accuracy.",
            "replacement",
        )
        == "unknown"
    )


def test_exclusion_stance_accepts_direct_own_use_with_modifier() -> None:
    assert (
        classify_exclusion_stance(
            "We explicitly use adversarial training for robust recognition.",
            "adversarial training",
        )
        == "conflict"
    )


def test_negation_topic_relevance_rejects_only_generic_overlap() -> None:
    assert not negation_topic_relevant(
        "Find learning systems based on BiaSwap without demographic attributes",
        "A learning-based interview system using demographic attributes",
        ["demographic attributes"],
    )


def test_negation_topic_relevance_requires_two_substantive_terms() -> None:
    assert negation_topic_relevant(
        "Robust image recognition without adversarial training",
        "Image recognition using adversarial training",
        ["adversarial training"],
    )


def test_negation_topic_relevance_rejects_generic_academic_overlap() -> None:
    assert not negation_topic_relevant(
        "Neural tangent kernel generalization without marginal likelihood in "
        "neural architecture search",
        "Bayesian sparsification for neural networks using the marginal likelihood "
        "across different neural network architectures",
        ["marginal likelihood"],
    )
    assert not negation_topic_relevant(
        "Prediction methods for intrinsic behavioral learning without reward",
        "We used reward-based optimization for a prediction model and learning method",
        ["reward"],
    )
    assert not negation_topic_relevant(
        "First lower bounds for fixed-confidence BAI without privacy",
        "Privacy-utility trade-offs with lower and upper bounds",
        ["privacy"],
    )


def test_negation_topic_relevance_preserves_short_acronym_as_required_anchor() -> None:
    assert "il" in negation_topic_terms(
        "Hierarchical IL without planning", ["planning"]
    )
    assert not negation_topic_relevant(
        "Hierarchical IL without planning",
        "Hierarchical planning for dynamic resource allocation involving uncertainty",
        ["planning"],
    )
