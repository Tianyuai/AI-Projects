from paper_search.learning.lexical_bridge import LexicalBridgeExample, SupervisedLexicalBridge
from paper_search.learning.lexical_bridge_validation import (
    BridgeFoldResult,
    bridge_training_gate,
    deterministic_bridge_fold,
)


def test_bridge_transfers_only_supported_title_terms_deterministically() -> None:
    bridge = SupervisedLexicalBridge.fit(
        [
            LexicalBridgeExample(
                query="autonomous driving camera perception",
                gold_titles=("LiDAR camera fusion for autonomous driving",),
            ),
            LexicalBridgeExample(
                query="autonomous vehicle visual perception",
                gold_titles=("Robust LiDAR fusion for vehicle perception",),
            ),
            LexicalBridgeExample(
                query="protein folding energy prediction",
                gold_titles=("Protein structure prediction with energy models",),
            ),
        ]
    )

    proposal = bridge.propose(
        "autonomous vehicle camera understanding",
        neighbors=3,
        max_expansion_terms=2,
        min_neighbor_support=2,
    )

    assert proposal is not None
    assert proposal.expansion_terms == ("fusion", "lidar")
    assert proposal.query_text.endswith("fusion lidar")
    assert proposal.neighbor_support == {"fusion": 2, "lidar": 2}


def test_bridge_abstains_when_no_title_term_has_independent_support() -> None:
    bridge = SupervisedLexicalBridge.fit(
        [
            LexicalBridgeExample(query="graph learning", gold_titles=("Graph attention",)),
            LexicalBridgeExample(query="language model", gold_titles=("Token routing",)),
        ]
    )

    assert (
        bridge.propose(
            "graph model",
            neighbors=2,
            max_expansion_terms=2,
            min_neighbor_support=2,
        )
        is None
    )


def test_hybrid_representation_handles_compound_and_spacing_variants() -> None:
    bridge = SupervisedLexicalBridge.fit(
        [
            LexicalBridgeExample(
                query="multimodal representation learning",
                gold_titles=("Cross modal retrieval with aligned embeddings",),
            ),
            LexicalBridgeExample(
                query="multi modal representation alignment",
                gold_titles=("Cross-modal retrieval using joint embeddings",),
            ),
            LexicalBridgeExample(
                query="protein structure prediction",
                gold_titles=("Protein folding with geometric networks",),
            ),
        ],
        representation="word_char",
        learning_objective="association",
    )

    proposal = bridge.propose(
        "multimodality representations",
        neighbors=2,
        max_expansion_terms=4,
        min_neighbor_support=2,
    )

    assert proposal is not None
    assert {"cross", "embeddings", "retrieval"} == set(proposal.expansion_terms)
    assert "modal" not in proposal.expansion_terms


def test_frequency_calibrated_objective_does_not_reward_repeated_generic_terms() -> None:
    distinguishing_terms = (
        "quasar",
        "quasar",
        "nebula",
        "pulsar",
        "galaxy",
        "cosmos",
        "photon",
        "plasma",
        "orbit",
        "meteor",
        "comet",
        "eclipse",
    )
    examples = [
        LexicalBridgeExample(
            query="specialized alpha method",
            gold_titles=(f"Learning evidence {term}",),
        )
        for term in distinguishing_terms
    ]
    bridge = SupervisedLexicalBridge.fit(
        examples,
        learning_objective="support_normalized_idf",
    )

    proposal = bridge.propose(
        "specialized alpha method",
        neighbors=12,
        max_expansion_terms=1,
        min_neighbor_support=2,
    )

    assert proposal is not None
    assert proposal.expansion_terms == ("quasar",)
    assert proposal.neighbor_support == {"quasar": 2}


def test_deterministic_folds_and_training_gate_are_precommitted() -> None:
    assert deterministic_bridge_fold("AutoScholarQuery_train_1") == 3
    assert deterministic_bridge_fold("AutoScholarQuery_train_1") == 3
    passing = [
        BridgeFoldResult(
            fold=fold,
            query_count=30,
            improved_query_count=7,
            mean_potential_delta=0.01,
            negative_stratum_count=0,
        )
        for fold in (1, 2, 3)
    ]
    failing = [*passing[:2], passing[2].__class__(3, 30, 5, 0.01, 0)]

    assert bridge_training_gate(passing).passed is True
    assert bridge_training_gate(failing).passed is False
    assert bridge_training_gate(failing).minimum_improvement_rate == 2 / 9
