from __future__ import annotations

import hashlib

import pytest

from paper_search.learning.lexical_bridge import (
    LexicalBridgeExample,
    SupervisedLexicalBridge,
)
from paper_search.learning.lexical_bridge_deployment import (
    freeze_lexical_bridge_model,
    load_lexical_bridge_model,
)


def _sha256(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _bridge() -> SupervisedLexicalBridge:
    return SupervisedLexicalBridge.fit(
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
        learning_objective="neighbor_idf",
    )


def test_frozen_lexical_bridge_round_trips_with_precommitted_configuration(
    tmp_path,
) -> None:
    model_path = tmp_path / "lexical-bridge.joblib"
    manifest_path = tmp_path / "lexical-bridge.json"

    manifest = freeze_lexical_bridge_model(
        _bridge(),
        model_path=model_path,
        manifest_path=manifest_path,
        training_query_count=3,
        raw_train_sha256=_sha256(b"raw-train"),
        train_partition_sha256=_sha256(b"train-partition"),
        training_oof_sha256=_sha256(b"training-oof"),
        independent_dev_sha256=_sha256(b"independent-dev"),
    )
    loaded = load_lexical_bridge_model(
        model_path=model_path,
        manifest_path=manifest_path,
    )

    assert manifest["configuration"] == {
        "learning_objective": "neighbor_idf",
        "max_expansion_terms": 6,
        "min_neighbor_support": 2,
        "neighbors": 12,
        "representation": "word_char",
    }
    assert loaded.source_sha256 == manifest["model_sha256"]
    proposal = loaded.bridge.propose(
        "multimodality representations",
        max_expansion_terms=loaded.max_expansion_terms,
    )
    assert proposal is not None
    assert "retrieval" in proposal.expansion_terms


def test_lexical_bridge_loader_rejects_tampered_artifact(tmp_path) -> None:
    model_path = tmp_path / "lexical-bridge.joblib"
    manifest_path = tmp_path / "lexical-bridge.json"
    freeze_lexical_bridge_model(
        _bridge(),
        model_path=model_path,
        manifest_path=manifest_path,
        training_query_count=3,
        raw_train_sha256=_sha256(b"raw-train"),
        train_partition_sha256=_sha256(b"train-partition"),
        training_oof_sha256=_sha256(b"training-oof"),
        independent_dev_sha256=_sha256(b"independent-dev"),
    )
    model_path.write_bytes(b"tampered")

    with pytest.raises(ValueError, match="hash mismatch"):
        load_lexical_bridge_model(
            model_path=model_path,
            manifest_path=manifest_path,
        )


def test_lexical_bridge_serialization_state_has_canonical_term_order() -> None:
    bridge = _bridge()

    assert all(
        isinstance(terms, tuple) and terms == tuple(sorted(terms))
        for terms in bridge._title_terms
    )
    assert list(bridge._title_idf) == sorted(bridge._title_idf)
