from paper_search.learning.query_term_pairwise_ranker import (
    QueryTermCandidate,
    QueryTermPairwiseRanker,
    build_query_term_candidates,
)
from paper_search.learning.lexical_bridge import (
    LexicalBridgeExample,
    SupervisedLexicalBridge,
)


def _candidate(
    query_id: str,
    query: str,
    term: str,
    *,
    relevant: bool,
) -> QueryTermCandidate:
    return QueryTermCandidate(
        query_id=query_id,
        query=query,
        term=term,
        relevant=relevant,
        neighbor_support=1,
        similarity_sum=0.8,
        maximum_similarity=0.8,
        title_idf=2.0,
    )


def test_pairwise_term_ranker_learns_query_term_compatibility() -> None:
    rows = []
    for index in range(12):
        rows.extend(
            [
                _candidate(
                    f"graph-{index}",
                    "graph node centrality",
                    "pagerank",
                    relevant=True,
                ),
                _candidate(
                    f"graph-{index}",
                    "graph node centrality",
                    "translation",
                    relevant=False,
                ),
                _candidate(
                    f"language-{index}",
                    "language sequence translation",
                    "translation",
                    relevant=True,
                ),
                _candidate(
                    f"language-{index}",
                    "language sequence translation",
                    "pagerank",
                    relevant=False,
                ),
            ]
        )
    ranker = QueryTermPairwiseRanker(dimension=1024, epochs=8, seed=7)

    pair_count = ranker.fit(rows)
    graph_scores = ranker.score(
        [
            _candidate("probe", "graph node centrality", "pagerank", relevant=False),
            _candidate(
                "probe", "graph node centrality", "translation", relevant=False
            ),
        ]
    )
    language_scores = ranker.score(
        [
            _candidate(
                "probe", "language sequence translation", "translation", relevant=False
            ),
            _candidate(
                "probe", "language sequence translation", "pagerank", relevant=False
            ),
        ]
    )

    assert pair_count == 24
    assert graph_scores[0] > graph_scores[1]
    assert language_scores[0] > language_scores[1]


def test_candidate_builder_allows_single_support_and_excludes_self() -> None:
    bridge = SupervisedLexicalBridge.fit(
        [
            LexicalBridgeExample(
                query="graph node centrality",
                gold_titles=("Quasar graph ranking",),
            ),
            LexicalBridgeExample(
                query="graph node importance",
                gold_titles=("Pagerank centrality",),
            ),
            LexicalBridgeExample(
                query="language translation",
                gold_titles=("Neural translation",),
            ),
        ]
    )

    candidates = build_query_term_candidates(
        bridge,
        query_id="graph-0",
        query="graph node centrality",
        gold_titles=("Quasar graph ranking",),
        neighbors=2,
        exclude_training_index=0,
    )

    assert "quasar" not in {row.term for row in candidates}
    assert "pagerank" in {row.term for row in candidates}
    assert all(row.neighbor_support >= 1 for row in candidates)
