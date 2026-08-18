"""CPU-only supervised query-to-title vocabulary bridging."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from math import log, sqrt
from typing import Literal

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer

from paper_search.learning.candidates import query_content_terms


@dataclass(frozen=True)
class LexicalBridgeExample:
    query: str
    gold_titles: tuple[str, ...]


@dataclass(frozen=True)
class LexicalBridgeProposal:
    query_text: str
    expansion_terms: tuple[str, ...]
    neighbor_support: dict[str, int]
    maximum_similarity: float


class SupervisedLexicalBridge:
    """Transfer title terms supported by multiple similar training queries."""

    def __init__(
        self,
        *,
        vectorizer: TfidfVectorizer | FeatureUnion,
        query_matrix: csr_matrix,
        title_terms: tuple[tuple[str, ...], ...],
        title_idf: dict[str, float],
        representation: Literal["word", "word_char"],
        learning_objective: Literal[
            "neighbor_idf", "association", "support_normalized_idf"
        ],
    ) -> None:
        self._vectorizer = vectorizer
        self._query_matrix = query_matrix
        self._title_terms = title_terms
        self._title_idf = title_idf
        self._representation = representation
        self._learning_objective = learning_objective

    @classmethod
    def fit(
        cls,
        examples: list[LexicalBridgeExample],
        *,
        representation: Literal["word", "word_char"] = "word",
        learning_objective: Literal[
            "neighbor_idf", "association", "support_normalized_idf"
        ] = "neighbor_idf",
    ) -> SupervisedLexicalBridge:
        if len(examples) < 2:
            raise ValueError("lexical bridge requires at least two examples")
        word_vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
            norm="l2",
        )
        if representation == "word":
            vectorizer: TfidfVectorizer | FeatureUnion = word_vectorizer
        elif representation == "word_char":
            vectorizer = FeatureUnion(
                [
                    ("word", word_vectorizer),
                    (
                        "char",
                        TfidfVectorizer(
                            analyzer="char_wb",
                            ngram_range=(3, 5),
                            min_df=1,
                            sublinear_tf=True,
                            norm="l2",
                        ),
                    ),
                ]
            )
        else:
            raise ValueError(f"unsupported representation: {representation}")
        if learning_objective not in {
            "neighbor_idf",
            "association",
            "support_normalized_idf",
        }:
            raise ValueError(f"unsupported learning objective: {learning_objective}")
        queries = [item.query for item in examples]
        query_matrix = vectorizer.fit_transform(queries).tocsr()
        title_terms = []
        for item in examples:
            terms = {
                term
                for title in item.gold_titles
                for term in query_content_terms(title)
            }
            if learning_objective == "association":
                # Learn only terms that add Gold-title coverage beyond the query.
                terms.difference_update(query_content_terms(item.query))
            title_terms.append(frozenset(terms))
        frozen_title_terms = tuple(tuple(sorted(terms)) for terms in title_terms)
        document_frequency = Counter(
            term for terms in frozen_title_terms for term in terms
        )
        count = len(examples)
        title_idf = {
            term: log((count + 1) / (document_frequency[term] + 1)) + 1
            for term in sorted(document_frequency)
        }
        return cls(
            vectorizer=vectorizer,
            query_matrix=query_matrix,
            title_terms=frozen_title_terms,
            title_idf=title_idf,
            representation=representation,
            learning_objective=learning_objective,
        )

    @property
    def representation(self) -> Literal["word", "word_char"]:
        return self._representation

    @property
    def learning_objective(
        self,
    ) -> Literal["neighbor_idf", "association", "support_normalized_idf"]:
        return self._learning_objective

    def propose(
        self,
        query: str,
        *,
        neighbors: int = 12,
        max_expansion_terms: int = 3,
        min_neighbor_support: int = 2,
        minimum_similarity: float = 0.05,
    ) -> LexicalBridgeProposal | None:
        if neighbors <= 0 or max_expansion_terms <= 0 or min_neighbor_support <= 0:
            raise ValueError("bridge bounds must be positive")
        query_vector = self._vectorizer.transform([query])
        similarities = np.asarray(
            (self._query_matrix @ query_vector.T).toarray()
        ).ravel()
        selected = np.argsort(-similarities, kind="stable")[:neighbors]
        original_terms = set(query_content_terms(query))
        support: dict[str, set[int]] = defaultdict(set)
        scores: Counter[str] = Counter()
        for index in selected:
            similarity = float(similarities[index])
            if similarity < minimum_similarity:
                continue
            for term in set(self._title_terms[int(index)]).difference(original_terms):
                support[term].add(int(index))
                idf = self._title_idf.get(term, 1.0)
                if self._learning_objective == "association":
                    scores[term] += similarity * idf * len(support[term])
                else:
                    scores[term] += similarity * idf
        eligible = [
            term
            for term in scores
            if len(support[term]) >= min_neighbor_support
        ]
        if self._learning_objective == "support_normalized_idf":
            eligible.sort(
                key=lambda term: (
                    -scores[term] / sqrt(len(support[term])),
                    term,
                )
            )
        else:
            eligible.sort(key=lambda term: (-scores[term], term))
        expansion_terms = tuple(eligible[:max_expansion_terms])
        if not expansion_terms:
            return None
        anchors = query_content_terms(query)[:5]
        return LexicalBridgeProposal(
            query_text=" ".join([*anchors, *expansion_terms]),
            expansion_terms=expansion_terms,
            neighbor_support={
                term: len(support[term]) for term in expansion_terms
            },
            maximum_similarity=float(similarities[selected[0]]),
        )


__all__ = [
    "LexicalBridgeExample",
    "LexicalBridgeProposal",
    "SupervisedLexicalBridge",
]
