from __future__ import annotations

import pytest

from paper_search.evaluation.official_adapter import (
    AstaPaperFindingRecord,
    adapt_asta_paper_finding_record,
    adapt_internal_to_asta_paper_finding,
)
from paper_search.evaluation.official_adapter import InternalPredictionRecord


def test_asta_agent_facing_input_adapts_without_treating_partial_gold_as_exhaustive() -> None:
    record = AstaPaperFindingRecord.model_validate(
        {
            "input": {
                "query_id": "29",
                "query": "visual question answering papers using EMD",
            }
        }
    )

    adapted = adapt_asta_paper_finding_record(
        record,
        source="allenai/asta-bench",
        split="paper_finder_validation",
        revision="v1.0.0",
    )

    assert adapted.query_id == "29"
    assert adapted.query == "visual question answering papers using EMD"
    assert adapted.relevant_paper_ids == []
    assert adapted.metadata["gold_semantics"] == "official_scorer_only"


def test_asta_output_preserves_ranking_and_requires_evidence_for_every_paper() -> None:
    prediction = InternalPredictionRecord(
        query_id="29",
        selected_paper_ids=["173990882", "222"],
    )

    output = adapt_internal_to_asta_paper_finding(
        prediction,
        markdown_evidence={
            "173990882": "Verbatim evidence one.",
            "222": "Verbatim evidence two.",
        },
    )

    assert output.model_dump(mode="json") == {
        "output": {
            "query_id": "29",
            "results": [
                {
                    "paper_id": "173990882",
                    "markdown_evidence": "Verbatim evidence one.",
                },
                {
                    "paper_id": "222",
                    "markdown_evidence": "Verbatim evidence two.",
                },
            ],
        }
    }

    with pytest.raises(ValueError, match="missing markdown evidence"):
        adapt_internal_to_asta_paper_finding(
            prediction,
            markdown_evidence={"173990882": "Only one."},
        )
