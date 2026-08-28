from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime
from pathlib import Path

from paper_search.domain.models import BudgetReservation, ProviderResult, UsageEstimate
from paper_search.learning.deployment import build_cpu_action_analyzer_decorator


async def _run(model_path: Path, result_path: Path) -> None:
    fallback_calls = 0

    async def fallback(
        query: str, reservation: BudgetReservation
    ) -> ProviderResult[dict[str, object]]:
        del query, reservation
        nonlocal fallback_calls
        fallback_calls += 1
        raise RuntimeError("LLM fallback requested")

    analyzer = build_cpu_action_analyzer_decorator(
        model_path=model_path,
        result_path=result_path,
        max_actions=5,
    )(fallback)
    reservation = BudgetReservation(
        reservation_id="cpu-smoke",
        action="query.analyze",
        reserved=UsageEstimate(llm_calls=1, input_tokens=100, output_tokens=100),
        expires_at=datetime(2030, 1, 1, tzinfo=UTC),
    )
    queries = (
        "Which paper proposed graph diffusion networks for retrieval?",
        "What works study federated learning aggregation methods?",
        "Find papers about diffusion models for molecule generation",
    )
    for query in queries:
        try:
            result = await analyzer(query, reservation)
            plan = result.data["search_plan"]
            assert isinstance(plan, dict)
            subqueries = plan["subqueries"]
            assert isinstance(subqueries, list)
            print(
                f"actions={len(subqueries)} "
                f"confidence={result.provenance['confidence']} "
                f"fallback={result.provenance['fallback_required']}"
            )
        except RuntimeError:
            print("fallback=true")
    print(f"fallback_calls={fallback_calls}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(_run(args.model, args.result))


if __name__ == "__main__":
    main()
