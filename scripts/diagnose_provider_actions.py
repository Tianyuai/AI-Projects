from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from paper_search.learning.action_diagnostics import diagnose_action_selection
from paper_search.learning.provider_action_dataset import (
    load_provider_action_labels_from_canary_runs,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    labels = load_provider_action_labels_from_canary_runs(
        partition_path=args.partition,
        provider_run_roots={
            "openalex": args.run_root / "openalex",
            "semantic_scholar": args.run_root / "semantic_scholar",
        },
    )
    groups = defaultdict(list)
    for label in labels:
        groups[label.query_id].append(label)

    summaries: list[dict[str, object]] = []
    for provider in ("openalex", "semantic_scholar"):
        for budget in (1, 2, 3):
            diagnostics = []
            for rows in groups.values():
                provider_rows = sorted(
                    (row for row in rows if row.provider == provider),
                    key=lambda row: row.action.action_id,
                )
                selected = [
                    (row.action.action_id, provider)
                    for row in provider_rows[:budget]
                ]
                diagnostics.append(
                    diagnose_action_selection(
                        provider_rows,
                        selected_action_provider_pairs=selected,
                        max_actions=budget,
                    )
                )
            summaries.append(
                {
                    "provider": provider,
                    "budget": budget,
                    "current_macro_recall": sum(
                        row.selected_recall for row in diagnostics
                    )
                    / len(diagnostics),
                    "oracle_macro_recall": sum(
                        row.oracle_recall for row in diagnostics
                    )
                    / len(diagnostics),
                    "queries_with_selection_gap": sum(
                        row.selection_gap > 0 for row in diagnostics
                    ),
                    "current_hits": sum(
                        row.selected_gold_hit_count for row in diagnostics
                    ),
                    "oracle_hits": sum(
                        row.oracle_gold_hit_count for row in diagnostics
                    ),
                }
            )

    for fixed_provider in ("openalex", "semantic_scholar"):
        diagnostics = []
        for rows in groups.values():
            fixed = sorted(
                (row for row in rows if row.provider == fixed_provider),
                key=lambda row: row.action.action_id,
            )[:3]
            diagnostics.append(
                diagnose_action_selection(
                    rows,
                    selected_action_provider_pairs=[
                        (row.action.action_id, fixed_provider) for row in fixed
                    ],
                    max_actions=3,
                )
            )
        summaries.append(
            {
                "fixed_provider": fixed_provider,
                "total_budget": 3,
                "current_macro_recall": sum(
                    row.selected_recall for row in diagnostics
                )
                / len(diagnostics),
                "cross_provider_oracle_macro_recall": sum(
                    row.oracle_recall for row in diagnostics
                )
                / len(diagnostics),
                "queries_with_selection_or_routing_gap": sum(
                    row.selection_gap > 0 for row in diagnostics
                ),
                "current_hits": sum(
                    row.selected_gold_hit_count for row in diagnostics
                ),
                "oracle_hits": sum(
                    row.oracle_gold_hit_count for row in diagnostics
                ),
            }
        )
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
