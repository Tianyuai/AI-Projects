# Verified-Identifier Offline Rescore v2

Status: passed
Gold associations: 143

| Source | TP | Macro F1 | Macro recall | Micro recall | MRR | NDCG | Direct | not_retrieved | filtered_out | ranked_outside_top50 | selected_top50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| formal_baseline_2026_08_10 | 17 | 0.01577682933599093 | 0.19708333333333333 | 0.11888111888111888 | 0.11377801120448179 | 0.11808437599703922 | 12 | 103 | 0 | 23 | 17 |
| formal_baseline_2026_08_09 | 19 | 0.01703513892763386 | 0.20958333333333332 | 0.13286713286713286 | 0.10691965516562289 | 0.11562842299523662 | 14 | 102 | 0 | 22 | 19 |
| legacy_title_2026_08_05 | 30 | 0.018820742827791524 | 0.2426984126984127 | 0.2097902097902098 | 0.08753212558359616 | 0.09602236825801648 | 21 | 94 | 0 | 19 | 30 |
| query_evolution_prompt_v2 | 19 | 0.01703513892763386 | 0.20958333333333332 | 0.13286713286713286 | 0.10691965516562289 | 0.11562842299523662 | 14 | 101 | 0 | 23 | 19 |

## Decision

- primary_loss_stage: not_retrieved
- next_direction: retrieval_query
- reason_codes: none
