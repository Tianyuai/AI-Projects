# Sealed Query Recomposition Offline Diagnostic

Conclusion: signal_insufficient
Reason code: usable_signal_below_legacy_benchmark
Current formal selected gold: 17
Legacy title selected gold: 30

| Method | TP | Macro F1 | Macro recall | Micro recall | MRR | NDCG | not_retrieved | filtered_out | ranked_outside_top50 | selected_top50 | usable_signal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| append_v2 | 19 | 0.01703513892763386 | 0.20958333333333332 | 0.13286713286713286 | 0.10691965516562289 | 0.11562842299523662 | 101 | 0 | 23 | 19 | false |
| round_robin_slots | 24 | 0.020170387847196502 | 0.24625 | 0.16783216783216784 | 0.10603802174626163 | 0.1225963674060259 | 101 | 0 | 18 | 24 | false |
| rrf_slots_k60 | 25 | 0.02081141348822214 | 0.25458333333333333 | 0.17482517482517482 | 0.12353650771725806 | 0.1400577921786296 | 101 | 0 | 17 | 25 | true |
