# Query Evolution Gate Contracts Repair Design

Date: 2026-08-10
Status: approved in conversation; awaiting written-spec review

## 1. Background

The Prompt v2 live probe completed 55 locked queries with 30 `generated` and 25 `no_op` outcomes. Capture and zero-network replay matched, every ledger receipt reached a terminal state, and no integrity, provenance, accounting, or snapshot failure occurred. The public report nevertheless returned Gate A `failed`.

Read-only diagnosis found three independent contract problems:

1. The frozen evaluation baseline contains 60 queries, while the bounded probe intentionally executes 55 locked queries. Runtime integrity records 55 terminals but Gate A compares that value with the 60-query baseline denominator, making Gate A fail deterministically.
2. Candidate retrieval and selected/Top-50 ranking are represented by one projection. The baseline side contains `selected_paper_ids`, so Gate B evaluates selected results instead of the full frozen `retrieved_paper_ids` set.
3. Gold counting resolves identifiers but then sums every original gold identifier. Multiple aliases resolving to the same canonical paper are counted more than once, while the lock thresholds use unique query-paper associations.

The same diagnosis confirmed a separate ranking bottleneck. Two gold associations entered the selected candidate projection, but they were appended after 50 baseline papers and occupied accepted positions 55 and 63. Both passed hard filters; neither entered Top-50. One of the two papers was already present in the frozen retrieved set, so the true retrieval gain was one unique association.

## 2. Goals

- Make Gate A validate the 55-query execution queue independently from the 60-query frozen metric denominator.
- Preserve the full retrieved candidate stream for Gate B while retaining the selected/Top-50 stream for ranking metrics and Gate C.
- Count gold as unique resolved `(query_id, canonical_paper_id)` associations everywhere.
- Derive preflight baseline counts from frozen evidence instead of hard-coded values.
- Re-evaluate the sealed Prompt v2 run offline without rewriting its evidence.
- Produce an aggregate-only append-order versus ranking-position comparison.

## 3. Non-goals

- Do not change the production ranking, RRF, hard filters, provider routing, prompt, retry, timeout, budget, ledger, snapshot, or promotion thresholds.
- Do not rebuild locks, read `.env`, send network requests, or run another canary or full probe.
- Do not rewrite the sealed live run, its snapshots, outcomes, result, or ledger receipts.
- Do not expose query IDs, paper IDs, titles, generated query text, or provider request IDs in committed documentation.

## 4. Chosen approach: explicit retrieved and selected streams

The evaluation domain will model retrieval and ranking as separate streams.

- `FrozenQueryRecord` will retain `retrieved_paper_ids` from the frozen execution record in addition to the existing selected-paper provider results.
- `QueryProjection` will expose a deduplicated `retrieved_ids` sequence. For the baseline projection it contains the frozen retrieved IDs. For the candidate projection it contains the frozen retrieved IDs followed by canonical IDs from query-evolution additions, with first occurrence preserved.
- `candidate_papers` and `top50_ids` will continue to be built from selected baseline papers plus additions. Their ordering and filtering behavior will not change.

This keeps Gate B tied to retrieval coverage and Gate C tied to selected ranking quality without adding parallel optional arguments to `evaluate_probe()`.

## 5. Gate contracts

### 5.1 Gate A: execution integrity

The frozen baseline must still reconstruct exactly 60 queries and 2,910 selected papers. These denominators protect metric comparability.

The bounded execution contract is separate:

- `locked_query_count` is `len(lock.query_ids)`, which is 55 for the full probe.
- `terminal_count` is the number of captured/replayed outcomes.
- When either execution count is supplied, both must be supplied, positive, and equal.
- Gate A must not compare either execution count with the 60-query baseline denominator.

All existing integrity predicates remain mandatory: capture/replay match, zero integrity/provenance/accounting/request/source failures, matching hashes, respected limits, aggregate-only reporting, and exact baseline reconstruction.

### 5.2 Gate B: retrieval coverage

Gate B will use unique resolved association sets:

- Baseline set: gold associations present in each query's frozen `retrieved_paper_ids`.
- Candidate set: gold associations present in frozen retrieved IDs union query-evolution additions.
- Newly retrieved count: size of `candidate_set - baseline_set`.
- Retention: `baseline_set` must be a subset of `candidate_set`.

The existing threshold remains unchanged: candidate unique retrieved gold must be strictly greater than 14 and contain at least one newly retrieved association.

### 5.3 Gate C: selected Top-50 quality

Gate C continues to use `top50_ids`, current ranking metrics, hard-filter rejection counts, and the existing production estimate. Gold counts and retention use the same unique resolved association helper as Gate B. No ranking behavior changes in this repair.

## 6. Canonical gold association helper

One internal helper will produce a set of resolved `(query_id, canonical_paper_id)` pairs from gold records and an identifier stream per query. Candidate counts, Top-50 counts, newly retrieved counts, and retention checks will all consume this helper.

This removes alias inflation while preserving the public aggregate-only report schema. The report field names and schema version remain unchanged; their values become semantically consistent with the lock thresholds.

## 7. Preflight baseline derivation

`preflight_probe()` will derive, not hard-code:

- `baseline_candidate_gold_count` from frozen `retrieved_paper_ids` using unique resolved associations.
- `baseline_top50_gold_count` from frozen `selected_paper_ids` using the same association helper.

The existing expected values remain 14 and 8 for the sealed source run. Preflight must reject a source that reconstructs different values rather than writing a misleading lock.

No lock schema field or schema version changes.

## 8. Offline evidence comparison

The sealed Prompt v2 run will be read only. An offline reconstruction must produce:

- Gate A: `passed`.
- Gate B: `passed`.
- Gate C: `failed`.
- Unique retrieved gold: 14 baseline, 15 candidate, true gain 1.
- Unique Top-50 gold: 8 baseline, 8 candidate, gain 0.
- Prior retrieved and Top-50 gold retained.
- Capture/replay remains matched.
- The two selected-stream gains remain accepted at positions 55 and 63 and remain outside Top-50.

The comparison output is aggregate-only and is not written back into the sealed run.

## 9. Error handling and compatibility

- Missing retrieved IDs in newly constructed frozen records are a validation error in production reconstruction.
- Unit-test fixtures may supply an empty retrieved list explicitly; there is no implicit fallback from selected results because that would hide the semantic boundary.
- Unknown query additions, malformed identifiers, hash drift, count drift, and replay mismatch keep their current failure behavior.
- Existing public report consumers require no schema migration.

## 10. Test strategy

Implementation follows TDD.

1. Add a Gate A regression with a 60-query baseline and `locked_query_count=terminal_count=55`; observe failure before changing production code.
2. Add a mismatch case proving 54 terminals for 55 locked queries fails Gate A.
3. Add canonical alias fixtures proving duplicate identifiers resolving to one paper count once.
4. Add a dual-stream fixture where one gold is retrieved but not selected; prove it is retained baseline coverage, not a newly retrieved association.
5. Add a true new retrieval and prove Gate B uses 14 to 15 while Top-50 remains 8 to 8.
6. Add preflight coverage proving 14/8 are derived from frozen inputs and drift is rejected.
7. Run focused evaluation and probe integration tests, Ruff, mypy, full offline pytest, and `git diff --check`.
8. Recompute the sealed run offline and verify the aggregate expectations in section 8 without modifying protected artifacts.

## 11. Acceptance criteria

- The focused regressions fail for the diagnosed reasons before implementation and pass afterward.
- The public report uses unique canonical association counts.
- Gate A distinguishes 60-query baseline comparability from 55-query execution completeness.
- Gate B uses the retrieved stream; Gate C uses selected Top-50.
- Preflight reconstructs 14/8 instead of embedding constants without evidence.
- The sealed Prompt v2 run evaluates as A passed, B passed, C failed, with true retrieval gain 1 and Top-50 gain 0.
- No tracked file outside the evaluation module, probe script, their tests, design, and implementation plan is modified.
- No live operation or protected evidence mutation occurs.
