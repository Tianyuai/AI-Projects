# Verified-Identifier Offline Rescore v2 Design

**Date:** 2026-08-11
**Status:** Approved
**Scope:** Task 4 offline rescore and bottleneck attribution only

## Decision

Task 4 will rescore the sealed development sources only after loading the identifier generation through `load_verified_identifier_generation()`. The rescore must not accept a separately supplied map, audit, or relaxed compatibility path.

The result is an aggregate, reproducible diagnosis of where each gold association is lost: retrieval, retention, ranking, or nowhere. It is evidence for choosing one later offline experiment; it does not itself change retrieval, filtering, ranking, prompts, or production behavior.

This design supersedes the unimplemented Task 4 in `docs/superpowers/plans/2026-08-10-identifier-map-semantic-recovery.md`. The old text remains as historical context but is not executable authority.

## Goals and Non-goals

The task must:

- prove that all scoring uses the passed v2 identifier generation and its six bound artifacts;
- independently verify each sealed source before reading stage data;
- recompute project metrics and a conserved four-stage funnel from the same source projection;
- publish only aggregate, privacy-safe evidence to fresh output paths;
- identify the current formal baseline's primary loss stage, or explicitly report that no unique conclusion is justified.

The task must not:

- access the network, `.env`, budget ledger, candidate locks, readiness, capture, replay, or live providers;
- rebuild or edit identifier evidence, snapshots, the verified map, or either semantic audit;
- tune thresholds or apply a reference-paper retrieval, retention, or ranking method;
- rewrite historical source artifacts, `HANDOFF.md`, or the retrieval roadmap;
- treat comparison among historical sources as a promotion decision.

## Frozen Identifier Generation

The command has one fixed v2 generation made of these paths:

| Role | Path |
|---|---|
| Public commit marker | `docs/evidence/identifier-map-semantic-audit-2026-08-11.json` |
| Development gold | `data/dev/gold.jsonl` |
| Sealed identity evidence | `data/annotation_work/identifier_semantics/identity-evidence.json` |
| Snapshot manifest | `data/annotation_work/identifier_semantics/snapshots/snapshot-manifest.json` |
| Private relation audit | `data/annotation_work/identifier_semantics/relation-audit.v2.json` |
| Verified identifier map | `data/annotation_work/identifier_semantics/dev-identifier-map.semantic-v2.json` |

The rescore calls `load_verified_identifier_generation()` before source loading. That loader remains the only authority for audit status, canonical bytes, hashes, sealed-evidence reconstruction, relation conservation, and exact map reconstruction. Task 4 consumes the returned `IdentifierMap`; it does not duplicate those checks or expose path overrides for individual generation members.

For the current generation, the passed audit binds 141 gold groups, 141 verified anchors, 90 provider candidates, 51 groups without an optional provider candidate, and 231 verified relations. These are identifier-generation facts, not the rescore funnel denominator.

## Sealed Sources

The source set and labels are fixed:

| Public label | Kind | Binding requirement |
|---|---|---|
| `formal_baseline_2026_08_10` | formal run | `validate_run_directory()` passes for `runs/dev-20260810T104256Z-d9e89476d484`; required run artifacts are hashed after validation |
| `formal_baseline_2026_08_09` | formal run | `validate_run_directory()` passes for `runs/dev-20260809T061903Z-9bd861e90299`; required run artifacts are hashed after validation |
| `legacy_title_2026_08_05` | legacy hash-bound run | `business-results.jsonl` and `executions.jsonl` match `docs/evidence/title-retention-offline-2026-08-09.json`; the row remains explicitly non-formal |
| `query_evolution_prompt_v2` | sealed probe | the probe lock self-hash, all locked source hashes, snapshot binding, result binding, and `capture_replay_match=matched` pass for `runs/_diag_query_evolution_query-evolution-prompt-v2-full-20260810` |

Source adapters expose one common in-memory projection containing ordered query IDs; raw, normalized-but-unresolved retrieved, post-filter, and selected Top-50 IDs; immutable safe binding hashes; and a closed source-status enum. Predictions are derived only from the stored selected Top-50 sequence and are not stored separately. Adapters receive the expected v2 gold query-ID sequence and reject duplicate, unknown, missing, or out-of-order query IDs. The source-neutral scorer owns identifier resolution, resolved subset checks, and cross-source denominator equality.

For formal and legacy runs, `business-results.jsonl.selected_paper_ids`, `executions.jsonl.post_filter_paper_ids`, and `executions.jsonl.retrieved_paper_ids` are the three stage sets. The invariant `selected <= post_filter <= retrieved` must hold after identifier resolution.

For the probe, the frozen baseline is reconstructed from its locked source and additions are reconstructed from sealed outcomes/snapshots through the existing query-evolution projection helpers. `QueryProjection` exposes the ordered `post_filter_ids` produced by that same hard-filter pass; the probe post-filter set is the stable union of hash-bound baseline execution post-filter IDs and these projected IDs. This preserves accepted additions that did not reach Top-50 without re-running or inferring a filter result. The scorer then resolves the merged projection and enforces the same subset invariant. The rescore must not infer missing papers from titles or issue replacement searches.

## Canonical Association Denominator

One scoring unit is a unique `(query_id, canonical gold terminal)` association. Gold identifiers are normalized and resolved through the verified v2 map, then deduplicated within each query. A terminal relevant to two queries therefore counts twice; aliases for the same terminal within one query count once.

Let `A` be this fixed association set. Every source assigns every member of `A` to exactly one stage using this precedence:

1. `selected_top50` if the terminal is in the resolved selected set;
2. `ranked_outside_top50` if it is in the resolved post-filter set but not selected;
3. `filtered_out` if it is in the resolved retrieved set but not post-filtered;
4. `not_retrieved` otherwise.

The report must prove:

```text
total_gold_associations
  = selected_top50
  + ranked_outside_top50
  + filtered_out
  + not_retrieved
```

It must also prove one and only one stage assignment per association, the subset invariant for every query, and an identical `total_gold_associations` across all four rows. No fixed value is hard-coded merely because an older map produced 143 associations; the value is derived once from the verified generation and then required to be shared by every source.

## Metrics and Direct-Hit Acceptance

Each source's predictions are built from that source's exact selected Top-50 sequence. `evaluate()` and `evaluate_ranking()` are the only metric authorities. Task 4 does not reimplement precision, recall, F1, MRR, or NDCG.

The aggregate row reports:

- true-positive count;
- macro F1 and macro recall;
- micro recall;
- macro MRR and macro NDCG;
- direct same-arXiv hit count;
- the four funnel counts and total denominator;
- exactly three `metric_quality_checks` for each formal row and none for legacy/probe rows;
- source kind, source verification status, capture/replay status where applicable, and safe binding hashes.

A direct same-arXiv hit is counted from a separate set `D={(query_id, arxiv_anchor(normalized_raw_arxiv_id))}` built from every raw arXiv gold ID. Duplicate raw arXiv forms that produce the same pair are collapsed. A member of `D` is a direct hit only when that exact normalized DataCite anchor occurs in the source's selected sequence before alias resolution. This intentionally narrower source-projection check remains 12 for the designated run, while v2 alias resolution may recover additional true positives. It is a diagnostic subset of true positives, not a second identity Gate.

Only `formal_baseline_2026_08_10` has a fixed acceptance check: `direct_same_arxiv_hit_count == 12`. The other sources report their recomputed count without inheriting 12. A failure of the designated check stops publication because it indicates a source-projection or scoring regression.

Metric values and funnel stages must come from the same selected set. For the two formal rows, an informational `metric_quality_checks` section recomputes only the three identifier-sensitive development rules from the fixed `configs/quality_gates_v1.yaml`: hard-filter recall loss from `filtered_out / total_gold_associations`, macro identifier-map recall from `evaluate()`, and micro identifier-map recall from `evaluate()`. The policy file is parsed through the existing policy model, must contain exactly one applicable rule for each required measure, and is bound in the report by `quality_policy_sha256`. Rule operators and thresholds are applied through a public wrapper around the existing Gate comparator, not a second comparison implementation. Other baseline-quality and all formal-validity rules are outside this rescore and are not reported as recomputed. Task 4 never opens a ledger or claims to re-certify a historical run, and it never rewrites `gates.json`.

## Result Contract

Successful execution creates fresh files only:

- `docs/evidence/identifier-map-semantic-rescore-2026-08-11.json`;
- `docs/identifier-map-semantic-rescore-2026-08-11.md`.

The JSON schema version is `identifier-semantic-rescore-v2`. It is a closed aggregate object containing:

- `scope=dev` and `status=passed`;
- the public v2 audit hash and the five other generation hashes already verified by the loader;
- `quality_policy_sha256` for the checked-in policy used by the three informational metric checks;
- `total_gold_associations`;
- exactly four source rows in the fixed order above;
- a decision object with `designated_source=formal_baseline_2026_08_10`, `primary_loss_stage`, `next_direction`, and closed reason codes.

The Markdown is a deterministic rendering of the same model, not an independently calculated report. It summarizes source validity, metric values, funnel counts, and the decision; it contains no query-level, paper-level, request-level, title, raw response, local absolute path, secret, or ledger data.

Both files are fully rendered and privacy-checked in memory before any write. Publication requires both fixed targets to be absent, then atomically writes the canonical JSON with no replacement as the formal evidence and writes the derived Markdown with no replacement. If the companion write fails after JSON publication, the rescore is not rerun: a `render-markdown` recovery mode validates the existing formal JSON and creates only the missing fixed Markdown from that model. It refuses an existing Markdown target. Neither mode chooses alternate names or overwrites a file.

## Bottleneck Decision Rule

The designated source is `formal_baseline_2026_08_10`. Its three loss buckets are `not_retrieved`, `filtered_out`, and `ranked_outside_top50`.

- A unique largest loss bucket becomes `primary_loss_stage`; otherwise it is `null`.
- `reason_codes` is a sorted list using the fixed order `largest_loss_tie`, then `source_sensitivity`. A designated-source tie produces exactly `['largest_loss_tie']`; no cross-source sensitivity check is made in that case.
- If the designated source has a unique primary stage, each other source independently derives a stage by the same unique-maximum rule. Any unique different stage adds `source_sensitivity`; tied comparison rows are ignored. A unique, consistent result has an empty reason list.
- `next_direction` is `retrieval_query`, `retention_filter`, or `ranking_selector` only when the designated stage is unique and `source_sensitivity` is absent. Otherwise it is `null`: Task 4 stops and cannot select a method automatically. A human may authorize a separately reviewed follow-up diagnostic or single-variable experiment, but that authorization is not implied by this report.
- `selected_top50` is never a loss-stage recommendation.

The resulting next direction is limited to one of:

- retrieval/query experiment for `not_retrieved`;
- retention/filter experiment for `filtered_out`;
- ranking/selector experiment for `ranked_outside_top50`;
- no automatic experiment when the result is tied or source-sensitive. Invalid inputs publish no report.

Reference-paper methods are considered only after this report passes. The next task selects one method matching the diagnosed stage and runs one single-variable offline comparison. No method is adopted merely because it appears in a reference.

## Stop Conditions and Human Intervention

Before formal JSON publication, stop with a nonzero result and publish no recognized report if any of the following occurs:

- the v2 generation loader rejects any artifact or binding;
- any source fails its designated formal, hash, lock, snapshot, or capture/replay check;
- source query sets differ from gold, stage inputs are malformed, or subset invariants fail;
- the quality policy is malformed, its three required applicable rules are absent or ambiguous, or its raw hash cannot be bound;
- association conservation, cross-source denominator equality, or metric/funnel set identity fails;
- the designated direct-hit count is not 12;
- privacy scanning fails or either output path already exists.

After formal JSON publication, a missing Markdown is the only recoverable partial state and is handled solely by the `render-markdown` mode described above. A malformed or privacy-invalid existing JSON is not recoverable by that mode and requires technical investigation.

Do not respond to a stop by rebuilding a candidate lock, refreshing readiness, repeating capture, or changing the map. Source or generation failures require technical inspection; a tied or source-sensitive valid diagnosis requires human choice before a method experiment.

## Module Boundaries

- `src/paper_search/evaluation/identifier_semantics.py` continues to own the verified-generation loader and identifier relation contract.
- A new focused evaluation module owns the common rescore model, source-neutral association classification, identifier resolution, metric assembly, conservation checks, and decision rule.
- `src/paper_search/evaluation/gates.py` exposes its existing rule comparator through a public helper; its comparison behavior and formal Gate contract do not change.
- A thin `scripts` module owns fixed local path selection, source-adapter orchestration, privacy validation, and no-replace publication. It is invoked with `python -m scripts.rescore_identifier_semantics` so the sealed-probe wrappers can be imported consistently by both tests and the CLI.
- Existing formal-run validation, metric, ranking, query-evolution projection, and privacy helpers are reused rather than copied.
- `scripts/analyze_gold_bottlenecks.py` remains a historical availability diagnostic. Task 4 does not refactor it or use its old independently loaded identifier map.

## Test Strategy

Implementation follows TDD with only the tests needed to prove the contract:

1. **Generation and source rejection:** failed/non-v2 generation; hash-swapped formal or legacy inputs; bad probe self-hash, source binding, or capture/replay match; mismatched query sets.
2. **Funnel correctness:** alias collapse, within-query deduplication, stage precedence, subset rejection, one-stage conservation, and equal denominators across all sources.
3. **Metric and Gate identity:** predictions use the exact selected sequence; reported measures equal `evaluate()` and `evaluate_ranking()`; the three formal metric-quality checks use the existing policy comparator without a ledger; the designated formal source has exactly 12 pre-alias direct same-arXiv hits while v2 may yield more total true positives.
4. **Decision behavior:** unique retrieval, retention, and ranking maxima; tie; source sensitivity; no recommendation from `selected_top50`.
5. **Publication safety:** closed aggregate schema, deterministic JSON/Markdown agreement, privacy rejection through `assert_public_json_safe()` and `assert_public_markdown_safe()`, existing-target rejection, JSON-preserving Markdown recovery, and an orchestration test that blocks network, `.env`, and ledger entry points.

After focused tests, run the affected evaluation/script tests, offline project suite, Ruff, mypy, and `git diff --check`. Execute the real four-source rescore once only after all checks pass. A successful rescore ends Task 4; choosing or implementing a reference method is a separate reviewed task.
