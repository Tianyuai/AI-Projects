# Verified-Identifier Offline Rescore v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce one trustworthy aggregate v2 rescore and four-stage bottleneck diagnosis from the fixed sealed development sources using only the passed identifier generation.

**Architecture:** Put source-neutral scoring, conservation, metric-quality checks, and the decision rule in one new evaluation module. Keep sealed-source loading and no-replace publication in one fixed-path script, reusing existing formal validation, probe projection, policy, metric, ranking, and privacy contracts.

**Tech Stack:** Python 3.12, Pydantic v2, pytest, Ruff, mypy, canonical JSON, existing Paper Search evaluation contracts.

## Global Constraints

- Authoritative design: `docs/superpowers/specs/2026-08-11-verified-identifier-rescore-design.md`.
- Work only in the existing `codex/query-evolution-gate-contracts` worktree; preserve unrelated dirty files.
- `load_verified_identifier_generation()` is the only identifier input. The CLI accepts no map, audit, evidence, manifest, source, output, `.env`, ledger, or network override.
- Fixed sources and order: `formal_baseline_2026_08_10`, `formal_baseline_2026_08_09`, `legacy_title_2026_08_05`, `query_evolution_prompt_v2`.
- Count unique resolved `(query_id, terminal)` associations once; require `selected <= post_filter <= retrieved` after resolution and conserve all four stages.
- The designated pre-alias direct DataCite-anchor check is exactly 12; other true positives may be recovered through v2 aliases.
- Recompute only the three identifier-sensitive quality rules: hard-filter recall loss, macro identifier-map recall, and micro identifier-map recall.
- Publish only `docs/evidence/identifier-map-semantic-rescore-2026-08-11.json` and `docs/identifier-map-semantic-rescore-2026-08-11.md`, with no replacement.
- Do not update `HANDOFF.md`, `docs/retrieval-roadmap.md`, historical runs, snapshots, the budget ledger, candidate locks, readiness, capture, replay, or production behavior.

---

### Task 1: Build the Pure Rescore and Decision Contract

**Files:**
- Modify: `src/paper_search/evaluation/gates.py`
- Modify: `tests/evaluation/test_gates.py`
- Create: `src/paper_search/evaluation/semantic_rescore.py`
- Create: `tests/evaluation/test_semantic_rescore.py`

**Interfaces:**
- `compare_quality_gate_rule(rule: QualityGateRule, value: Decimal) -> bool`
- `score_source(gold, identifier_map, source, *, policy) -> SemanticRescoreRun`
- `decide_bottleneck(runs: Sequence[SemanticRescoreRun]) -> RescoreDecision`
- `build_rescore_report(*, gold, identifier_map, sources, policy, generation_hashes, quality_policy_sha256) -> SemanticRescoreReport`

- [ ] **Step 1: Write RED tests for the comparator and core invariants**

Add tests with synthetic `EvaluationQuery`, `IdentifierMap.from_bytes()`, and `SourceProjection` values. The assertions must cover:

```python
def test_public_gate_comparator_preserves_required_boundaries() -> None:
    rules = {rule.rule_id: rule for rule in POLICY.rules}
    assert compare_quality_gate_rule(
        rules["hard-filter-recall-loss"], Decimal("0.02")
    )
    assert not compare_quality_gate_rule(
        rules["hard-filter-recall-loss"], Decimal("0.0201")
    )
    assert compare_quality_gate_rule(
        rules["macro-recall-positive"], Decimal("0.0001")
    )
    assert not compare_quality_gate_rule(
        rules["macro-recall-positive"], Decimal("0")
    )


def test_score_source_conserves_resolved_associations() -> None:
    row = score_source(GOLD, VERIFIED_MAP, FORMAL_SOURCE, policy=POLICY)
    stages = row.pipeline_stages
    assert stages.total_gold_associations == (
        stages.not_retrieved
        + stages.filtered_out
        + stages.ranked_outside_top50
        + stages.selected_top50
    )
    assert row.direct_same_arxiv_hit_count == 1
    assert [check.rule_id for check in row.metric_quality_checks] == [
        "hard-filter-recall-loss",
        "macro-recall-positive",
        "micro-recall-positive",
    ]
```

The same module must test alias collapse within a query, the same terminal counted separately across queries, exact stage precedence, query-set mismatch, subset violation, unequal source denominators, fixed source order, designated tie, and cross-source sensitivity.

- [ ] **Step 2: Run RED**

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest tests/evaluation/test_gates.py tests/evaluation/test_semantic_rescore.py -q
```

Expected: collection fails because the public comparator and rescore module do not exist.

- [ ] **Step 3: Implement the minimal closed contract**

Rename the existing private `_compare()` to `compare_quality_gate_rule()` and update `evaluate_gates()` to call it without changing operator behavior.

In `semantic_rescore.py`, define these closed `DomainModel` types exactly:

- `SourceProjection`: fixed label/kind/status, capture-replay status, safe binding hashes, ordered query IDs, and retrieved/post-filter/selected sequences by query.
- `PipelineStageCounts`: total plus `not_retrieved`, `filtered_out`, `ranked_outside_top50`, `selected_top50`.
- `MetricQualityCheck`: rule identity, exact measure numerator/denominator/value, operator, threshold, and pass status.
- `SemanticRescoreRun`: source metadata, TP, macro F1/recall, micro recall, macro MRR/NDCG, direct-hit count, funnel, and quality checks.
- `GenerationHashes`: public audit, gold, identity evidence, snapshot manifest, private audit, and candidate-map hashes.
- `RescoreDecision`: designated source, nullable primary loss stage, nullable next direction, and ordered closed reason codes.
- `SemanticRescoreReport`: literal schema `identifier-semantic-rescore-v2`, dev/passed status, generation and policy hashes, shared denominator, four ordered rows, and decision.

Implementation rules:

1. Build predictions from the exact selected sequence and call existing `evaluate()` and `evaluate_ranking()`.
2. Build the direct set from raw normalized arXiv gold IDs and test exact DataCite-anchor membership before alias resolution.
3. Assign stages with precedence selected, post-filter, retrieved, absent.
4. For formal rows, select exactly one applicable policy rule for each required measure and call `compare_quality_gate_rule()`; legacy/probe rows have no metric-quality checks.
5. A designated tie yields `primary_loss_stage=null`, `next_direction=null`, `['largest_loss_tie']`. A unique designated stage contradicted by another unique stage yields `next_direction=null`, `['source_sensitivity']`. Otherwise return the mapped single direction and no reason code.

- [ ] **Step 4: Run GREEN and commit**

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest tests/evaluation/test_gates.py tests/evaluation/test_semantic_rescore.py -q
git add src/paper_search/evaluation/gates.py tests/evaluation/test_gates.py src/paper_search/evaluation/semantic_rescore.py tests/evaluation/test_semantic_rescore.py
git commit -m "feat: add verified identifier rescore core"
```

Expected: selected tests pass and only Task 1 files are committed.

---

### Task 2: Load and Verify the Four Sealed Sources

**Files:**
- Modify: `src/paper_search/evaluation/query_evolution_probe.py`
- Modify: `scripts/probe_query_evolution.py`
- Modify: `tests/evaluation/test_query_evolution_probe.py`
- Modify: `tests/integration/test_query_evolution_probe.py`
- Create: `scripts/rescore_identifier_semantics.py`
- Create: `tests/scripts/test_rescore_identifier_semantics.py`

**Interfaces:**
- `load_formal_source(label: SourceLabel, run_dir: Path) -> SourceProjection`
- `load_legacy_source(run_dir: Path, evidence_path: Path) -> SourceProjection`
- `load_probe_source(run_dir: Path) -> SourceProjection`
- `load_fixed_sources(root: Path = ROOT) -> tuple[SourceProjection, ...]`

- [ ] **Step 1: Write RED source-binding tests**

Tests must prove:

- formal validation runs before any source projection is accepted;
- formal query order and business/execution query sets match;
- legacy business/execution drift is rejected against `title-retention-offline-2026-08-09.json`;
- probe lock self-hash, expected directory, source hashes, result schema, `capture_replay_match=matched`, equal capture/replay business hashes, manifest/set identity, every response hash, and exact outcome order are required;
- error-bearing probe search outcomes, unknown/duplicate queries, and subset violations are rejected;
- the orchestrator returns exactly the four fixed labels in order.

Run:

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest tests/evaluation/test_query_evolution_probe.py tests/integration/test_query_evolution_probe.py tests/scripts/test_rescore_identifier_semantics.py -q
```

Expected: RED because the public helper names and rescore script are absent.

- [ ] **Step 2: Expose only the existing probe helpers needed offline**

Make these mechanical renames and update their existing call sites:

| File | Existing | Public name |
|---|---|---|
| `src/paper_search/evaluation/query_evolution_probe.py` | `_offline_provider_result` | `offline_provider_result` |
| `scripts/probe_query_evolution.py` | `_verify_probe_source_bindings` | `verify_probe_source_bindings` |
| `scripts/probe_query_evolution.py` | `_frozen_inputs` | `frozen_probe_inputs` |
| `scripts/probe_query_evolution.py` | `_capture_replay_hash` | `probe_outcome_hash` |

Add `offline_provider_result` to the evaluation module's `__all__`. Do not alter live capture, retry, budget, or existing Gate behavior.

- [ ] **Step 3: Implement the three strict adapters**

Formal adapters call `validate_run_directory()` first, then parse `BusinessResultRecord` and `EvaluationExecutionRecord`; bind hashes for `run.json`, `gates.json`, `predictions.jsonl`, `executions.jsonl`, and `business-results.jsonl`.

The legacy adapter verifies the exact business/execution hashes recorded by the title-retention evidence before parsing, binds those two hashes plus the evidence-file hash, and labels the row `legacy_hash_bound`.

The probe adapter:

1. Loads and verifies the copied lock and fixed source bindings.
2. Validates `result.json` with a closed model and requires matched equal capture/replay hashes.
3. Instantiates `DependencySnapshotReader` with the result's manifest hash and set ID, then reads every manifest entry to verify every response hash.
4. Recomputes `probe_outcome_hash()` from exact ordered outcomes.
5. Converts error-free sealed search data to `Paper` and `offline_provider_result()`, then uses `reconstruct_frozen_baseline()` and `merge_probe_results()`.
6. Uses merged retrieved/Top-50 IDs and the union of bound baseline post-filter IDs with accepted additions.

The probe row binds the copied lock, result, outcomes, probe snapshot manifest, and all lock-declared source hashes. Every adapter requires the exact gold query set and records only aggregate-safe hash keys in `SourceProjection.binding_hashes`.

Add the fixed orchestrator:

```python
def load_fixed_sources(root: Path = ROOT) -> tuple[SourceProjection, ...]:
    return (
        load_formal_source(
            "formal_baseline_2026_08_10",
            root / "runs/dev-20260810T104256Z-d9e89476d484",
        ),
        load_formal_source(
            "formal_baseline_2026_08_09",
            root / "runs/dev-20260809T061903Z-9bd861e90299",
        ),
        load_legacy_source(
            root / "runs/dev-20260805T035209Z-7af4b103f6cc",
            root / "docs/evidence/title-retention-offline-2026-08-09.json",
        ),
        load_probe_source(
            root
            / "runs/_diag_query_evolution_query-evolution-prompt-v2-full-20260810"
        ),
    )
```

- [ ] **Step 4: Run GREEN and commit**

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest tests/evaluation/test_query_evolution_probe.py tests/integration/test_query_evolution_probe.py tests/scripts/test_rescore_identifier_semantics.py -q
git add src/paper_search/evaluation/query_evolution_probe.py scripts/probe_query_evolution.py tests/evaluation/test_query_evolution_probe.py tests/integration/test_query_evolution_probe.py scripts/rescore_identifier_semantics.py tests/scripts/test_rescore_identifier_semantics.py
git commit -m "feat: load sealed identifier rescore sources"
```

Expected: selected tests pass without network or ledger access.

---

### Task 3: Build and Publish the Fixed Aggregate Report

**Files:**
- Modify: `scripts/rescore_identifier_semantics.py`
- Modify: `tests/scripts/test_rescore_identifier_semantics.py`

**Interfaces:**
- `build_fixed_report() -> SemanticRescoreReport`
- `canonical_report_bytes(report: SemanticRescoreReport) -> bytes`
- `render_markdown(report: SemanticRescoreReport) -> str`
- `publish_report(report, *, json_path, markdown_path) -> None`
- `render_markdown_from_json(json_path, markdown_path) -> None`
- CLI commands: `run`, `render-markdown`; no additional arguments.

- [ ] **Step 1: Write RED orchestration and publication tests**

Tests must prove strict generation failure stops before source loading; the parser exposes no path/network/env/ledger flags; the report is canonical and privacy-safe; existing targets are never overwritten; JSON is written as formal evidence; and `render-markdown` validates canonical existing JSON and creates only a missing Markdown companion.

```python
def test_generation_failure_stops_before_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        rescore,
        "load_verified_identifier_generation",
        lambda **kwargs: (_ for _ in ()).throw(
            ValueError("identifier semantic audit is not passed")
        ),
    )
    monkeypatch.setattr(
        rescore,
        "load_fixed_sources",
        lambda root: (_ for _ in ()).throw(
            AssertionError("sources must not be read")
        ),
    )
    with pytest.raises(ValueError, match="identifier semantic audit"):
        rescore.build_fixed_report()
```

- [ ] **Step 2: Implement fixed inputs and report construction**

Use exactly these inputs and outputs:

```python
PUBLIC_AUDIT = ROOT / "docs/evidence/identifier-map-semantic-audit-2026-08-11.json"
GOLD = ROOT / "data/dev/gold.jsonl"
IDENTITY_EVIDENCE = ROOT / "data/annotation_work/identifier_semantics/identity-evidence.json"
SNAPSHOT_MANIFEST = ROOT / "data/annotation_work/identifier_semantics/snapshots/snapshot-manifest.json"
PRIVATE_AUDIT = ROOT / "data/annotation_work/identifier_semantics/relation-audit.v2.json"
VERIFIED_MAP = ROOT / "data/annotation_work/identifier_semantics/dev-identifier-map.semantic-v2.json"
QUALITY_POLICY = ROOT / "configs/quality_gates_v1.yaml"
OUT_JSON = ROOT / "docs/evidence/identifier-map-semantic-rescore-2026-08-11.json"
OUT_MARKDOWN = ROOT / "docs/identifier-map-semantic-rescore-2026-08-11.md"
```

`build_fixed_report()` must call the strict generation loader before `load_fixed_sources()`, parse the raw quality-policy bytes with the existing policy parser, bind the raw hashes for all six generation artifacts and the policy, call `build_rescore_report()`, and require designated direct hits equal 12 before returning.

- [ ] **Step 3: Implement deterministic privacy-safe publication**

- Canonical JSON: sorted keys, compact separators, UTF-8, finite values, one trailing LF.
- Markdown: deterministic rendering from the validated report model only; summarize source status, metrics, funnel, and decision.
- Run both existing public privacy scanners before writing.
- Use the existing no-replace hard-link pattern: fsync a sibling temporary file, hard-link it to the absent target, then remove the temporary file.
- `run` refuses either existing target, writes formal JSON, then derived Markdown.
- `render-markdown` requires canonical privacy-valid formal JSON and an absent Markdown target; it never reruns sources or overwrites either file.
- Catch only expected `OSError`/`ValueError`, print the safe fixed error, and return exit code 3; success returns 0.

- [ ] **Step 4: Run GREEN and commit**

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest tests/evaluation/test_semantic_rescore.py tests/scripts/test_rescore_identifier_semantics.py tests/evaluation/test_gates.py tests/evaluation/test_query_evolution_probe.py tests/integration/test_query_evolution_probe.py -q
git add scripts/rescore_identifier_semantics.py tests/scripts/test_rescore_identifier_semantics.py
git commit -m "feat: publish verified identifier rescore"
```

Expected: all selected tests pass; no formal report has been executed yet.

---

### Task 4: Verify and Execute One Offline Rescore

**Files:**
- Create after successful execution: `docs/evidence/identifier-map-semantic-rescore-2026-08-11.json`
- Create after successful execution: `docs/identifier-map-semantic-rescore-2026-08-11.md`

**Interfaces:**
- Consumes: completed Tasks 1-3 and fixed local sealed artifacts.
- Produces: one formal aggregate JSON and one derived Markdown report; no other state change.

- [ ] **Step 1: Run focused and full offline verification**

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest tests/evaluation/test_gates.py tests/evaluation/test_semantic_rescore.py tests/evaluation/test_query_evolution_probe.py tests/integration/test_query_evolution_probe.py tests/scripts/test_rescore_identifier_semantics.py -q
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest -m "not online" -q
& 'D:\AI Projects\Projects\.venv\Scripts\ruff.exe' check src scripts tests
& 'D:\AI Projects\Projects\.venv\Scripts\mypy.exe' src scripts
git diff --check
```

Expected: every command exits 0. Stop before execution on the first failure.

- [ ] **Step 2: Execute exactly once**

Require both output targets to be absent, then run:

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' scripts/rescore_identifier_semantics.py run
```

Expected: exit 0 without network, ledger, readiness, capture, replay, or historical-source mutation.

- [ ] **Step 3: Validate result and decision boundary**

```powershell
@'
import json
from pathlib import Path

report = json.loads(
    Path("docs/evidence/identifier-map-semantic-rescore-2026-08-11.json")
    .read_text(encoding="utf-8")
)
assert report["schema_version"] == "identifier-semantic-rescore-v2"
assert report["status"] == "passed"
assert len(report["runs"]) == 4
assert report["runs"][0]["label"] == "formal_baseline_2026_08_10"
assert report["runs"][0]["direct_same_arxiv_hit_count"] == 12
assert all(
    row["pipeline_stages"]["total_gold_associations"]
    == report["total_gold_associations"]
    for row in report["runs"]
)
print(report["decision"])
'@ | & 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -
```

If `next_direction` is null, stop for human review. Do not select a reference method automatically.

- [ ] **Step 4: Reverify and commit only aggregate evidence**

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest tests/evaluation/test_semantic_rescore.py tests/scripts/test_rescore_identifier_semantics.py -q
git diff --check
git add docs/evidence/identifier-map-semantic-rescore-2026-08-11.json docs/identifier-map-semantic-rescore-2026-08-11.md
git commit -m "docs: record verified identifier rescore"
```

Do not stage `HANDOFF.md`, `docs/retrieval-roadmap.md`, `data/budget_ledger.sqlite3`, `deliverables/`, or the historical 2026-08-10 audit.

## Stop Boundary

This plan ends after the offline report. A later reviewed task may select one reference-paper method only when the diagnosis has a non-null, non-sensitive direction, and may run one single-variable offline comparison. Production changes or live capture require a separate design and authorization.
