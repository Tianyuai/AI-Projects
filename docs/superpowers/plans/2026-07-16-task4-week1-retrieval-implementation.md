# Task 4 Week-One Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic, auditable Week-1 pipeline that deduplicates `Paper` records, applies hard filters with uncertainty penalties, ranks candidates with keyword coverage and BM25, evaluates a split through the existing metric contract, and exposes a minimal safe list UI.

**Architecture:** Deduplication, filtering, and lexical ranking are pure functions returning frozen domain models. A thin runner owns provider calls, per-query budgets, snapshot binding, metrics, and atomic artifacts; the UI consumes an injected search service and contains no algorithm logic. Real dev baseline execution remains gated on the Task 2 manifest reaching a frozen state.

**Tech Stack:** Python 3.11, Pydantic 2, `rank-bm25`, FastAPI, httpx, SQLite, pytest, Ruff, mypy strict.

## Global Constraints

- Work only on `codex/task4-week1-retrieval`, based on Task 3 commit `6e9fad4`.
- Never read, print, stage, or commit `.env`; code receives secrets only from the process environment.
- Do not touch `docs/superpowers/specs/2026-07-15-task2-evaluation-design.md` or its line-ending metadata change.
- Do not merge branches. Push only Task 4 commits after all verification and review gates pass.
- Offline tests, Ruff, and mypy use `uv run --no-sync --no-env-file`.
- Keep the existing `Paper`, `QuerySpec`, `ProviderResult`, `EvaluationQuery`, `PredictionRecord`, `EvaluationResult`, `IdentifierMap`, and snapshot contracts backward compatible.
- Every removal and merge is auditable by a stable machine-readable reason.
- Formal artifacts are UTF-8, deterministic, atomic, and reject non-identical overwrite.
- Synthetic fixtures prove engineering behavior but cannot satisfy the real dev baseline, fuzzy-match human audit, Recall-loss target, data-freeze task, or Week-1 stage gate.

---

## File Map

- Create `src/paper_search/processing/deduplicate.py`: pure four-level deduplication, deterministic clustering, field merge, and audit models.
- Modify `src/paper_search/processing/__init__.py`: export Task 4 processing contracts.
- Create `src/paper_search/processing/filter.py`: hard-filter decisions and uncertainty multipliers.
- Create `src/paper_search/ranking/__init__.py`: export lexical-ranking contracts.
- Create `src/paper_search/ranking/lexical.py`: tokenizer, keyword coverage, BM25 normalization, and stable ranking.
- Create `src/paper_search/evaluation/runner.py`: pure candidate pipeline, injected async split runner, CLI, immutable artifacts, and snapshot binding.
- Create `src/paper_search/ui/__init__.py`: UI package boundary.
- Create `src/paper_search/ui/app.py`: injected FastAPI query form and safe result list.
- Create `tests/unit/test_deduplicate.py`: all match levels, non-match guards, clustering, merge precedence, and order.
- Create `tests/unit/test_filter.py`: every rejection and uncertainty rule.
- Create `tests/unit/test_lexical.py`: tokenization, scoring, normalization, penalties, and deterministic ties.
- Create `tests/evaluation/test_runner.py`: pipeline orchestration, metrics, usage, snapshots, artifact safety, and CLI gates.
- Create `tests/integration/test_week1_pipeline.py`: fixture end-to-end from queries and provider results to F1.
- Create `tests/ui/test_app.py`: form, results, escaping, and safe failures.
- Create `tests/fixtures/week1/gold.jsonl`: synthetic non-restricted evaluation queries.
- Create `tests/fixtures/week1/openalex_results.json`: synthetic normalized candidate inputs without credentials.
- Modify `PRD.md`: check only engineering items demonstrated by tests; leave real baseline, freeze, human audit, and Week-1 gate unchecked.

---

### Task 1: Deterministic Paper Deduplication

**Files:**
- Create: `src/paper_search/processing/deduplicate.py`
- Modify: `src/paper_search/processing/__init__.py`
- Test: `tests/unit/test_deduplicate.py`

**Interfaces:**
- Consumes: `Paper`, `DomainModel`, `NonEmptyStr`, `IdentifierMap`, `normalize_paper_id()`, and `normalize_title()`.
- Produces: `MergeDecision`, `DeduplicationResult`, and `deduplicate_papers(papers, *, id_map=None, fuzzy_title_threshold=0.98)`.

- [ ] **Step 1: Write failing DOI and external-ID tests**

Add a `_paper()` factory that always creates a valid `Paper`, then assert DOI precedence and mapped cross-source IDs:

```python
def test_same_doi_merges_and_preserves_first_cluster_position() -> None:
    papers = [
        _paper("openalex:W1", title="First", doi="10.1000/a", sources=["openalex"]),
        _paper("s2:S1", title="Richer", doi="https://doi.org/10.1000/A", abstract="body", sources=["semantic_scholar"]),
    ]

    result = deduplicate_papers(papers)

    assert len(result.papers) == 1
    assert result.papers[0].doi == "10.1000/a"
    assert result.papers[0].sources == ["semantic_scholar", "openalex"]
    assert result.decisions[0].match_rule == "doi"
    assert result.decisions[0].member_ids == ["openalex:W1", "s2:S1"]


def test_identifier_map_merges_cross_source_aliases(tmp_path: Path) -> None:
    path = tmp_path / "id-map.json"
    path.write_text('{"openalex:W2":"s2:S2"}', encoding="utf-8")
    result = deduplicate_papers(
        [_paper("openalex:W2"), _paper("s2:S2")],
        id_map=IdentifierMap.from_path(path),
    )
    assert len(result.papers) == 1
    assert result.decisions[0].match_rule == "external_id"
```

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```powershell
uv run --no-sync --no-env-file pytest tests/unit/test_deduplicate.py -k "doi or identifier_map" -v
```

Expected: collection fails with `ModuleNotFoundError: paper_search.processing.deduplicate`.

- [ ] **Step 3: Implement identifiers, frozen result models, and deterministic union-find**

Implement canonical identifier extraction and a private `_DisjointSet`. For every pair, choose exactly one first matching rule in priority order. Store the rule on the edge; after clustering, choose the highest-priority edge used by the cluster for `MergeDecision`.

```python
MatchRule = Literal["doi", "external_id", "exact_title", "fuzzy_title"]
_RULE_PRIORITY: dict[MatchRule, int] = {
    "doi": 0,
    "external_id": 1,
    "exact_title": 2,
    "fuzzy_title": 3,
}


class MergeDecision(DomainModel):
    representative_id: NonEmptyStr
    member_ids: list[NonEmptyStr]
    match_rule: MatchRule
    match_value: NonEmptyStr


class DeduplicationResult(DomainModel):
    papers: list[Paper]
    decisions: list[MergeDecision]
```

Normalize DOI with `normalize_paper_id(value, kind="doi")`. Build external identifiers from DOI, `openalex_id`, `semantic_scholar_id`, and supported canonical IDs; resolve every identifier through the optional `IdentifierMap`.

- [ ] **Step 4: Run focused DOI/external-ID tests and verify GREEN**

Run the Step 2 command. Expected: both selected tests pass.

- [ ] **Step 5: Add failing exact/fuzzy-title and guard tests**

```python
def test_normalized_exact_title_merges() -> None:
    result = deduplicate_papers([
        _paper("openalex:W3", title="Graph-Based Retrieval"),
        _paper("s2:S3", title="graph based retrieval"),
    ])
    assert result.decisions[0].match_rule == "exact_title"


def test_fuzzy_title_requires_same_year_and_author_surname() -> None:
    left = _paper("openalex:W4", title="Neural Paper Retrieval for Science", authors=["Ada Lovelace"], publication_year=2024)
    close = _paper("s2:S4", title="Neural Paper Retrieval in Science", authors=["A. Lovelace"], publication_year=2024)
    wrong_year = _paper("openalex:W5", title=close.title, authors=close.authors, publication_year=2023)

    assert len(deduplicate_papers([left, close], fuzzy_title_threshold=0.80).papers) == 1
    assert len(deduplicate_papers([left, wrong_year], fuzzy_title_threshold=0.80).papers) == 2


def test_fuzzy_title_does_not_merge_when_author_is_missing() -> None:
    left = _paper("openalex:W6", title="A Study of Search", authors=[], publication_year=2024)
    right = _paper("s2:S6", title="Study of Search", authors=["Ada Lovelace"], publication_year=2024)
    assert len(deduplicate_papers([left, right], fuzzy_title_threshold=0.70).papers) == 2
```

- [ ] **Step 6: Implement title matching and validation**

Use `difflib.SequenceMatcher(None, left, right).ratio()` on `normalize_title()` values. Derive author surnames by NFKC/casefold, punctuation-to-space normalization, and the final non-empty token. Reject boolean/non-finite/out-of-range thresholds with `ValueError`; require `0.0 <= threshold <= 1.0`.

- [ ] **Step 7: Add failing clustering, representative, field merge, and stable-order tests**

Cover a three-member transitive cluster, single-member omission from decisions, richer representative selection, scalar fallback, author/source ordered union, conflicting identifiers, and identical repeat output via `model_dump(mode="json")`.

```python
def test_output_is_deterministic_and_singletons_have_no_decision() -> None:
    papers = [_paper("openalex:W7"), _paper("openalex:W8", title="Unique")]
    first = deduplicate_papers(papers).model_dump(mode="json")
    second = deduplicate_papers(papers).model_dump(mode="json")
    assert first == second
    assert first["decisions"] == []
```

- [ ] **Step 8: Implement representative selection and merged `Paper` construction**

Rank representatives by `(has_doi, stable_external_id_count, has_abstract, author_count, has_year, has_venue, has_citation_count)` descending and original index ascending. Fill missing scalar fields from members in original order. Ordered-union authors and sources. Keep the representative identifier on conflicts.

- [ ] **Step 9: Run Task 1 tests, Ruff, and mypy**

```powershell
uv run --no-sync --no-env-file pytest tests/unit/test_deduplicate.py -v
uv run --no-sync --no-env-file ruff check src/paper_search/processing/deduplicate.py src/paper_search/processing/__init__.py tests/unit/test_deduplicate.py
uv run --no-sync --no-env-file mypy src/paper_search/processing
```

Expected: all commands exit `0`.

- [ ] **Step 10: Commit Task 1**

```powershell
git add -- src/paper_search/processing/deduplicate.py src/paper_search/processing/__init__.py tests/unit/test_deduplicate.py
git commit -m "feat: add deterministic paper deduplication"
```

---

### Task 2: Hard Filters and Uncertainty Penalties

**Files:**
- Create: `src/paper_search/processing/filter.py`
- Modify: `src/paper_search/processing/__init__.py`
- Test: `tests/unit/test_filter.py`

**Interfaces:**
- Consumes: `Paper`, `QuerySpec`, `DomainModel`, and `normalize_title()`.
- Produces: `AcceptedPaper`, `RejectedPaper`, `FilterResult`, and `apply_hard_filters(papers, query)`.

- [ ] **Step 1: Write failing tests for every hard-rejection code**

Use one parametrized test with expected codes `retracted`, `missing_stable_id`, `year_out_of_range`, `venue_mismatch`, and `excluded_term`. Verify the first rule wins when a paper violates multiple rules.

```python
@pytest.mark.parametrize(
    ("paper", "query", "code"),
    [
        (_paper(is_retracted=True), _query(), "retracted"),
        (_paper(canonical_id="title:unstable", doi=None, openalex_id=None), _query(), "missing_stable_id"),
        (_paper(publication_year=2019), _query(year_from=2020), "year_out_of_range"),
        (_paper(venue="Other"), _query(venues=["NeurIPS"]), "venue_mismatch"),
        (_paper(title="Survey paper"), _query(exclusions=["survey"]), "excluded_term"),
    ],
)
def test_hard_filter_reason(paper: Paper, query: QuerySpec, code: str) -> None:
    result = apply_hard_filters([paper], query)
    assert result.accepted == []
    assert result.rejected[0].reason_code == code
    assert result.rejected[0].reason
```

- [ ] **Step 2: Run tests and verify RED**

```powershell
uv run --no-sync --no-env-file pytest tests/unit/test_filter.py -k hard_filter -v
```

Expected: missing module or missing `apply_hard_filters`.

- [ ] **Step 3: Implement frozen models and ordered rejection rules**

```python
class AcceptedPaper(DomainModel):
    paper: Paper
    uncertainty_reasons: list[NonEmptyStr]
    score_multiplier: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class RejectedPaper(DomainModel):
    paper: Paper
    reason_code: NonEmptyStr
    reason: NonEmptyStr


class FilterResult(DomainModel):
    accepted: list[AcceptedPaper]
    rejected: list[RejectedPaper]
```

Match venues and exclusion text after `normalize_title()`. Stable IDs are DOI, OpenAlex ID, Semantic Scholar ID, or canonical IDs prefixed by `doi:`, `openalex:`, `s2:`, or `arxiv:`.

- [ ] **Step 4: Add failing uncertainty and order tests**

```python
def test_missing_constrained_fields_are_downweighted_not_removed() -> None:
    result = apply_hard_filters(
        [_paper(publication_year=None, venue=None, abstract=None, is_retracted=None)],
        _query(year_from=2020, venues=["NeurIPS"], exclusions=["survey"]),
    )
    accepted = result.accepted[0]
    assert accepted.uncertainty_reasons == [
        "missing_year",
        "missing_venue",
        "unknown_retraction_status",
        "missing_abstract_for_exclusion",
    ]
    assert accepted.score_multiplier == pytest.approx(0.7)


def test_filter_keeps_input_order_with_separate_audit_lists() -> None:
    result = apply_hard_filters([_paper("openalex:W1"), _paper("openalex:W2", is_retracted=True)], _query())
    assert [item.paper.canonical_id for item in result.accepted] == ["openalex:W1"]
    assert [item.paper.canonical_id for item in result.rejected] == ["openalex:W2"]
```

- [ ] **Step 5: Implement uncertainty accumulation and multiplier**

Append reasons in the exact order specified by the design. Calculate `max(0.7, 0.9 ** len(reasons))`; do not mutate the `Paper`.

- [ ] **Step 6: Run Task 2 verification**

```powershell
uv run --no-sync --no-env-file pytest tests/unit/test_filter.py -v
uv run --no-sync --no-env-file ruff check src/paper_search/processing/filter.py src/paper_search/processing/__init__.py tests/unit/test_filter.py
uv run --no-sync --no-env-file mypy src/paper_search/processing
```

- [ ] **Step 7: Commit Task 2**

```powershell
git add -- src/paper_search/processing/filter.py src/paper_search/processing/__init__.py tests/unit/test_filter.py
git commit -m "feat: add auditable hard filters"
```

---

### Task 3: Keyword Coverage and BM25 Ranking

**Files:**
- Create: `src/paper_search/ranking/__init__.py`
- Create: `src/paper_search/ranking/lexical.py`
- Test: `tests/unit/test_lexical.py`

**Interfaces:**
- Consumes: `QuerySpec` and `Sequence[AcceptedPaper]`.
- Produces: `LexicalScore`, `tokenize_text(value)`, and `rank_lexically(query, candidates)`.

- [ ] **Step 1: Write failing tokenizer and keyword-coverage tests**

```python
def test_tokenizer_is_unicode_normalized_and_deterministic() -> None:
    assert tokenize_text("Graph-based ＡＩ, graph!") == ["graph", "based", "ai", "graph"]


def test_keyword_coverage_uses_unique_query_tokens() -> None:
    ranked = rank_lexically(_query("graph graph retrieval"), [_accepted(title="Graph methods")])
    assert ranked[0].keyword_coverage == pytest.approx(0.5)
```

- [ ] **Step 2: Run tests and verify RED**

```powershell
uv run --no-sync --no-env-file pytest tests/unit/test_lexical.py -k "tokenizer or coverage" -v
```

- [ ] **Step 3: Implement tokenizer, query/document builders, and frozen score model**

```python
SCORING_VERSION = "week1-lexical-v1"


class LexicalScore(DomainModel):
    paper: Paper
    bm25_score: float = Field(allow_inf_nan=False)
    normalized_bm25: float = Field(ge=0, le=1, allow_inf_nan=False)
    keyword_coverage: float = Field(ge=0, le=1, allow_inf_nan=False)
    uncertainty_multiplier: float = Field(ge=0, le=1, allow_inf_nan=False)
    final_score: float = Field(ge=0, le=1, allow_inf_nan=False)
```

Use NFKC, casefold, and `re.findall(r"[^\W_]+", value, flags=re.UNICODE)`.

- [ ] **Step 4: Add failing BM25, normalization, penalty, empty, and tie tests**

Verify raw BM25 is retained, normalized values are within `[0, 1]`, equal raw scores normalize to zero, uncertainty changes order, empty candidates return `[]`, and exact ties retain input order before canonical-ID fallback.

```python
def test_uncertainty_multiplier_changes_final_order() -> None:
    candidates = [
        _accepted("openalex:W1", title="graph retrieval", multiplier=0.7),
        _accepted("openalex:W2", title="graph retrieval", multiplier=1.0),
    ]
    ranked = rank_lexically(_query("graph retrieval"), candidates)
    assert [item.paper.canonical_id for item in ranked] == ["openalex:W2", "openalex:W1"]
```

- [ ] **Step 5: Implement BM25 and stable composite ranking**

Build `BM25Okapi(document_tokens)` once. Convert returned NumPy scalars to Python `float`. Min-max normalize when `maximum > minimum`, otherwise use `0.0`. Compute:

```python
final_score = (
    0.7 * normalized_bm25 + 0.3 * keyword_coverage
) * candidate.score_multiplier
```

Sort by negative final score, negative coverage, negative raw BM25, original index, then canonical ID.

- [ ] **Step 6: Run Task 3 verification**

```powershell
uv run --no-sync --no-env-file pytest tests/unit/test_lexical.py -v
uv run --no-sync --no-env-file ruff check src/paper_search/ranking tests/unit/test_lexical.py
uv run --no-sync --no-env-file mypy src/paper_search/ranking
```

- [ ] **Step 7: Commit Task 3**

```powershell
git add -- src/paper_search/ranking tests/unit/test_lexical.py
git commit -m "feat: add deterministic lexical ranking"
```

---

### Task 4: Pure Candidate Pipeline and Injected Evaluation Runner

**Files:**
- Create: `src/paper_search/evaluation/runner.py`
- Test: `tests/evaluation/test_runner.py`

**Interfaces:**
- Consumes: `deduplicate_papers()`, `apply_hard_filters()`, `rank_lexically()`, `evaluate()`, `HardBudgetController`, `ProviderResult[list[Paper]]`, `SQLiteResponseCache`, and `RuntimeConfig`.
- Produces: `PipelineResult`, `QueryRunRecord`, `RunResult`, `process_candidates()`, and `async run_evaluation(gold, *, provider, cache, config, output, id_map=None) -> RunResult`.

- [ ] **Step 1: Write failing pure-pipeline test**

```python
def test_process_candidates_composes_dedup_filter_and_rank() -> None:
    result = process_candidates(
        _query_spec("graph retrieval"),
        [_paper("openalex:W1", doi="10.1000/a"), _paper("s2:S1", doi="10.1000/a")],
    )
    assert len(result.deduplication.papers) == 1
    assert result.filtering.rejected == []
    assert [item.paper.canonical_id for item in result.ranked] == [result.deduplication.papers[0].canonical_id]
```

- [ ] **Step 2: Run pure-pipeline test and verify RED**

```powershell
uv run --no-sync --no-env-file pytest tests/evaluation/test_runner.py::test_process_candidates_composes_dedup_filter_and_rank -v
```

- [ ] **Step 3: Implement `PipelineResult` and `process_candidates()`**

```python
class PipelineResult(DomainModel):
    deduplication: DeduplicationResult
    filtering: FilterResult
    ranked: list[LexicalScore]


def process_candidates(
    query: QuerySpec,
    papers: Sequence[Paper],
    *,
    id_map: IdentifierMap | None = None,
) -> PipelineResult:
    deduplicated = deduplicate_papers(papers, id_map=id_map)
    filtered = apply_hard_filters(deduplicated.papers, query)
    ranked = rank_lexically(query, filtered.accepted)
    return PipelineResult(deduplication=deduplicated, filtering=filtered, ranked=ranked)
```

- [ ] **Step 4: Write failing async multi-query and failure-isolation tests**

Define a `SearchProvider` protocol matching Task 3:

```python
class SearchProvider(Protocol):
    async def search(
        self,
        query: str,
        filters: dict[str, object],
        limit: int,
        reservation: BudgetReservation,
    ) -> ProviderResult[list[Paper]]: ...
```

The fake returns non-empty papers for `q1` and a structured error plus empty data for `q2`. Assert both predictions exist, only the failed query is empty, aggregate metrics include both, usage sums calls/latency, and query order matches gold order.

- [ ] **Step 5: Implement per-query runner models, budget reservation, and isolation**

For each query create a fresh `HardBudgetController(config.budget)`, reserve `UsageEstimate(search_api_calls=config.budget.max_search_api_calls, elapsed_ms=config.budget.max_elapsed_seconds * 1000)`, call the provider, settle actual usage, and process candidates. Convert top `max_output_papers` to `PredictionRecord`. Parse `provenance["cache_keys"]` as a JSON string list and reject malformed values.

Use these exact result contracts so later artifact and UI work does not infer untyped dictionaries:

```python
class QueryRunRecord(DomainModel):
    query_id: NonEmptyStr
    prediction: PredictionRecord
    pipeline: PipelineResult
    usage: UsageActual
    latency_ms: NonNegativeInt
    cache_keys: list[NonEmptyStr]
    errors: list[ErrorDetail]


class RunResult(DomainModel):
    evaluation: EvaluationResult
    query_runs: list[QueryRunRecord]
    usage: UsageActual
    snapshot_manifest: NonEmptyStr


async def run_evaluation(
    gold: Sequence[EvaluationQuery],
    *,
    provider: SearchProvider,
    cache: SQLiteResponseCache,
    config: RuntimeConfig,
    output: Path,
    id_map: IdentifierMap | None = None,
) -> RunResult: ...
```

Aggregate `UsageActual` by summing numeric fields and preserving `cost_cny=None` unless every query reports a cost. `snapshot_manifest` is the POSIX relative path `snapshot_manifest.json`, never an absolute path.

The `EvaluationQuery` fallback `QuerySpec` uses the raw query as `original_query` and `research_goal`, empty optional lists, and no inferred hard filters. This prevents Task 4 from pretending to implement Task 5 parsing.

- [ ] **Step 6: Write failing snapshot/artifact tests**

Populate a temporary `SQLiteResponseCache` with two safe raw fixture responses. Assert `run_evaluation()` writes:

```text
predictions.jsonl
metrics.json
usage.json
run.json
deduplication.jsonl
filtering.jsonl
snapshot_manifest.json
snapshots/openalex-0001.json
```

Validate the manifest, exact input hashes, config hash, scoring version, relative manifest path, sorted JSON keys, trailing newline, idempotent identical rerun, and refusal to overwrite changed output. Scan every text artifact for a sentinel API key.

- [ ] **Step 7: Implement frozen artifact assembly**

Reuse `write_jsonl_atomic()` for predictions. Add one private `_write_frozen_json(path, payload)` that serializes with `ensure_ascii=False`, `sort_keys=True`, `indent=2`, `allow_nan=False`, and a trailing newline, then delegates to `write_frozen_bytes()`. Export the ordered unique cache keys through `cache.export_snapshot()`, validate it before writing metrics, and store only relative paths in `run.json`.

Audit JSONL records contain `query_id`, paper IDs, merge decisions, rejection codes, and uncertainty reasons; they do not contain raw provider bytes, request URLs, headers, or secrets.

- [ ] **Step 8: Run Task 4 verification**

```powershell
uv run --no-sync --no-env-file pytest tests/evaluation/test_runner.py -v
uv run --no-sync --no-env-file ruff check src/paper_search/evaluation/runner.py tests/evaluation/test_runner.py
uv run --no-sync --no-env-file mypy src/paper_search/evaluation/runner.py
```

- [ ] **Step 9: Commit Task 4**

```powershell
git add -- src/paper_search/evaluation/runner.py tests/evaluation/test_runner.py
git commit -m "feat: add week-one evaluation runner"
```

---

### Task 5: CLI, Frozen-Data Gate, and End-to-End Fixture

**Files:**
- Modify: `src/paper_search/evaluation/runner.py`
- Create: `tests/integration/test_week1_pipeline.py`
- Create: `tests/fixtures/week1/gold.jsonl`
- Create: `tests/fixtures/week1/openalex_results.json`
- Modify: `tests/evaluation/test_runner.py`

**Interfaces:**
- Consumes: `load_runtime_config(config_path, env_file=None)`, `OpenAlexProvider`, `SQLiteResponseCache`, and `run_evaluation()`.
- Produces: `main(argv=None) -> int` and the required `python -m paper_search.evaluation.runner --config ... --split ... --output ...` command.

- [ ] **Step 1: Write failing manifest and split-resolution tests**

In a temporary working directory create `data/manifest.json` and `data/dev/gold.jsonl`. Assert statuses other than `frozen` return CLI exit code `2` with `data manifest is not frozen`, missing gold returns `2`, unknown split returns `2`, and no synthetic fallback output is created.

```python
def test_cli_refuses_unfrozen_dev_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    _write_manifest(tmp_path, status="waiting_for_human_label_freeze")
    monkeypatch.chdir(tmp_path)
    assert main(["--config", str(CONFIG), "--split", "dev", "--output", "out"]) == 2
    assert "data manifest is not frozen" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()
```

- [ ] **Step 2: Implement parser and frozen split validation**

Parser options are exactly `--config`, `--split`, and `--output`. Load `data/manifest.json`, require `status == "frozen"`, resolve `partitions[split].gold_path` under `data/`, reject absolute or escaping paths, verify the file exists, and compare declared SHA-256 if present.

- [ ] **Step 3: Write failing process-environment and provider-construction tests**

Monkeypatch `OPENALEX_API_KEY` with a sentinel and patch the provider factory. Assert `load_runtime_config` is called with `env_file=None`, the key reaches only the provider constructor, and no captured output or created artifact includes it. With the variable absent, expect exit code `2` and a safe `OPENALEX_API_KEY is required` message.

- [ ] **Step 4: Implement async CLI boundary**

Create one `httpx.AsyncClient` with explicit connect/read/write/pool timeouts, one `SQLiteResponseCache(output.parent / ".cache" / "openalex.sqlite3")`, and one `OpenAlexProvider`. Call `asyncio.run(run_evaluation(...))`. Catch only expected `OSError`, `ValueError`, `FileExistsError`, `KeyError`, and Pydantic validation errors at the CLI boundary; return `2` without raw exception repr.

- [ ] **Step 5: Add fixed fixture end-to-end test**

The synthetic fixture contains two queries, a duplicate DOI pair, a retracted paper, an unknown-year accepted paper, and relevant/non-relevant candidates. Use a fake provider plus temporary SQLite cache entries, run the full split, and assert:

```python
assert result.evaluation.summary.query_count == 2
assert result.evaluation.summary.macro_f1 > 0
assert result.query_runs[0].pipeline.deduplication.decisions
assert result.query_runs[0].pipeline.filtering.rejected[0].reason_code == "retracted"
validate_snapshot_manifest(output / "snapshot_manifest.json")
```

- [ ] **Step 6: Run Task 5 verification**

```powershell
uv run --no-sync --no-env-file pytest tests/evaluation/test_runner.py tests/integration/test_week1_pipeline.py -v
uv run --no-sync --no-env-file ruff check src/paper_search/evaluation/runner.py tests/evaluation/test_runner.py tests/integration/test_week1_pipeline.py
uv run --no-sync --no-env-file mypy src/paper_search/evaluation/runner.py
```

- [ ] **Step 7: Commit Task 5**

```powershell
git add -- src/paper_search/evaluation/runner.py tests/evaluation/test_runner.py tests/integration/test_week1_pipeline.py tests/fixtures/week1
git commit -m "feat: add frozen week-one baseline CLI"
```

---

### Task 6: Minimal Safe Collaborator UI

**Files:**
- Create: `src/paper_search/ui/__init__.py`
- Create: `src/paper_search/ui/app.py`
- Create: `tests/ui/test_app.py`

**Interfaces:**
- Consumes: an injected async `SearchService` returning `PipelineResult`.
- Produces: `create_app(search_service) -> FastAPI` and module-level `app` with an unavailable default service.

- [ ] **Step 1: Write failing home-page and injected-result tests**

Use `httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")`. Assert `GET /` returns a form with one named `query` input. Post URL-encoded query data to `/search`; assert the fake service receives the stripped query once and the response lists title, authors, year, venue, sources, final score, uncertainty reasons, the deduplication `match_rule`, and rejected-paper `reason_code`.

- [ ] **Step 2: Run UI tests and verify RED**

```powershell
uv run --no-sync --no-env-file pytest tests/ui/test_app.py -k "home or result" -v
```

- [ ] **Step 3: Implement UI protocol, form parsing, and escaped renderer**

Avoid `python-multipart`: read `await request.body()`, decode UTF-8, and use `urllib.parse.parse_qs`. Reject empty or repeated `query` fields with status `400`. Escape every external string through `html.escape(..., quote=True)`. Render ranked papers first, followed by collapsible `<details>` sections for merge decisions, rejected papers, and accepted-paper uncertainty reasons; do not recompute any rule in the renderer.

```python
class SearchService(Protocol):
    async def __call__(self, query: str) -> PipelineResult: ...


def create_app(search_service: SearchService) -> FastAPI:
    application = FastAPI()

    @application.get("/", response_class=HTMLResponse)
    async def home() -> str:
        return _render_form()

    @application.post("/search", response_class=HTMLResponse)
    async def search(request: Request) -> HTMLResponse:
        query = _parse_query(await request.body())
        result = await search_service(query)
        return HTMLResponse(_render_results(query, result))

    return application
```

- [ ] **Step 4: Add failing XSS, safe-error, and algorithm-boundary tests**

Return a paper containing `<script>` in title and assert the raw tag is absent while escaped text is present. Raise an exception containing a sentinel credential and assert the response is status `503`, includes only `search is temporarily unavailable`, and excludes the sentinel. Use AST or source assertions to ensure the UI does not import `rank_bm25` or define dedup/filter/rank functions.

- [ ] **Step 5: Implement safe failure handling and unavailable default app**

Catch service exceptions at the route, return a constant safe message, and never interpolate exception text. The module-level `app` uses a default async service that raises a private availability error; production wiring is deferred to the application composition task, not duplicated in UI.

- [ ] **Step 6: Run Task 6 verification**

```powershell
uv run --no-sync --no-env-file pytest tests/ui/test_app.py -v
uv run --no-sync --no-env-file ruff check src/paper_search/ui tests/ui/test_app.py
uv run --no-sync --no-env-file mypy src/paper_search/ui
```

- [ ] **Step 7: Commit Task 6**

```powershell
git add -- src/paper_search/ui tests/ui/test_app.py
git commit -m "feat: add safe week-one results UI"
```

---

### Task 7: PRD Evidence, Full Verification, and Independent Review

**Files:**
- Modify: `PRD.md`
- Modify only if review finds defects: Task 4 implementation/test files from Tasks 1–6.

**Interfaces:**
- Consumes: all Task 4 components and tests.
- Produces: truthful PRD state, final verification evidence, and a review-clean branch.

- [ ] **Step 1: Run the complete focused Task 4 suite**

```powershell
uv run --no-sync --no-env-file pytest tests/unit/test_deduplicate.py tests/unit/test_filter.py tests/unit/test_lexical.py tests/evaluation/test_runner.py tests/integration/test_week1_pipeline.py tests/ui/test_app.py -v
```

Expected: all selected tests pass with no online skip involved.

- [ ] **Step 2: Run full repository verification**

```powershell
uv run --no-sync --no-env-file pytest -q
uv run --no-sync --no-env-file ruff check .
uv run --no-sync --no-env-file mypy src
git diff --check
```

Expected: pytest passes with only the existing explicitly marked online test permitted to skip; Ruff, mypy, and diff check exit `0`.

- [ ] **Step 3: Run safety and scope audits**

```powershell
git diff --name-only 6e9fad4...HEAD
git status --short
git grep -n -I -E "OPENALEX_API_KEY=.+|Authorization:[[:space:]]*(Bearer|Basic)[[:space:]]+" -- . ":(exclude).env*"
```

Expected: only Task 4 docs/code/tests/fixtures and truthful PRD changes appear; the pre-existing Task 2 design metadata modification remains unstaged; credential grep returns no secret-bearing values.

- [ ] **Step 4: Update only demonstrated PRD checkboxes**

Mark complete:

- DOI/cross-source/exact/fuzzy dedup tests;
- hard filtering and uncertainty downweighting;
- keyword coverage and BM25;
- required CLI implementation;
- per-query/aggregate/call/latency output;
- collaborator query box and paper list.

Leave unchecked and annotate as blocked by the unfrozen collaborator dataset:

- real dev baseline execution;
- dev/validation/sampling freeze;
- Week-1 stage gate;
- human fuzzy-title accuracy audit and real Recall-loss evidence.

- [ ] **Step 5: Commit PRD evidence**

```powershell
git add -- PRD.md
git commit -m "docs: record Task 4 engineering evidence"
```

- [ ] **Step 6: Request independent code review**

Reviewer prompt:

```text
Review codex/task4-week1-retrieval against PRD.md Task 4 and
docs/superpowers/specs/2026-07-16-task4-week1-retrieval-design.md.
Inspect 6e9fad4...HEAD. Prioritize data-loss/false-merge risks, filter semantics,
determinism, budget and snapshot integrity, secret leakage, artifact overwrite
protection, CLI failure behavior, UI XSS, tests that can pass falsely, and scope
creep. Report Critical/Important/Minor findings with file and line references.
Do not modify files.
```

- [ ] **Step 7: Resolve findings with TDD**

For every valid finding, first add or tighten a test that fails for the reported behavior, run it to confirm RED, implement the smallest fix, run focused tests to confirm GREEN, then commit only the touched Task 4 files:

```powershell
git add -- <exact Task 4 test files> <exact Task 4 implementation files>
git commit -m "fix: address Task 4 review findings"
```

If a finding is invalid, record the concrete contract and test evidence rather than changing code.

- [ ] **Step 8: Repeat final verification after review fixes**

Repeat Steps 1–3 from the post-fix HEAD. Do not reuse pre-fix output.

- [ ] **Step 9: Push without merging**

```powershell
git push -u origin codex/task4-week1-retrieval
```

Report engineering completion separately from the still-blocked real dev baseline and Week-1 stage gate.

---

## Execution Order and Checkpoints

Execute Tasks 1–3 as the first checkpoint, Task 4 as the second, Tasks 5–6 as the third, and Task 7 as the final verification/review checkpoint. At each checkpoint, inspect `git status --short`, confirm the unrelated Task 2 design modification is neither staged nor changed by Task 4, and report exact test counts rather than only saying “tests pass.”

Do not run the real online baseline during implementation. It requires both a process-environment OpenAlex key and a genuinely frozen `data/manifest.json`; having only the key is insufficient for formal acceptance.
