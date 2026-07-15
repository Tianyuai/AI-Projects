# Task 2 Dataset Adaptation and Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible, offline-first evaluation subsystem for PaSa data, ranked paper predictions, deterministic dataset freezing, and human-annotation agreement checks.

**Architecture:** Add a focused `paper_search.evaluation` package with a strict internal contract. External PaSa and future competition formats are converted at adapter boundaries; normalization, JSONL validation, scoring, sampling, and annotation logic remain provider-independent. Networked preparation is isolated in one script, while all metric tests run offline.

**Tech Stack:** Python 3.11, Pydantic 2, pytest, standard-library `argparse`/`hashlib`/`json`/`urllib`, Ruff, mypy, uv.

## Global Constraints

- Work only on branch `codex/task2-evaluation` in `D:\AI Projects\.worktrees\task2-evaluation`.
- Fixed PaSa repository: `CarlanLark/pasa-dataset` at revision `232428b0c867268c3b8ded90db4d98c1b30501d6`.
- Fixed random seed: `20260714`; sample sizes: dev 60, validation 30, simulated test 50.
- Internal IDs are exactly `doi:`, `arxiv:`, `openalex:`, `s2:`, or `title:` namespaced strings.
- Gold IDs reject duplicates after normalization; ranked predictions retain duplicates until scoring, then keep the first occurrence.
- Unknown prediction query IDs, malformed JSONL, unknown fields, mapping conflicts/cycles, insufficient strata, and inconsistent frozen outputs fail closed.
- Core evaluation is offline and must never read API keys or call the network.
- Real `.env`, gated raw data, gold query text, annotation workbooks, and tokens stay untracked.
- All generated JSON uses UTF-8, deterministic key ordering, a trailing newline, and atomic replacement.
- Human-label status remains `waiting_for_human_label_freeze` until the two annotators finish and agreement is accepted.

---

## File Map

- Create `src/paper_search/evaluation/__init__.py`: public evaluation API.
- Create `src/paper_search/evaluation/dataset.py`: internal models, ID normalization, JSONL, hashes, ID maps, deterministic sampling, frozen writes.
- Create `src/paper_search/evaluation/official_adapter.py`: PaSa and fixed prediction adapters.
- Create `src/paper_search/evaluation/metrics.py`: query metrics, aggregate evaluation, CLI.
- Create `src/paper_search/evaluation/annotation.py`: annotation schema and Cohen's kappa.
- Create `scripts/prepare_task2_data.py`: gated download, integrity validation, sampling, manifests, work-package orchestration.
- Create `tests/evaluation/test_dataset.py`: models, identifiers, JSONL, mapping, sampling, freezing.
- Create `tests/evaluation/test_official_adapter.py`: PaSa and prediction format conversion.
- Create `tests/evaluation/test_metrics.py`: per-query, macro/micro, mapping, missing/unknown queries.
- Create `tests/evaluation/test_cli.py`: deterministic CLI output and failure behavior.
- Create `tests/evaluation/test_annotation.py`: annotation validation and kappa.
- Create `tests/evaluation/test_prepare_data.py`: offline downloader boundary and manifest orchestration.
- Create `tests/fixtures/evaluation/gold.jsonl`, `tests/fixtures/evaluation/predictions.jsonl`, `tests/fixtures/evaluation/id_map.json`: non-restricted synthetic fixtures.
- Modify `.env.example`: add `HF_TOKEN=`.
- Modify `.gitignore`: ignore restricted Task 2 outputs while retaining safe manifests, split ID lists, and stress data.
- Create `data/README.md`, `data/annotation_guide.md`, `data/manifest.example.json`: reproduction and human-workflow contracts without restricted records.
- Create `data/stress/queries.jsonl`: 24 original stress queries with category/language tags and no PaSa text.

---

### Task 1: Internal models and paper identifier normalization

**Files:**
- Create: `src/paper_search/evaluation/__init__.py`
- Create: `src/paper_search/evaluation/dataset.py`
- Create: `tests/evaluation/test_dataset.py`

**Interfaces:**
- Produces: `normalize_paper_id(value: str, *, kind: str | None = None) -> str`
- Produces: `normalize_title(value: str) -> str`
- Produces: `EvaluationQuery` and `PredictionRecord` frozen Pydantic models.

- [x] **Step 1: Write failing normalization tests**

```python
@pytest.mark.parametrize(
    ("raw", "kind", "expected"),
    [
        ("https://doi.org/10.1000/ABC", None, "doi:10.1000/abc"),
        ("arXiv:2501.10120v3", None, "arxiv:2501.10120"),
        ("https://arxiv.org/pdf/1706.03762v5.pdf", None, "arxiv:1706.03762"),
        ("https://openalex.org/w123", None, "openalex:W123"),
        ("https://www.semanticscholar.org/paper/example/ABC123", None, "s2:ABC123"),
        ("Ａ Study:  On RAG!", "title", "title:a study on rag"),
    ],
)
def test_normalize_paper_id(raw: str, kind: str | None, expected: str) -> None:
    assert normalize_paper_id(raw, kind=kind) == expected

@pytest.mark.parametrize("raw", ["", "ordinary untyped title", "openalex:123", "doi:not-a-doi"])
def test_normalize_paper_id_rejects_empty_ambiguous_or_invalid_values(raw: str) -> None:
    with pytest.raises(ValueError):
        normalize_paper_id(raw)
```

- [x] **Step 2: Run the tests and verify RED**

Run: `uv run pytest tests/evaluation/test_dataset.py -v`

Expected: collection fails because `paper_search.evaluation.dataset` does not exist.

- [x] **Step 3: Implement the normalization API and frozen models**

```python
class EvaluationQuery(DomainModel):
    query_id: NonEmptyStr
    query: NonEmptyStr
    relevant_paper_ids: list[NonEmptyStr] = Field(default_factory=list)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("relevant_paper_ids")
    @classmethod
    def normalize_gold_ids(cls, values: list[str]) -> list[str]:
        normalized = [normalize_paper_id(value) for value in values]
        if len(normalized) != len(set(normalized)):
            raise ValueError("relevant_paper_ids contains duplicate canonical IDs")
        return normalized


class PredictionRecord(DomainModel):
    query_id: NonEmptyStr
    predicted_paper_ids: list[NonEmptyStr] = Field(default_factory=list)

    @field_validator("predicted_paper_ids")
    @classmethod
    def normalize_prediction_ids(cls, values: list[str]) -> list[str]:
        return [normalize_paper_id(value) for value in values]
```

Implement `normalize_paper_id` with anchored regular expressions for DOI, modern/legacy arXiv, `W` plus digits for OpenAlex, non-empty Semantic Scholar IDs, and explicit `title` normalization using Unicode NFKC, `casefold`, punctuation-to-space, and whitespace collapse. Bare strings without a recognized provider prefix must raise `ValueError`.

- [x] **Step 4: Add model-boundary tests**

```python
def test_evaluation_query_normalizes_ids_and_is_frozen() -> None:
    query = EvaluationQuery(
        query_id=" q1 ",
        query=" RAG evaluation ",
        relevant_paper_ids=["arXiv:2501.10120v2"],
        metadata={"split": "dev"},
    )
    assert query.query_id == "q1"
    assert query.relevant_paper_ids == ["arxiv:2501.10120"]
    with pytest.raises(ValidationError):
        query.query = "changed"


def test_evaluation_query_rejects_duplicates_after_normalization() -> None:
    with pytest.raises(ValidationError, match="duplicate"):
        EvaluationQuery(
            query_id="q1",
            query="RAG",
            relevant_paper_ids=["arXiv:2501.10120", "2501.10120v2"],
        )


def test_prediction_record_preserves_ranked_duplicates() -> None:
    record = PredictionRecord(
        query_id="q1",
        predicted_paper_ids=["doi:10.1000/A", "https://doi.org/10.1000/a"],
    )
    assert record.predicted_paper_ids == ["doi:10.1000/a", "doi:10.1000/a"]
```

- [x] **Step 5: Run GREEN and quality checks**

Run: `uv run pytest tests/evaluation/test_dataset.py -v`

Expected: all Task 1 tests pass.

Run: `uv run ruff check src/paper_search/evaluation tests/evaluation`

Expected: `All checks passed!`

Run: `uv run mypy src/paper_search/evaluation`

Expected: `Success: no issues found`.

- [x] **Step 6: Commit Task 1**

```powershell
git add src/paper_search/evaluation/__init__.py src/paper_search/evaluation/dataset.py tests/evaluation/test_dataset.py
git commit -m "feat: add evaluation data models and identifier normalization"
```

---

### Task 2: Strict JSONL I/O, hashing, and identifier maps

**Files:**
- Modify: `src/paper_search/evaluation/dataset.py`
- Modify: `tests/evaluation/test_dataset.py`

**Interfaces:**
- Produces: `read_jsonl(path: Path, model_type: type[T]) -> list[T]`
- Produces: `write_jsonl_atomic(path: Path, records: Sequence[BaseModel]) -> None`
- Produces: `sha256_file(path: Path) -> str`
- Produces: `IdentifierMap.from_path(path: Path) -> IdentifierMap` and `resolve(value: str) -> str`.

- [ ] **Step 1: Write failing JSONL and duplicate-query tests**

```python
def test_read_jsonl_reports_line_and_rejects_unknown_fields(tmp_path: Path) -> None:
    path = tmp_path / "gold.jsonl"
    path.write_text('{"query_id":"q1","query":"x","extra":1}\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"gold.jsonl:1"):
        read_jsonl(path, EvaluationQuery)


def test_read_jsonl_rejects_duplicate_query_ids(tmp_path: Path) -> None:
    path = tmp_path / "gold.jsonl"
    line = '{"query_id":"q1","query":"x","relevant_paper_ids":[],"metadata":{}}\n'
    path.write_text(line + line, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate query_id: q1"):
        read_jsonl(path, EvaluationQuery)
```

- [ ] **Step 2: Verify RED, then implement strict line-by-line loading**

Run: `uv run pytest tests/evaluation/test_dataset.py -k "jsonl" -v`

Expected: failures because `read_jsonl` is missing.

Implementation rules: reject blank lines, JSON values other than objects, Pydantic validation failures, and duplicate `query_id`; wrap every error as `ValueError(f"{path}:{line_number}: {reason}")` without including secrets.

- [ ] **Step 3: Add atomic-write determinism test and implementation**

```python
def test_write_jsonl_atomic_is_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "out.jsonl"
    records = [EvaluationQuery(query_id="q1", query="x")]
    write_jsonl_atomic(path, records)
    first = path.read_bytes()
    write_jsonl_atomic(path, records)
    assert path.read_bytes() == first
    assert first.endswith(b"\n")
```

Serialize each model with `model_dump(mode="json")`, `ensure_ascii=False`, `sort_keys=True`, and compact separators; write a sibling temporary file, flush and `os.fsync`, then use `Path.replace`.

- [ ] **Step 4: Add ID-map chain, conflict, and cycle tests**

```python
def test_identifier_map_resolves_normalized_chains(tmp_path: Path) -> None:
    path = tmp_path / "map.json"
    path.write_text(
        '{"https://doi.org/10.1/A":"arxiv:2501.10120",'
        '"arxiv:2501.10120":"openalex:W1"}',
        encoding="utf-8",
    )
    mapping = IdentifierMap.from_path(path)
    assert mapping.resolve("doi:10.1/a") == "openalex:W1"


def test_identifier_map_rejects_normalized_conflicts_and_cycles(tmp_path: Path) -> None:
    conflict = tmp_path / "conflict.json"
    conflict.write_text(
        '{"doi:10.1/A":"openalex:W1","https://doi.org/10.1/a":"openalex:W2"}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="conflict"):
        IdentifierMap.from_path(conflict)
    cycle = tmp_path / "cycle.json"
    cycle.write_text(
        '{"doi:10.1/a":"openalex:W1","openalex:W1":"doi:10.1/a"}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="cycle"):
        IdentifierMap.from_path(cycle)
```

Parse maps with `json.loads(..., object_pairs_hook=list)` so duplicate raw keys are observable, normalize both endpoints, reject one normalized alias targeting different canonical IDs, detect cycles with depth-first color states, and path-compress successful resolutions.

- [ ] **Step 5: Run GREEN, full regression, and commit**

Run: `uv run pytest tests/evaluation/test_dataset.py -v`

Expected: all dataset tests pass.

Run: `uv run pytest -q`

Expected: Task 1 and all prior Task 1 project tests pass.

```powershell
git add src/paper_search/evaluation/dataset.py tests/evaluation/test_dataset.py
git commit -m "feat: add strict evaluation file contracts"
```

---

### Task 3: PaSa and prediction adapters

**Files:**
- Create: `src/paper_search/evaluation/official_adapter.py`
- Create: `tests/evaluation/test_official_adapter.py`

**Interfaces:**
- Consumes: `EvaluationQuery`, `PredictionRecord`, `normalize_paper_id`.
- Produces: `PaSaRecord`, `InternalPredictionRecord`.
- Produces: `adapt_pasa_record(record, *, source, split, revision) -> EvaluationQuery`.
- Produces: `adapt_prediction_record(record) -> PredictionRecord`.

- [ ] **Step 1: Write failing PaSa conversion tests**

```python
def test_pasa_prefers_arxiv_ids_and_copies_provenance() -> None:
    source = PaSaRecord(
        qid="q1",
        question="Find RAG evaluations",
        answer=["Paper A", "Paper B"],
        answer_arxiv_id=["2501.10120v2", "1706.03762"],
        source_meta={"published_time": "2025-01-01"},
    )
    result = adapt_pasa_record(source, source="AutoScholarQuery", split="dev", revision="abc")
    assert result.relevant_paper_ids == ["arxiv:2501.10120", "arxiv:1706.03762"]
    assert result.metadata == {
        "dataset_revision": "abc",
        "source": "AutoScholarQuery",
        "source_meta": {"published_time": "2025-01-01"},
        "split": "dev",
    }


def test_pasa_uses_title_only_when_corresponding_arxiv_id_is_missing() -> None:
    source = PaSaRecord(
        qid="q1",
        question="Find papers",
        answer=["Paper A", "Paper B"],
        answer_arxiv_id=["2501.10120", ""],
        source_meta={},
    )
    result = adapt_pasa_record(source, source="RealScholarQuery", split="test", revision="abc")
    assert result.relevant_paper_ids == ["arxiv:2501.10120", "title:paper b"]
```

- [ ] **Step 2: Verify RED and implement strict source models**

Run: `uv run pytest tests/evaluation/test_official_adapter.py -v`

Expected: module import fails.

`PaSaRecord` uses `extra="forbid"` and a before-validator that converts a single string in `answer` or `answer_arxiv_id` to a one-item list. It rejects unequal non-empty list lengths. `adapt_pasa_record` pairs each answer with its arXiv ID and only uses `title:` when that position has no formal ID.

- [ ] **Step 3: Write and implement prediction-boundary tests**

```python
def test_fixed_prediction_schema_maps_selected_ids_without_deduplication() -> None:
    source = InternalPredictionRecord(
        query_id="q1",
        selected_paper_ids=["arxiv:2501.10120v2", "arxiv:2501.10120"],
    )
    result = adapt_prediction_record(source)
    assert result.predicted_paper_ids == ["arxiv:2501.10120", "arxiv:2501.10120"]


def test_prediction_schema_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        InternalPredictionRecord.model_validate(
            {"query_id": "q1", "selected_paper_ids": [], "scores": []}
        )
```

- [ ] **Step 4: Run GREEN, regression, and commit**

Run: `uv run pytest tests/evaluation/test_official_adapter.py -v`

Expected: all adapter tests pass.

Run: `uv run pytest -q`

Expected: all tests pass.

```powershell
git add src/paper_search/evaluation/official_adapter.py tests/evaluation/test_official_adapter.py
git commit -m "feat: adapt PaSa and ranked prediction records"
```

---

### Task 4: Per-query metrics and aggregate evaluator

**Files:**
- Create: `src/paper_search/evaluation/metrics.py`
- Create: `tests/evaluation/test_metrics.py`

**Interfaces:**
- Consumes: normalized `EvaluationQuery`, `PredictionRecord`, optional `IdentifierMap`.
- Produces: `deduplicate_ranked(values: Sequence[str]) -> list[str]`.
- Produces: `score_query(gold: Sequence[str], predicted: Sequence[str]) -> QueryMetrics`.
- Produces: `evaluate(gold, predictions, *, id_map=None) -> EvaluationResult`.

- [ ] **Step 1: Write failing empty-set and ranked-deduplication tests**

```python
@pytest.mark.parametrize(
    ("gold", "predicted", "expected_f1"),
    [([], [], 1.0), (["doi:10.1/a"], [], 0.0), ([], ["doi:10.1/a"], 0.0)],
)
def test_empty_set_contract(gold: list[str], predicted: list[str], expected_f1: float) -> None:
    result = score_query(gold, predicted)
    assert result.f1 == expected_f1


def test_recall_at_k_uses_first_ranked_occurrence() -> None:
    result = score_query(
        ["openalex:W1", "openalex:W2"],
        ["openalex:W1", "openalex:W1", "openalex:W3", "openalex:W2"],
    )
    assert result.predicted_ids == ["openalex:W1", "openalex:W3", "openalex:W2"]
    assert result.recall_at_5 == 1.0
```

- [ ] **Step 2: Verify RED and implement minimal query scoring**

Run: `uv run pytest tests/evaluation/test_metrics.py -k "empty_set or recall_at_k" -v`

Expected: import fails because metric functions do not exist.

Compute `tp`, `fp`, `fn`; use the approved both-empty convention; compute Recall@5/10/20 from de-duplicated ranked IDs; return hit IDs in prediction order.

- [ ] **Step 3: Add macro/micro and missing-prediction tests**

```python
def test_evaluate_reports_macro_micro_and_missing_predictions() -> None:
    gold = [
        EvaluationQuery(query_id="q1", query="one", relevant_paper_ids=["openalex:W1"]),
        EvaluationQuery(query_id="q2", query="two", relevant_paper_ids=[]),
    ]
    predictions = [PredictionRecord(query_id="q1", predicted_paper_ids=["openalex:W1"])]
    result = evaluate(gold, predictions)
    assert result.summary.query_count == 2
    assert result.summary.missing_prediction_count == 1
    assert result.summary.macro_f1 == 1.0
    assert result.summary.micro_f1 == 1.0


def test_evaluate_rejects_unknown_prediction_query() -> None:
    gold = [EvaluationQuery(query_id="q1", query="one")]
    predictions = [PredictionRecord(query_id="unknown")]
    with pytest.raises(ValueError, match="unknown prediction query_id"):
        evaluate(gold, predictions)
```

- [ ] **Step 4: Implement aggregate models and ID-map application**

Define frozen `QueryMetrics`, `MetricSummary`, and `EvaluationResult` models. Macro values are arithmetic means across query results. Micro values are calculated after summing TP/FP/FN. Resolve every gold and prediction ID through the optional map before set comparison, then de-duplicate predictions by first occurrence.

- [ ] **Step 5: Run GREEN and commit**

Run: `uv run pytest tests/evaluation/test_metrics.py -v`

Expected: all metric tests pass.

Run: `uv run pytest -q`

Expected: all tests pass.

```powershell
git add src/paper_search/evaluation/metrics.py tests/evaluation/test_metrics.py
git commit -m "feat: add reproducible paper retrieval metrics"
```

---

### Task 5: Deterministic evaluation CLI

**Files:**
- Modify: `src/paper_search/evaluation/metrics.py`
- Create: `tests/evaluation/test_cli.py`
- Create: `tests/fixtures/evaluation/gold.jsonl`
- Create: `tests/fixtures/evaluation/predictions.jsonl`
- Create: `tests/fixtures/evaluation/id_map.json`

**Interfaces:**
- Produces command: `python -m paper_search.evaluation.metrics --gold PATH --pred PATH --out PATH [--id-map PATH]`.
- Produces output keys: `contract_version`, `input_hashes`, `summary`, `per_query`.

- [ ] **Step 1: Add synthetic fixtures and failing CLI test**

```python
def test_cli_writes_stable_metrics(tmp_path: Path) -> None:
    output = tmp_path / "metrics.json"
    args = [
        sys.executable,
        "-m",
        "paper_search.evaluation.metrics",
        "--gold",
        "tests/fixtures/evaluation/gold.jsonl",
        "--pred",
        "tests/fixtures/evaluation/predictions.jsonl",
        "--out",
        str(output),
    ]
    first = subprocess.run(args, check=False, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    first_bytes = output.read_bytes()
    second = subprocess.run(args, check=False, capture_output=True, text=True)
    assert second.returncode == 0, second.stderr
    assert output.read_bytes() == first_bytes
    payload = json.loads(first_bytes)
    assert payload["contract_version"] == "task2-evaluation-v1"
    assert "macro_f1" in payload["summary"]
```

- [ ] **Step 2: Verify RED and implement parser/orchestration**

Run: `uv run pytest tests/evaluation/test_cli.py -v`

Expected: subprocess exits non-zero because no CLI entry point exists.

Load gold as `EvaluationQuery`; load fixed external predictions as `InternalPredictionRecord` and adapt them; optionally load `IdentifierMap`; include SHA-256 for every input; sort `per_query` by query ID; atomically write indented UTF-8 JSON.

- [ ] **Step 3: Add invalid-input failure test**

```python
def test_cli_fails_without_replacing_existing_output(tmp_path: Path) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text("not-json\n", encoding="utf-8")
    output = tmp_path / "metrics.json"
    output.write_text("preserve-me\n", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "paper_search.evaluation.metrics", "--gold", str(bad),
         "--pred", "tests/fixtures/evaluation/predictions.jsonl", "--out", str(output)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert output.read_text(encoding="utf-8") == "preserve-me\n"
```

- [ ] **Step 4: Run GREEN, smoke command, and commit**

Run: `uv run pytest tests/evaluation/test_cli.py -v`

Expected: all CLI tests pass.

Run: `uv run python -m paper_search.evaluation.metrics --gold tests/fixtures/evaluation/gold.jsonl --pred tests/fixtures/evaluation/predictions.jsonl --out experiments/smoke/metrics.json`

Expected: exit code 0 and output containing macro F1 and per-query rows.

```powershell
git add src/paper_search/evaluation/metrics.py tests/evaluation/test_cli.py tests/fixtures/evaluation
git commit -m "feat: add deterministic evaluation CLI"
```

---

### Task 6: Deterministic stratified sampling and frozen-file protection

**Files:**
- Modify: `src/paper_search/evaluation/dataset.py`
- Modify: `tests/evaluation/test_dataset.py`

**Interfaces:**
- Produces: `answer_count_bucket(count: int) -> Literal["1", "2-3", "4-7", "8+"]`.
- Produces: `stratified_sample(records, size, *, seed, key, stratum) -> list[T]`.
- Produces: `write_frozen_bytes(path: Path, content: bytes) -> Literal["created", "matched"]`.

- [ ] **Step 1: Write deterministic/largest-remainder sampling tests**

```python
def test_stratified_sample_is_deterministic_and_proportional() -> None:
    records = [Sample(f"q{i}", "1" if i < 6 else "2-3") for i in range(10)]
    first = stratified_sample(records, 5, seed=20260714, key=lambda x: x.qid,
                              stratum=lambda x: x.bucket)
    second = stratified_sample(list(reversed(records)), 5, seed=20260714,
                               key=lambda x: x.qid, stratum=lambda x: x.bucket)
    assert [x.qid for x in first] == [x.qid for x in second]
    assert Counter(x.bucket for x in first) == {"1": 3, "2-3": 2}
```

- [ ] **Step 2: Verify RED and implement stable sampling**

Run: `uv run pytest tests/evaluation/test_dataset.py -k "stratified" -v`

Expected: failure because sampling API is missing.

Sort source records by stable key before seeded shuffling. Allocate floor quotas, then distribute remaining slots by descending fractional remainder and lexical stratum name. Raise `ValueError` if `size` exceeds total records or allocated quota exceeds a stratum.

- [ ] **Step 3: Write frozen-output idempotence tests and implementation**

```python
def test_frozen_write_allows_identical_rerun_and_rejects_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "frozen.json"
    assert write_frozen_bytes(path, b"same\n") == "created"
    assert write_frozen_bytes(path, b"same\n") == "matched"
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_frozen_bytes(path, b"different\n")
    assert path.read_bytes() == b"same\n"
```

- [ ] **Step 4: Run GREEN and commit**

Run: `uv run pytest tests/evaluation/test_dataset.py -v`

Expected: all dataset tests pass.

```powershell
git add src/paper_search/evaluation/dataset.py tests/evaluation/test_dataset.py
git commit -m "feat: add deterministic evaluation data freezing"
```

---

### Task 7: PaSa download and preparation orchestration

**Files:**
- Create: `scripts/prepare_task2_data.py`
- Create: `tests/evaluation/test_prepare_data.py`
- Modify: `.env.example`
- Modify: `.gitignore`
- Create: `data/manifest.example.json`

**Interfaces:**
- Consumes `HF_TOKEN` only in the network boundary.
- Produces frozen raw metadata, sampled gold JSONL, split ID lists, and `data/manifest.json`.
- Uses injectable `download_file(repo_id, revision, path, token) -> bytes` for offline tests.

- [ ] **Step 1: Write failing token and download-boundary tests**

```python
def test_prepare_requires_hf_token_without_leaking_value(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="HF_TOKEN is required") as error:
        prepare(output_root=tmp_path, token=None, downloader=forbidden_downloader)
    assert "hf_" not in str(error.value)


def test_prepare_requests_only_fixed_files_and_revision(tmp_path: Path) -> None:
    calls: list[tuple[str, str, str]] = []
    def downloader(repo_id: str, revision: str, path: str, token: str) -> bytes:
        calls.append((repo_id, revision, path))
        return FIXTURE_FILES[path]
    prepare(output_root=tmp_path, token="secret", downloader=downloader,
            expected_counts={path: 2 for path in FIXTURE_FILES}, dev_size=1,
            validation_size=1, simulated_test_size=2)
    assert calls == [(PASA_REPO_ID, PASA_REVISION, path) for path in PASA_FILES]
```

- [ ] **Step 2: Verify RED and implement preparation with dependency injection**

Run: `uv run pytest tests/evaluation/test_prepare_data.py -v`

Expected: import fails because the script does not exist.

Use standard-library HTTPS with `Authorization: Bearer ...` only inside the request object. Convert downloaded lines through `PaSaRecord` and `adapt_pasa_record`, validate expected counts, hash raw bytes, sample dev/validation, keep all simulated-test records, and call frozen-write helpers.

- [ ] **Step 3: Add manifest and idempotence tests**

```python
def test_prepare_manifest_has_reproducibility_fields_and_is_idempotent(tmp_path: Path) -> None:
    first = prepare_with_fixtures(tmp_path)
    second = prepare_with_fixtures(tmp_path)
    assert first == second
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["repo_id"] == PASA_REPO_ID
    assert manifest["revision"] == PASA_REVISION
    assert manifest["random_seed"] == 20260714
    assert manifest["sampling_algorithm"] == "answer-count-largest-remainder-v1"
    assert manifest["status"] == "waiting_for_human_label_freeze"
    assert all("sha256" in item for item in manifest["source_files"])
```

- [ ] **Step 4: Update secret/data ignore contracts**

Append `HF_TOKEN=` to `.env.example`. Add exact ignore entries for `data/raw/`, `data/dev/gold.jsonl`, `data/validation/gold.jsonl`, `data/simulated_test/`, and `data/annotation_work/`; do not ignore `data/manifest.json`, `data/splits/`, or `data/stress/`.

- [ ] **Step 5: Run offline GREEN and commit**

Run: `uv run pytest tests/evaluation/test_prepare_data.py -v`

Expected: all tests pass without network access.

Run: `git check-ignore data/raw/example.jsonl data/dev/gold.jsonl data/annotation_work/a.jsonl`

Expected: all three paths are ignored.

```powershell
git add scripts/prepare_task2_data.py tests/evaluation/test_prepare_data.py .env.example .gitignore data/manifest.example.json
git commit -m "feat: add reproducible PaSa data preparation"
```

---

### Task 8: Annotation schema and Cohen's kappa

**Files:**
- Create: `src/paper_search/evaluation/annotation.py`
- Create: `tests/evaluation/test_annotation.py`
- Create: `data/annotation_guide.md`

**Interfaces:**
- Produces: `AnnotationRecord` with the 11 approved fields.
- Produces: `cohen_kappa(first: Sequence[str], second: Sequence[str]) -> float`.
- Produces: `compare_annotations(left, right, *, fields) -> AgreementReport`.

- [ ] **Step 1: Write failing schema and perfect-agreement tests**

```python
def test_annotation_schema_rejects_invalid_years_and_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        AnnotationRecord.model_validate({**valid_annotation(), "year_from": 2030,
                                         "year_to": 2020})
    with pytest.raises(ValidationError):
        AnnotationRecord.model_validate({**valid_annotation(), "invented": True})


def test_cohen_kappa_is_one_for_perfect_agreement() -> None:
    assert cohen_kappa(["method", "topic"], ["method", "topic"]) == 1.0
```

- [ ] **Step 2: Verify RED and implement schema/kappa**

Run: `uv run pytest tests/evaluation/test_annotation.py -v`

Expected: module import fails.

Use the Task 1 year range and non-empty conventions. Compute observed agreement and expected agreement from both raters' marginal frequencies; when expected agreement is 1, return 1 only for perfect observed agreement and otherwise raise `ValueError`.

- [ ] **Step 3: Add alignment and threshold tests**

```python
def test_compare_annotations_aligns_by_query_id_and_flags_low_agreement() -> None:
    report = compare_annotations(
        left_records(),
        right_records_with_domain_disagreement(),
        fields=("query_type", "domain"),
    )
    assert report.compared_query_count == 2
    assert report.fields["query_type"].accepted is True
    assert report.fields["domain"].accepted is False
    assert report.fields["domain"].threshold == 0.80
```

Reject missing, extra, or duplicate query IDs instead of silently comparing partial overlaps.

- [ ] **Step 4: Write concrete annotation guide and commit**

The guide defines allowed query types, domain-label format, every constraint field, examples for inclusive/exclusive years, annotator independence, disagreement adjudication, and the `0.80` re-annotation gate.

Run: `uv run pytest tests/evaluation/test_annotation.py -v`

Expected: all annotation tests pass.

```powershell
git add src/paper_search/evaluation/annotation.py tests/evaluation/test_annotation.py data/annotation_guide.md
git commit -m "feat: add annotation agreement validation"
```

---

### Task 9: Safe project assets, stress set, and final acceptance

**Files:**
- Create: `data/README.md`
- Create: `data/stress/queries.jsonl`
- Modify: `src/paper_search/evaluation/__init__.py`
- Modify: `PRD.md` only to check Task 2 boxes and record `waiting_for_human_label_freeze`; do not alter approved requirements.

**Interfaces:**
- Produces 24 original, non-PaSa stress queries covering topic, method, dataset, time/venue, combined constraints, relationship, exclusion, Chinese/English, paraphrase, long-query, ambiguity, and missing-metadata tags.
- Produces reproducible commands for data preparation and offline evaluation.

- [ ] **Step 1: Add a contract test for the stress set**

```python
def test_stress_set_has_24_unique_original_queries() -> None:
    rows = [json.loads(line) for line in Path("data/stress/queries.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()]
    assert len(rows) == 24
    assert len({row["query_id"] for row in rows}) == 24
    assert {"topic", "method", "dataset", "time_venue", "combined", "relationship",
            "exclusion"} <= {tag for row in rows for tag in row["tags"]}
    assert {"zh", "en"} <= {row["language"] for row in rows}
```

- [ ] **Step 2: Verify RED, add the 24 records, and make the test GREEN**

Run: `uv run pytest tests/evaluation/test_dataset.py -k "stress_set" -v`

Expected before data creation: file-not-found failure. Expected after data creation: pass.

- [ ] **Step 3: Document reproducible and safe usage**

`data/README.md` contains: license/access notice, fixed revision, exact preparation command, exact evaluation command, committed versus ignored file matrix, teammate reproduction steps, and the explicit human-label freeze status.

- [ ] **Step 4: Run complete acceptance verification**

Run: `uv run pytest tests/evaluation -v`

Expected: all evaluation tests pass.

Run: `uv run python -m paper_search.evaluation.metrics --gold tests/fixtures/evaluation/gold.jsonl --pred tests/fixtures/evaluation/predictions.jsonl --out experiments/smoke/metrics.json`

Expected: exit code 0; output contains deterministic input hashes, macro F1, micro F1, Recall@5/10/20, and per-query results.

Run: `uv run pytest -q`

Expected: all project tests pass.

Run: `uv run ruff check .`

Expected: `All checks passed!`

Run: `uv run mypy src`

Expected: `Success: no issues found`.

Run: `git status --short --ignored; git grep -n -E "hf_[A-Za-z0-9]{20,}|Authorization: Bearer" -- . ":(exclude).env.example"`

Expected: generated restricted files are ignored and no tracked secret match is printed.

- [ ] **Step 5: Commit documentation and safe assets**

```powershell
git add data/README.md data/stress/queries.jsonl src/paper_search/evaluation/__init__.py PRD.md tests/evaluation/test_dataset.py
git commit -m "docs: finalize task 2 evaluation workflow"
```

---

## Review Checkpoints

1. After Task 2: review the canonical-ID contract before adapters depend on it.
2. After Task 5: inspect fixture metrics and CLI JSON before data preparation begins.
3. After Task 7: inspect `git status --ignored` and manifest fields before any real gated download.
4. After Task 9: review the complete diff and use `finishing-a-development-branch` before merge or push.

## Self-Review Results

- Spec coverage: all approved design sections map to Tasks 1–9; online download, offline evaluation, deterministic sampling/freezing, annotation agreement, stress data, security, and final acceptance each have an explicit owner and test gate.
- Placeholder scan: no deferred implementation markers or unspecified error-handling steps remain.
- Type consistency: adapters produce `EvaluationQuery`/`PredictionRecord`; metrics consume those same types; CLI uses the adapter rather than a second prediction contract; sampling and preparation share the same `EvaluationQuery` records.
- Scope control: model training, online retrieval, LLM-generated labels, frontend work, and future official competition Schema are excluded from this implementation.
