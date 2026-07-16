# Task 3 OpenAlex Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a budget-bounded OpenAlex search provider with deterministic normalization, SQLite replay cache, immutable snapshots, and offline contract tests.

**Architecture:** Keep OpenAlex HTTP orchestration in `retrieval/openalex.py`, raw response persistence and snapshot export in `storage/cache.py`, and Work-to-`Paper` conversion in the pure `processing/normalize.py` module. The provider exposes ordered cache keys through `ProviderResult.provenance["cache_keys"]`, so later runners can freeze exactly the responses used without mutable provider state.

**Tech Stack:** Python 3.11, Pydantic 2, httpx 0.28, SQLite (`sqlite3`), pytest, Ruff, mypy strict.

## Global Constraints

- Work only on `codex/task3-openalex`; never commit the existing Task 2 design line-ending metadata change.
- Do not read or print local secret files; the optional online test reads only the process environment.
- OpenAlex search only: no references, citations, Semantic Scholar, Task 4 ranking/deduplication, UI, or generic provider framework.
- Every production behavior follows RED → verify RED → minimal GREEN → verify GREEN.
- Each actual HTTP attempt consumes one reserved search call; cache/cooldown hits consume zero.
- Per-page size is at most 50, total `limit` is `1..300`, and each page gets at most three attempts.
- Successful responses cache for seven days; 429 stores only a 60-second cooldown; timeout, 5xx, invalid JSON, and invalid top-level responses are not cached.
- API keys never enter cache keys, SQLite values, errors, provenance, snapshots, or committed fixtures.

---

### Task 1: Pure OpenAlex Work Normalization

**Files:**
- Create: `src/paper_search/processing/__init__.py`
- Create: `src/paper_search/processing/normalize.py`
- Create: `tests/unit/test_normalize.py`
- Create: `tests/fixtures/openalex/works_page_1.json`
- Create: `tests/fixtures/openalex/works_missing_abstract.json`

**Interfaces:**
- Consumes: `paper_search.domain.models.Paper` and `normalize_paper_id()` from Task 2.
- Produces: `reconstruct_abstract(index: object) -> str | None` and `normalize_openalex_work(raw_work: Mapping[str, object]) -> Paper`.

- [ ] **Step 1: Add realistic safe fixtures and failing normalization tests**

```python
def test_normalize_openalex_work_maps_complete_record() -> None:
    raw = load_fixture("works_page_1.json")["results"][0]
    paper = normalize_openalex_work(raw)
    assert paper.canonical_id == "doi:10.1000/example"
    assert paper.openalex_id == "W123"
    assert paper.abstract == "retrieval augmented generation"
    assert paper.authors == ["Ada Lovelace", "Grace Hopper"]
    assert paper.venue == "Journal of Safe Fixtures"
    assert paper.sources == ["openalex"]


def test_missing_abstract_is_valid() -> None:
    raw = load_fixture("works_missing_abstract.json")["results"][0]
    assert normalize_openalex_work(raw).abstract is None


@pytest.mark.parametrize("field", ["title", "id"])
def test_record_without_title_or_stable_id_is_rejected(field: str) -> None:
    raw = complete_work()
    raw[field] = None
    with pytest.raises(ValueError):
        normalize_openalex_work(raw)
```

- [ ] **Step 2: Run RED**

Run: `uv run --no-sync pytest tests/unit/test_normalize.py -v`
Expected: collection fails because `paper_search.processing.normalize` does not exist.

- [ ] **Step 3: Implement minimal pure normalization**

```python
def reconstruct_abstract(index: object) -> str | None:
    if index is None:
        return None
    if not isinstance(index, Mapping):
        raise ValueError("abstract_inverted_index must be a mapping or null")
    positioned: dict[int, str] = {}
    for token, positions in index.items():
        if not isinstance(token, str) or not isinstance(positions, list):
            raise ValueError("invalid abstract inverted index")
        for position in positions:
            if isinstance(position, bool) or not isinstance(position, int) or position < 0:
                raise ValueError("abstract positions must be non-negative integers")
            if position in positioned:
                raise ValueError("abstract positions must be unique")
            positioned[position] = token
    return " ".join(positioned[position] for position in sorted(positioned)) or None


def normalize_openalex_work(raw_work: Mapping[str, object]) -> Paper:
    title_value = raw_work.get("title") or raw_work.get("display_name")
    if not isinstance(title_value, str) or not title_value.strip():
        raise ValueError("OpenAlex work must have a title")
    openalex_value = raw_work.get("id")
    if not isinstance(openalex_value, str):
        raise ValueError("OpenAlex work must have an id")
    openalex_id = normalize_paper_id(f"openalex:{openalex_value}").removeprefix("openalex:")
    doi_value = raw_work.get("doi")
    doi = None
    if isinstance(doi_value, str) and doi_value.strip():
        doi = normalize_paper_id(f"doi:{doi_value}").removeprefix("doi:")
    canonical_id = f"doi:{doi}" if doi is not None else f"openalex:{openalex_id}"
    authorships = raw_work.get("authorships")
    authors = extract_authors(authorships)
    venue, url = extract_primary_location(raw_work.get("primary_location"), openalex_value)
    return Paper(
        canonical_id=canonical_id,
        title=title_value,
        abstract=reconstruct_abstract(raw_work.get("abstract_inverted_index")),
        authors=authors,
        publication_year=optional_int(raw_work.get("publication_year"), "publication_year"),
        venue=venue,
        doi=doi,
        openalex_id=openalex_id,
        url=url,
        citation_count=optional_int(raw_work.get("cited_by_count"), "cited_by_count"),
        is_retracted=optional_bool(raw_work.get("is_retracted"), "is_retracted"),
        sources=["openalex"],
    )
```

Define `extract_authors`, `extract_primary_location`, `optional_int`, and `optional_bool` immediately above the public function. They accept only the documented JSON shapes, reject booleans for numeric fields, discard blank author names, and return typed values consumed by the shown function.

- [ ] **Step 4: Run GREEN and static checks**

Run: `uv run --no-sync pytest tests/unit/test_normalize.py -v`
Expected: all normalization tests pass.

Run: `uv run --no-sync ruff check src/paper_search/processing tests/unit/test_normalize.py`
Expected: `All checks passed!`

Run: `uv run --no-sync mypy src/paper_search/processing`
Expected: no issues.

- [ ] **Step 5: Commit**

```powershell
git add src/paper_search/processing tests/unit/test_normalize.py tests/fixtures/openalex
git commit -m "feat: normalize OpenAlex works"
```

---

### Task 2: Deterministic SQLite Response Cache

**Files:**
- Create: `src/paper_search/storage/__init__.py`
- Create: `src/paper_search/storage/cache.py`
- Create: `tests/unit/test_cache.py`

**Interfaces:**
- Produces `CachedResponse`, `canonical_request_params()`, `make_cache_key()`, and `SQLiteResponseCache`.
- Later tasks call `get_response`, `put_response`, `get_cooldown`, and `set_cooldown`.

- [ ] **Step 1: Write failing key, TTL, cooldown, and secret tests**

```python
def test_cache_key_is_order_independent_and_excludes_api_key() -> None:
    first = make_cache_key("openalex", "/works", {"search": "rag", "api_key": "secret", "per_page": 2}, "v1")
    second = make_cache_key("openalex", "/works", {"per_page": 2, "search": "rag"}, "v1")
    assert first == second
    assert "secret" not in first


def test_success_response_expires_after_seven_days(tmp_path: Path) -> None:
    clock = MutableClock(datetime(2026, 7, 16, tzinfo=UTC))
    cache = SQLiteResponseCache(tmp_path / "cache.sqlite3", clock=clock)
    cache.put_response(key="k", provider="openalex", endpoint="/works", params={"search": "rag"}, raw_response=b"{}", requested_at=clock(), ttl=timedelta(days=7), safe_headers={})
    assert cache.get_response("k") is not None
    clock.advance(timedelta(days=7, microseconds=1))
    assert cache.get_response("k") is None


def test_cooldown_is_active_for_sixty_seconds(tmp_path: Path) -> None:
    cache.set_cooldown("k", now + timedelta(seconds=60))
    assert cache.get_cooldown("k") == now + timedelta(seconds=60)
```

- [ ] **Step 2: Run RED**

Run: `uv run --no-sync pytest tests/unit/test_cache.py -v`
Expected: collection fails because `paper_search.storage.cache` does not exist.

- [ ] **Step 3: Implement schema and transactional operations**

```python
@dataclass(frozen=True)
class CachedResponse:
    cache_key: str
    provider: str
    endpoint: str
    params: dict[str, str]
    status_code: int
    raw_response: bytes
    response_hash: str
    safe_headers: dict[str, str]
    requested_at: datetime
    expires_at: datetime


def canonical_request_params(params: Mapping[str, object]) -> dict[str, str]:
    return {key: str(value) for key, value in sorted(params.items()) if key != "api_key"}


def make_cache_key(provider: str, endpoint: str, params: Mapping[str, object], cache_version: str) -> str:
    payload = {"provider": provider, "endpoint": endpoint, "params": canonical_request_params(params), "cache_version": cache_version}
    return "sha256:" + hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
```

`SQLiteResponseCache` initializes `responses` and `cooldowns` tables, uses a new SQLite connection per operation, commits each write transaction, stores canonical JSON, and treats expired responses/cooldowns as misses.

- [ ] **Step 4: Run GREEN and static checks**

Run: `uv run --no-sync pytest tests/unit/test_cache.py -v`
Expected: all cache core tests pass.

Run: `uv run --no-sync ruff check src/paper_search/storage tests/unit/test_cache.py`
Run: `uv run --no-sync mypy src/paper_search/storage`
Expected: both pass.

- [ ] **Step 5: Commit**

```powershell
git add src/paper_search/storage tests/unit/test_cache.py
git commit -m "feat: add OpenAlex response cache"
```

---

### Task 3: Immutable Snapshot Export

**Files:**
- Modify: `src/paper_search/storage/cache.py`
- Modify: `tests/unit/test_cache.py`

**Interfaces:**
- Produces `SQLiteResponseCache.export_snapshot(cache_keys: Sequence[str], run_dir: Path) -> Path` and `validate_snapshot_manifest(path: Path) -> None`.

- [ ] **Step 1: Write failing deterministic export tests**

```python
def test_snapshot_export_is_deterministic_and_validated(tmp_path: Path) -> None:
    cache = populated_cache(tmp_path)
    manifest = cache.export_snapshot(["page-1", "page-2"], tmp_path / "run")
    first = manifest.read_bytes()
    assert [item["cache_key"] for item in json.loads(first)["entries"]] == ["page-1", "page-2"]
    validate_snapshot_manifest(manifest)
    assert cache.export_snapshot(["page-1", "page-2"], tmp_path / "run").read_bytes() == first


def test_snapshot_refuses_different_existing_content(tmp_path: Path) -> None:
    cache = populated_cache(tmp_path)
    cache.export_snapshot(["page-1"], tmp_path / "run")
    with pytest.raises(FileExistsError):
        cache.export_snapshot(["page-2"], tmp_path / "run")
```

- [ ] **Step 2: Run RED**

Run: `uv run --no-sync pytest tests/unit/test_cache.py -k snapshot -v`
Expected: fails because export methods are absent.

- [ ] **Step 3: Implement exact-byte snapshots and atomic manifest**

Each entry writes raw bytes to `snapshots/openalex-NNNN.json`. The manifest contains contract version, ordered cache key, endpoint, canonical params, requested time, response hash, relative path, and file hash. Identical files are accepted; differing existing files raise `FileExistsError`. Validation recomputes every file hash and response hash.

- [ ] **Step 4: Run GREEN**

Run: `uv run --no-sync pytest tests/unit/test_cache.py -v`
Expected: all cache and snapshot tests pass.

- [ ] **Step 5: Commit**

```powershell
git add src/paper_search/storage/cache.py tests/unit/test_cache.py
git commit -m "feat: export immutable provider snapshots"
```

---

### Task 4: Basic OpenAlex Search, Paging, and Replay

**Files:**
- Create: `src/paper_search/retrieval/__init__.py`
- Create: `src/paper_search/retrieval/openalex.py`
- Create: `tests/unit/test_openalex.py`
- Create: `tests/fixtures/openalex/works_page_2.json`
- Create: `tests/fixtures/openalex/works_empty.json`

**Interfaces:**
- Consumes `SQLiteResponseCache`, `normalize_openalex_work`, `BudgetReservation`, and `ProviderResult[list[Paper]]`.
- Produces `OpenAlexProvider.search()` and `OPENALEX_SELECT_FIELDS`.

- [ ] **Step 1: Write failing request, pagination, empty, and replay tests**

```python
def test_search_builds_safe_bounded_request_and_maps_results(tmp_path: Path) -> None:
    seen: list[httpx.Request] = []
    provider = provider_with_transport(tmp_path, fixture_transport(seen, "works_page_1.json"))
    result = asyncio.run(provider.search("RAG", {"year_from": 2020, "year_to": 2024}, 2, reservation(1)))
    assert [paper.openalex_id for paper in result.data] == ["W123", "W124"]
    assert seen[0].url.params["per_page"] == "2"
    assert seen[0].url.params["filter"] == "from_publication_date:2020-01-01,to_publication_date:2024-12-31"
    assert seen[0].url.params["select"] == OPENALEX_SELECT_FIELDS
    assert result.usage.search_api_calls == 1
    assert "api_key" not in json.dumps(result.provenance)


def test_search_replays_cache_without_network_call(tmp_path: Path) -> None:
    online = provider_with_transport(tmp_path, fixture_transport([], "works_page_1.json"))
    first = asyncio.run(online.search("RAG", {}, 2, reservation(1)))
    offline = provider_with_transport(tmp_path, raising_transport())
    replay = asyncio.run(offline.search("RAG", {}, 2, reservation(0)))
    assert replay.data == first.data
    assert replay.cache_hit is True
    assert replay.usage.search_api_calls == 0
```

- [ ] **Step 2: Run RED**

Run: `uv run --no-sync pytest tests/unit/test_openalex.py -v`
Expected: collection fails because `paper_search.retrieval.openalex` does not exist.

- [ ] **Step 3: Implement input validation and successful page loop**

Implement `OpenAlexProvider.search()` with this fixed control flow:

```text
strip and validate query → validate filters and limit → cursor="*"
→ build safe page params and cache key → read valid cache entry
→ on miss, enforce reservation and issue one GET → validate JSON page → cache seven days
→ normalize each Work while collecting invalid_work errors
→ append ordered cache key and page hash → stop on limit/empty/no cursor
→ construct ProviderResult with aggregate hash and compact cache_keys JSON
```

Only `year_from` and `year_to` filters are accepted. `_build_params()` emits the fixed `select`, `per_page=min(50, remaining)`, and cursor. `_parse_page()` accepts only a JSON object whose `results` is a list and whose optional `meta.next_cursor` is a string or null. The page loop stops on limit, empty page, missing cursor, budget boundary, or failed page. Successful valid top-level responses are cached for seven days.

- [ ] **Step 4: Run GREEN and static checks**

Run: `uv run --no-sync pytest tests/unit/test_openalex.py -v`
Expected: basic search, pagination, empty, and replay tests pass.

Run: `uv run --no-sync ruff check src/paper_search/retrieval tests/unit/test_openalex.py`
Run: `uv run --no-sync mypy src/paper_search/retrieval`
Expected: both pass.

- [ ] **Step 5: Commit**

```powershell
git add src/paper_search/retrieval tests/unit/test_openalex.py tests/fixtures/openalex
git commit -m "feat: search and replay OpenAlex works"
```

---

### Task 5: Retry, Budget, Partial Results, and Error Contracts

**Files:**
- Modify: `src/paper_search/retrieval/openalex.py`
- Modify: `tests/unit/test_openalex.py`
- Create: `tests/fixtures/openalex/error_429.json`
- Create: `tests/fixtures/openalex/error_500.json`
- Create: `tests/fixtures/openalex/works_invalid_record.json`

**Interfaces:**
- Extends `OpenAlexProvider.search()` without changing its signature.

- [ ] **Step 1: Write one failing test per error behavior**

```python
def test_429_retries_at_most_three_times_and_sets_cooldown(tmp_path: Path) -> None:
    result, attempts, sleeps = run_scripted_search(tmp_path, [429, 429, 429], reserved_calls=3)
    assert attempts == 3
    assert sleeps == [1.0, 2.0]
    assert result.errors[-1].code == "rate_limited"
    cooled = run_again_with_forbidden_network(tmp_path, reserved_calls=0)
    assert cooled.errors[-1].code == "rate_limited"
    assert cooled.usage.search_api_calls == 0


def test_timeout_then_success_counts_both_attempts(tmp_path: Path) -> None:
    result = run_scripted_search(tmp_path, [httpx.ReadTimeout("slow"), 200], reserved_calls=2)
    assert result.usage.search_api_calls == 2
    assert result.data


def test_second_page_failure_returns_first_page_and_error(tmp_path: Path) -> None:
    result = run_scripted_search(tmp_path, [page_one, 500, 500], reserved_calls=3)
    assert [paper.openalex_id for paper in result.data] == ["W123", "W124"]
    assert result.errors[-1].code == "server_error"


def test_invalid_work_is_skipped_without_losing_valid_siblings(tmp_path: Path) -> None:
    result = run_fixture_search(tmp_path, "works_invalid_record.json")
    assert len(result.data) == 1
    assert result.errors[0].code == "invalid_work"
```

Also cover 400, 403, malformed JSON, invalid top-level results, zero remaining budget, request ID propagation, and secret absence from serialized errors.

- [ ] **Step 2: Run RED**

Run: `uv run --no-sync pytest tests/unit/test_openalex.py -k "retry or timeout or partial or invalid or budget or cooldown" -v`
Expected: new behavior tests fail with missing retry/error handling.

- [ ] **Step 3: Implement bounded attempt loop and structured errors**

Use a private `_fetch_page()` that checks cooldown and remaining reservation before each attempt. Retry only 429, 5xx, and `httpx.TimeoutException`; sleep only when another retry and budget unit remain. Store cooldown on every 429. Return a page outcome containing raw bytes/cache key/errors/attempt count rather than raising normal provider failures.

- [ ] **Step 4: Run GREEN and complete focused suite**

Run: `uv run --no-sync pytest tests/unit/test_openalex.py -v`
Expected: all provider tests pass with no warnings.

Run: `uv run --no-sync pytest -m "not online" tests/unit/test_openalex.py tests/unit/test_cache.py tests/unit/test_normalize.py -v`
Expected: all Task 3 offline tests pass.

- [ ] **Step 5: Commit**

```powershell
git add src/paper_search/retrieval/openalex.py tests/unit/test_openalex.py tests/fixtures/openalex
git commit -m "feat: bound OpenAlex retries and errors"
```

---

### Task 6: Optional Live Smoke, Public API, PRD Status, and Final Verification

**Files:**
- Modify: `pyproject.toml`
- Modify: `src/paper_search/retrieval/__init__.py`
- Modify: `src/paper_search/storage/__init__.py`
- Modify: `src/paper_search/processing/__init__.py`
- Create: `tests/integration/test_openalex_live.py`
- Modify: `PRD.md`

**Interfaces:**
- Exports the Task 3 provider, cache, snapshot validator, and normalizer.

- [ ] **Step 1: Write the optional online smoke contract before helper code**

```python
@pytest.mark.online
def test_three_live_queries_produce_safe_snapshot(tmp_path: Path) -> None:
    api_key = os.environ.get("OPENALEX_API_KEY")
    if not api_key:
        pytest.skip("OPENALEX_API_KEY is not set in the process environment")
    results = asyncio.run(run_live_queries(api_key, tmp_path, LIVE_QUERIES))
    assert len(results) == 3
    assert all(result.data for result in results)
    serialized = (tmp_path / "provider.json").read_text(encoding="utf-8")
    assert api_key not in serialized
```

The test file contains exactly three original safe queries and writes counts, latency, errors, hashes, and snapshot paths only.

- [ ] **Step 2: Register marker and implement the live-only orchestration helper**

Add `markers = ["online: requires explicit external provider credentials and network"]` to pytest configuration. The helper constructs a temporary SQLite cache, runs three searches, exports the union of provenance cache keys, validates the manifest, and writes atomic safe summary JSON.

- [ ] **Step 3: Run offline acceptance**

Run: `uv run --no-sync pytest -m "not online" tests/unit/test_openalex.py tests/unit/test_cache.py tests/unit/test_normalize.py -v`
Expected: all offline Task 3 tests pass.

Run: `uv run --no-sync pytest -q`
Expected: all project tests pass; online test is skipped when no process key is present.

Run: `uv run --no-sync ruff check .`
Expected: `All checks passed!`

Run: `uv run --no-sync mypy src`
Expected: no issues.

- [ ] **Step 4: Run live smoke only when the process environment already has a key**

Run: `uv run --no-sync pytest -m online tests/integration/test_openalex_live.py -v`
Expected with key/network: three queries pass and safe outputs are generated. Expected without process key: one explicit skip. Do not load a local secret file to force this test.

- [ ] **Step 5: Update only truthful PRD checkboxes**

Check the fixture, search, cache, snapshot, normalization, retry, and offline-test boxes only after their commands pass. Check the three-query smoke and online-test boxes only after a real keyed run succeeds. Do not claim the first-week stage gate or Task 4 is complete.

- [ ] **Step 6: Security and diff review**

Run tracked high-entropy token and literal authorization-header scans, excluding only their rule-definition documentation. Verify cache databases, live outputs, and smoke snapshots are ignored or outside the repository. Run `git diff --check` and confirm the pre-existing Task 2 design metadata is unstaged.

- [ ] **Step 7: Commit**

```powershell
git add pyproject.toml PRD.md src/paper_search/retrieval/__init__.py src/paper_search/storage/__init__.py src/paper_search/processing/__init__.py tests/integration/test_openalex_live.py
git commit -m "test: finalize OpenAlex search workflow"
```

---

## Review Checkpoints

1. After Task 1, confirm canonical IDs match Task 2 normalization.
2. After Task 3, inspect manifest bytes and overwrite protection before the provider depends on it.
3. After Task 5, review retry counts, partial results, and secret-free errors independently.
4. After Task 6, run the complete verification suite and use `finishing-a-development-branch`.
