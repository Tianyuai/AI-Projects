from __future__ import annotations

import asyncio
import errno
import hashlib
import inspect
import importlib
import json
import shutil
import sqlite3
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any, IO

import pytest
import yaml

import paper_search.evaluation.runner as runner_module
from paper_search.application.contracts import (
    SearchErrorResponse,
    SearchExecutionResult,
    SearchFailure,
)
from paper_search.config import RuntimeConfig, load_budget, load_runtime_config
from paper_search.domain.models import (
    BudgetReservation,
    ErrorDetail,
    Paper,
    ProviderResult,
    QuerySpec,
    UsageActual,
)
from paper_search.evaluation.dataset import EvaluationQuery, PredictionRecord
from paper_search.evaluation.metrics import CONTRACT_VERSION
from paper_search.evaluation.runner import (
    EvaluationRunRequest,
    EvaluationRunResult,
    RunIdentity,
    process_candidates,
    run_evaluation,
)
from paper_search.processing import (
    FUZZY_TITLE_THRESHOLD,
    MINIMUM_UNCERTAINTY_MULTIPLIER,
    UNCERTAINTY_REASON_MULTIPLIER,
)
from paper_search.ranking import (
    BM25_WEIGHT,
    KEYWORD_COVERAGE_WEIGHT,
    SCORING_VERSION,
    TOKENIZER_VERSION,
)
from paper_search.storage import SQLiteResponseCache
from paper_search.storage.cache import validate_snapshot_manifest
from paper_search.storage.dependency_snapshot import DependencyCaptureStore


CONFIG = Path(__file__).parents[2] / "configs" / "base.yaml"
GIT_SHA = "a" * 40


def _paper(
    canonical_id: str,
    *,
    title: str = "Graph retrieval",
    doi: str | None = None,
    is_retracted: bool | None = False,
) -> Paper:
    return Paper(
        canonical_id=canonical_id,
        title=title,
        doi=doi,
        is_retracted=is_retracted,
    )


def _query_spec(text: str) -> QuerySpec:
    return QuerySpec(original_query=text, research_goal=text)


SENTINEL_API_KEY = "sentinel-api-key-must-not-leak"


def _runtime_config(*, with_secret: bool = False) -> RuntimeConfig:
    return RuntimeConfig(
        budget_profile="low",
        budget=load_budget(Path("configs/budget_low.yaml")),
        llm_base_url="https://llm.invalid/v1",
        llm_model_primary="test-primary",
        llm_model_fallback="test-fallback",
        openalex_api_key=SENTINEL_API_KEY if with_secret else None,
    )


def _provider_result(
    papers: list[Paper],
    *,
    calls: int,
    latency_ms: int,
    cache_keys: str = "[]",
    provider: str = "openalex",
    endpoint: str = "/works",
    response_hash: str | None = None,
    cost_cny: float | None = None,
    errors: list[ErrorDetail] | None = None,
) -> ProviderResult[list[Paper]]:
    if response_hash is None:
        response_hash = _aggregate_hash([])
    return ProviderResult(
        data=papers,
        usage=UsageActual(
            search_api_calls=calls,
            cost_cny=cost_cny,
            elapsed_ms=latency_ms,
        ),
        provenance={
            "provider": provider,
            "endpoint": endpoint,
            "model_id": "openalex-api",
            "requested_at": "2026-07-17T00:00:00+00:00",
            "response_hash": response_hash,
            "cache_keys": cache_keys,
        },
        cache_hit=False,
        latency_ms=latency_ms,
        errors=[] if errors is None else errors,
    )


class FakeProvider:
    def __init__(self, results: dict[str, ProviderResult[list[Paper]]]) -> None:
        self._results = results
        self.calls: list[tuple[str, dict[str, object], int, BudgetReservation]] = []

    async def search(
        self,
        query: str,
        filters: dict[str, object],
        limit: int,
        reservation: BudgetReservation,
    ) -> ProviderResult[list[Paper]]:
        self.calls.append((query, filters, limit, reservation))
        return self._results[query]


def _populated_cache(tmp_path: Path) -> SQLiteResponseCache:
    now = datetime(2026, 7, 17, tzinfo=UTC)
    cache = SQLiteResponseCache(tmp_path / "cache.sqlite3", clock=lambda: now)
    for index, key in enumerate(("page-1", "page-2"), start=1):
        cache.put_response(
            key=key,
            provider="openalex",
            endpoint="/works",
            cache_version="v1",
            params={
                "search": "q1",
                "cursor": f"cursor-{index}",
                "api_key": SENTINEL_API_KEY,
            },
            raw_response=(f'{{"page":{index},"results":[]}}\n').encode(),
            requested_at=now + timedelta(seconds=index),
            ttl=timedelta(days=7),
            safe_headers={"authorization": SENTINEL_API_KEY},
        )
    return cache


def _canonical_jsonl_bytes(records: list[EvaluationQuery | PredictionRecord]) -> bytes:
    return b"".join(
        json.dumps(
            record.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
        for record in records
    )


def _sha256(content: bytes) -> str:
    return f"sha256:{hashlib.sha256(content).hexdigest()}"


def _aggregate_hash(hashes: list[str]) -> str:
    if len(hashes) == 1:
        return hashes[0]
    return _sha256(json.dumps(hashes, separators=(",", ":")).encode("utf-8"))


def _run_identity(gold: list[EvaluationQuery] | None = None) -> RunIdentity:
    del gold
    return RunIdentity(
        split="dev",
        git_sha=GIT_SHA,
        gold_sha256=f"sha256:{'c' * 64}",
        manifest_sha256=f"sha256:{'b' * 64}",
        dataset_revision="dataset-r1",
        zero_answer_policy="allow",
    )


def _write_cli_manifest(
    root: Path,
    *,
    status: str = "frozen",
    revision: str = "dataset-r1",
    gold_path: str = "dev/gold.jsonl",
    gold_sha256: str | None = None,
    gold_content: str = (
        '{"query_id":"query-1","query":"graph retrieval",'
        '"relevant_paper_ids":["openalex:W1"]}\n'
    ),
    ids: object = None,
    partition_updates: dict[str, object] | None = None,
    omit_partition_fields: set[str] | None = None,
) -> Path:
    data = root / "data"
    gold = data / "dev" / "gold.jsonl"
    gold.parent.mkdir(parents=True)
    gold.write_text(gold_content, encoding="utf-8")
    id_values = ["query-1"] if ids is None else ids
    ids_file = data / "dev" / "ids.json"
    ids_file.write_text(json.dumps(id_values), encoding="utf-8")
    partition: dict[str, object] = {
        "count": 1,
        "gold_path": gold_path,
        "gold_sha256": _sha256(gold.read_bytes()) if gold_sha256 is None else gold_sha256,
        "ids_path": "dev/ids.json",
        "ids_sha256": _sha256(ids_file.read_bytes()),
        "labels_complete": True,
        "zero_answer_policy": "reject",
    }
    if partition_updates is not None:
        partition.update(partition_updates)
    for field in omit_partition_fields or set():
        partition.pop(field, None)
    (data / "manifest.json").write_text(
        json.dumps(
            {
                "revision": revision,
                "status": status,
                "partitions": {"dev": partition},
            }
        ),
        encoding="utf-8",
    )
    return gold


@pytest.fixture(autouse=True)
def _fixed_git_sha(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner_module, "_current_git_sha", lambda: GIT_SHA, raising=False)


def _run_cli_from(
    root: Path,
    *,
    split: str = "dev",
    id_map: str | None = None,
) -> int:
    argv = [
        "--config",
        str(CONFIG),
        "--split",
        split,
        "--output",
        "out",
    ]
    if id_map is not None:
        argv.extend(["--id-map", id_map])
    return runner_module.main(argv)


def test_cli_refuses_unfrozen_dev_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_cli_manifest(tmp_path, status="waiting_for_human_label_freeze")
    monkeypatch.chdir(tmp_path)

    assert _run_cli_from(tmp_path) == 2

    assert "data manifest is not frozen" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_cli_rejects_unknown_split_without_creating_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_cli_manifest(tmp_path)
    monkeypatch.chdir(tmp_path)

    assert _run_cli_from(tmp_path, split="not-a-split") == 2

    assert "unknown data split" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_cli_rejects_missing_gold_without_creating_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    gold = _write_cli_manifest(tmp_path)
    gold.unlink()
    monkeypatch.chdir(tmp_path)

    assert _run_cli_from(tmp_path) == 2

    assert "gold file does not exist" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("gold_path", ["../outside.jsonl", "C:/outside.jsonl"])
def test_cli_rejects_gold_path_outside_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    gold_path: str,
) -> None:
    _write_cli_manifest(tmp_path, gold_path=gold_path)
    monkeypatch.chdir(tmp_path)

    assert _run_cli_from(tmp_path) == 2

    assert "gold path must stay under data" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_cli_rejects_gold_hash_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_cli_manifest(tmp_path, gold_sha256=f"sha256:{'0' * 64}")
    monkeypatch.chdir(tmp_path)

    assert _run_cli_from(tmp_path) == 2

    assert "gold file SHA-256 mismatch" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("field", ["gold_sha256", "ids_sha256"])
def test_cli_rejects_missing_mandatory_partition_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    field: str,
) -> None:
    _write_cli_manifest(tmp_path, omit_partition_fields={field})
    monkeypatch.chdir(tmp_path)

    assert _run_cli_from(tmp_path) == 2

    assert "data split manifest is invalid" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_cli_rejects_empty_frozen_gold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_cli_manifest(tmp_path, gold_content="")
    monkeypatch.chdir(tmp_path)

    assert _run_cli_from(tmp_path) == 2

    assert "gold file must not be empty" in capsys.readouterr().err


@pytest.mark.parametrize("count", [0, -1, True, "1"])
def test_cli_requires_positive_integer_partition_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    count: object,
) -> None:
    _write_cli_manifest(tmp_path, partition_updates={"count": count})
    monkeypatch.chdir(tmp_path)

    assert _run_cli_from(tmp_path) == 2

    assert "data split manifest is invalid" in capsys.readouterr().err


@pytest.mark.parametrize("ids_path", ["../outside.json", "C:/outside.json"])
def test_cli_rejects_ids_path_outside_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    ids_path: str,
) -> None:
    _write_cli_manifest(tmp_path, partition_updates={"ids_path": ids_path})
    monkeypatch.chdir(tmp_path)

    assert _run_cli_from(tmp_path) == 2

    assert "ID path must stay under data" in capsys.readouterr().err


def test_cli_rejects_id_file_hash_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_cli_manifest(
        tmp_path,
        partition_updates={"ids_sha256": f"sha256:{'0' * 64}"},
    )
    monkeypatch.chdir(tmp_path)

    assert _run_cli_from(tmp_path) == 2

    assert "ID file SHA-256 mismatch" in capsys.readouterr().err


@pytest.mark.parametrize("labels_complete", [False, None, "true"])
def test_cli_requires_complete_labels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    labels_complete: object,
) -> None:
    _write_cli_manifest(
        tmp_path,
        partition_updates={"labels_complete": labels_complete},
    )
    monkeypatch.chdir(tmp_path)

    assert _run_cli_from(tmp_path) == 2

    assert "labels must be complete" in capsys.readouterr().err


@pytest.mark.parametrize("policy", [None, "", "ignore", True])
def test_cli_rejects_invalid_zero_answer_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    policy: object,
) -> None:
    _write_cli_manifest(tmp_path, partition_updates={"zero_answer_policy": policy})
    monkeypatch.chdir(tmp_path)

    assert _run_cli_from(tmp_path) == 2

    assert "zero-answer policy is invalid" in capsys.readouterr().err


@pytest.mark.parametrize("revision", ["", "   "])
def test_cli_requires_nonempty_dataset_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    revision: str,
) -> None:
    _write_cli_manifest(tmp_path, revision=revision)
    monkeypatch.chdir(tmp_path)

    assert _run_cli_from(tmp_path) == 2

    assert "dataset revision is invalid" in capsys.readouterr().err


def test_cli_rejects_gold_count_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_cli_manifest(
        tmp_path,
        ids=["query-1", "query-2"],
        partition_updates={"count": 2},
    )
    monkeypatch.chdir(tmp_path)

    assert _run_cli_from(tmp_path) == 2

    assert "gold record count mismatch" in capsys.readouterr().err


@pytest.mark.parametrize(
    "ids",
    [{"query-1": True}, ["query-1", "query-1"], [""], [1]],
)
def test_cli_rejects_invalid_or_duplicate_id_list(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    ids: object,
) -> None:
    _write_cli_manifest(tmp_path, ids=ids)
    monkeypatch.chdir(tmp_path)

    assert _run_cli_from(tmp_path) == 2

    assert "ID list is invalid" in capsys.readouterr().err


def test_cli_rejects_id_count_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_cli_manifest(tmp_path, ids=["query-1", "query-2"])
    monkeypatch.chdir(tmp_path)

    assert _run_cli_from(tmp_path) == 2

    assert "ID count mismatch" in capsys.readouterr().err


def test_cli_rejects_ordered_gold_id_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    gold_content = (
        '{"query_id":"query-1","query":"one",'
        '"relevant_paper_ids":["openalex:W1"]}\n'
        '{"query_id":"query-2","query":"two",'
        '"relevant_paper_ids":["openalex:W2"]}\n'
    )
    _write_cli_manifest(
        tmp_path,
        gold_content=gold_content,
        ids=["query-2", "query-1"],
        partition_updates={"count": 2},
    )
    monkeypatch.chdir(tmp_path)

    assert _run_cli_from(tmp_path) == 2

    assert "ordered query IDs do not match" in capsys.readouterr().err


def test_cli_rejects_zero_answer_record_under_reject_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_cli_manifest(
        tmp_path,
        gold_content='{"query_id":"query-1","query":"graph retrieval"}\n',
    )
    monkeypatch.chdir(tmp_path)

    assert _run_cli_from(tmp_path) == 2

    assert "zero-answer gold record is not allowed" in capsys.readouterr().err


@pytest.mark.parametrize("git_sha", ["", "not-a-sha", "a" * 39, "g" * 40])
def test_cli_rejects_invalid_git_sha_before_runtime_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    git_sha: str,
) -> None:
    _write_cli_manifest(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runner_module, "_current_git_sha", lambda: git_sha, raising=False)

    def fail_loader(*args: object, **kwargs: object) -> RuntimeConfig:
        raise AssertionError(f"runtime config constructed: {args} {kwargs}")

    monkeypatch.setattr(runner_module, "load_runtime_config", fail_loader)

    assert _run_cli_from(tmp_path) == 2

    assert "Git SHA is invalid" in capsys.readouterr().err


def test_frozen_split_returns_complete_typed_run_identity(tmp_path: Path) -> None:
    gold = _write_cli_manifest(
        tmp_path,
        gold_content='{"query_id":"query-1","query":"graph retrieval"}\n',
        partition_updates={"zero_answer_policy": "allow"},
    )
    manifest = tmp_path / "data" / "manifest.json"

    assert hasattr(runner_module, "_resolve_frozen_split")
    frozen = runner_module._resolve_frozen_split(tmp_path / "data", "dev", GIT_SHA)

    assert frozen.gold_path == gold.resolve()
    assert [record.query_id for record in frozen.gold] == ["query-1"]
    assert frozen.identity.model_dump(mode="json", exclude_none=True) == {
        "dataset_revision": "dataset-r1",
        "git_sha": GIT_SHA,
        "gold_sha256": _sha256(gold.read_bytes()),
        "manifest_sha256": _sha256(manifest.read_bytes()),
        "split": "dev",
        "zero_answer_policy": "allow",
    }


def test_legacy_run_evaluation_requires_explicit_run_identity_at_runtime() -> None:
    parameter = inspect.signature(run_evaluation).parameters.get("identity")

    assert parameter is not None
    assert parameter.default is None
    with pytest.raises(TypeError, match="legacy evaluation requires identity"):
        asyncio.run(run_evaluation([]))


def test_behavior_constants_are_publicly_exported() -> None:
    deduplication = importlib.import_module("paper_search.processing.deduplicate")
    filtering = importlib.import_module("paper_search.processing.filter")
    lexical = importlib.import_module("paper_search.ranking.lexical")

    assert deduplication.FUZZY_TITLE_THRESHOLD == 0.98
    assert deduplication.DEDUPLICATION_VERSION == "week1-dedup-v1"
    assert filtering.UNCERTAINTY_REASON_MULTIPLIER == 0.9
    assert filtering.MINIMUM_UNCERTAINTY_MULTIPLIER == 0.7
    assert filtering.FILTERING_VERSION == "week1-filter-v1"
    assert lexical.BM25_WEIGHT == 0.7
    assert lexical.KEYWORD_COVERAGE_WEIGHT == 0.3
    assert lexical.TOKENIZER_VERSION == "unicode-nfkc-alnum-v1"


def test_cli_parser_exposes_required_options_and_optional_id_map() -> None:
    parser = runner_module._build_parser()

    actions = {
        option: action
        for action in parser._actions
        for option in action.option_strings
        if option not in {"--help", "-h"}
    }
    assert set(actions) == {"--config", "--split", "--output", "--id-map"}
    assert all(actions[option].required for option in {"--config", "--split", "--output"})
    assert actions["--id-map"].required is False


def test_cli_binds_confined_identifier_map_into_metrics_and_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_cli_manifest(
        tmp_path,
        gold_content=(
            '{"query_id":"query-1","query":"graph retrieval",'
            '"relevant_paper_ids":["arxiv:2501.10120"]}\n'
        ),
    )
    map_path = tmp_path / "data" / "annotation_work" / "dev-map.json"
    map_path.parent.mkdir(parents=True)
    map_path.write_text(
        (
            '{"arxiv:2501.10120":"openalex:W1",'
            '"openalex:W2":"openalex:W1"}'
        ),
        encoding="utf-8",
    )
    map_hash = _sha256(map_path.read_bytes())
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENALEX_API_KEY", SENTINEL_API_KEY)

    def fake_provider_factory(**kwargs: object) -> FakeProvider:
        del kwargs
        return FakeProvider(
            {
                "graph retrieval": _provider_result(
                    [
                        _paper("openalex:W1", title="graph retrieval"),
                        _paper("openalex:W2", title="unrelated title"),
                    ],
                    calls=1,
                    latency_ms=1,
                )
            }
        )

    monkeypatch.setattr(
        runner_module,
        "OpenAlexProvider",
        fake_provider_factory,
        raising=False,
    )

    assert _run_cli_from(
        tmp_path,
        id_map="data/annotation_work/dev-map.json",
    ) == 0

    run = json.loads((tmp_path / "out" / "run.json").read_text(encoding="utf-8"))
    metrics = json.loads(
        (tmp_path / "out" / "metrics.json").read_text(encoding="utf-8")
    )
    assert run["identity"]["id_map_sha256"] == map_hash
    assert run["input_hashes"]["id_map"] == map_hash
    assert metrics["input_hashes"]["id_map"] == map_hash
    assert metrics["summary"]["macro_recall"] == 1.0
    assert metrics["summary"]["micro_recall"] == 1.0
    deduplication = json.loads(
        (tmp_path / "out" / "deduplication.jsonl").read_text(encoding="utf-8")
    )
    assert len(deduplication["paper_ids"]) == 1
    assert deduplication["merge_decisions"][0]["match_rule"] == "external_id"
    assert set(deduplication["merge_decisions"][0]["member_ids"]) == {
        "openalex:W1",
        "openalex:W2",
    }


def test_cli_uses_original_identifier_map_bytes_after_file_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_cli_manifest(
        tmp_path,
        gold_content=(
            '{"query_id":"query-1","query":"graph retrieval",'
            '"relevant_paper_ids":["arxiv:2501.10120"]}\n'
        ),
    )
    map_path = tmp_path / "data" / "annotation_work" / "replacement-map.json"
    map_path.parent.mkdir(parents=True)
    original_bytes = b'{"arxiv:2501.10120":"openalex:W1"}'
    replacement_bytes = b'{"arxiv:2501.10120":"openalex:W2"}'
    map_path.write_bytes(original_bytes)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENALEX_API_KEY", SENTINEL_API_KEY)

    def fake_provider_factory(**kwargs: object) -> FakeProvider:
        del kwargs
        return FakeProvider(
            {
                "graph retrieval": _provider_result(
                    [_paper("openalex:W1", title="graph retrieval")],
                    calls=1,
                    latency_ms=1,
                )
            }
        )

    original_read_map = runner_module._read_confined_identifier_map

    def replace_map_after_read(data_root: Path, raw_path: Path) -> bytes:
        content = original_read_map(data_root, raw_path)
        map_path.write_bytes(replacement_bytes)
        return content

    monkeypatch.setattr(
        runner_module,
        "_read_confined_identifier_map",
        replace_map_after_read,
    )
    monkeypatch.setattr(
        runner_module,
        "OpenAlexProvider",
        fake_provider_factory,
        raising=False,
    )

    assert _run_cli_from(
        tmp_path,
        id_map="data/annotation_work/replacement-map.json",
    ) == 0

    run = json.loads((tmp_path / "out" / "run.json").read_text(encoding="utf-8"))
    metrics = json.loads(
        (tmp_path / "out" / "metrics.json").read_text(encoding="utf-8")
    )
    assert map_path.read_bytes() == replacement_bytes
    assert run["identity"]["id_map_sha256"] == _sha256(original_bytes)
    assert metrics["summary"]["macro_recall"] == 1.0
    assert metrics["summary"]["micro_recall"] == 1.0


def test_cli_rejects_opened_identifier_map_target_outside_data_before_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_cli_manifest(
        tmp_path,
        gold_content=(
            '{"query_id":"query-1","query":"graph retrieval",'
            '"relevant_paper_ids":["arxiv:2501.10120"]}\n'
        ),
    )
    map_path = tmp_path / "data" / "annotation_work" / "race-map.json"
    map_path.parent.mkdir(parents=True)
    map_path.write_text(
        '{"arxiv:2501.10120":"openalex:W1"}',
        encoding="utf-8",
    )
    outside_map = tmp_path / "outside-private-map.json"
    outside_map.write_text(
        '{"arxiv:2501.10120":"openalex:W1"}',
        encoding="utf-8",
    )
    resolved_map_path = map_path.resolve()
    original_open = Path.open

    def open_swapped_map(
        path: Path,
        mode: str = "r",
        buffering: int = -1,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ) -> IO[Any]:
        target = outside_map if path == resolved_map_path and mode == "rb" else path
        return original_open(
            target,
            mode=mode,
            buffering=buffering,
            encoding=encoding,
            errors=errors,
            newline=newline,
        )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENALEX_API_KEY", SENTINEL_API_KEY)
    monkeypatch.setattr(Path, "open", open_swapped_map)
    monkeypatch.setattr(
        runner_module,
        "OpenAlexProvider",
        lambda **kwargs: pytest.fail(f"provider constructed: {sorted(kwargs)}"),
        raising=False,
    )

    assert _run_cli_from(
        tmp_path,
        id_map="data/annotation_work/race-map.json",
    ) == 2
    assert (
        capsys.readouterr().err.strip()
        == "evaluation failed: identifier map path must stay under data"
    )


def test_cli_rejects_identifier_map_when_final_handle_target_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_cli_manifest(
        tmp_path,
        gold_content=(
            '{"query_id":"query-1","query":"graph retrieval",'
            '"relevant_paper_ids":["arxiv:2501.10120"]}\n'
        ),
    )
    map_path = tmp_path / "data" / "annotation_work" / "map.json"
    map_path.parent.mkdir(parents=True)
    map_path.write_text(
        '{"arxiv:2501.10120":"openalex:W1"}',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENALEX_API_KEY", SENTINEL_API_KEY)

    def fail_final_path(source: IO[bytes]) -> Path:
        del source
        raise OSError("final target unavailable")

    monkeypatch.setattr(
        runner_module,
        "_final_path_from_open_file",
        fail_final_path,
        raising=False,
    )
    monkeypatch.setattr(
        runner_module,
        "OpenAlexProvider",
        lambda **kwargs: pytest.fail(f"provider constructed: {sorted(kwargs)}"),
        raising=False,
    )

    assert _run_cli_from(
        tmp_path,
        id_map="data/annotation_work/map.json",
    ) == 2
    assert capsys.readouterr().err.strip() == "evaluation failed"


@pytest.mark.parametrize(
    ("raw_path", "expected"),
    [
        (r"\\?\C:\data\map.json", r"C:\data\map.json"),
        (r"\\?\UNC\server\share\data\map.json", r"\\server\share\data\map.json"),
    ],
)
def test_normalize_windows_final_path_prefixes(
    raw_path: str,
    expected: str,
) -> None:
    assert runner_module._normalize_windows_final_path(raw_path) == expected


@pytest.mark.parametrize(
    "raw_path",
    ["../outside-map.json", "C:/outside-map.json"],
)
def test_cli_rejects_identifier_map_outside_data_before_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    raw_path: str,
) -> None:
    _write_cli_manifest(tmp_path)
    (tmp_path / "outside-map.json").write_text("{}", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENALEX_API_KEY", SENTINEL_API_KEY)
    monkeypatch.setattr(
        runner_module,
        "OpenAlexProvider",
        lambda **kwargs: pytest.fail(f"provider constructed: {sorted(kwargs)}"),
        raising=False,
    )

    assert _run_cli_from(tmp_path, id_map=raw_path) == 2
    assert "identifier map path must stay under data" in capsys.readouterr().err


def test_cli_rejects_identifier_map_symlink_to_outside_data_before_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_cli_manifest(tmp_path)
    outside_map = tmp_path / "outside-map.json"
    outside_map.write_text("{}", encoding="utf-8")
    map_link = tmp_path / "data" / "annotation_work" / "outside-link.json"
    map_link.parent.mkdir(parents=True)
    try:
        map_link.symlink_to(outside_map)
    except OSError as error:
        if error.winerror == 1314 or error.errno in {errno.EACCES, errno.EPERM}:
            pytest.skip(f"platform denied symlink creation: {error}")
        raise
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENALEX_API_KEY", SENTINEL_API_KEY)
    monkeypatch.setattr(
        runner_module,
        "OpenAlexProvider",
        lambda **kwargs: pytest.fail(f"provider constructed: {sorted(kwargs)}"),
        raising=False,
    )

    assert _run_cli_from(
        tmp_path,
        id_map="data/annotation_work/outside-link.json",
    ) == 2
    assert "identifier map path must stay under data" in capsys.readouterr().err


def test_cli_rejects_partial_identifier_map_before_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_cli_manifest(
        tmp_path,
        gold_content=(
            '{"query_id":"query-1","query":"graph retrieval",'
            '"relevant_paper_ids":["arxiv:2501.10120"]}\n'
        ),
    )
    map_path = tmp_path / "data" / "annotation_work" / "partial.json"
    map_path.parent.mkdir(parents=True)
    map_path.write_text(
        '{"arxiv:2501.99999":"openalex:W1"}',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENALEX_API_KEY", SENTINEL_API_KEY)
    monkeypatch.setattr(
        runner_module,
        "OpenAlexProvider",
        lambda **kwargs: pytest.fail(f"provider constructed: {sorted(kwargs)}"),
        raising=False,
    )

    assert _run_cli_from(
        tmp_path,
        id_map="data/annotation_work/partial.json",
    ) == 2
    captured = capsys.readouterr()
    assert "identifier map does not cover frozen gold identifiers" in captured.err
    assert "2501.10120" not in captured.out + captured.err


def test_cli_rejects_missing_identifier_map_before_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_cli_manifest(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENALEX_API_KEY", SENTINEL_API_KEY)
    monkeypatch.setattr(
        runner_module,
        "OpenAlexProvider",
        lambda **kwargs: pytest.fail(f"provider constructed: {sorted(kwargs)}"),
        raising=False,
    )

    assert _run_cli_from(
        tmp_path,
        id_map="data/annotation_work/missing.json",
    ) == 2
    assert "identifier map file does not exist" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_cli_rejects_identifier_map_directory_before_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_cli_manifest(tmp_path)
    map_directory = tmp_path / "data" / "annotation_work" / "map-directory"
    map_directory.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENALEX_API_KEY", SENTINEL_API_KEY)
    monkeypatch.setattr(
        runner_module,
        "OpenAlexProvider",
        lambda **kwargs: pytest.fail(f"provider constructed: {sorted(kwargs)}"),
        raising=False,
    )

    assert _run_cli_from(
        tmp_path,
        id_map="data/annotation_work/map-directory",
    ) == 2
    assert "identifier map file does not exist" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "payload",
    [
        "not-json",
        (
            '{"arxiv:2501.10120":"openalex:W1",'
            '"https://arxiv.org/abs/2501.10120":"openalex:W2"}'
        ),
        (
            '{"arxiv:2501.10120":"openalex:W1",'
            '"openalex:W1":"arxiv:2501.10120"}'
        ),
    ],
)
def test_cli_redacts_invalid_identifier_map_before_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    payload: str,
) -> None:
    _write_cli_manifest(tmp_path)
    map_path = tmp_path / "data" / "annotation_work" / "invalid.json"
    map_path.parent.mkdir(parents=True)
    map_path.write_text(payload, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENALEX_API_KEY", SENTINEL_API_KEY)
    monkeypatch.setattr(
        runner_module,
        "OpenAlexProvider",
        lambda **kwargs: pytest.fail(f"provider constructed: {sorted(kwargs)}"),
        raising=False,
    )

    assert _run_cli_from(
        tmp_path,
        id_map="data/annotation_work/invalid.json",
    ) == 2
    captured = capsys.readouterr()
    assert captured.err.strip() == "evaluation failed"
    assert payload not in captured.out + captured.err
    assert not (tmp_path / "out").exists()


def test_cli_loads_process_environment_and_builds_provider_without_secret_leak(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_cli_manifest(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENALEX_API_KEY", SENTINEL_API_KEY)
    load_calls: list[tuple[Path, object]] = []
    provider_calls: list[dict[str, object]] = []

    def recording_loader(path: Path, *, env_file: object) -> RuntimeConfig:
        load_calls.append((path, env_file))
        return load_runtime_config(path, env_file=env_file)

    def fake_provider_factory(**kwargs: object) -> FakeProvider:
        provider_calls.append(kwargs)
        cache = kwargs["cache"]
        assert isinstance(cache, SQLiteResponseCache)
        assert cache.path == tmp_path / ".cache" / "openalex.sqlite3"
        client = kwargs["client"]
        assert client.timeout.connect is not None
        assert client.timeout.read is not None
        assert client.timeout.write is not None
        assert client.timeout.pool is not None
        return FakeProvider(
            {
                "graph retrieval": _provider_result(
                    [_paper("openalex:W1", title="graph retrieval")],
                    calls=1,
                    latency_ms=1,
                )
            }
        )

    monkeypatch.setattr(runner_module, "load_runtime_config", recording_loader, raising=False)
    monkeypatch.setattr(runner_module, "OpenAlexProvider", fake_provider_factory, raising=False)

    assert _run_cli_from(tmp_path) == 0

    captured = capsys.readouterr()
    assert load_calls == [(CONFIG, None)]
    assert len(provider_calls) == 1
    assert provider_calls[0]["api_key"] == SENTINEL_API_KEY
    assert SENTINEL_API_KEY not in captured.out
    assert SENTINEL_API_KEY not in captured.err
    scanned_files = {
        artifact.relative_to(tmp_path).as_posix(): artifact.read_bytes()
        for artifact in tmp_path.rglob("*")
        if artifact.is_file()
    }
    assert ".cache/openalex.sqlite3" in scanned_files
    for relative_path, content in scanned_files.items():
        assert SENTINEL_API_KEY.encode() not in content, relative_path


def test_cli_requires_openalex_key_from_process_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_cli_manifest(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)

    assert _run_cli_from(tmp_path) == 2

    assert capsys.readouterr().err.strip().endswith("OPENALEX_API_KEY is required")
    assert not (tmp_path / "out").exists()


def test_cli_redacts_expected_provider_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_cli_manifest(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENALEX_API_KEY", SENTINEL_API_KEY)

    def failing_provider_factory(**kwargs: object) -> None:
        del kwargs
        raise ValueError(f"provider rejected {SENTINEL_API_KEY}")

    monkeypatch.setattr(
        runner_module,
        "OpenAlexProvider",
        failing_provider_factory,
        raising=False,
    )

    assert _run_cli_from(tmp_path) == 2

    captured = capsys.readouterr()
    assert "evaluation failed" in captured.err
    assert SENTINEL_API_KEY not in captured.out
    assert SENTINEL_API_KEY not in captured.err
    for artifact in tmp_path.rglob("*"):
        if artifact.is_file():
            assert SENTINEL_API_KEY.encode() not in artifact.read_bytes()


def _artifact_inputs(
    tmp_path: Path,
) -> tuple[list[EvaluationQuery], FakeProvider, SQLiteResponseCache, RuntimeConfig, Path]:
    gold = [
        EvaluationQuery(
            query_id="query-1",
            query="q1",
            relevant_paper_ids=["openalex:W1"],
        )
    ]
    cache = _populated_cache(tmp_path)
    first = cache.get_snapshot_response("page-1")
    second = cache.get_snapshot_response("page-2")
    assert first is not None and second is not None
    provider = FakeProvider(
        {
            "q1": _provider_result(
                [_paper("openalex:W1", title="q1 graph retrieval")],
                calls=2,
                latency_ms=19,
                cache_keys='["page-1","page-2","page-1"]',
                response_hash=_aggregate_hash(
                    [first.response_hash, second.response_hash, first.response_hash]
                ),
            )
        }
    )
    return (
        gold,
        provider,
        cache,
        _runtime_config(with_secret=True),
        tmp_path / "run",
    )


def _run_artifact_evaluation(
    gold: list[EvaluationQuery],
    provider: FakeProvider,
    cache: SQLiteResponseCache,
    config: RuntimeConfig,
    output: Path,
) -> runner_module.RunResult:
    return asyncio.run(
        run_evaluation(
            gold,
            identity=_run_identity(gold),
            provider=provider,
            cache=cache,
            config=config,
            output=output,
        )
    )


def test_process_candidates_composes_dedup_filter_and_rank() -> None:
    result = process_candidates(
        _query_spec("graph retrieval"),
        [
            _paper("openalex:W1", doi="10.1000/a"),
            _paper("s2:S1", doi="10.1000/a"),
        ],
    )

    assert len(result.deduplication.papers) == 1
    assert result.filtering.rejected == []
    assert [item.paper.canonical_id for item in result.ranked] == [
        result.deduplication.papers[0].canonical_id
    ]


def test_run_evaluation_preserves_order_and_isolates_structured_failure(
    tmp_path: Path,
) -> None:
    failure = ErrorDetail(
        code="provider_unavailable",
        message="provider unavailable",
        retryable=True,
        provider="openalex",
    )
    provider = FakeProvider(
        {
            "q1": _provider_result(
                [_paper("openalex:W1", title="q1 graph retrieval")],
                calls=1,
                latency_ms=11,
            ),
            "q2": _provider_result([], calls=2, latency_ms=17, errors=[failure]),
        }
    )
    gold = [
        EvaluationQuery(
            query_id="query-1",
            query="q1",
            relevant_paper_ids=["openalex:W1"],
        ),
        EvaluationQuery(
            query_id="query-2",
            query="q2",
            relevant_paper_ids=["openalex:W2"],
        ),
    ]

    result = asyncio.run(
        run_evaluation(
            gold,
            identity=_run_identity(gold),
            provider=provider,
            cache=SQLiteResponseCache(tmp_path / "cache.sqlite3"),
            config=_runtime_config(),
            output=tmp_path / "run",
        )
    )

    assert [item.query_id for item in result.query_runs] == ["query-1", "query-2"]
    assert result.query_runs[0].prediction.predicted_paper_ids == ["openalex:W1"]
    assert result.query_runs[1].prediction.predicted_paper_ids == []
    assert result.query_runs[1].errors == [failure]
    assert result.evaluation.summary.query_count == 2
    assert result.evaluation.summary.missing_prediction_count == 0
    assert result.usage.search_api_calls == 3
    assert result.usage.elapsed_ms == 28
    assert [call[0] for call in provider.calls] == ["q1", "q2"]
    assert all(call[1] == {} for call in provider.calls)
    assert all(call[2] == _runtime_config().budget.max_output_papers for call in provider.calls)
    assert all(
        call[3].reserved.search_api_calls == _runtime_config().budget.max_search_api_calls
        for call in provider.calls
    )
    assert json.loads(provider.calls[0][3].model_dump_json())["action"] == (
        "evaluation-search:query-1"
    )


def test_run_evaluation_rejects_malformed_cache_key_provenance(tmp_path: Path) -> None:
    provider = FakeProvider(
        {
            "q1": _provider_result(
                [],
                calls=1,
                latency_ms=1,
                cache_keys="{}",
            )
        }
    )

    with pytest.raises(ValueError, match="cache_keys"):
        asyncio.run(
            run_evaluation(
                [EvaluationQuery(query_id="query-1", query="q1")],
                identity=_run_identity(),
                provider=provider,
                cache=SQLiteResponseCache(tmp_path / "cache.sqlite3"),
                config=_runtime_config(),
                output=tmp_path / "run",
            )
        )


def test_run_evaluation_requires_cache_key_provenance(tmp_path: Path) -> None:
    provider_result = _provider_result([], calls=1, latency_ms=1)
    provider_result = provider_result.model_copy(
        update={
            "provenance": {
                key: value
                for key, value in provider_result.provenance.items()
                if key != "cache_keys"
            }
        }
    )
    provider = FakeProvider({"q1": provider_result})

    with pytest.raises(ValueError, match="cache_keys"):
        asyncio.run(
            run_evaluation(
                [EvaluationQuery(query_id="query-1", query="q1")],
                identity=_run_identity(),
                provider=provider,
                cache=SQLiteResponseCache(tmp_path / "cache.sqlite3"),
                config=_runtime_config(),
                output=tmp_path / "run",
            )
        )


def test_run_evaluation_rejects_snapshot_response_hash_mismatch(tmp_path: Path) -> None:
    cache = _populated_cache(tmp_path)
    provider = FakeProvider(
        {
            "q1": _provider_result(
                [],
                calls=1,
                latency_ms=1,
                cache_keys='["page-1"]',
                response_hash=f"sha256:{'0' * 64}",
            )
        }
    )
    output = tmp_path / "run"

    with pytest.raises(ValueError, match="response hash"):
        asyncio.run(
            run_evaluation(
                [EvaluationQuery(query_id="query-1", query="q1")],
                identity=_run_identity(),
                provider=provider,
                cache=cache,
                config=_runtime_config(),
                output=output,
            )
        )

    assert not output.exists()


@pytest.mark.parametrize(
    ("provider_name", "endpoint"),
    [("semantic_scholar", "/works"), ("openalex", "/different")],
)
def test_run_evaluation_rejects_snapshot_provider_or_endpoint_mismatch(
    tmp_path: Path,
    provider_name: str,
    endpoint: str,
) -> None:
    cache = _populated_cache(tmp_path)
    cached = cache.get_snapshot_response("page-1")
    assert cached is not None
    provider = FakeProvider(
        {
            "q1": _provider_result(
                [],
                calls=1,
                latency_ms=1,
                cache_keys='["page-1"]',
                provider=provider_name,
                endpoint=endpoint,
                response_hash=cached.response_hash,
            )
        }
    )

    with pytest.raises(ValueError, match="provenance mismatch"):
        asyncio.run(
            run_evaluation(
                [EvaluationQuery(query_id="query-1", query="q1")],
                identity=_run_identity(),
                provider=provider,
                cache=cache,
                config=_runtime_config(),
                output=tmp_path / "run",
            )
        )


def test_run_evaluation_rejects_missing_snapshot_cache_key_safely(tmp_path: Path) -> None:
    provider = FakeProvider(
        {
            "q1": _provider_result(
                [],
                calls=1,
                latency_ms=1,
                cache_keys='["missing"]',
            )
        }
    )

    with pytest.raises(ValueError, match="cache key is missing"):
        asyncio.run(
            run_evaluation(
                [EvaluationQuery(query_id="query-1", query="q1")],
                identity=_run_identity(),
                provider=provider,
                cache=SQLiteResponseCache(tmp_path / "cache.sqlite3"),
                config=_runtime_config(),
                output=tmp_path / "run",
            )
        )


def test_run_evaluation_validates_multi_page_hash_order(tmp_path: Path) -> None:
    cache = _populated_cache(tmp_path)
    first = cache.get_snapshot_response("page-1")
    second = cache.get_snapshot_response("page-2")
    assert first is not None and second is not None
    provider = FakeProvider(
        {
            "q1": _provider_result(
                [],
                calls=1,
                latency_ms=1,
                cache_keys='["page-1","page-2"]',
                response_hash=_aggregate_hash([second.response_hash, first.response_hash]),
            )
        }
    )

    with pytest.raises(ValueError, match="response hash"):
        asyncio.run(
            run_evaluation(
                [EvaluationQuery(query_id="query-1", query="q1")],
                identity=_run_identity(),
                provider=provider,
                cache=cache,
                config=_runtime_config(),
                output=tmp_path / "run",
            )
        )


def test_run_evaluation_records_valid_shared_cache_key_associations(tmp_path: Path) -> None:
    cache = _populated_cache(tmp_path)
    cached = cache.get_snapshot_response("page-1")
    assert cached is not None
    provider = FakeProvider(
        {
            query: _provider_result(
                [],
                calls=0,
                latency_ms=1,
                cache_keys='["page-1"]',
                response_hash=cached.response_hash,
            )
            for query in ("q1", "q2")
        }
    )
    gold = [
        EvaluationQuery(query_id="query-1", query="q1"),
        EvaluationQuery(query_id="query-2", query="q2"),
    ]
    output = tmp_path / "run"

    _run_artifact_evaluation(gold, provider, cache, _runtime_config(), output)

    run = json.loads((output / "run.json").read_bytes())
    assert run["query_snapshots"] == [
        {
            "cache_keys": ["page-1"],
            "endpoint": "/works",
            "page_hashes": [cached.response_hash],
            "provider": "openalex",
            "query_id": query_id,
            "response_hash": cached.response_hash,
        }
        for query_id in ("query-1", "query-2")
    ]
    assert len(list((output / "snapshots").glob("*.json"))) == 1


def test_run_evaluation_revalidates_shared_key_immediately_before_export(
    tmp_path: Path,
) -> None:
    cache = _populated_cache(tmp_path)
    original = cache.get_snapshot_response("page-1")
    assert original is not None

    class MutatingProvider:
        def __init__(self) -> None:
            self.calls = 0

        async def search(
            self,
            query: str,
            filters: dict[str, object],
            limit: int,
            reservation: BudgetReservation,
        ) -> ProviderResult[list[Paper]]:
            del query, filters, limit, reservation
            self.calls += 1
            if self.calls == 1:
                response_hash = original.response_hash
            else:
                cache.put_response(
                    key="page-1",
                    provider="openalex",
                    endpoint="/works",
                    cache_version="v1",
                    params={"search": "q2"},
                    raw_response=b'{"page":"replaced","results":[]}',
                    requested_at=datetime(2026, 7, 17, 1, tzinfo=UTC),
                    ttl=timedelta(days=7),
                    safe_headers={},
                )
                replaced = cache.get_snapshot_response("page-1")
                assert replaced is not None
                response_hash = replaced.response_hash
            return _provider_result(
                [],
                calls=0,
                latency_ms=1,
                cache_keys='["page-1"]',
                response_hash=response_hash,
            )

    gold = [
        EvaluationQuery(query_id="query-1", query="q1"),
        EvaluationQuery(query_id="query-2", query="q2"),
    ]
    output = tmp_path / "run"

    with pytest.raises(ValueError, match="changed after query"):
        asyncio.run(
            run_evaluation(
                gold,
                identity=_run_identity(gold),
                provider=MutatingProvider(),
                cache=cache,
                config=_runtime_config(),
                output=output,
            )
        )

    assert not output.exists()


def test_run_evaluation_rejects_key_deleted_after_query_with_safe_error(
    tmp_path: Path,
) -> None:
    cache = _populated_cache(tmp_path)
    original = cache.get_snapshot_response("page-1")
    assert original is not None

    class DeletingProvider:
        def __init__(self) -> None:
            self.calls = 0

        async def search(
            self,
            query: str,
            filters: dict[str, object],
            limit: int,
            reservation: BudgetReservation,
        ) -> ProviderResult[list[Paper]]:
            del query, filters, limit, reservation
            self.calls += 1
            if self.calls == 1:
                return _provider_result(
                    [],
                    calls=0,
                    latency_ms=1,
                    cache_keys='["page-1"]',
                    response_hash=original.response_hash,
                )
            with sqlite3.connect(cache.path) as connection:
                connection.execute(
                    "DELETE FROM responses WHERE cache_key = ?",
                    ("page-1",),
                )
            return _provider_result([], calls=0, latency_ms=1)

    gold = [
        EvaluationQuery(query_id="query-1", query="q1"),
        EvaluationQuery(query_id="query-2", query="q2"),
    ]
    output = tmp_path / "run"

    with pytest.raises(ValueError, match="cache key changed after query"):
        asyncio.run(
            run_evaluation(
                gold,
                identity=_run_identity(gold),
                provider=DeletingProvider(),
                cache=cache,
                config=_runtime_config(),
                output=output,
            )
        )

    assert not output.exists()


def test_run_evaluation_rejects_cached_hash_not_matching_raw_bytes(
    tmp_path: Path,
) -> None:
    cache = _populated_cache(tmp_path)
    stale_hash = f"sha256:{'0' * 64}"
    with sqlite3.connect(cache.path) as connection:
        connection.execute(
            "UPDATE responses SET response_hash = ? WHERE cache_key = ?",
            (stale_hash, "page-1"),
        )
    provider = FakeProvider(
        {
            "q1": _provider_result(
                [],
                calls=0,
                latency_ms=1,
                cache_keys='["page-1"]',
                response_hash=stale_hash,
            )
        }
    )
    gold = [EvaluationQuery(query_id="query-1", query="q1")]
    output = tmp_path / "run"

    with pytest.raises(ValueError, match="cached response bytes"):
        asyncio.run(
            run_evaluation(
                gold,
                identity=_run_identity(gold),
                provider=provider,
                cache=cache,
                config=_runtime_config(),
                output=output,
            )
        )

    assert not output.exists()


def test_run_evaluation_serializes_nonempty_audit_jsonl_records(tmp_path: Path) -> None:
    cache = _populated_cache(tmp_path)
    cached = cache.get_snapshot_response("page-1")
    assert cached is not None
    provider = FakeProvider(
        {
            "q1": _provider_result(
                [
                    _paper("openalex:W70", doi="10.1000/shared"),
                    _paper("s2:S70", doi="10.1000/shared"),
                    _paper("openalex:W71", title="Retracted paper", is_retracted=True),
                    _paper("openalex:W72", title="Uncertain paper", is_retracted=None),
                ],
                calls=0,
                latency_ms=1,
                cache_keys='["page-1"]',
                response_hash=cached.response_hash,
            )
        }
    )
    gold = [EvaluationQuery(query_id="query-1", query="q1")]
    output = tmp_path / "run"

    _run_artifact_evaluation(gold, provider, cache, _runtime_config(), output)

    deduplication = json.loads((output / "deduplication.jsonl").read_text().strip())
    filtering = json.loads((output / "filtering.jsonl").read_text().strip())
    assert deduplication["merge_decisions"][0]["match_rule"] == "doi"
    assert filtering["rejected"] == [
        {"paper_id": "openalex:W71", "reason_code": "retracted"}
    ]
    assert {
        item["paper_id"]: item["uncertainty_reasons"]
        for item in filtering["accepted"]
    }["openalex:W72"] == ["unknown_retraction_status"]


def test_run_evaluation_sums_all_known_costs(tmp_path: Path) -> None:
    provider = FakeProvider(
        {
            "q1": _provider_result([], calls=0, latency_ms=1, cost_cny=0.01),
            "q2": _provider_result([], calls=0, latency_ms=1, cost_cny=0.02),
        }
    )
    gold = [
        EvaluationQuery(query_id="query-1", query="q1"),
        EvaluationQuery(query_id="query-2", query="q2"),
    ]

    result = asyncio.run(
        run_evaluation(
            gold,
            identity=_run_identity(gold),
            provider=provider,
            cache=SQLiteResponseCache(tmp_path / "cache.sqlite3"),
            config=_runtime_config(),
            output=tmp_path / "run",
        )
    )

    assert result.usage.cost_cny == pytest.approx(Decimal("0.03"))


def test_run_evaluation_preserves_unknown_mixed_cost(tmp_path: Path) -> None:
    provider = FakeProvider(
        {
            "q1": _provider_result([], calls=0, latency_ms=1, cost_cny=0.01),
            "q2": _provider_result([], calls=0, latency_ms=1, cost_cny=None),
        }
    )
    gold = [
        EvaluationQuery(query_id="query-1", query="q1"),
        EvaluationQuery(query_id="query-2", query="q2"),
    ]

    result = asyncio.run(
        run_evaluation(
            gold,
            identity=_run_identity(gold),
            provider=provider,
            cache=SQLiteResponseCache(tmp_path / "cache.sqlite3"),
            config=_runtime_config(),
            output=tmp_path / "run",
        )
    )

    assert result.usage.cost_cny is None


def test_empty_evaluation_preserves_unknown_cost(tmp_path: Path) -> None:
    result = asyncio.run(
        run_evaluation(
            [],
            identity=_run_identity([]),
            provider=FakeProvider({}),
            cache=SQLiteResponseCache(tmp_path / "cache.sqlite3"),
            config=_runtime_config(),
            output=tmp_path / "run",
        )
    )

    assert result.usage.cost_cny is None


def test_run_evaluation_writes_deterministic_secret_safe_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gold, provider, cache, config, output = _artifact_inputs(tmp_path)

    result = _run_artifact_evaluation(gold, provider, cache, config, output)

    expected_paths = {
        "predictions.jsonl",
        "metrics.json",
        "usage.json",
        "run.json",
        "deduplication.jsonl",
        "filtering.jsonl",
        "snapshot_manifest.json",
        "snapshots/openalex-0001.json",
        "snapshots/openalex-0002.json",
    }
    artifacts = {
        path.relative_to(output).as_posix(): path.read_bytes()
        for path in output.rglob("*")
        if path.is_file()
    }
    assert set(artifacts) == expected_paths
    validate_snapshot_manifest(output / "snapshot_manifest.json")
    assert (output / "snapshots/openalex-0001.json").read_bytes() == (b'{"page":1,"results":[]}\n')

    predictions = [item.prediction for item in result.query_runs]
    exact_identity_hash = _run_identity(gold).gold_sha256
    assert exact_identity_hash != _sha256(_canonical_jsonl_bytes(gold))
    expected_hashes = {
        "gold": exact_identity_hash,
        "predictions": _sha256(_canonical_jsonl_bytes(predictions)),
    }
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["contract_version"] == CONTRACT_VERSION
    assert metrics["input_hashes"] == expected_hashes
    snapshot_hash = _sha256(artifacts["snapshot_manifest.json"])
    assert metrics["snapshot_manifest"] == "snapshot_manifest.json"
    assert metrics["snapshot_manifest_sha256"] == snapshot_hash
    assert metrics["summary"] == result.evaluation.summary.model_dump(mode="json")

    run = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert run["contract_version"] == "week1-run-v2"
    assert run["config_hash"] == config.config_hash()
    assert run["input_hashes"] == expected_hashes
    assert run["identity"] == _run_identity(gold).model_dump(
        mode="json",
        exclude_none=True,
    )
    assert run["rules"] == {
        "deduplication": {
            "fuzzy_title_threshold": FUZZY_TITLE_THRESHOLD,
            "version": "week1-dedup-v1",
        },
        "filtering": {
            "minimum_uncertainty_multiplier": MINIMUM_UNCERTAINTY_MULTIPLIER,
            "uncertainty_reason_multiplier": UNCERTAINTY_REASON_MULTIPLIER,
            "version": "week1-filter-v1",
        },
        "scoring": {
            "bm25_weight": BM25_WEIGHT,
            "keyword_coverage_weight": KEYWORD_COVERAGE_WEIGHT,
            "scoring_version": SCORING_VERSION,
            "tokenizer_version": TOKENIZER_VERSION,
        },
    }
    assert run["scoring_version"] == SCORING_VERSION
    assert run["snapshot_manifest"] == "snapshot_manifest.json"
    assert run["snapshot_manifest_sha256"] == snapshot_hash
    assert all(not Path(value).is_absolute() for value in run["artifacts"].values())
    assert str(output) not in json.dumps(run)

    usage = json.loads((output / "usage.json").read_text(encoding="utf-8"))
    assert usage["total"] == result.usage.model_dump(mode="json")
    assert usage["queries"][0]["latency_ms"] == 19
    assert usage["queries"][0]["errors"] == []

    deduplication = json.loads((output / "deduplication.jsonl").read_text().strip())
    assert deduplication["query_id"] == "query-1"
    assert deduplication["paper_ids"] == ["openalex:W1"]
    assert deduplication["merge_decisions"] == []
    filtering = json.loads((output / "filtering.jsonl").read_text().strip())
    assert filtering["accepted"] == [{"paper_id": "openalex:W1", "uncertainty_reasons": []}]
    assert filtering["rejected"] == []

    for relative, content in artifacts.items():
        assert SENTINEL_API_KEY.encode() not in content, relative
        if relative.endswith((".json", ".jsonl")):
            assert content.endswith(b"\n"), relative
    for relative in (
        "metrics.json",
        "usage.json",
        "run.json",
        "snapshot_manifest.json",
    ):
        payload = json.loads(artifacts[relative])
        assert list(payload) == sorted(payload), relative

    def fail_if_rewritten(path: Path, records: object) -> None:
        raise AssertionError(f"identical predictions were rewritten: {path} {records}")

    monkeypatch.setattr(runner_module, "write_jsonl_atomic", fail_if_rewritten)
    repeated = _run_artifact_evaluation(gold, provider, cache, config, output)
    assert repeated == result
    assert {
        path.relative_to(output).as_posix(): path.read_bytes()
        for path in output.rglob("*")
        if path.is_file()
    } == artifacts


@pytest.mark.parametrize(
    "relative_path",
    [
        "predictions.jsonl",
        "metrics.json",
        "usage.json",
        "run.json",
        "deduplication.jsonl",
        "filtering.jsonl",
        "snapshot_manifest.json",
        "snapshots/openalex-0001.json",
    ],
)
def test_run_evaluation_refuses_nonidentical_formal_artifact_before_prediction_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative_path: str,
) -> None:
    gold, provider, cache, config, output = _artifact_inputs(tmp_path)
    _run_artifact_evaluation(gold, provider, cache, config, output)
    (output / relative_path).write_text("changed\n", encoding="utf-8")

    def fail_if_called(path: Path, records: object) -> None:
        raise AssertionError(f"prediction writer called before preflight: {path} {records}")

    monkeypatch.setattr(runner_module, "write_jsonl_atomic", fail_if_called)
    with pytest.raises(FileExistsError):
        _run_artifact_evaluation(gold, provider, cache, config, output)


def test_run_evaluation_preflights_existing_artifacts_before_snapshot_write(
    tmp_path: Path,
) -> None:
    gold, provider, cache, config, output = _artifact_inputs(tmp_path)
    output.mkdir()
    (output / "metrics.json").write_text("changed\n", encoding="utf-8")

    with pytest.raises(FileExistsError):
        _run_artifact_evaluation(gold, provider, cache, config, output)

    assert not (output / "snapshot_manifest.json").exists()
    assert not (output / "snapshots").exists()
def test_formal_run_request_and_result_contracts(tmp_path: Path) -> None:
    request = EvaluationRunRequest(
        split="dev",
        mode="replay",
        lock_path=tmp_path / "replay.lock.yaml",
        output_root=tmp_path / "runs",
        snapshot_manifest_path=tmp_path / "snapshot-manifest.json",
        network_authorized=False,
    )
    result = EvaluationRunResult(
        run_id="formal-1",
        run_path=tmp_path / "runs" / "formal-1",
        status="complete",
        gate_result="passed",
    )

    assert request.mode == "replay"
    assert result.status == "complete"


def test_formal_runner_exposes_canonical_request_and_injection_boundary() -> None:
    parameters = inspect.signature(run_evaluation).parameters

    assert next(iter(parameters)) == "request"
    assert "composition_root" in parameters
    assert "attempt_store_factory" in parameters
    assert "clock" in parameters


def test_ordered_service_batch_continues_after_query_scoped_exception() -> None:
    calls: list[str] = []

    class FakeService:
        async def execute(self, request: object, *, run_id: str | None = None):
            query_id = getattr(request, "query_id")
            calls.append(query_id)
            if query_id == "q1":
                return SearchExecutionResult(
                    outcome=SearchFailure(
                        query_id=query_id,
                        run_id=run_id or "formal-1",
                        error=SearchErrorResponse(
                            code="internal_error",
                            detail="safe",
                            retryable=False,
                            run_id=run_id,
                        ),
                        usage=UsageActual(),
                        stop_reason="internal_error",
                    ),
                    diagnostics=[],
                    business_result_sha256=None,
                )
            return SearchExecutionResult(
                outcome=SearchFailure(
                    query_id=query_id,
                    run_id=run_id or "formal-1",
                    error=SearchErrorResponse(
                        code="dependency_failure",
                        detail="safe",
                        retryable=True,
                        run_id=run_id,
                    ),
                    usage=UsageActual(),
                    stop_reason="dependency_failure",
                ),
                diagnostics=[],
                business_result_sha256=None,
            )

    records = asyncio.run(
        runner_module._execute_service_batch(
            [
                EvaluationQuery(query_id="q1", query="one", metadata={"split": "dev"}),
                EvaluationQuery(query_id="q2", query="two", metadata={"split": "dev"}),
            ],
            service=FakeService(),
            run_id="formal-1",
            mode="replay",
        )
    )

    assert calls == ["q1", "q2"]
    assert [record.prediction.query_id for record in records] == ["q1", "q2"]
    assert [record.failure.error_code for record in records if record.failure] == [
        "internal_error",
        "dependency_failure",
    ]


def test_ordered_service_batch_stops_on_unexpected_integrity_exception() -> None:
    class BrokenService:
        async def execute(self, request: object, *, run_id: str | None = None):
            del request, run_id
            raise ValueError("business hash mismatch")

    with pytest.raises(ValueError, match="business hash mismatch"):
        asyncio.run(
            runner_module._execute_service_batch(
                [EvaluationQuery(query_id="q1", query="one")],
                service=BrokenService(),
                run_id="formal-1",
                mode="replay",
            )
        )


def test_formal_capture_binding_reuses_exact_workspace_store(tmp_path: Path) -> None:
    store = DependencyCaptureStore(tmp_path / "snapshots")
    binding = runner_module._FormalCaptureBinding(run_id="formal-1", store=store)

    assert binding.claim_snapshot_store() is store
    assert binding.claim_snapshot_store() is store


def test_service_batch_settles_and_appends_before_starting_next_query() -> None:
    events: list[str] = []

    class FakeService:
        async def execute(self, request: object, *, run_id: str | None = None):
            query_id = getattr(request, "query_id")
            events.append(f"execute:{query_id}")
            return SearchExecutionResult(
                outcome=SearchFailure(
                    query_id=query_id,
                    run_id=run_id or "formal-1",
                    error=SearchErrorResponse(
                        code="dependency_failure",
                        detail="safe",
                        retryable=True,
                        run_id=run_id,
                    ),
                    usage=UsageActual(search_api_calls=1),
                    stop_reason="dependency_failure",
                ),
                diagnostics=[],
                business_result_sha256=None,
            )

    records = asyncio.run(
        runner_module._execute_service_batch(
            [
                EvaluationQuery(query_id="q1", query="one"),
                EvaluationQuery(query_id="q2", query="two"),
            ],
            service=FakeService(),
            run_id="formal-1",
            mode="replay",
            on_start=lambda index: events.append(f"reserve:q{index + 1}"),
            on_record=lambda index, record: events.append(
                f"settle-and-append:{record.execution.query_id}"
            ),
        )
    )

    assert [record.execution.usage.search_api_calls for record in records] == [1, 1]
    assert events == [
        "reserve:q1",
        "execute:q1",
        "settle-and-append:q1",
        "reserve:q2",
        "execute:q2",
        "settle-and-append:q2",
    ]


def test_cancellation_closes_every_outstanding_reservation(tmp_path: Path) -> None:
    ledger = runner_module.SQLiteBudgetLedger(tmp_path / "ledger.sqlite3")
    estimates = [
        runner_module.UsageEstimate(
            search_api_calls=value,
            cost_cny=Decimal(value) / Decimal("10"),
        )
        for value in (1, 2, 3)
    ]
    reservations = [
        ledger.reserve(
            run_id="formal-1",
            query_id=f"q{index}",
            estimate=estimate,
            run_cap_cny=Decimal("18"),
        )
        for index, estimate in enumerate(estimates, start=1)
    ]
    ledger.settle(
        reservations[0],
        UsageActual(search_api_calls=1, cost_cny=Decimal("0.1")),
    )

    runner_module._close_outstanding_reservations(
        ledger=ledger,
        reservations=reservations,
        estimates=estimates,
        settled={0},
        current_index=1,
    )

    report = ledger.report("formal-1")
    assert report.actual.search_api_calls == 3
    assert report.actual.cost_cny == Decimal("0.3")


def test_formal_inputs_reject_current_source_sha_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_bytes = Path("tests/fixtures/application/candidate.lock.yaml").read_bytes()
    lock = runner_module.CandidateLock.model_validate(yaml.safe_load(lock_bytes))
    lock_path = tmp_path / "candidate.lock.yaml"
    lock_path.write_bytes(lock_bytes)
    monkeypatch.setattr(
        runner_module,
        "load_verified_input_lock_bytes",
        lambda content, *, artifact_root: SimpleNamespace(
            lock=lock,
            artifact_bytes={},
        ),
    )
    monkeypatch.setattr(runner_module, "_current_git_sha", lambda: "different")

    with pytest.raises(ValueError, match="current source SHA"):
        runner_module._load_formal_inputs(
            EvaluationRunRequest(
                split="dev",
                mode="live",
                lock_path=lock_path,
                output_root=tmp_path / "runs",
                snapshot_manifest_path=None,
                network_authorized=True,
            )
        )


def test_formal_inputs_reject_dirty_tracked_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_bytes = Path("tests/fixtures/application/candidate.lock.yaml").read_bytes()
    lock = runner_module.CandidateLock.model_validate(yaml.safe_load(lock_bytes))
    lock_path = tmp_path / "candidate.lock.yaml"
    lock_path.write_bytes(lock_bytes)
    monkeypatch.setattr(
        runner_module,
        "load_verified_input_lock_bytes",
        lambda content, *, artifact_root: SimpleNamespace(
            lock=lock,
            artifact_bytes={},
        ),
    )
    monkeypatch.setattr(runner_module, "_current_git_sha", lambda: lock.source_git_sha)
    monkeypatch.setattr(
        runner_module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=" M tracked.py\n"),
    )

    with pytest.raises(ValueError, match="tracked source must be clean"):
        runner_module._load_formal_inputs(
            EvaluationRunRequest(
                split="dev",
                mode="live",
                lock_path=lock_path,
                output_root=tmp_path / "runs",
                snapshot_manifest_path=None,
                network_authorized=True,
            )
        )


def test_existing_claim_recovers_complete_published_run_then_rejects_reuse(
    tmp_path: Path,
) -> None:
    validation_hash = "sha256:" + "a" * 64
    shutil.copytree(
        Path("tests/fixtures/formal_run/capture"),
        tmp_path / "capture",
    )
    manifest_path = tmp_path / "capture" / "run.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["config_hash"] = validation_hash
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    store = runner_module.ValidationAttemptStore(tmp_path)
    store.claim(
        validation_lock_sha256=validation_hash,
        run_id="capture",
        claimed_at=datetime(2026, 8, 1, tzinfo=UTC),
    )

    with pytest.raises(
        runner_module.ValidationAttemptConflictError,
        match="irrevocable attempt",
    ):
        runner_module._reject_or_recover_existing_attempt(
            store=store,
            validation_lock_sha256=validation_hash,
            output_root=tmp_path,
        )

    assert store.read(validation_hash).state == "complete"
