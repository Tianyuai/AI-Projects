from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

import paper_search.evaluation.runner as runner_module
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
from paper_search.evaluation.runner import process_candidates, run_evaluation
from paper_search.ranking import SCORING_VERSION
from paper_search.storage import SQLiteResponseCache
from paper_search.storage.cache import validate_snapshot_manifest


CONFIG = Path(__file__).parents[2] / "configs" / "base.yaml"


def _paper(
    canonical_id: str,
    *,
    title: str = "Graph retrieval",
    doi: str | None = None,
) -> Paper:
    return Paper(
        canonical_id=canonical_id,
        title=title,
        doi=doi,
        is_retracted=False,
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
    errors: list[ErrorDetail] | None = None,
) -> ProviderResult[list[Paper]]:
    return ProviderResult(
        data=papers,
        usage=UsageActual(search_api_calls=calls, elapsed_ms=latency_ms),
        provenance={
            "provider": "openalex",
            "endpoint": "/works",
            "model_id": "openalex-api",
            "requested_at": "2026-07-17T00:00:00+00:00",
            "response_hash": f"sha256:{'0' * 64}",
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


def _write_cli_manifest(
    root: Path,
    *,
    status: str = "frozen",
    gold_path: str = "dev/gold.jsonl",
    gold_sha256: str | None = None,
) -> Path:
    data = root / "data"
    gold = data / "dev" / "gold.jsonl"
    gold.parent.mkdir(parents=True)
    gold.write_text(
        '{"query_id":"query-1","query":"graph retrieval"}\n',
        encoding="utf-8",
    )
    partition: dict[str, object] = {"gold_path": gold_path}
    if gold_sha256 is not None:
        partition["gold_sha256"] = gold_sha256
    (data / "manifest.json").write_text(
        json.dumps(
            {
                "status": status,
                "partitions": {"dev": partition},
            }
        ),
        encoding="utf-8",
    )
    return gold


def _run_cli_from(root: Path, *, split: str = "dev") -> int:
    return runner_module.main(
        [
            "--config",
            str(CONFIG),
            "--split",
            split,
            "--output",
            "out",
        ]
    )


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


def test_cli_parser_exposes_only_required_week1_options() -> None:
    parser = runner_module._build_parser()

    options = {
        option
        for action in parser._actions
        for option in action.option_strings
        if option != "--help" and option != "-h"
    }
    assert options == {"--config", "--split", "--output"}
    assert all(action.required for action in parser._actions if action.dest != "help")


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
    provider = FakeProvider(
        {
            "q1": _provider_result(
                [_paper("openalex:W1", title="q1 graph retrieval")],
                calls=2,
                latency_ms=19,
                cache_keys='["page-1","page-2","page-1"]',
            )
        }
    )
    return (
        gold,
        provider,
        _populated_cache(tmp_path),
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
                provider=provider,
                cache=SQLiteResponseCache(tmp_path / "cache.sqlite3"),
                config=_runtime_config(),
                output=tmp_path / "run",
            )
        )


def test_empty_evaluation_preserves_unknown_cost(tmp_path: Path) -> None:
    result = asyncio.run(
        run_evaluation(
            [],
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
    expected_hashes = {
        "gold": _sha256(_canonical_jsonl_bytes(gold)),
        "predictions": _sha256(_canonical_jsonl_bytes(predictions)),
    }
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["contract_version"] == CONTRACT_VERSION
    assert metrics["input_hashes"] == expected_hashes
    assert metrics["summary"] == result.evaluation.summary.model_dump(mode="json")

    run = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert run["config_hash"] == config.config_hash()
    assert run["input_hashes"] == expected_hashes
    assert run["scoring_version"] == SCORING_VERSION
    assert run["snapshot_manifest"] == "snapshot_manifest.json"
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
