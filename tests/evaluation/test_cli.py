from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import paper_search.evaluation.metrics as metrics_module


FIXTURE_ROOT = Path("tests/fixtures/evaluation")
SUBPROCESS_ENV = {**os.environ, "PYTHONPATH": str(Path("src").resolve())}


def _run_cli(output: Path, *extra_args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "paper_search.evaluation.metrics",
            "--gold",
            str(FIXTURE_ROOT / "gold.jsonl"),
            "--pred",
            str(FIXTURE_ROOT / "predictions.jsonl"),
            "--out",
            str(output),
            *extra_args,
        ],
        check=False,
        capture_output=True,
        env=SUBPROCESS_ENV,
        text=True,
    )


def test_cli_writes_stable_metrics(tmp_path: Path) -> None:
    output = tmp_path / "metrics.json"

    first = _run_cli(output)
    assert first.returncode == 0, first.stderr
    assert first.stderr == ""
    first_bytes = output.read_bytes()

    second = _run_cli(output)
    assert second.returncode == 0, second.stderr
    assert output.read_bytes() == first_bytes

    payload = json.loads(first_bytes)
    assert payload["contract_version"] == "task2-evaluation-v1"
    assert set(payload["input_hashes"]) == {"gold", "predictions"}
    assert "macro_f1" in payload["summary"]
    assert "micro_f1" in payload["summary"]
    assert list(payload["per_query"]) == ["q1", "q2"]


def test_cli_hashes_optional_identifier_map(tmp_path: Path) -> None:
    output = tmp_path / "metrics.json"

    result = _run_cli(
        output,
        "--id-map",
        str(FIXTURE_ROOT / "id_map.json"),
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_bytes())
    assert set(payload["input_hashes"]) == {"gold", "predictions", "id_map"}
    assert payload["per_query"]["q1"]["gold_ids"][0] == "openalex:W1"


def test_cli_uses_one_identifier_map_byte_snapshot_for_metrics_and_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gold = tmp_path / "gold.jsonl"
    gold.write_text(
        '{"query_id":"q1","query":"private query",'
        '"relevant_paper_ids":["arxiv:2501.10120"]}\n',
        encoding="utf-8",
    )
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        '{"query_id":"q1","selected_paper_ids":["openalex:W1"]}\n',
        encoding="utf-8",
    )
    identifier_map = tmp_path / "private-map.json"
    original_bytes = b'{"arxiv:2501.10120":"openalex:W1"}'
    replacement_bytes = b'{"arxiv:2501.10120":"openalex:W2"}'
    identifier_map.write_bytes(original_bytes)
    output = tmp_path / "metrics.json"
    original_read_bytes = Path.read_bytes
    map_reads = 0

    def replace_map_after_read(path: Path) -> bytes:
        nonlocal map_reads
        content = original_read_bytes(path)
        if path == identifier_map:
            map_reads += 1
            path.write_bytes(replacement_bytes)
        return content

    monkeypatch.setattr(Path, "read_bytes", replace_map_after_read)

    exit_code = metrics_module.main(
        [
            "--gold",
            str(gold),
            "--pred",
            str(predictions),
            "--out",
            str(output),
            "--id-map",
            str(identifier_map),
        ]
    )

    payload = json.loads(original_read_bytes(output))
    assert exit_code == 0
    assert map_reads == 1
    assert payload["per_query"]["q1"]["gold_ids"] == ["openalex:W1"]
    assert payload["input_hashes"]["id_map"] == (
        f"sha256:{hashlib.sha256(original_bytes).hexdigest()}"
    )


@pytest.mark.parametrize(
    ("payload", "expected_error"),
    [
        (
            (
                b'{"doi:10.1000/private-secret":"openalex:W1",'
                b'"DOI:10.1000/private-secret":"openalex:W2"}'
            ),
            "evaluation failed: identifier map is invalid\n",
        ),
        (None, "evaluation failed: identifier map is unavailable\n"),
    ],
)
def test_cli_redacts_invalid_or_missing_identifier_map(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    payload: bytes | None,
    expected_error: str,
) -> None:
    identifier_map = tmp_path / "private-map-do-not-leak.json"
    if payload is not None:
        identifier_map.write_bytes(payload)
    output = tmp_path / "metrics.json"

    exit_code = metrics_module.main(
        [
            "--gold",
            str(FIXTURE_ROOT / "gold.jsonl"),
            "--pred",
            str(FIXTURE_ROOT / "predictions.jsonl"),
            "--out",
            str(output),
            "--id-map",
            str(identifier_map),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert captured.err == expected_error
    assert str(identifier_map) not in captured.err
    assert "private-secret" not in captured.err
    assert "openalex:W1" not in captured.err
    assert "openalex:W2" not in captured.err
    assert not output.exists()


def test_package_metric_exports_are_lazy_and_public() -> None:
    script = """
import sys
import paper_search.evaluation as evaluation

assert "paper_search.evaluation.metrics" not in sys.modules
from paper_search.evaluation import metrics

for name in (
    "EvaluationResult",
    "MetricSummary",
    "QueryMetrics",
    "deduplicate_ranked",
    "evaluate",
    "score_query",
):
    assert getattr(evaluation, name) is getattr(metrics, name)
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        env=SUBPROCESS_ENV,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_cli_fails_without_replacing_existing_output(tmp_path: Path) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text("not-json\n", encoding="utf-8")
    output = tmp_path / "metrics.json"
    output.write_text("preserve-me\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "paper_search.evaluation.metrics",
            "--gold",
            str(bad),
            "--pred",
            str(FIXTURE_ROOT / "predictions.jsonl"),
            "--out",
            str(output),
        ],
        check=False,
        capture_output=True,
        env=SUBPROCESS_ENV,
        text=True,
    )

    assert result.returncode == 2
    assert output.read_text(encoding="utf-8") == "preserve-me\n"
