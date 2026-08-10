from __future__ import annotations

import inspect
import hashlib
import json
import sqlite3
from collections.abc import Mapping
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from paper_search.control.ledger import DEV_RUN_CAP_CNY, SQLiteBudgetLedger
from paper_search.domain.models import UsageActual, UsageEstimate
from paper_search.storage.dependency_snapshot import (
    DependencyRequestIdentity,
    DependencySnapshotManifestV2,
)
from scripts.capture_dev_identifier_identity import (
    S2_FIELDS,
    IdentifierCaptureRuntime,
    IdentifierInventory,
    IdentityCaptureLock,
    TransportResponse,
    _load_lock,
    build_capture_lock,
    build_identifier_inventory,
    capture_identity,
    ledger_checkpoint,
    main,
)


CAPTURED_AT = datetime(2026, 8, 10, 12, 0, tzinfo=UTC)
S2_ARXIV_BYTES = (
    b'[ {"paperId":"S2-A","externalIds":{"ArXiv":"2501.00001",'
    b'"DOI":"10.1000/a"}}, {"paperId":"S2-B","externalIds":'
    b'{"ArXiv":"2501.00002","DOI":"10.1000/b"}} ]'
)
S2_DOI_BYTES = (
    b'[{"paperId":"S2-A","externalIds":{"ArXiv":"2501.00001",'
    b'"DOI":"10.1000/a"}},{"paperId":"S2-B","externalIds":'
    b'{"ArXiv":"2501.00002","DOI":"10.1000/b"}}]'
)
OPENALEX_BYTES = (
    b'{"meta":{"count":1},"results":'
    b'[{"id":"https://openalex.org/W1","locations":'
    b'[{"landing_page_url":"https://arxiv.org/abs/2501.00001"}]}]}'
)


class RecordingTransport:
    def __init__(
        self,
        *,
        arxiv_response: bytes = S2_ARXIV_BYTES,
        doi_response: bytes = S2_DOI_BYTES,
        openalex_response: bytes = OPENALEX_BYTES,
        statuses: Mapping[str, list[int]] | None = None,
    ) -> None:
        self.arxiv_response = arxiv_response
        self.doi_response = doi_response
        self.openalex_response = openalex_response
        self.statuses = {key: list(values) for key, values in (statuses or {}).items()}
        self.semantic_scholar_batches: list[list[str]] = []
        self.semantic_scholar_requests: list[tuple[dict[str, str], dict[str, object]]] = []
        self.openalex_ids: list[str] = []
        self.openalex_urls: list[str] = []
        self.openalex_query_param_names: list[list[str]] = []

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        query_params: Mapping[str, str] | None,
        json_body: object | None,
    ) -> TransportResponse:
        del headers
        if method == "POST":
            assert isinstance(json_body, dict)
            self.semantic_scholar_requests.append(
                (dict(query_params or {}), dict(json_body))
            )
            ids = json_body["ids"]
            assert isinstance(ids, list)
            normalized_ids = [str(identifier) for identifier in ids]
            self.semantic_scholar_batches.append(normalized_ids)
            route = "s2-arxiv" if normalized_ids[0].startswith("ARXIV:") else "s2-doi"
            content = self.arxiv_response if route == "s2-arxiv" else self.doi_response
        else:
            route = "openalex"
            self.openalex_urls.append(url)
            filter_value = (query_params or {})["filter"]
            identifier = filter_value.removeprefix("openalex:")
            self.openalex_ids.append(identifier)
            self.openalex_query_param_names.append(sorted((query_params or {}).keys()))
            content = self.openalex_response
        route_statuses = self.statuses.get(route, [])
        status_code = route_statuses.pop(0) if route_statuses else 200
        if status_code >= 400:
            content = b'{"error":"provider failure for 10.9999/private-value"}'
        return TransportResponse(
            status_code=status_code,
            content=content,
            headers={"content-type": "application/json"},
        )


class TamperingTransport(RecordingTransport):
    def __init__(self, snapshot_root: Path) -> None:
        super().__init__()
        self.snapshot_root = snapshot_root
        self._tampered = False

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        query_params: Mapping[str, str] | None,
        json_body: object | None,
    ) -> TransportResponse:
        if method == "POST" and isinstance(json_body, dict):
            ids = json_body.get("ids")
            if (
                isinstance(ids, list)
                and ids
                and str(ids[0]).startswith("DOI:")
                and not self._tampered
            ):
                next(self.snapshot_root.rglob("*.bin")).write_bytes(b"tampered")
                self._tampered = True
        return super().request(
            method,
            url,
            headers=headers,
            query_params=query_params,
            json_body=json_body,
        )


def _write_inventory(tmp_path: Path, *, include_openalex: bool = True) -> Path:
    aliases = ["doi:10.1000/candidate-only"]
    if include_openalex:
        aliases.append("openalex:W1")
    inventory = IdentifierInventory(
        schema_version="identifier-identity-inventory-v1",
        scope="dev",
        source_hashes={
            "dev_gold": "sha256:" + "a" * 64,
            "candidate_map": "sha256:" + "b" * 64,
        },
        arxiv_ids=["arxiv:2501.00002", "arxiv:2501.00001"],
        candidate_aliases=aliases,
    )
    path = tmp_path / "identifier-inventory.json"
    path.write_text(inventory.model_dump_json(), encoding="utf-8")
    return path


def _lock_bytes(lock: IdentityCaptureLock) -> bytes:
    return (
        json.dumps(
            lock.model_dump(mode="json", exclude_none=True),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def _lock_digest(lock: IdentityCaptureLock) -> str:
    return f"sha256:{hashlib.sha256(_lock_bytes(lock)).hexdigest()}"


def _empty_ledger(path: Path) -> None:
    ledger = SQLiteBudgetLedger(path)
    ledger.close()


def _lock_and_runtime(
    tmp_path: Path,
    transport: RecordingTransport | None = None,
    *,
    include_openalex: bool = True,
) -> tuple[IdentityCaptureLock, IdentifierCaptureRuntime, RecordingTransport]:
    inventory_path = _write_inventory(tmp_path, include_openalex=include_openalex)
    ledger_path = tmp_path / "ledger.sqlite3"
    _empty_ledger(ledger_path)
    lock = build_capture_lock(
        inventory_path=inventory_path,
        ledger_path=ledger_path,
        output_root="private",
    )
    selected_transport = transport or RecordingTransport()
    runtime = IdentifierCaptureRuntime(
        project_root=tmp_path,
        inventory_path=inventory_path,
        ledger_path=ledger_path,
        expected_lock_sha256=_lock_digest(lock),
        output_root="private",
        semantic_scholar_base_url="https://api.semanticscholar.org",
        semantic_scholar_endpoint="/graph/v1/paper/batch",
        semantic_scholar_api_key_env="SEMANTIC_SCHOLAR_API_KEY",
        openalex_base_url="https://api.openalex.org",
        openalex_endpoint_template="/works",
        openalex_api_key_env="OPENALEX_API_KEY",
        credential_values={
            "SEMANTIC_SCHOLAR_API_KEY": "mock-s2-key",
            "OPENALEX_API_KEY": "mock-openalex-key",
        },
        transport=selected_transport,
        allow_network=True,
        clock=lambda: CAPTURED_AT,
    )
    return lock, runtime, selected_transport


def _ledger_states(path: Path) -> list[str]:
    with sqlite3.connect(path) as connection:
        return [
            str(row[0])
            for row in connection.execute("SELECT state FROM reservations ORDER BY rowid")
        ]


def test_inventory_consumes_only_dev_gold_and_current_candidate_map(
    tmp_path: Path,
) -> None:
    gold_path = tmp_path / "gold.jsonl"
    map_path = tmp_path / "identifier-map.json"
    output_path = tmp_path / "inventory.json"
    gold_path.write_bytes(
        b'{"query_id":"private-query","query":"private text",'
        b'"relevant_paper_ids":["ARXIV:2501.00002","arxiv:2501.00001"]}\n'
    )
    map_path.write_bytes(
        b'{"arxiv:2501.00001":"doi:10.48550/arxiv.2501.00001",'
        b'"doi:10.1000/a":"doi:10.48550/arxiv.2501.00001",'
        b'"openalex:W1":"doi:10.48550/arxiv.2501.00001"}\n'
    )

    inventory = build_identifier_inventory(
        gold_path=gold_path,
        candidate_map_path=map_path,
        out_path=output_path,
    )

    assert tuple(inspect.signature(build_identifier_inventory).parameters) == (
        "gold_path",
        "candidate_map_path",
        "out_path",
    )
    assert "predictions" not in inspect.getsource(build_identifier_inventory)
    assert inventory.arxiv_ids == ["arxiv:2501.00001", "arxiv:2501.00002"]
    assert inventory.candidate_aliases == ["doi:10.1000/a", "openalex:W1"]
    assert IdentifierInventory.model_validate_json(output_path.read_bytes()) == inventory


def test_preflight_locks_dev_inputs_without_network_or_ledger_mutation(
    tmp_path: Path,
) -> None:
    inventory_path = _write_inventory(tmp_path)
    ledger_path = tmp_path / "ledger.sqlite3"
    _empty_ledger(ledger_path)
    before = ledger_checkpoint(ledger_path)

    lock = build_capture_lock(
        inventory_path=inventory_path,
        ledger_path=ledger_path,
        output_root="private",
    )

    assert lock.schema_version == "identifier-identity-capture-lock-v2"
    assert lock.scope == "dev"
    assert lock.semantic_scholar_batch_max == 2
    assert lock.semantic_scholar_http_attempt_max == 4
    assert lock.semantic_scholar_arxiv_ids == sorted(set(lock.semantic_scholar_arxiv_ids))
    assert lock.openalex_request_max == 2
    assert lock.openalex_endpoint_template == "/works"
    assert ledger_checkpoint(ledger_path) == before


def test_preflight_writes_canonical_lock_with_exclusive_create(tmp_path: Path) -> None:
    inventory_path = _write_inventory(tmp_path)
    ledger_path = tmp_path / "ledger.sqlite3"
    lock_path = tmp_path / "capture.lock.json"
    _empty_ledger(ledger_path)

    lock = build_capture_lock(
        inventory_path=inventory_path,
        ledger_path=ledger_path,
        output_root="private",
        out_lock=lock_path,
    )

    assert json.loads(lock_path.read_bytes()) == lock.model_dump(mode="json")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        build_capture_lock(
            inventory_path=inventory_path,
            ledger_path=ledger_path,
            output_root="private",
            out_lock=lock_path,
        )


def test_lock_loader_rejects_v1_schema_even_with_matching_digest(
    tmp_path: Path,
) -> None:
    lock, _, _ = _lock_and_runtime(tmp_path)
    payload = lock.model_dump(mode="json")
    payload["schema_version"] = "identifier-identity-capture-lock-v1"
    content = (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )
    lock_path = tmp_path / "v1-capture.lock.json"
    lock_path.write_bytes(content)
    expected_digest = f"sha256:{hashlib.sha256(content).hexdigest()}"

    with pytest.raises(ValueError) as error:
        _load_lock(lock_path, expected_lock_sha256=expected_digest)

    assert str(error.value) == "identity capture authorization mismatch"


def test_capture_uses_arxiv_batch_then_only_discovered_doi_batch(
    tmp_path: Path,
) -> None:
    lock, runtime, transport = _lock_and_runtime(tmp_path)

    result = capture_identity(lock, runtime)

    assert transport.semantic_scholar_batches == [
        ["ARXIV:2501.00001", "ARXIV:2501.00002"],
        ["DOI:10.1000/a", "DOI:10.1000/b"],
    ]
    assert transport.semantic_scholar_requests == [
        (
            {"fields": S2_FIELDS},
            {"ids": ["ARXIV:2501.00001", "ARXIV:2501.00002"]},
        ),
        (
            {"fields": S2_FIELDS},
            {"ids": ["DOI:10.1000/a", "DOI:10.1000/b"]},
        ),
    ]
    assert transport.openalex_ids == ["W1"]
    assert transport.openalex_urls == ["https://api.openalex.org/works"]
    assert transport.openalex_query_param_names == [["api_key", "filter", "per_page"]]
    assert result.derived_doi_lock.ids == ["DOI:10.1000/a", "DOI:10.1000/b"]
    assert result.semantic_scholar_batch_count == 2
    assert result.semantic_scholar_http_attempt_count == 2
    assert result.openalex_request_count == 1
    assert all(state == "settled" for state in _ledger_states(runtime.ledger_path))

    snapshot_root = tmp_path / "private" / "snapshots"
    manifest = DependencySnapshotManifestV2.model_validate_json(
        (snapshot_root / "snapshot-manifest.json").read_bytes()
    )
    assert {
        entry.request.model_or_adapter
        for entry in manifest.entries
        if entry.request.dependency == "semantic_scholar"
    } == {
        "semantic-scholar-identity-arxiv-v1",
        "semantic-scholar-identity-doi-v1",
    }
    openalex_entries = [
        entry for entry in manifest.entries if entry.request.dependency == "openalex"
    ]
    assert [entry.request for entry in openalex_entries] == [
        DependencyRequestIdentity.from_canonical_request(
            dependency="openalex",
            operation="search",
            method="GET",
            endpoint="/works",
            model_or_adapter="openalex-identity-v1",
            canonical_request={"filter": "openalex:W1", "per_page": "1"},
        )
    ]
    raw_responses = {
        (snapshot_root / entry.response_path).read_bytes() for entry in manifest.entries
    }
    assert S2_ARXIV_BYTES in raw_responses
    assert S2_DOI_BYTES in raw_responses
    assert OPENALEX_BYTES in raw_responses
    assert [
        (
            ref.semantic_scholar_arxiv_item_index,
            ref.semantic_scholar_doi_item_index,
        )
        for ref in result.evidence_refs
        if ref.alias.startswith("doi:")
    ] == [(0, 0), (1, 1)]


def test_capture_never_queries_an_unsealed_or_hand_added_doi(tmp_path: Path) -> None:
    lock, runtime, transport = _lock_and_runtime(tmp_path)
    runtime = runtime.with_manual_doi_addition("DOI:10.1000/not-discovered")

    with pytest.raises(ValueError, match="derived DOI lock mismatch"):
        capture_identity(lock, runtime)

    assert transport.semantic_scholar_batches == [
        ["ARXIV:2501.00001", "ARXIV:2501.00002"]
    ]
    assert all(state != "reserved" for state in _ledger_states(runtime.ledger_path))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("semantic_scholar_base_url", "https://example.invalid"),
        ("openalex_endpoint_template", "/search"),
        ("semantic_scholar_api_key_env", "OTHER_KEY"),
        ("output_root", "other"),
    ],
)
def test_capture_rejects_runtime_scope_outside_locked_authorization(
    tmp_path: Path, field: str, value: str
) -> None:
    lock, runtime, transport = _lock_and_runtime(tmp_path)

    with pytest.raises(ValueError, match="identity capture authorization mismatch"):
        capture_identity(lock, runtime.model_copy(update={field: value}))

    assert transport.semantic_scholar_batches == []
    assert _ledger_states(runtime.ledger_path) == []


def test_runtime_representation_redacts_credentials(tmp_path: Path) -> None:
    _, runtime, _ = _lock_and_runtime(tmp_path)

    assert "mock-s2-key" not in repr(runtime)
    assert "mock-openalex-key" not in repr(runtime)


def test_runtime_requires_independently_approved_lock_digest(tmp_path: Path) -> None:
    _, runtime, _ = _lock_and_runtime(tmp_path)
    payload = runtime.model_dump(mode="python")
    payload.pop("expected_lock_sha256", None)

    with pytest.raises(ValidationError):
        IdentifierCaptureRuntime.model_validate(payload)


def test_cli_run_requires_independently_approved_lock_digest(tmp_path: Path) -> None:
    lock, runtime, _ = _lock_and_runtime(tmp_path)
    lock_path = tmp_path / "capture.lock.json"
    lock_path.write_bytes(_lock_bytes(lock))

    with pytest.raises(SystemExit):
        main(
            [
                "run",
                "--lock",
                str(lock_path),
                "--ledger",
                str(runtime.ledger_path),
                "--allow-network",
            ]
        )


def test_cli_rejects_noncanonical_lock_bytes_even_with_matching_digest(
    tmp_path: Path,
) -> None:
    lock, runtime, _ = _lock_and_runtime(tmp_path)
    lock_path = tmp_path / "capture.lock.json"
    noncanonical_content = _lock_bytes(lock) + b" "
    lock_path.write_bytes(noncanonical_content)
    expected = f"sha256:{hashlib.sha256(noncanonical_content).hexdigest()}"

    with pytest.raises(ValueError) as error:
        main(
            [
                "run",
                "--lock",
                str(lock_path),
                "--expected-lock-sha256",
                expected,
                "--ledger",
                str(runtime.ledger_path),
                "--allow-network",
            ]
        )

    assert str(error.value) == "identity capture authorization mismatch"


@pytest.mark.parametrize("mutation", ["identifier_list", "inflated_cap"])
def test_capture_rederives_inventory_authorization_before_network(
    tmp_path: Path, mutation: str
) -> None:
    lock, runtime, transport = _lock_and_runtime(tmp_path)
    if mutation == "identifier_list":
        edited_lock = lock.model_copy(
            update={
                "semantic_scholar_arxiv_ids": [
                    *lock.semantic_scholar_arxiv_ids,
                    "ARXIV:2501.99999",
                ]
            }
        )
    else:
        edited_lock = lock.model_copy(
            update={"openalex_request_max": lock.openalex_request_max + 1}
        )
    runtime = runtime.model_copy(
        update={"expected_lock_sha256": _lock_digest(edited_lock)}
    )

    with pytest.raises(ValueError) as error:
        capture_identity(edited_lock, runtime)

    assert str(error.value) == "identity capture authorization mismatch"
    assert transport.semantic_scholar_batches == []
    assert _ledger_states(runtime.ledger_path) == []


def test_capture_rejects_wrong_expected_lock_digest_before_network(
    tmp_path: Path,
) -> None:
    lock, runtime, transport = _lock_and_runtime(tmp_path)
    runtime = runtime.model_copy(
        update={"expected_lock_sha256": "sha256:" + "f" * 64}
    )

    with pytest.raises(ValueError) as error:
        capture_identity(lock, runtime)

    assert str(error.value) == "identity capture authorization mismatch"
    assert transport.semantic_scholar_batches == []
    assert _ledger_states(runtime.ledger_path) == []


def test_zero_discovered_dois_skips_second_batch_without_empty_request(
    tmp_path: Path,
) -> None:
    transport = RecordingTransport(arxiv_response=b"[null,null]")
    lock, runtime, _ = _lock_and_runtime(
        tmp_path,
        transport,
        include_openalex=False,
    )

    result = capture_identity(lock, runtime)

    assert transport.semantic_scholar_batches == [
        ["ARXIV:2501.00001", "ARXIV:2501.00002"]
    ]
    assert result.derived_doi_lock.ids == []
    assert result.semantic_scholar_batch_count == 1


def test_retry_consumes_attempt_allowance_and_terminalizes_each_reservation(
    tmp_path: Path,
) -> None:
    transport = RecordingTransport(statuses={"s2-arxiv": [500, 200]})
    lock, runtime, _ = _lock_and_runtime(tmp_path, transport, include_openalex=False)

    result = capture_identity(lock, runtime)

    assert result.semantic_scholar_http_attempt_count == 3
    assert _ledger_states(runtime.ledger_path) == ["failed", "settled", "settled"]


def test_terminal_http_failure_has_value_free_error_and_no_reserved_receipts(
    tmp_path: Path,
) -> None:
    transport = RecordingTransport(statuses={"s2-arxiv": [500, 500]})
    lock, runtime, _ = _lock_and_runtime(tmp_path, transport, include_openalex=False)

    with pytest.raises(ValueError) as error:
        capture_identity(lock, runtime)

    assert str(error.value) == "semantic scholar request failed"
    assert "10.9999" not in str(error.value)
    assert _ledger_states(runtime.ledger_path) == ["failed", "failed"]


def test_malformed_json_fails_closed_with_terminal_receipts(tmp_path: Path) -> None:
    transport = RecordingTransport(arxiv_response=b"not-json")
    lock, runtime, _ = _lock_and_runtime(tmp_path, transport, include_openalex=False)

    with pytest.raises(ValueError) as error:
        capture_identity(lock, runtime)

    assert str(error.value) == "semantic scholar response is invalid"
    assert _ledger_states(runtime.ledger_path) == ["failed", "failed"]


def test_missing_semantic_scholar_side_is_preserved_without_false_evidence(
    tmp_path: Path,
) -> None:
    transport = RecordingTransport(
        arxiv_response=(
            b'[{"paperId":"S2-A","externalIds":{"ArXiv":"2501.00001",'
            b'"DOI":"10.1000/a"}},null]'
        ),
        doi_response=(
            b'[{"paperId":"S2-A","externalIds":{"ArXiv":"2501.00001",'
            b'"DOI":"10.1000/a"}}]'
        ),
    )
    lock, runtime, _ = _lock_and_runtime(tmp_path, transport, include_openalex=False)

    result = capture_identity(lock, runtime)

    doi_refs = [ref for ref in result.evidence_refs if ref.alias.startswith("doi:")]
    assert result.derived_doi_lock.ids == ["DOI:10.1000/a"]
    assert [(ref.arxiv_id, ref.alias) for ref in doi_refs] == [
        ("arxiv:2501.00001", "doi:10.1000/a")
    ]


@pytest.mark.parametrize(
    "results",
    [
        [],
        [
            {"id": "https://openalex.org/W1", "locations": []},
            {"id": "https://openalex.org/W2", "locations": []},
        ],
        [{"id": "https://openalex.org/W999", "locations": []}],
    ],
    ids=["zero", "multiple", "mismatch"],
)
def test_openalex_exact_lookup_rejects_non_exact_result_envelope(
    tmp_path: Path, results: list[dict[str, object]]
) -> None:
    transport = RecordingTransport(
        openalex_response=json.dumps(
            {"meta": {"count": len(results)}, "results": results},
            separators=(",", ":"),
        ).encode("utf-8")
    )
    lock, runtime, _ = _lock_and_runtime(tmp_path, transport)

    with pytest.raises(ValueError) as error:
        capture_identity(lock, runtime)

    assert str(error.value) == "openalex response is invalid"
    assert transport.openalex_ids == ["W1", "W1"]
    assert all(state != "reserved" for state in _ledger_states(runtime.ledger_path))


def test_openalex_exact_lookup_accepts_one_matching_result(tmp_path: Path) -> None:
    transport = RecordingTransport(
        openalex_response=(
            b'{"meta":{"count":1},"results":'
            b'[{"id":"https://openalex.org/W1","locations":'
            b'[{"landing_page_url":"https://arxiv.org/abs/2501.00001"}]}]}'
        )
    )
    lock, runtime, _ = _lock_and_runtime(tmp_path, transport)

    result = capture_identity(lock, runtime)

    assert transport.openalex_ids == ["W1"]
    assert [
        (ref.arxiv_id, ref.alias)
        for ref in result.evidence_refs
        if ref.alias.startswith("openalex:")
    ] == [("arxiv:2501.00001", "openalex:W1")]
    assert all(state == "settled" for state in _ledger_states(runtime.ledger_path))


def test_edited_request_cap_is_rejected_before_ledger_or_transport(
    tmp_path: Path,
) -> None:
    lock, runtime, transport = _lock_and_runtime(tmp_path)
    lock = lock.model_copy(update={"openalex_request_max": 0})
    runtime = runtime.model_copy(update={"expected_lock_sha256": _lock_digest(lock)})

    with pytest.raises(ValueError, match="identity capture authorization mismatch"):
        capture_identity(lock, runtime)

    assert transport.semantic_scholar_batches == []
    assert _ledger_states(runtime.ledger_path) == []


def test_snapshot_tampering_fails_closed_after_terminal_accounting(tmp_path: Path) -> None:
    snapshot_root = tmp_path / "private" / "snapshots"
    transport = TamperingTransport(snapshot_root)
    lock, runtime, _ = _lock_and_runtime(tmp_path, transport)

    with pytest.raises(ValueError) as error:
        capture_identity(lock, runtime)

    assert str(error.value) == "identity snapshot is invalid"
    assert all(state != "reserved" for state in _ledger_states(runtime.ledger_path))


def test_ledger_checkpoint_drift_is_rejected_before_network(tmp_path: Path) -> None:
    lock, runtime, transport = _lock_and_runtime(tmp_path)
    ledger = SQLiteBudgetLedger(runtime.ledger_path)
    reservation = ledger.reserve(
        run_id="drift",
        query_id="drift",
        estimate=UsageEstimate(search_api_calls=1, cost_cny=Decimal("0.01")),
        run_cap_cny=DEV_RUN_CAP_CNY,
    )
    ledger.settle(
        reservation,
        UsageActual(search_api_calls=1, cost_cny=Decimal("0.01")),
    )
    ledger.close()

    with pytest.raises(ValueError, match="ledger checkpoint mismatch"):
        capture_identity(lock, runtime)

    assert transport.semantic_scholar_batches == []
    assert _ledger_states(runtime.ledger_path) == ["settled"]


def test_capture_rejects_different_ledger_with_same_checkpoint(tmp_path: Path) -> None:
    lock, runtime, transport = _lock_and_runtime(tmp_path)
    alternate_ledger_path = tmp_path / "alternate-ledger.sqlite3"
    _empty_ledger(alternate_ledger_path)

    with pytest.raises(ValueError, match="identity capture authorization mismatch"):
        capture_identity(
            lock,
            runtime.model_copy(update={"ledger_path": alternate_ledger_path}),
        )

    assert transport.semantic_scholar_batches == []
    assert _ledger_states(alternate_ledger_path) == []
