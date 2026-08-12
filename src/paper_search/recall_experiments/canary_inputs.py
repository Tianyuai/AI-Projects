"""Unified query inputs for scored and unscored candidate-recall canaries."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import Field

from paper_search.domain.models import DomainModel, NonEmptyStr
from paper_search.recall_experiments.inputs.formal_run import FormalRunInputSource
from paper_search.recall_experiments.recipes import LoadedSampleBinding, load_sample_binding


class RecallCase(DomainModel):
    """One public query; Gold is optional and never enters generation payloads."""

    query_id: NonEmptyStr
    query: NonEmptyStr
    gold_paper_ids: tuple[NonEmptyStr, ...] | None = Field(default=None, min_length=1)


class LoadedCanaryInput(DomainModel):
    input_kind: Literal["single", "jsonl", "frozen"]
    cases: tuple[RecallCase, ...]
    evaluation_status: Literal["available", "not_available"]
    input_sha256: str
    identifier_map_bytes: bytes | None = None
    identifier_map_sha256: str | None = None


def load_single_case(query: str, query_id: str = "user-query-001") -> tuple[RecallCase, ...]:
    return (RecallCase(query_id=query_id, query=query),)


def load_jsonl_cases(path: str | Path) -> tuple[RecallCase, ...]:
    return _load_jsonl_case_bytes(Path(path).read_bytes())


def _load_jsonl_case_bytes(source: bytes) -> tuple[RecallCase, ...]:
    cases: list[RecallCase] = []
    for line in source.splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError("each query JSONL row must be an object")
        cases.append(RecallCase.model_validate(payload))
    return _validated_cases(cases)


def load_frozen_cases(
    sample_path: str | Path, *, workspace_root: Path
) -> tuple[RecallCase, ...]:
    loaded = load_sample_binding(sample_path)
    return _loaded_frozen_cases(loaded, workspace_root=workspace_root)


def _loaded_frozen_cases(
    loaded: LoadedSampleBinding, *, workspace_root: Path
) -> tuple[RecallCase, ...]:
    dataset = FormalRunInputSource(workspace_root).load_queries(loaded.binding)
    return _validated_cases(
        [
            RecallCase(
                query_id=query.query_id,
                query=query.query,
                gold_paper_ids=tuple(query.relevant_paper_ids),
            )
            for query in dataset.queries
        ]
    )


def load_canary_input(
    *,
    input_kind: Literal["single", "jsonl", "frozen"],
    query: str | None,
    query_id: str | None,
    source_path: Path | None,
    identifier_map_path: Path | None,
    workspace_root: Path,
) -> LoadedCanaryInput:
    if input_kind == "single":
        if query is None or source_path is not None:
            raise ValueError("single input requires query and forbids source_path")
        cases = load_single_case(query, query_id or "user-query-001")
        source_bytes = _canonical_case_bytes(cases)
    elif input_kind == "jsonl":
        if source_path is None or query is not None or query_id is not None:
            raise ValueError("JSONL input requires only source_path")
        source_bytes = source_path.read_bytes()
        cases = _load_jsonl_case_bytes(source_bytes)
    else:
        if source_path is None or query is not None or query_id is not None:
            raise ValueError("frozen input requires only source_path")
        loaded_sample = load_sample_binding(source_path)
        source_bytes = loaded_sample.binding_bytes
        cases = _loaded_frozen_cases(loaded_sample, workspace_root=workspace_root)
        frozen = loaded_sample.binding.frozen_inputs
        if frozen is None:
            raise ValueError("frozen sample lacks identifier map binding")
        identifier_map_path = (workspace_root / frozen.identifier_map.path).resolve()
    scored = cases[0].gold_paper_ids is not None
    identifier_bytes: bytes | None = None
    identifier_sha: str | None = None
    if scored:
        if identifier_map_path is None:
            raise ValueError("identifier map is required for scored input")
        identifier_bytes = identifier_map_path.read_bytes()
        identifier_sha = _sha256(identifier_bytes)
        if input_kind == "frozen":
            assert frozen is not None
            if identifier_sha != frozen.identifier_map.sha256:
                raise ValueError("frozen identifier map bytes do not match the binding")
            if source_path is None or source_path.read_bytes() != source_bytes:
                raise ValueError("frozen sample binding changed while loading")
    return LoadedCanaryInput(
        input_kind=input_kind,
        cases=cases,
        evaluation_status="available" if scored else "not_available",
        input_sha256=_sha256(source_bytes),
        identifier_map_bytes=identifier_bytes,
        identifier_map_sha256=identifier_sha,
    )


def _validated_cases(cases: list[RecallCase]) -> tuple[RecallCase, ...]:
    if not cases:
        raise ValueError("at least one query is required")
    query_ids = [case.query_id for case in cases]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("query IDs must be unique")
    gold_states = {case.gold_paper_ids is not None for case in cases}
    if len(gold_states) != 1:
        raise ValueError("all cases must consistently include Gold or omit Gold")
    return tuple(cases)


def _canonical_case_bytes(cases: tuple[RecallCase, ...]) -> bytes:
    return json.dumps(
        [case.model_dump(mode="json") for case in cases],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(content: bytes) -> str:
    return "sha256:" + hashlib.sha256(content).hexdigest()


__all__ = [
    "LoadedCanaryInput",
    "RecallCase",
    "load_canary_input",
    "load_frozen_cases",
    "load_jsonl_cases",
    "load_single_case",
]
