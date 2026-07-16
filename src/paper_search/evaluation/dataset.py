from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import unicodedata
from collections.abc import Sequence
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, Field, JsonValue, ValidationError, field_validator

from paper_search.domain.models import DomainModel, NonEmptyStr


ModelT = TypeVar("ModelT", bound=BaseModel)


_DOI_PATTERN = re.compile(r"^10\.\d{4,9}/\S+$", re.IGNORECASE)
_DOI_PREFIXES = (
    "https://doi.org/",
    "http://doi.org/",
    "http://dx.doi.org/",
    "doi:",
)
_ARXIV_PATTERN = re.compile(
    r"^(?:\d{4}\.\d{4,5}|[a-z][a-z0-9.-]*/\d{7})$",
    re.IGNORECASE,
)
_ARXIV_PREFIXES = (
    "https://arxiv.org/abs/",
    "http://arxiv.org/abs/",
    "https://arxiv.org/pdf/",
    "http://arxiv.org/pdf/",
    "arxiv:",
)
_OPENALEX_PATTERN = re.compile(r"^W\d+$", re.IGNORECASE)
_OPENALEX_PREFIXES = (
    "https://openalex.org/",
    "http://openalex.org/",
    "openalex:",
)
_S2_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
_S2_URL_PATTERN = re.compile(
    r"^https?://(?:www\.)?semanticscholar\.org/paper/(?:[^/]+/)?([^/?#]+)",
    re.IGNORECASE,
)


def normalize_title(value: str) -> str:
    """Normalize a title for diagnostic fallback matching."""
    normalized = unicodedata.normalize("NFKC", value).casefold()
    without_punctuation = "".join(
        " " if unicodedata.category(character)[0] in {"P", "S"} else character
        for character in normalized
    )
    collapsed = " ".join(without_punctuation.split())
    if not collapsed:
        raise ValueError("title must contain letters or numbers")
    return collapsed


def normalize_paper_id(value: str, *, kind: str | None = None) -> str:
    """Return the canonical namespaced representation of a paper identifier."""
    candidate = value.strip()
    lowered = candidate.casefold()
    for prefix in _DOI_PREFIXES:
        if lowered.startswith(prefix):
            candidate = candidate[len(prefix) :]
            break

    if (kind is None or kind.casefold() == "doi") and _DOI_PATTERN.fullmatch(candidate):
        return f"doi:{candidate.casefold()}"

    candidate = value.strip()
    lowered = candidate.casefold()
    for prefix in _ARXIV_PREFIXES:
        if lowered.startswith(prefix):
            candidate = candidate[len(prefix) :]
            break
    if candidate.casefold().endswith(".pdf"):
        candidate = candidate[:-4]
    candidate = re.sub(r"v\d+$", "", candidate, flags=re.IGNORECASE)

    if (kind is None or kind.casefold() == "arxiv") and _ARXIV_PATTERN.fullmatch(candidate):
        return f"arxiv:{candidate.casefold()}"

    candidate = value.strip()
    lowered = candidate.casefold()
    for prefix in _OPENALEX_PREFIXES:
        if lowered.startswith(prefix):
            candidate = candidate[len(prefix) :]
            break
    if (kind is None or kind.casefold() == "openalex") and _OPENALEX_PATTERN.fullmatch(
        candidate
    ):
        return f"openalex:{candidate.upper()}"

    candidate = value.strip()
    semantic_url_match = _S2_URL_PATTERN.match(candidate)
    if semantic_url_match is not None:
        candidate = semantic_url_match.group(1)
    elif candidate.casefold().startswith("s2:"):
        candidate = candidate[3:]
    if (
        (semantic_url_match is not None or value.strip().casefold().startswith("s2:")
         or kind is not None and kind.casefold() in {"s2", "semantic_scholar"})
        and _S2_PATTERN.fullmatch(candidate)
    ):
        return f"s2:{candidate}"

    candidate = value.strip()
    is_explicit_title = candidate.casefold().startswith("title:")
    if is_explicit_title:
        candidate = candidate[6:]
    if is_explicit_title or kind is not None and kind.casefold() == "title":
        return f"title:{normalize_title(candidate)}"

    raise ValueError(f"unsupported or invalid paper identifier: {value!r}")


class EvaluationQuery(DomainModel):
    """One normalized evaluation query and its relevant paper identifiers."""

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
    """One ranked prediction list; duplicates are retained until scoring."""

    query_id: NonEmptyStr
    predicted_paper_ids: list[NonEmptyStr] = Field(default_factory=list)

    @field_validator("predicted_paper_ids")
    @classmethod
    def normalize_prediction_ids(cls, values: list[str]) -> list[str]:
        return [normalize_paper_id(value) for value in values]


def _format_validation_error(error: ValidationError) -> str:
    error_types = sorted(
        {
            str(detail["type"])
            for detail in error.errors(include_url=False, include_input=False)
        }
    )
    return f"record validation failed ({', '.join(error_types)})"


def read_jsonl(path: Path, model_type: type[ModelT]) -> list[ModelT]:
    """Load strict JSONL records and reject duplicate query identifiers."""
    records: list[ModelT] = []
    seen_query_ids: set[str] = set()

    with path.open("rb") as source:
        for line_number, raw_line in enumerate(source, start=1):
            try:
                line = raw_line.decode("utf-8")
            except UnicodeDecodeError:
                raise ValueError(
                    f"{path}:{line_number}: invalid UTF-8"
                ) from None
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank line is not allowed")
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON: {error.msg}"
                ) from None
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")

            try:
                record = model_type.model_validate(payload)
            except ValidationError as error:
                reason = _format_validation_error(error)
                raise ValueError(f"{path}:{line_number}: {reason}") from None

            query_id = getattr(record, "query_id", None)
            if isinstance(query_id, str):
                if query_id in seen_query_ids:
                    raise ValueError(
                        f"{path}:{line_number}: duplicate query_id: {query_id}"
                    )
                seen_query_ids.add(query_id)
            records.append(record)

    return records


def write_jsonl_atomic(path: Path, records: Sequence[BaseModel]) -> None:
    """Write deterministic JSONL through a flushed sibling temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            for record in records:
                payload = json.dumps(
                    record.model_dump(mode="json"),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                temporary.write(payload.encode("utf-8") + b"\n")
            temporary.flush()
            os.fsync(temporary.fileno())

        temporary_path.replace(path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def sha256_file(path: Path) -> str:
    """Return a namespaced SHA-256 digest of the exact file bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


class _JsonObjectPairs(list[tuple[str, object]]):
    """Distinguish JSON objects from arrays while preserving duplicate keys."""


class IdentifierMap:
    """Resolve normalized paper identifier aliases to their terminal identifier."""

    def __init__(self, resolved: dict[str, str]) -> None:
        self._resolved = resolved.copy()

    @classmethod
    def from_path(cls, path: Path) -> IdentifierMap:
        """Load, validate, and fully resolve an identifier mapping JSON object."""
        try:
            payload: object = json.loads(
                path.read_text(encoding="utf-8"),
                object_pairs_hook=_JsonObjectPairs,
            )
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}: invalid JSON: {error.msg}") from None

        if not isinstance(payload, _JsonObjectPairs):
            raise ValueError(f"{path}: expected a JSON object")

        direct: dict[str, str] = {}
        for raw_alias, raw_target in payload:
            if not isinstance(raw_target, str):
                raise ValueError(
                    f"{path}: identifier map keys and values must be strings"
                )
            try:
                alias = normalize_paper_id(raw_alias)
                target = normalize_paper_id(raw_target)
            except ValueError as error:
                raise ValueError(f"{path}: {error}") from None

            existing = direct.get(alias)
            if existing is not None and existing != target:
                raise ValueError(f"{path}: identifier map conflict for {alias}")
            direct[alias] = target

        resolved: dict[str, str] = {}

        def resolve_chain(identifier: str) -> str:
            chain: list[str] = []
            positions: dict[str, int] = {}
            current = identifier

            while current not in resolved:
                if current in positions:
                    raise ValueError(
                        f"{path}: identifier map cycle involving {current}"
                    )
                positions[current] = len(chain)
                chain.append(current)

                target = direct.get(current)
                if target is None:
                    resolved[current] = current
                    break
                current = target

            terminal = resolved[current]
            for member in reversed(chain):
                resolved[member] = terminal
            return terminal

        for alias in direct:
            resolve_chain(alias)

        return cls({alias: resolved[alias] for alias in direct})

    def resolve(self, value: str) -> str:
        """Normalize an identifier and return its terminal mapped value."""
        normalized = normalize_paper_id(value)
        return self._resolved.get(normalized, normalized)
