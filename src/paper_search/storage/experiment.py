"""Small immutable experiment-record store for offline pipeline runs."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping


_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


class ExperimentRecordStore:
    """Persist canonical JSON records without silently overwriting a run."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def write(self, run_id: str, record: Mapping[str, Any]) -> Path:
        if not _RUN_ID.fullmatch(run_id):
            raise ValueError("run_id must be a single safe filename")
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{run_id}.json"
        content = json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"
        if path.exists():
            if path.read_bytes() == content:
                return path
            raise FileExistsError(f"refusing to overwrite experiment record: {path}")
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_bytes(content)
        temporary.replace(path)
        return path

    def read(self, run_id: str) -> dict[str, Any]:
        if not _RUN_ID.fullmatch(run_id):
            raise ValueError("run_id must be a single safe filename")
        try:
            payload = json.loads((self.root / f"{run_id}.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("experiment record is unavailable or invalid") from error
        if not isinstance(payload, dict):
            raise ValueError("experiment record must contain an object")
        return payload
