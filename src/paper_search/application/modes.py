"""Immutable replay and live mode bindings."""

from __future__ import annotations

from typing import Self

from pydantic import model_validator

from paper_search.domain.models import (
    DomainModel,
    NonEmptyStr,
    SearchMode,
    Sha256,
)


class ModeBinding(DomainModel):
    """One fail-closed execution-mode identity for an application lifetime."""

    mode: SearchMode
    network_authorized: bool
    snapshot_set_id: NonEmptyStr | None
    snapshot_manifest_sha256: Sha256 | None

    @model_validator(mode="after")
    def validate_mode_fields(self) -> Self:
        has_snapshot = (
            self.snapshot_set_id is not None
            and self.snapshot_manifest_sha256 is not None
        )
        has_partial_snapshot = (
            self.snapshot_set_id is None
        ) != (self.snapshot_manifest_sha256 is None)
        if has_partial_snapshot:
            raise ValueError("snapshot identity must be complete")
        if self.mode == "replay":
            if self.network_authorized:
                raise ValueError("replay mode cannot authorize network access")
            if not has_snapshot:
                raise ValueError("replay mode requires an immutable snapshot identity")
        elif not self.network_authorized:
            raise ValueError("live mode requires network authorization")
        return self
