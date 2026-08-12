from __future__ import annotations

import pytest
from pydantic import ValidationError

from paper_search.live_identity import LiveDependencyEvidence, LiveProviderDescriptor


HASH_A = "sha256:" + "a" * 64
HASH_B = "sha256:" + "b" * 64


def descriptor(**changes: object) -> LiveProviderDescriptor:
    payload: dict[str, object] = {
        "identity_schema_version": "live-provider-descriptor-v1",
        "provider": "openalex",
        "dependency": "openalex",
        "adapter": "openalex-works-v1",
        "version": "live-capture-search-v1",
        "model": None,
        "endpoints": ("https://api.openalex.org/works",),
        "operations": ("search",),
    }
    payload.update(changes)
    return LiveProviderDescriptor.model_validate(payload)


def test_live_provider_descriptor_is_strict_frozen_and_canonical() -> None:
    value = descriptor()
    assert value.model_dump(mode="json") == {
        "identity_schema_version": "live-provider-descriptor-v1",
        "provider": "openalex",
        "dependency": "openalex",
        "adapter": "openalex-works-v1",
        "version": "live-capture-search-v1",
        "model": None,
        "endpoints": ["https://api.openalex.org/works"],
        "operations": ["search"],
    }
    with pytest.raises(ValidationError):
        value.provider = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    "changes",
    [
        {"endpoints": ()},
        {"operations": ()},
        {"endpoints": ("https://api.openalex.org/works",) * 2},
        {"operations": ("search", "search")},
        {"api_key": "secret"},
    ],
)
def test_live_provider_descriptor_rejects_incomplete_duplicate_or_extra_fields(
    changes: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        descriptor(**changes)


def test_live_dependency_evidence_binds_provider_pricing_and_controller() -> None:
    evidence = LiveDependencyEvidence(
        identity_schema_version="live-dependency-evidence-v1",
        provider=descriptor(),
        pricing_policy_sha256=HASH_A,
        controller_policy_sha256=HASH_B,
        formal_live=True,
    )
    serialized = evidence.model_dump_json()
    assert HASH_A in serialized and HASH_B in serialized
    assert "secret" not in serialized


def test_live_dependency_evidence_requires_formal_live() -> None:
    with pytest.raises(ValidationError, match="formal_live"):
        LiveDependencyEvidence(
            identity_schema_version="live-dependency-evidence-v1",
            provider=descriptor(),
            pricing_policy_sha256=HASH_A,
            controller_policy_sha256=HASH_B,
            formal_live=False,
        )
