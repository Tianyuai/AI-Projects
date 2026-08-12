# Scheme B Live Provider Identity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Scheme B candidate-recall harness capable of constructing and validating a complete live runtime identity from the actual DeepSeek, OpenAlex, Semantic Scholar, pricing, and budget objects without issuing an online request.

**Architecture:** Add a provider-neutral immutable self-description contract, make the existing live capture providers/analyzer expose identity evidence derived from their own validated fields, and let Scheme B budget-owning adapters convert that evidence into strict `LiveDependencyIdentity` values. The live runtime factory accepts only the exact self-identifying adapters it will execute, checks shared object and policy identity, and produces the existing strict `LiveRuntimeIdentity`; the CLI still requires separate runtime `--allow-live` authorization.

**Tech Stack:** Python 3.11, Pydantic v2 strict models, `httpx` injected clients, existing `ActualCostPricer`, `HardBudgetController`, dependency capture adapters, pytest, Ruff, and mypy.

## Global Constraints

- The binding design is `docs/superpowers/specs/2026-08-12-recall-live-provider-identity-design.md`.
- This plan does not authorize or execute DeepSeek, OpenAlex, or Semantic Scholar requests.
- Do not read `.env`, API keys from the environment, `data/budget_ledger.sqlite3`, frozen query text, Gold records, or protected response bodies during implementation or verification.
- Use only injected fake `httpx` transports/clients and synthetic credentials in tests; every new test remains outside the `online` marker.
- Preserve these user-owned untracked paths exactly: `data/budget_ledger.sqlite3`, `deliverables/`, and `docs/evidence/identifier-map-semantic-audit-2026-08-10.json`.
- Never serialize credentials, authorization headers, e-mail addresses, query/Prompt content, response bodies, paper identifiers, request IDs, reservations, or private paths into runtime identity.
- Provider identity is derived by the actual execution object. Constructors and composition functions accept no caller-authored identity mapping or override.
- The actual OpenAlex, Semantic Scholar, and LLM live adapters must share one exact `ActualCostPricer` object and one exact `HardBudgetController` object.
- The controller must have `formal_live=True`; its policy fingerprint continues to bind the complete budget, formal-live mode, reservation TTL, and controller version.
- Unknown actual cost remains fail-closed and is never normalized to zero.
- Keep search, citation, and LLM backends independently replaceable. Do not add imports between retrieval handlers or add method IDs/action branches to the runner.
- Do not change Scheme B recipes, Prompts, sample selection, action limits, candidate-pool policy, evaluation thresholds, historical evidence, or Oracle catalog status.
- Every production change follows RED -> GREEN TDD. Tasks 1-5 are executed as one continuous infrastructure patch with focused test checkpoints; request one independent whole-feature review after the focused and aggregate gates pass.
- Use `D:\AI Projects\Projects\.venv\Scripts\python.exe` for all project tests and checks.

## Framework reuse invariant

This plan repairs one missing cross-cutting capability of the existing Scheme B framework: safe live-provider construction. It does not create a new search-method framework.

After this one-time repair, a normal candidate-recall experiment must use this fixed path:

```text
frozen sample binding
        +
method module/configuration
        |
existing generator registry
        |
existing text/title/citation handlers
        |
existing live-or-replay backends
        |
existing candidate pool
        |
existing candidate-recall evaluator
        |
existing artifacts and matched historical compare
```

For future methods:

- a Prompt/search-slot change modifies only Prompt/YAML;
- pre-generated DeepSeek search terms use the existing manual-action artifact;
- a new generation algorithm implements only `QueryGenerator` and registers it;
- a new retrieval mechanism implements only the relevant backend or handler and registers it;
- filtering/ranking enters only through the existing post-pool stage seam;
- frozen/local search changes only the input source or backend;
- no method may add a method-specific runner, candidate-pool builder, evaluator, artifact layout, comparison command, or standalone experiment script.

A new method does not require a new design/spec/implementation plan when it fits an existing interface. It requires only a small module/config change, focused interface tests, the three-query canary, and—if viable—the existing repeat/comparison workflow. A new design is required only if the method changes an interface or evidence contract shared by multiple modules.

## Execution shape

Tasks 1-5 are implementation checkpoints inside one bounded infrastructure change, not five separately designed frameworks. Execute them continuously in this order and stop only for a failing contract or architectural contradiction:

1. define the shared identity value types;
2. expose identity from the already-existing retrieval and LLM live objects;
3. bind the already-existing Scheme B budgeted adapters to those objects;
4. reopen the already-existing live composition path under the existing CLI dual gate;
5. run focused and aggregate offline checks;
6. request one independent review and then one fresh final gate.

Do not write another design or implementation plan between these checkpoints.

## File and interface map

- Create `src/paper_search/live_identity.py`: provider-neutral strict descriptor/evidence models and read-only protocols. This module imports domain/control types but never imports Scheme B.
- Modify `src/paper_search/retrieval/snapshot_adapters.py`: expose OpenAlex/Semantic Scholar live evidence from the actual priced capture provider.
- Modify `src/paper_search/llm/client.py`: expose the OpenAI-compatible client's provider descriptor without credentials.
- Modify `src/paper_search/llm/snapshot_adapters.py`: expose DeepSeek live evidence from the actual analyzer, pricer, and controller.
- Modify `src/paper_search/recall_experiments/retrieval/backends.py`: convert retrieval evidence into strict Scheme B dependency identities and expose exact object references for composition checks.
- Modify `src/paper_search/recall_experiments/generation/backends.py`: do the same for the LLM backend.
- Modify `src/paper_search/recall_experiments/identity.py`: centralize conversion from generic evidence to the existing `LiveDependencyIdentity`; retain strict execution-identity validation.
- Modify `src/paper_search/control/pricing.py`: expose immutable canonical policy bytes from the actual pricer.
- Modify `src/paper_search/recall_experiments/composition.py`: restore a safe `build_live_runtime` and `_valid_live_runtime_identity` using the actual adapters that will execute.
- Modify focused unit/recall tests named in each task. Do not create a standalone experiment script.

---

### Task 1: Define provider-neutral immutable live identity evidence

**Files:**
- Create: `src/paper_search/live_identity.py`
- Test: `tests/unit/test_live_identity.py`

**Interfaces:**
- Produces: `LiveProviderDescriptor`, `LiveDependencyEvidence`, `LiveIdentityController`, and `SelfIdentifyingLiveDependency`.
- Consumed by: retrieval and LLM live capture adapters in Tasks 2-3; Scheme B adapters in Task 4.

- [ ] **Step 1: Write strict-model RED tests**

Create `tests/unit/test_live_identity.py` with tests that establish exact schema, immutability, forbidden extras, normalized tuples, strict dependency roles, and privacy-safe serialization:

```python
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
```

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest tests/unit/test_live_identity.py -q
```

Expected: collection fails because `paper_search.live_identity` does not exist.

- [ ] **Step 3: Implement the strict generic identity contract**

Create `src/paper_search/live_identity.py` with this public surface:

```python
from __future__ import annotations

from typing import Literal, Protocol

from pydantic import ConfigDict, model_validator

from paper_search.domain.models import DomainModel, NonEmptyStr, Sha256


class _FrozenIdentityModel(DomainModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class LiveProviderDescriptor(_FrozenIdentityModel):
    identity_schema_version: Literal["live-provider-descriptor-v1"]
    provider: NonEmptyStr
    dependency: Literal["openalex", "semantic_scholar", "llm"]
    adapter: NonEmptyStr
    version: NonEmptyStr
    model: NonEmptyStr | None
    endpoints: tuple[NonEmptyStr, ...]
    operations: tuple[NonEmptyStr, ...]

    @model_validator(mode="after")
    def validate_surface(self) -> "LiveProviderDescriptor":
        if not self.endpoints or len(set(self.endpoints)) != len(self.endpoints):
            raise ValueError("live provider endpoints must be nonempty and unique")
        if not self.operations or len(set(self.operations)) != len(self.operations):
            raise ValueError("live provider operations must be nonempty and unique")
        return self


class LiveDependencyEvidence(_FrozenIdentityModel):
    identity_schema_version: Literal["live-dependency-evidence-v1"]
    provider: LiveProviderDescriptor
    pricing_policy_sha256: Sha256
    controller_policy_sha256: Sha256
    formal_live: Literal[True]


class LiveIdentityController(Protocol):
    @property
    def policy_fingerprint(self) -> str: ...

    @property
    def formal_live(self) -> bool: ...


class SelfIdentifyingLiveDependency(Protocol):
    @property
    def live_identity_evidence(self) -> LiveDependencyEvidence: ...
```

Export only these four names. Do not import Scheme B or credentials/config modules.

- [ ] **Step 4: Run Task 1 GREEN and static checks**

Run:

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest tests/unit/test_live_identity.py -q
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m ruff check src/paper_search/live_identity.py tests/unit/test_live_identity.py
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m mypy src/paper_search/live_identity.py
```

Expected: all commands exit `0`.

- [ ] **Step 5: Commit Task 1**

```powershell
git add -- src/paper_search/live_identity.py tests/unit/test_live_identity.py
git diff --cached --check
git commit -m "feat: define live provider identity evidence"
```

---

### Task 2: Make the priced OpenAlex and Semantic Scholar capture provider self-identifying

**Files:**
- Modify: `src/paper_search/retrieval/snapshot_adapters.py`
- Modify: `tests/unit/test_retrieval_snapshot_adapters.py`
- Test: `tests/unit/test_live_identity.py`

**Interfaces:**
- Consumes: `LiveProviderDescriptor`, `LiveDependencyEvidence`, `LiveIdentityController` from Task 1; `ActualCostPricer.policy_sha256` and controller `policy_fingerprint`.
- Produces: read-only `LiveCaptureSearchProvider.live_identity_evidence`, `live_pricer`, and `live_controller` properties.

- [ ] **Step 1: Add RED tests for deterministic dependency-specific evidence**

Add focused tests using the existing in-memory capture store/cache and `httpx.AsyncClient` fixtures. Construct providers but do not call their methods:

```python
def test_openalex_live_capture_identity_is_derived_and_credential_free(
    live_capture_store: DependencyCaptureStore,
    pricer: ActualCostPricer,
    formal_controller: HardBudgetController,
) -> None:
    first = LiveCaptureSearchProvider(
        dependency="openalex",
        client=httpx.AsyncClient(transport=httpx.MockTransport(no_request)),
        capture_store=live_capture_store,
        pricer=pricer,
        controller=formal_controller,
        api_key="secret-one",
        mailto="private@example.invalid",
    )
    second = LiveCaptureSearchProvider(
        dependency="openalex",
        client=httpx.AsyncClient(transport=httpx.MockTransport(no_request)),
        capture_store=live_capture_store,
        pricer=pricer,
        controller=formal_controller,
        api_key="secret-two",
        mailto="other@example.invalid",
    )
    assert first.live_identity_evidence == second.live_identity_evidence
    payload = first.live_identity_evidence.model_dump_json()
    assert "secret" not in payload and "example.invalid" not in payload
    assert first.live_pricer is pricer
    assert first.live_controller is formal_controller


def test_search_provider_descriptor_changes_with_dependency_or_adapter_version(...) -> None:
    openalex = make_live_provider(dependency="openalex", adapter_version="openalex-works-v1")
    semantic = make_live_provider(
        dependency="semantic_scholar", adapter_version="semantic-graph-v1"
    )
    changed = make_live_provider(dependency="openalex", adapter_version="openalex-works-v2")
    assert openalex.live_identity_evidence.provider != semantic.live_identity_evidence.provider
    assert openalex.live_identity_evidence.provider != changed.live_identity_evidence.provider
```

Use a `no_request(request)` callback that raises `AssertionError`; assert its call count remains zero.

- [ ] **Step 2: Run retrieval identity tests and verify RED**

Run the two new node IDs. Expected: `AttributeError` for `live_identity_evidence`.

- [ ] **Step 3: Add controller identity to the settlement protocol**

Extend `ProviderSettlementController` with read-only `policy_fingerprint` and `formal_live`. Do not accept an independent fingerprint argument. Existing fake controllers must implement the properties or be replaced by the existing `HardBudgetController` in tests that construct a live capture provider.

- [ ] **Step 4: Derive immutable provider evidence in `LiveCaptureSearchProvider`**

Add a private dependency descriptor table with canonical, identifier-free endpoint templates:

```python
_LIVE_DESCRIPTOR_SURFACES = {
    "openalex": {
        "provider": "openalex",
        "version": "live-capture-search-v1",
        "model": None,
        "endpoints": ("https://api.openalex.org/works",),
        "operations": ("search",),
    },
    "semantic_scholar": {
        "provider": "semantic_scholar",
        "version": "live-capture-search-v1",
        "model": None,
        "endpoints": (
            "https://api.semanticscholar.org/graph/v1/paper/search",
            "https://api.semanticscholar.org/graph/v1/paper/batch",
            "https://api.semanticscholar.org/graph/v1/paper/{paper_id}/references",
            "https://api.semanticscholar.org/graph/v1/paper/{paper_id}/citations",
        ),
        "operations": ("search", "batch", "references", "citations"),
    },
}
```

At construction, validate `controller.formal_live is True` before accepting a live capture provider. Build one frozen `LiveDependencyEvidence` and store it. Expose:

```python
@property
def live_identity_evidence(self) -> LiveDependencyEvidence: ...

@property
def live_pricer(self) -> ActualCostPricer: ...

@property
def live_controller(self) -> ProviderSettlementController: ...
```

The properties return the actual objects used by `_LiveOperation`; no copies and no caller identity values.

- [ ] **Step 5: Add privacy and policy-drift tests**

Test that changing the pricing policy or controller budget/TTL changes evidence; `formal_live=False` construction fails before the fake transport; serialized evidence contains none of the synthetic credential, mail, capture root, or request content sentinels.

- [ ] **Step 6: Run Task 2 GREEN**

Run:

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest tests/unit/test_retrieval_snapshot_adapters.py tests/unit/test_live_identity.py -q
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m ruff check src/paper_search/retrieval/snapshot_adapters.py tests/unit/test_retrieval_snapshot_adapters.py tests/unit/test_live_identity.py
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m mypy src/paper_search/retrieval/snapshot_adapters.py src/paper_search/live_identity.py
```

Expected: all exit `0`; no socket/request callback is reached.

- [ ] **Step 7: Commit Task 2**

```powershell
git add -- src/paper_search/retrieval/snapshot_adapters.py tests/unit/test_retrieval_snapshot_adapters.py tests/unit/test_live_identity.py
git diff --cached --check
git commit -m "feat: attest live scholarly providers"
```

---

### Task 3: Make the DeepSeek-compatible analyzer self-identifying

**Files:**
- Modify: `src/paper_search/llm/client.py`
- Modify: `src/paper_search/llm/snapshot_adapters.py`
- Modify: `tests/unit/test_llm_client.py`
- Modify: `tests/unit/test_llm_snapshot_adapters.py`

**Interfaces:**
- Consumes: Task 1 identity models, actual `OpenAICompatibleLLMClient`, `ActualCostPricer`, and controller.
- Produces: `OpenAICompatibleLLMClient.live_provider_descriptor`; `LiveCaptureLLMAnalyzer.live_identity_evidence`, `live_pricer`, and `live_controller`.

- [ ] **Step 1: Write RED tests for client-owned descriptor**

Add tests that construct clients with `httpx.MockTransport` and synthetic keys:

```python
def test_deepseek_client_descriptor_binds_endpoint_model_and_adapter_not_key() -> None:
    first = make_client(base_url="https://api.deepseek.com", model="deepseek-chat", key="one")
    second = make_client(base_url="https://api.deepseek.com/", model="deepseek-chat", key="two")
    assert first.live_provider_descriptor == second.live_provider_descriptor
    assert first.live_provider_descriptor.model == "deepseek-chat"
    assert first.live_provider_descriptor.endpoints == (
        "https://api.deepseek.com/chat/completions",
    )
    assert first.live_provider_descriptor.operations == ("generate_json",)
    assert "one" not in first.live_provider_descriptor.model_dump_json()


def test_llm_descriptor_changes_with_endpoint_or_model() -> None:
    baseline = make_client("https://api.deepseek.com", "deepseek-chat", "key")
    assert baseline.live_provider_descriptor != make_client(
        "https://api.deepseek.com", "deepseek-reasoner", "key"
    ).live_provider_descriptor
    assert baseline.live_provider_descriptor != make_client(
        "https://dashscope.aliyuncs.com/compatible-mode/v1", "deepseek-v3", "key"
    ).live_provider_descriptor
```

Expected RED: property does not exist.

- [ ] **Step 2: Implement the client descriptor**

Keep the normalized full transport endpoint in an immutable string already derived during construction. Add a read-only descriptor property using:

- provider `deepseek` for `api.deepseek.com`;
- provider `dashscope` for the existing DashScope-compatible host;
- provider `openai_compatible` for other validated HTTPS hosts;
- dependency `llm`;
- adapter `openai-compatible-json`;
- version `openai-compatible-client-v1`;
- actual model;
- exact normalized `/chat/completions` transport endpoint;
- operation `generate_json`.

Do not expose the API key or prompt content.

- [ ] **Step 3: Write analyzer evidence RED tests**

Construct `LiveCaptureLLMAnalyzer` using an in-memory capture store, shared pricer, formal controller, and DeepSeek client; assert the evidence uses the client descriptor and actual policy hashes, and that `live_pricer`/`live_controller` are the exact objects. Add `formal_live=False` and fake-controller tests that fail during construction without HTTP dispatch.

- [ ] **Step 4: Implement analyzer evidence**

Extend `RequestSettlementController` with `policy_fingerprint` and `formal_live`. Make `HardBudgetSettlementAdapter` forward both properties so existing application composition remains valid. In `LiveCaptureLLMAnalyzer.__init__`, reject non-formal controllers, construct one frozen `LiveDependencyEvidence`, and expose the three read-only properties.

- [ ] **Step 5: Prove Prompt/credential privacy and deterministic drift behavior**

Tests must serialize the descriptor/evidence and scan for the synthetic API key, prompt instructions, prompt SHA source bytes, capture path, and request payload. Changing model, endpoint, pricing policy, budget, or TTL must change the appropriate descriptor/evidence.

- [ ] **Step 6: Run Task 3 GREEN**

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest tests/unit/test_llm_client.py tests/unit/test_llm_snapshot_adapters.py -q
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m ruff check src/paper_search/llm/client.py src/paper_search/llm/snapshot_adapters.py tests/unit/test_llm_client.py tests/unit/test_llm_snapshot_adapters.py
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m mypy src/paper_search/llm/client.py src/paper_search/llm/snapshot_adapters.py
```

Expected: all exit `0`; no mock transport request occurs in identity tests.

- [ ] **Step 7: Commit Task 3**

```powershell
git add -- src/paper_search/llm/client.py src/paper_search/llm/snapshot_adapters.py tests/unit/test_llm_client.py tests/unit/test_llm_snapshot_adapters.py
git diff --cached --check
git commit -m "feat: attest live deepseek analyzer"
```

---

### Task 4: Bind Scheme B budgeted adapters to actual live evidence

**Files:**
- Modify: `src/paper_search/recall_experiments/identity.py`
- Modify: `src/paper_search/recall_experiments/retrieval/backends.py`
- Modify: `src/paper_search/recall_experiments/generation/backends.py`
- Modify: `tests/recall_experiments/test_retrieval_registry.py`
- Modify: `tests/recall_experiments/test_llm_backend.py`
- Modify: `tests/recall_experiments/test_cli.py`

**Interfaces:**
- Consumes: generic `LiveDependencyEvidence` from Tasks 2-3 and the existing strict Scheme B identity models.
- Produces: `dependency_identity`, `live_pricer`, and `live_controller` on `BudgetedSearchBackend`, `BudgetedCitationBackend`, and `BudgetedLLMBackend` when wrapping actual live capture objects.

- [ ] **Step 1: Add a strict conversion function and RED tests**

In `identity.py`, plan the exact function:

```python
def dependency_identity_from_evidence(
    evidence: LiveDependencyEvidence,
) -> LiveDependencyIdentity:
    return LiveDependencyIdentity(
        identity_schema_version="live-dependency-runtime-identity-v1",
        provider=evidence.provider.provider,
        dependency=evidence.provider.dependency,
        adapter=evidence.provider.adapter,
        model=evidence.provider.model,
        version=evidence.provider.version,
        endpoints=evidence.provider.endpoints,
        operations=evidence.provider.operations,
        pricing_policy_sha256=evidence.pricing_policy_sha256,
        controller_policy_sha256=evidence.controller_policy_sha256,
    )
```

Tests assert exact canonical output, extra/malformed evidence rejection occurs in the generic model, and no protected field can appear.

- [ ] **Step 2: Write RED tests for retrieval adapter binding**

Construct real `LiveCaptureSearchProvider` objects with fake transports, then wrap them in `BudgetedSearchBackend`/`BudgetedCitationBackend`. Assert:

- dependency identity comes from provider evidence;
- adapter controller is the exact provider controller;
- adapter rejects a different outer controller even if its remaining budget happens to match;
- OpenAlex cannot occupy the citation role;
- Semantic Scholar citation identity includes references/citations operations;
- legacy/replay fake providers remain usable for offline execution but expose no live identity and cannot enter live composition.

Expected RED: no dependency identity property and no constructor cross-check.

- [ ] **Step 3: Implement retrieval adapter binding**

In `_BudgetedProviderCall`, preserve the existing controller. In each concrete constructor:

1. detect the generic self-identifying protocol without accepting a separate identity argument;
2. when live evidence exists, require `provider.live_controller is controller` and `formal_live=True`;
3. retain `provider.live_pricer` and the validated converted identity;
4. validate role-specific dependency (`openalex` search, `semantic_scholar` citation);
5. expose read-only properties that raise a fixed `ValueError("live identity unavailable")` for non-live/replay providers.

Do not change search/citation execution or settlement behavior.

- [ ] **Step 4: Write RED tests for LLM backend binding**

Wrap a real `LiveCaptureLLMAnalyzer` in `BudgetedLLMBackend`. Assert exact analyzer/controller/pricer object binding, dependency `llm`, DeepSeek provider/model propagation, and rejection of a mismatched outer controller. Remove the old caller `pricing_policy_sha256` argument from live construction tests; callers must not declare it.

- [ ] **Step 5: Implement LLM backend binding**

Keep replay analyzers supported. For a self-identifying live analyzer, derive identity and actual object references exactly as retrieval adapters do. Delete the caller-authored `pricing_policy_sha256` identity field/parameter after updating all call sites; replay estimate behavior remains unchanged.

- [ ] **Step 6: Run Task 4 GREEN**

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest tests/recall_experiments/test_retrieval_registry.py tests/recall_experiments/test_llm_backend.py tests/recall_experiments/test_cli.py -q
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m ruff check src/paper_search/recall_experiments/identity.py src/paper_search/recall_experiments/retrieval/backends.py src/paper_search/recall_experiments/generation/backends.py tests/recall_experiments/test_retrieval_registry.py tests/recall_experiments/test_llm_backend.py tests/recall_experiments/test_cli.py
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m mypy src/paper_search/recall_experiments/identity.py src/paper_search/recall_experiments/retrieval/backends.py src/paper_search/recall_experiments/generation/backends.py
```

Expected: all exit `0`.

- [ ] **Step 7: Commit Task 4**

```powershell
git add -- src/paper_search/recall_experiments/identity.py src/paper_search/recall_experiments/retrieval/backends.py src/paper_search/recall_experiments/generation/backends.py tests/recall_experiments/test_retrieval_registry.py tests/recall_experiments/test_llm_backend.py tests/recall_experiments/test_cli.py
git diff --cached --check
git commit -m "feat: bind recall adapters to live evidence"
```

---

### Task 5: Restore fail-closed live runtime composition and CLI dual authorization

**Files:**
- Modify: `src/paper_search/recall_experiments/composition.py`
- Modify: `src/paper_search/recall_experiments/identity.py`
- Modify: `tests/recall_experiments/test_cli.py`
- Modify: `tests/recall_experiments/test_runner.py`

**Interfaces:**
- Consumes: exact self-identifying budgeted adapters from Task 4.
- Produces: safe `build_live_runtime(*, search_backend, citation_backend, llm_backend) -> RecallRuntime`; working `_valid_live_runtime_identity`.

- [ ] **Step 1: Write valid-runtime RED test**

Build all three actual live capture objects using one fake `httpx.AsyncClient`, one in-memory capture store, one composite synthetic pricing policy, and one `HardBudgetController(formal_live=True)`. Wrap them in Task 4 adapters, then call:

```python
runtime = build_live_runtime(
    search_backend=search_backend,
    citation_backend=citation_backend,
    llm_backend=llm_backend,
)
identity = LiveRuntimeIdentity.model_validate(runtime.identity)
assert set(identity.dependencies) == {"search", "citation", "llm"}
assert runtime.search_backend is search_backend
assert runtime.citation_backend is citation_backend
assert runtime.llm_backend is llm_backend
assert runtime.controller is controller
assert runtime.pricing_policy_bytes == canonical_pricing_policy_bytes(policy)
```

Expected RED: current `build_live_runtime` always raises `live_runtime_unavailable`.

- [ ] **Step 2: Write fail-before-dispatch RED matrix**

Parameterize these cases, with fake provider methods that increment a counter and fail the test if reached:

- search/citation/LLM adapter missing live evidence;
- wrong dependency role;
- distinct pricer object, even with identical policy bytes;
- distinct controller object, even with identical policy fingerprint;
- different pricing policy;
- different budget, TTL, or formal-live mode;
- caller tries to pass an `identity`/`backend_identity` keyword;
- injected `RecallRuntime` contains a valid-looking identity mapping but adapters do not own it;
- incomplete or modified `LiveRuntimeIdentity`.

Expected RED: current API cannot distinguish the valid path and `_valid_live_runtime_identity` always returns `False`.

- [ ] **Step 3: Implement the safe runtime factory**

Change the public signature to accept only the three concrete budgeted adapters. Validate:

1. each exposes a strict dependency identity;
2. search dependency is OpenAlex, citation is Semantic Scholar, LLM dependency is `llm` with provider `deepseek` for this Scheme B canary contract;
3. all three expose the same exact controller object using `is`;
4. all three expose the same exact pricer object using `is`;
5. controller is `formal_live=True`;
6. evidence hashes equal the actual pricer/controller properties;
7. the strict `LiveRuntimeIdentity` validates.

Add this exact read-only property to `ActualCostPricer`:

```python
@property
def canonical_policy_bytes(self) -> bytes:
    return canonical_pricing_policy_bytes(self._policy)
```

Return `RecallRuntime` containing the same adapter objects, strict canonical runtime identity, actual controller, and `pricer.canonical_policy_bytes`. Do not expose the internal `PricingPolicy` object or accept pricing bytes separately in `build_live_runtime`.

Classify absent self-identification as `RecallTerminalError("live_runtime_unavailable")` and present-but-mismatched evidence as `RecallTerminalError("config_mismatch")`.

- [ ] **Step 4: Implement `_valid_live_runtime_identity` as object-and-model verification**

It must validate the strict runtime identity and verify it matches identities freshly read from the three exact contained adapters plus actual pricer/controller properties. A copied mapping alone is insufficient.

- [ ] **Step 5: Restore CLI dual-gate tests**

Tests must prove:

- a recipe cannot activate live without `--allow-live`;
- `--allow-live` without a runtime factory returns `live_runtime_unavailable`;
- a malformed/unverifiable injected runtime returns `config_mismatch` before provider methods;
- a valid injected factory plus `--allow-live` reaches fake DeepSeek/OpenAlex methods and writes an identity-bound report;
- the same valid factory without `--allow-live` reaches no provider method;
- provider authentication/rate-limit/network errors after valid composition are retained as `infrastructure_failure`, not misreported as identity failure.

Use only `httpx.MockTransport`; all fake responses are local bytes and no socket is opened.

- [ ] **Step 6: Verify snapshot and historical paths did not change**

Run focused replay, strict identity, no-historical, attempt-state, and historical comparison tests. Expected: exact snapshot requests still reject novel actions; legacy/v1 compare rules remain fail-closed; historical conclusions remain unchanged.

- [ ] **Step 7: Run Task 5 GREEN**

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest tests/recall_experiments/test_cli.py tests/recall_experiments/test_runner.py tests/recall_experiments/test_historical_replay.py tests/recall_experiments/test_generation.py -q
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest tests/recall_experiments -q
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m ruff check src/paper_search/recall_experiments src/paper_search/live_identity.py tests/recall_experiments
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m mypy src/paper_search/recall_experiments src/paper_search/live_identity.py
git diff --check
```

Expected: all exit `0`; no online test runs.

- [ ] **Step 8: Commit Task 5**

```powershell
git add -- src/paper_search/control/pricing.py src/paper_search/recall_experiments/composition.py src/paper_search/recall_experiments/identity.py tests/recall_experiments/test_cli.py tests/recall_experiments/test_runner.py
git diff --cached --check
git commit -m "feat: compose verified recall live runtime"
```

---

### Task 6: Run offline acceptance, privacy checks, and update the live-canary handoff

**Files:**
- Modify after all gates pass: `HANDOFF.md`
- Modify after all gates pass: `docs/retrieval-roadmap.md`
- Test: all modified unit/recall tests and the complete offline repository suite

**Interfaces:**
- Consumes: Tasks 1-5.
- Produces: an evidence-backed statement that the framework is ready to request separate Task 13 canary authorization, not a performance result.

- [ ] **Step 1: Run the focused identity and privacy gate**

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest `
  tests/unit/test_live_identity.py `
  tests/unit/test_retrieval_snapshot_adapters.py `
  tests/unit/test_llm_client.py `
  tests/unit/test_llm_snapshot_adapters.py `
  tests/recall_experiments/test_retrieval_registry.py `
  tests/recall_experiments/test_llm_backend.py `
  tests/recall_experiments/test_cli.py `
  tests/recall_experiments/test_runner.py -q
```

Expected: exit `0`, no network markers, no skipped identity tests.

- [ ] **Step 2: Run aggregate offline verification**

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest tests/recall_experiments -q
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest -m "not online" -q
$trackedPython = @(git ls-files '*.py')
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m ruff check -- $trackedPython
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m mypy src scripts
git diff --check
```

Expected: every command exits `0`. Record exact pass/skip/deselect counts rather than copying the earlier `2496/35/1` baseline.

- [ ] **Step 3: Run mechanical architecture and privacy checks**

Confirm with AST/import tests and repository search:

- `live_identity.py` imports no Scheme B, application composition, credential, Gold, or frozen-input module;
- provider/analyzer identity properties reference no API key, mail, query, Prompt, payload, response, paper ID, capture root, request ID, or reservation field;
- Scheme B handlers still do not import one another;
- runner still contains no method ID, provider constructor, action-type branch, metric formula, or credential/config reader;
- no new standalone experiment script exists;
- all valid-live tests use `httpx.MockTransport` and no `online` marker;
- the three user-owned untracked paths retain their initial status and exact content fingerprints.

- [ ] **Step 4: Update handoff documents only after verification**

Add a dated section that states:

- self-attested DeepSeek/OpenAlex/Semantic Scholar live runtime construction is implemented;
- identities bind actual adapter endpoint/model/version, composite pricing policy, controller budget/formal-live/TTL, and exact execution objects;
- offline fake-live composition and dual authorization passed;
- no API request, `.env` read, ledger write, Oracle generation, or Scheme B performance experiment occurred;
- Task 13 still requires explicit providers, total CNY/token/call budget, ledger path, output root, and authorization;
- first canary remains Query Evolution method, three frozen queries, one capture followed by exact offline replay.

- [ ] **Step 5: Commit verified handoff state**

```powershell
git add -- HANDOFF.md docs/retrieval-roadmap.md
git diff --cached --check
git commit -m "docs: prepare verified scheme b live canary"
```

- [ ] **Step 6: Request whole-feature independent review**

Review the implementation range from the Task 1 base through Task 6 head against the design. The reviewer must explicitly inspect credential privacy, provider-owned identity, exact object binding, pricing/controller equality, pre-dispatch failure, snapshot non-regression, and absence of live calls. Fix every Critical/Important finding with RED -> GREEN tests, then re-review.

- [ ] **Step 7: Run the final fresh gate after review fixes**

Repeat the full offline pytest, tracked Ruff, `mypy src scripts`, and `git diff --check`. Do not rely on Task 6 Step 2 output if review fixes changed any tracked file.

---

## Completion boundary

This implementation plan ends when the self-identifying live runtime is fully implemented, independently approved, and all offline gates pass. It does not execute Task 13.

The subsequent separately authorized canary must declare all of the following before any command is run:

- DeepSeek, OpenAlex, and Semantic Scholar network authorization;
- exact model and approved composite pricing-policy SHA-256;
- total CNY, LLM-call/token, search-call, elapsed-time, and attempt limits;
- ledger path and output root;
- the frozen `dev-smoke-3.yaml` identity;
- Query Evolution method-matched recipe/prompt/action limits;
- authorization to seal provider responses and replay them offline.

Without that explicit authorization, the completed code remains offline-ready only.

## Fixed method-test workflow after this plan

The first use of the repaired framework is not another implementation project. It is the existing Task 13 canary executed through configuration and frozen artifacts:

1. Select the already frozen `dev-smoke-3.yaml`; do not select queries after seeing results.
2. Select one inventory-supported method. Start with Query Evolution because it has the strongest exact action/provider-response evidence.
3. Bind its existing generation Prompt/method settings in the recipe. Do not edit the runner or create a Query Evolution script.
4. Before retrieval, generate search actions through the existing DeepSeek generator or prepare the same output through the existing manual-action path.
5. Validate and freeze that action artifact before inspecting candidate-recall results.
6. Under separate authorization, run one live capture through the common search/title/citation registry and common candidate-pool builder.
7. Seal provider request/response and usage evidence, then replay the exact capture offline through the same runtime interface.
8. Report the three-query candidate recall and infrastructure state. Treat this as an end-to-end viability check, not a general performance conclusion.
9. If the canary is viable, obtain three valid repeats within five attempts using the same recipe and existing runner. Do not tune search terms after seeing results.
10. Compare only to the matched historical method when the required historical identity, denominator, and per-query Gold-hit evidence are present; otherwise retain `not_comparable` or `insufficient_historical_evidence`.
11. To try the next method, replace only its Prompt/YAML/manual actions or registered generator/backend module and repeat Steps 1-10. Reuse every other component and command.

The expected engineering delta for a Prompt/slot-only method after this plan is zero Python production files and no new script. The expected delta for a genuinely new method family is one focused module plus its registry entry and interface tests.
