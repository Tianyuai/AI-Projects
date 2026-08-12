# Scheme B Live Provider Identity Design

**Date:** 2026-08-12
**Status:** Approved in conversation; pending written-spec review
**Scope:** Make the modular candidate-recall harness capable of constructing a verifiable live runtime. This design does not authorize or execute DeepSeek, OpenAlex, or Semantic Scholar calls.

## 1. Goal

Replace the current unconditional `live_runtime_unavailable` boundary with a complete, fail-closed identity layer for the actual DeepSeek, OpenAlex, Semantic Scholar, pricing, and budget objects used by one Scheme B run.

The resulting framework may enter a separately authorized three-query live canary only when every live dependency proves its execution identity from its own immutable runtime configuration. Caller-written identity dictionaries are never accepted.

## 2. Non-goals

- Do not execute the Task 13 live canary.
- Do not read `.env`, API keys, the budget ledger, or protected query/Gold content during implementation or offline verification.
- Do not change Scheme B prompts, search methods, action limits, candidate-pool policy, evaluation thresholds, or historical conclusions.
- Do not complete the Oracle Gold-document catalog.
- Do not build a monolithic provider factory or merge the independent text, title, citation, generation, candidate-pool, or evaluator modules.
- Do not claim candidate-recall performance, historical compatibility, filtering quality, ranking quality, or competition generalization.

## 3. Chosen architecture

Use execution-object self-attestation.

Each live provider/analyzer exposes a deterministic, read-only description of its own provider-specific runtime surface. Each budget-owning adapter combines that provider description with the actual `ActualCostPricer.policy_sha256` and actual `HardBudgetController.policy_fingerprint` that it uses. The Scheme B live runtime factory only combines and cross-validates those adapter identities.

The caller cannot pass, override, or patch a backend identity. The factory does not inspect private attributes and does not reconstruct identity from parallel configuration arguments.

```text
OpenAlex provider --------> provider descriptor --\
Semantic Scholar provider -> provider descriptor --- budgeted adapters
DeepSeek analyzer ---------> provider descriptor --/       |
ActualCostPricer -------------------------------------------|
HardBudgetController ---------------------------------------|
                                                            v
                                             LiveDependencyIdentity x 3
                                                            |
                                             cross-policy validation
                                                            v
                                                 LiveRuntimeIdentity
                                                            |
                                                 Scheme B run context
```

### Rejected alternatives

1. **Caller-populated wrapper identity:** concentrated changes, but could not prove that the wrapped object matched caller declarations.
2. **One exclusive construction factory:** could be safe, but would recouple configuration, credentials, providers, budgets, and experiment orchestration into a monolith and weaken independent replacement/testing.

## 4. Provider self-description

Add a small, immutable provider descriptor contract distinct from the existing final `LiveDependencyIdentity`. A provider descriptor contains only facts owned by the provider/analyzer itself:

- identity schema version;
- provider name;
- dependency kind: `openalex`, `semantic_scholar`, or `llm`;
- adapter implementation name;
- adapter contract version;
- model identifier for the LLM, otherwise `null`;
- normalized endpoint origins/paths used by the object;
- supported operations used by Scheme B.

Required operations are:

- OpenAlex search adapter: paper search;
- Semantic Scholar adapter: paper search, references, and citations;
- DeepSeek analyzer: structured JSON generation.

The descriptor must be derived from constructor-validated immutable fields. It must not contain:

- API keys, authorization headers, e-mail addresses, or other credentials;
- query text, Prompt content, request payloads, response data, paper identifiers, or Gold data;
- mutable retry counters, cooldown timestamps, usage totals, request IDs, or reservation IDs.

Changing a credential alone must not change the descriptor. Changing an endpoint, model, adapter implementation, contract version, or supported operation must change its canonical serialized identity.

## 5. Adapter-owned dependency identity

The budget-owning live adapters expose the existing strict `LiveDependencyIdentity`. They construct it from:

1. the wrapped object's provider descriptor;
2. the exact `ActualCostPricer` instance used for settlement;
3. the exact `HardBudgetController` instance used for reservations and settlement.

The final dependency identity fields remain:

- `identity_schema_version`;
- `provider`;
- `dependency`;
- `adapter`;
- `model`;
- `version`;
- `endpoints`;
- `operations`;
- `pricing_policy_sha256`;
- `controller_policy_sha256`.

The identity property is read-only. The adapter constructor accepts provider, pricer, controller, and usage estimates, but accepts no identity mapping or identity override.

### Adapter separation

- Search and citation remain independent backend interfaces.
- OpenAlex is admitted for the Scheme B search role.
- Semantic Scholar is admitted for the Scheme B citation role and may also support search without changing the role binding.
- The LLM backend binds the actual DeepSeek-compatible analyzer, pricer, and controller.
- Provider classes remain unaware of Scheme B recipes, candidate pools, Gold, evaluators, and comparison thresholds.

## 6. Shared pricing and budget policy

One approved composite pricing policy is parsed once and used to construct the single `ActualCostPricer` shared by all three live adapters. The policy contains the required rates for LLM, OpenAlex, and Semantic Scholar dependencies. All dependency identities therefore bind the same `pricing_policy_sha256`.

One `HardBudgetController` is shared by all three adapters. It must use `formal_live=True`. Its existing policy fingerprint binds:

- the full search budget;
- formal-live enforcement mode;
- reservation TTL;
- controller identity/version.

All dependency identities therefore bind the same `controller_policy_sha256`.

The live runtime factory rejects:

- distinct pricing-policy hashes;
- distinct controller-policy hashes;
- `formal_live=False`;
- missing or malformed provider descriptors;
- a dependency kind assigned to the wrong Scheme B role;
- duplicate, missing, or unexpected roles;
- caller-supplied identity data.

Unknown cost is never treated as zero. If actual usage cannot be priced by the bound composite policy, settlement fails closed.

## 7. Runtime composition

`build_live_runtime` accepts only the three already budget-owning, self-identifying adapters required by Scheme B:

- search backend;
- citation backend;
- LLM backend.

It reads each adapter's immutable dependency identity, validates the role/dependency mapping, checks the shared pricing/controller fingerprints, and constructs the existing strict `LiveRuntimeIdentity`.

The resulting `RecallRuntime` contains the same three objects whose identities were validated. No replacement may occur between validation and execution. The runtime identity is canonicalized by the strict model before it is placed into `ExecutionIdentity` and the run report.

The CLI retains its two independent gates:

1. recipe data may request `live_provider`, but cannot authorize it;
2. runtime `--allow-live` is required and an injected runtime factory must return a fully valid self-identifying runtime.

Without both gates, the provider methods are never called.

## 8. Error and terminal-state contract

- The official construction path cannot form all required adapters: `live_runtime_unavailable`.
- A factory returns malformed, incomplete, mismatched, caller-declared, or non-formal-live identity: `config_mismatch` before any provider method.
- Missing API key or provider authentication failure after a valid runtime is constructed: `infrastructure_failure` with retained attempt artifacts.
- Rate limit, timeout, network, or provider failure: `infrastructure_failure`; it consumes a scheduled attempt but not a valid-repeat ordinal.
- Pricing or accounting mismatch: fail closed as an accounting/infrastructure failure; retain actual known usage and receipts.
- Cancellation: finalize or release reservations according to the existing adapter contract and re-raise cancellation.

Errors and public artifacts must not expose credentials, query text, Prompt text, response bodies, paper identifiers, private paths, or provider request IDs.

## 9. Offline verification

Use only fake transports/clients and synthetic configuration. No test may open a socket, read `.env`, create a real provider request, or mutate the user's three existing untracked paths.

Required tests:

1. OpenAlex, Semantic Scholar, and DeepSeek objects each emit deterministic provider descriptors.
2. Endpoint, model, adapter version, or operation changes alter the provider descriptor.
3. Credential changes do not alter identity and credentials never appear in serialized identities or error text.
4. Search, citation, and LLM adapters derive dependency identity from their actual provider, pricer, and controller objects.
5. Pricing-policy, budget, `formal_live`, or TTL changes alter the relevant policy fingerprint.
6. Different pricing/controller policies, wrong dependency roles, missing roles, malformed descriptors, and any caller identity override fail before a provider call.
7. A complete valid runtime passes composition and a fake live execution reaches the expected fake provider methods only when `--allow-live` is present.
8. A complete valid runtime is still rejected when `--allow-live` is absent.
9. Runtime and dependency identity serialization contains no protected values.
10. Snapshot replay, historical comparison, strict `ExecutionIdentity`, attempt artifacts, and offline CLI behavior remain unchanged.

Acceptance commands must include focused provider/adapter/composition tests, the complete `tests/recall_experiments` suite, tracked-file Ruff, `mypy src scripts`, `git diff --check`, and the full `pytest -m "not online"` gate.

## 10. Modularity and future replacement

Provider descriptors and adapter dependency identities are protocols/models, not Scheme B method conditionals. A future local scholarly index or different LLM may be admitted by implementing the same self-description contract and receiving an explicit dependency/role contract version.

Changing a provider implementation must not require edits to:

- search-term generators;
- retrieval action handlers;
- candidate-pool construction;
- post-pool stages;
- evaluator or historical comparison logic;
- frozen input sources.

Changing a search method or Prompt must not require edits to the live identity layer.

## 11. Completion boundary and next step

This feature is complete when a synthetic, fully self-identifying live runtime passes all offline gates and the official Scheme B composition path can construct the same contract from real configured objects without issuing a request.

Completion authorizes no network or fee-bearing action. The next separately approved operation is Task 13 Step 1:

- use the Query Evolution generation method as the first method-matched canary;
- select the existing three-query frozen smoke binding;
- declare DeepSeek/OpenAlex/Semantic Scholar authorization, total budget, ledger path, and output root;
- generate and freeze actions;
- execute one live retrieval attempt;
- seal exact requests/responses and usage;
- replay the new snapshot offline;
- report candidate recall only, without a general performance or compatibility claim.
