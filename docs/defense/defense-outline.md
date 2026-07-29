# Defense Outline

## Problem Definition

Explain the goal of resource-aware scholarly search: turn a constrained
research request into an auditable set of candidate papers while respecting
hard budgets, reproducibility requirements, and safe degraded behavior.

State the evidence boundary up front: the current runnable demonstration is
offline and synthetic, and R2 is diagnostic rather than a relevance result.

## Architecture

Trace one request through validated query specification and planning, injected
providers, budget reservation and settlement, deduplication, filtering, fusion,
optional ranking stages, and the structured response.

Distinguish the loopback-only mock server from a real-provider deployment. The
mock composition is useful for deterministic interface demonstration but does
not establish external API readiness.

## Baseline

Describe the intended fixed-strategy baseline as a future controlled
comparison, not as a completed experiment. Identifier-map wiring and the R3
baseline are external dependencies that were incomplete at this task's start.

Explain that R2 can reveal retrieval or integration diagnostics, but its zero
relevance metrics result from an identifier namespace mismatch and cannot be
used as a performance conclusion.

## Innovation Module

Describe the offline `EvolutionCoordinator` as an injected coordinator around
a single-round executor. It analyzes constraint coverage, plans an optional
next round, estimates budget use, and emits auditable stopping decisions.

It does not replace the single-run orchestrator and remains disabled in the
main API and runtime configuration. There is no real fixed-strategy comparison
for adaptive evolution yet.

## Experimental Method

Present the planned method: preserve frozen inputs and identifier manifests;
run fixed and adaptive variants on identical data, budgets, configuration,
seeds, and measurement rules; then review failures before interpreting results.

Do not present R2 diagnostics, synthetic mock responses, or test fixtures as
formal evaluation. Begin formal comparisons only after R3 and the required
operator-authorized fresh-cache and provider gates.

## Failure Analysis

Discuss identifier namespace mismatch as a data-contract failure that can
produce zero relevance accounting without measuring retrieval quality.

Include later quality analysis for the seven `invalid_work` records. Also note
that relationship visualization remains gated and must not be used as evidence
until its stage gate passes.

## Cost and Reproducibility

Describe the hard-budget accounting boundary and the planned preservation of
configuration hashes, git revision, frozen inputs, snapshots, cache manifests,
and response hashes. Unknown costs must remain unknown rather than being
reported as zero.

Explain that real-provider cost collection and fresh-cache execution require
explicit operator authorization and credential handling outside this document.
No secret values belong in artifacts, commands, logs, or slides.

## Limitations

The evidence currently excludes the R3 baseline, formal relevance results,
formal fixed-versus-adaptive comparison, real-provider readiness proof, and a
stage-approved relationship visualization. Adaptive behavior stays disabled
until the relevant integration and evaluation gates pass.

## Formal Results to Add After R3

After R3 authorizes formal evaluation, add these named artifacts with their
documented inputs, definitions, and provenance:

- Formal metric table
- Cost table
- Ablation table

Do not add numeric placeholders, empty metric cells, or comparative conclusions
before those artifacts are produced from the authorized frozen inputs.
