# DOI Exact-Endpoint Acceptance Design

## Goal

Remove false DOI integrity failures from the aggregate OpenAlex availability diagnostic without weakening OpenAlex-ID checks, changing production retrieval, or making another network request.

## Decision

For a request sent to OpenAlex's single-Work endpoint with a normalized DOI:

- HTTP 200 plus a valid OpenAlex Work ID means `available`.
- The response's top-level `doi` is descriptive canonical metadata, not a second identity gate.
- A missing, different, or unparseable top-level DOI does not invalidate the successful exact-endpoint resolution.
- A missing Work ID is `integrity_failure / missing_expected_field`.
- An unparseable Work ID is `integrity_failure / unparseable_identifier`.

For a request sent with an OpenAlex Work ID, the existing strict contract remains unchanged: the response must contain a valid OpenAlex Work ID that normalizes to the requested ID; a different ID is `integrity_failure / canonical_mismatch`.

This follows the OpenAlex single-Work contract: the endpoint accepts an OpenAlex ID or external ID such as a DOI, while the returned `doi` is the Work's canonical external ID. Reference: <https://developers.openalex.org/api-reference/works/get-a-single-work>.

## Scope

Modify only the aggregate diagnostic classifier in `scripts/analyze_gold_bottlenecks.py`, its synthetic tests, and the active project-state documentation needed to record the new contract. Do not change provider adapters, production search, candidate generation, ranking, report schema, privacy rules, budget accounting, locks, or historical evidence files.

The implementation must not read `.env`, call OpenAlex, rerun the availability probe, rebuild `runs/candidate.lock.yaml`, run readiness, or perform capture/replay/validation.

## Classification Flow

1. Normalize the requested identifier with the existing `normalize_paper_id` function.
2. Require the HTTP 200 payload to be an object.
3. Read and normalize the response `id` as an OpenAlex Work ID.
4. For a DOI request, accept any valid normalized OpenAlex Work ID because the exact DOI endpoint supplied the request-to-Work binding.
5. For an OpenAlex-ID request, require the normalized response Work ID to equal the normalized requested ID.
6. Keep the existing aggregate status and integrity-reason maps unchanged.

The classifier must not inspect titles, authors, URLs beyond identifier normalization, or any other provider fields. It must not retain or emit the requested DOI or returned Work ID.

## Error Handling

- Non-object payload: `integrity_failure / missing_expected_field`.
- Missing or non-string response `id`: `integrity_failure / missing_expected_field`.
- Response `id` rejected by `normalize_paper_id`, or normalized to a non-OpenAlex identifier: `integrity_failure / unparseable_identifier`.
- OpenAlex-ID request whose response Work ID differs: `integrity_failure / canonical_mismatch`.
- DOI request with a valid response Work ID: `available`, regardless of top-level DOI content.

HTTP 404, retries, authentication failures, quotas, ledger settlement, and global fail-closed behavior remain unchanged.

## Testing

Use only `httpx.MockTransport` and synthetic identifiers. TDD coverage must prove:

- DOI request plus valid Work ID and matching DOI is available.
- DOI request plus valid Work ID and missing DOI is available.
- DOI request plus valid Work ID and different canonical DOI is available.
- DOI request plus valid Work ID and unparseable DOI is available.
- DOI request with missing Work ID is `missing_expected_field`.
- DOI request with malformed or non-OpenAlex response ID is `unparseable_identifier`.
- OpenAlex-ID exact match remains available.
- OpenAlex-ID mismatch remains `canonical_mismatch`.
- Existing aggregate schema, integrity-count conservation, privacy checks, retry behavior, Ruff, mypy, and the full test suite remain green.

## Acceptance Criteria

The change is complete when the focused synthetic tests pass, the full offline verification passes, and no network, ledger, lock, or historical diagnostic artifact has changed. The existing aggregate report remains a historical record of the single approved online probe; a new diagnostic conclusion requires separate authorization for a future online probe.
