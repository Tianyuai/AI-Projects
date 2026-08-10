# Identity Capture Request-Failure Classification Report

## Scope

- Implemented value-free request-failure categories in `scripts/capture_dev_identifier_identity.py`.
- Added regression coverage in `tests/scripts/test_capture_dev_identifier_identity.py`.
- Did not access the network, `.env`, preflight, capture, or validation data.

## Root Cause

`_request_json` terminalized reservations correctly, but collapsed every transport
exception and every non-2xx response into the same provider-level error. The final
failure therefore lost the distinction between timeout, network, rate-limit,
client, server, and unexpected HTTP status failures.

## TDD Evidence

### RED

Command:

`python -m pytest tests/scripts/test_capture_dev_identifier_identity.py -q`

Observed before the production change: `9 failed, 29 passed`. Every new assertion
received the old unclassified `semantic scholar request failed` message.

### GREEN

The minimal production change maps:

- `httpx.TimeoutException` to `timeout`;
- other request and generic transport exceptions to `network_error`;
- HTTP 429 to `rate_limited`;
- other 4xx to `client_error`;
- 5xx to `server_error`;
- other non-2xx statuses to `unexpected_status`.

The category is computed for each failed attempt, and only the final exhausted
attempt's fixed category is raised. Provider response bodies, URLs, identifiers,
headers, credentials, and exception text are never interpolated.

Focused result: `38 passed`.

## Verification

- Related pytest gate: `132 passed`.
- Ruff on both changed Python files: passed.
- mypy on `src` and the changed script: no issues in 97 source files.
- `git diff --check`: passed.

## Review

- Critical findings: none.
- Important findings: none.
- Retry count, request/attempt caps, ledger reservation terminalization, lock v2,
  request shape, validators, snapshot identity, and credential handling remain
  unchanged.
- User-owned dirty paths were excluded from staging.
