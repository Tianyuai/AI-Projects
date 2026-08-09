# Gold Availability Bottleneck Attribution

- Schema: `gold-bottleneck-attribution-v2`
- Source run: `dev-20260809T061903Z-9bd861e90299`
- Diagnostic complete: `True`

## Denominators

- Queries: 60
- Raw gold identifiers: 143
- Normalized query–work associations: 139
- Unique works: 134

## Availability

- available: 134
- exact_not_found: 0
- unknown_transient: 0
- invalid_identifier: 0
- integrity_failure: 0

## Integrity failure breakdown

- missing_expected_field: doi=0, openalex=0
- unparseable_identifier: doi=0, openalex=0
- canonical_mismatch: doi=0, openalex=0

## Pipeline stages

- selected_top50: 8
- ranked_outside_top50: 6
- filtered_out: 0
- not_retrieved: 125

## Usage

- Planned unique requests: 134
- HTTP attempts: 135 / 402
- Retries: 1
- Timeouts: 0

- Recommended direction: `retrieval_query_evolution_probe`
- Reason codes: `available_not_retrieved_dominant`

## Limitations

- Frozen dev run only.
- Exact DOI/OpenAlex-ID lookup only.
- One provider only; no alternate-source probe.
- No live capture, replay, compare, or production change was performed.
