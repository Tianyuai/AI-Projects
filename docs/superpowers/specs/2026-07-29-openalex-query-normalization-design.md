# OpenAlex Query Normalization Design

## Context

The first real fresh-cache development baseline completed structurally but all
60 OpenAlex requests returned `invalid_request`. Live diagnostics established
that every frozen development query contains one question mark. OpenAlex treats
`?` and `*` as wildcard operators in its default stemmed search and rejects
those requests.

The frozen gold data is valid and remains unchanged. The failed run is retained
as immutable diagnostic evidence and must not be reported as an effective
baseline.

## Design

Normalize only the outbound OpenAlex search text:

1. Strip surrounding whitespace.
2. Replace every `?` and `*` with one space.
3. Collapse consecutive whitespace to one ASCII space.
4. Reject the query if normalization leaves it empty.

The provider cache key and request provenance use the normalized outbound
parameter. The original frozen query remains unchanged in evaluation inputs and
identity hashes. No LLM is involved.

This is preferred over `search.exact`, which would disable stemming and reduce
baseline recall, and over removing only a terminal question mark, which would
leave the same failure mode elsewhere in user input.

## Error Handling and Tests

Add focused unit tests proving that:

- question marks and asterisks never reach the OpenAlex request;
- whitespace is normalized deterministically;
- a wildcard-only query is rejected before any HTTP request;
- ordinary search text is unchanged.

Run the OpenAlex unit suite, static checks, and the broader test suite before a
new real baseline.

## Real Baseline Rerun

Run from `D:\AI Projects\Projects` and load credentials with the relative
argument `uv run --no-sync --env-file .env`. This avoids the observed Windows
path parsing failure for an absolute `.env` path containing spaces. The runner
and configuration paths remain absolute.

Create a new Git-external run directory with a new empty SQLite cache and output
directory. Never overwrite or reuse the failed run. Validate the snapshot
manifest, artifact hashes, provider error counts, non-empty retrieval coverage,
aggregate metrics, usage, and source Git SHA before reporting the result.
