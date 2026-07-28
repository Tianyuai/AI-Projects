# Task 3 implementation report

Date: 2026-07-28

Scope completed:

- Added strict default-off `EmbeddingConfig`.
- Integrated `RuntimeConfig.embedding`.
- Added explicit embedding defaults to `configs/base.yaml`.
- Added config-hash coverage for embedding state.
- Did not touch `.env`, private data, Week 2 branches, real records, or immutable evaluation design.

Files changed:

- `src/paper_search/config.py`
- `tests/unit/test_config.py`
- `configs/base.yaml`

Test-first evidence:

1. Added failing tests in `tests/unit/test_config.py` for:
   - default-off embedding config values
   - explicit CUDA embedding YAML with CPU fallback
   - rejecting unsafe batch sizes
2. Ran the focused config suite and confirmed the expected red state:
   - `pytest tests/unit/test_config.py -q`
   - Result: 2 failures, both caused by `RuntimeConfig` not yet exposing `embedding`

Implementation evidence:

- Added `EmbeddingConfig` as a frozen, extra-forbidden Pydantic model.
- Added `embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)` to `RuntimeConfig`.
- Left environment-variable mapping unchanged; no new secret or env wiring was added for embedding.
- Added explicit embedding defaults to `configs/base.yaml`.

Verification evidence:

- Focused config tests:
  - `pytest tests/unit/test_config.py -q`
  - Result: `14 passed`
- Ruff:
  - `ruff check src/paper_search/config.py tests/unit/test_config.py`
  - Result: passed
- Mypy:
  - `mypy src/paper_search/config.py`
  - Result: passed
- Diff hygiene:
  - `git diff --check`
  - Result: no whitespace errors; only line-ending warnings from Git on this environment

Notes:

- The new embedding values are now part of the canonical runtime config payload, so they participate in `config_hash()`.
- No task 4 or task 5 behavior was implemented.
