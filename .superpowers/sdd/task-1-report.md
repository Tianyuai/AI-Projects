# Task 1 Report — Deterministic embedding ranking core

Status: complete

Scope: implemented Task 1 only. Tasks 2-5 were not touched.

Changed files:

- `src/paper_search/ranking/embedding.py`
- `src/paper_search/ranking/__init__.py`
- `tests/unit/test_embedding.py`

Commit:

- `f6f54eb` — `feat: add deterministic embedding ranking`

Verification:

1. Red test run

   Command:

   ```powershell
   $env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
   & 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/unit/test_embedding.py -q
   ```

   Result:

   - First attempt failed at the uv cache initialization step because the default cache location was not writable in the sandbox.
   - Retried with elevated access.
   - Final red result:
     - `ModuleNotFoundError: No module named 'paper_search.ranking.embedding'`

2. Green focused test run

   Command:

   ```powershell
   $env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
   & 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/unit/test_embedding.py -q
   ```

   Result:

   - `7 passed in 0.32s`

3. Static checks

   Ruff:

   ```powershell
   $env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
   & 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file ruff check src/paper_search/ranking/embedding.py tests/unit/test_embedding.py
   ```

   Result:

   - `All checks passed!`

   Mypy:

   ```powershell
   $env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
   & 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file mypy src/paper_search/ranking/embedding.py
   ```

   Result:

   - `Success: no issues found in 1 source file`

Notes / concerns:

- I left unrelated untracked worktree items alone, including `.superpowers/` and `docs/superpowers/plans/2026-07-28-week3-task9-embedding-ranking.md`.
- Git warned that the staged files will be normalized from LF to CRLF on the next checkout/touch in this Windows worktree.
- The focused Task 1 test suite covers the deterministic ranking core requested in the brief; I did not expand into Tasks 2-5.

## Fix section — reviewer follow-up

Issue summary:

- `fallback_to_cpu` now controls the CUDA-to-CPU retry path.
- Factory and encode failures are translated into `EmbeddingUnavailableError` or `EmbeddingOutOfMemoryError` internally before fallback/degradation handling.
- The unit tests now cover CUDA unavailable fallback, CUDA OOM fallback, and CPU failure degradation with sanitized warning codes.

Fix commit:

- `fix: add embedding fallback handling`

Verification commands and results:

1. Red pass for the new fallback tests

   Command:

   ```powershell
   $env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
   & 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/unit/test_embedding.py -q
   ```

   Result before the fix:

   - `3 failed, 7 passed in 0.63s`
   - Failures:
     - CUDA unavailable fallback raised `RuntimeError: cuda device not available`
     - CUDA OOM fallback raised `MemoryError: cuda out of memory`
     - CPU degradation path raised `RuntimeError: cuda unavailable`

2. Green pass after implementing fallback and degradation handling

   Command:

   ```powershell
   $env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
   & 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/unit/test_embedding.py -q
   ```

   Result after the fix:

   - `10 passed in 0.33s`

3. Static checks after the fix

   Ruff:

   ```powershell
   $env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
   & 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file ruff check src/paper_search/ranking/embedding.py tests/unit/test_embedding.py
   ```

   Result:

   - `All checks passed!`

   Mypy:

   ```powershell
   $env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
   & 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file mypy src/paper_search/ranking/embedding.py
   ```

   Result:

   - `Success: no issues found in 1 source file`
