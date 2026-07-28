# AI Projects — Paper Search

本仓库实现可复现的学术论文检索与 Week 1 评测流程。当前正式数据状态仍为 `waiting_for_human_label_freeze`；工程测试通过不代表人工标注、真实 baseline 或 Week 1 gate 已完成。

## CPU-first quickstart

第三方验收必须从 fresh clone 或全新 worktree 开始，并使用 Python 3.11：

```powershell
git clone https://github.com/Tianyuai/AI-Projects.git AI-Projects-week1-check
Set-Location AI-Projects-week1-check
git fetch origin
git switch --track origin/codex/week1-collaboration
uv sync --locked --extra cpu
```

裸 `uv sync` 只安装 core-only 依赖，不包含 embedding/torch；完整检索、健康检查与测试必须显式选择 `cpu` 或 `cuda` profile。CPU 是 Windows/Linux 的必选验收配置，不要求 NVIDIA GPU。

```powershell
uv run --no-sync --no-env-file python -m paper_search.health
uv run --no-sync --no-env-file pytest -q
uv run --no-sync --no-env-file ruff check .
uv run --no-sync --no-env-file mypy src
```

离线全量测试中，只有需要 `OPENALEX_API_KEY` 的显式 online 测试可以因缺少进程凭据而跳过。`--no-env-file` 表示测试子进程不主动加载 `.env`，不等同于网络隔离。

prepared manifest、ID 清单和冻结规则可用以下测试复核：

```powershell
uv run --no-sync --no-env-file pytest tests/evaluation/test_prepare_data.py tests/evaluation/test_freeze.py -q
```

## Optional NVIDIA CUDA profile

只有明确需要并拥有兼容 NVIDIA 环境时才选择 CUDA：

```powershell
uv sync --locked --extra cuda
uv run --no-sync --no-env-file python -m paper_search.health --require-accelerator cuda
```

`cpu` 与 `cuda` extras 互斥，不能同时选择。Windows AMD GPU 按 CPU 路径验收；仓库不宣称原生 ROCm 支持。

## Week 3 optional embedding ranking

Embedding ranking is disabled by default. The offline unit and integration
tests use deterministic injected components and do not download a model, call
an API, or load `.env`.

CPU is the default path for local embedding ranking. CUDA is opt-in, retries
once on CPU after CUDA OOM or CUDA unavailability, and the local encoder must
be released before another local model is loaded. Do not keep Embedding and a
local Reranker resident on the same 4 GB GPU at the same time.

Benchmark output keeps only a safe model identifier and warning codes. It does
not emit local model paths, raw exception text, or query-like free text.

Offline focused verification command:

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/unit/test_embedding.py tests/unit/test_sentence_transformer.py tests/evaluation/test_embedding_benchmark.py tests/integration/test_orchestrator.py -q
```

Opt-in real local-model benchmarking is separate from the default offline test
suite. Use an already-downloaded local model path plus synthetic text, and
record only aggregate latency, process peak RSS, CUDA peak allocation, status,
and fallback state.

## 数据与人工标注

协作者的立即执行清单见 `docs/TEAMMATE_ONBOARDING.md`，安全数据契约见 `data/README.md`。真实查询、gold、原始数据和人工标签不得进入 Git。不得读取、打印、搜索或复制 `.env`；只有明确授权的在线或数据准备子进程可以通过 `--env-file` 加载它。
