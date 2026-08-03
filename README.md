# AI Projects — Paper Search

本仓库实现一套 offline-first、可审计的学术论文检索系统。当前代码已经提供统一的 replay/live 组合、FastAPI 服务与浏览器 UI、依赖快照、正式评测工作区、运行验证和 capture/replay 等价比较；默认实验仍是 `main-baseline`，所有可选模块默认关闭。

当前公开数据状态仍为 `waiting_for_human_label_freeze`，仓库内 Gate 0 报告为 blocked。工程测试、合成 capture/replay 和 replay 浏览器验收通过，不等同于真实数据冻结、真实 provider 运行、正式 dev/validation 指标或可选模块晋升完成。

## CPU-first quickstart

第三方验收应从 fresh clone 或全新 worktree 开始，并使用 Python 3.11：

```powershell
git clone https://github.com/Tianyuai/AI-Projects.git paper-search-check
Set-Location paper-search-check
uv sync --locked --extra cpu
```

裸 `uv sync` 只安装 core-only 依赖；完整检索、健康检查与测试需要显式选择 `cpu` 或 `cuda` profile。CPU 是 Windows/Linux 的便携验收配置，不要求 NVIDIA GPU。

```powershell
uv run --no-sync --no-env-file python -m paper_search.health
uv run --no-sync --no-env-file pytest -q
uv run --no-sync --no-env-file ruff check .
uv run --no-sync --no-env-file mypy src
uv run --no-sync --no-env-file paper-search --help
```

`--no-env-file` 阻止命令主动加载 `.env`，但不会清除继承的环境变量，也不等同于网络隔离。默认离线测试中，凭据门控的显式 online 测试可因未提供进程凭据而跳过。

CUDA 仅用于明确选择且兼容的 NVIDIA 环境；CPU 与 CUDA extras 不应同时安装：

```powershell
uv sync --locked --extra cuda
uv run --no-sync --no-env-file python -m paper_search.health --require-accelerator cuda
```

## Unified replay service

`paper-search serve` 是唯一正式服务入口。它始终绑定一个经验证的 replay lock 与 snapshot manifest；replay 请求不构造 live client，也不访问外部依赖。

```powershell
paper-search serve `
  --lock <capture>/replay.lock.yaml `
  --mode replay `
  --snapshot-manifest <capture>/snapshot-manifest.json `
  --capture-output-root <runs-root> `
  --host 127.0.0.1 `
  --port 8000
```

浏览器 UI 位于 `http://127.0.0.1:8000/`，与直接调用 `POST /v1/search` 共用同一个应用服务和响应契约。`GET /health/live` 只表示进程存活；`GET /health/ready` 返回安全的 replay 快照绑定与依赖状态。

Live 请求只有同时满足三项条件才会执行：输入 lock 的 `runtime_allow_live: true`、服务启动时显式 `--allow-live`、单次请求显式 `mode: live`。每个 live 请求使用独立服务和预算，并在 HTTP 200 前完成捕获、封存、校验与原子发布；失败或取消不能发布为 complete。

真实 live 会产生网络和成本，必须另行获得针对目标环境、硬预算和凭据的明确授权。默认命令和 UI 选择均为 replay。

## Formal evaluation and replay verification

正式运行使用同一生产应用服务，不存在独立评测检索管线：

```powershell
paper-search evaluate --lock <input-lock> --split dev --mode live --output-root <runs-root> --allow-network
paper-search evaluate --lock <capture>/replay.lock.yaml --split dev --mode replay --output-root <runs-root> --snapshot-manifest <capture>/snapshot-manifest.json
paper-search verify-run <run-directory>
paper-search compare-replay <capture-run> <replay-run>
```

`verify-run` 校验锁、快照、配置身份、逐查询记录、聚合指标和发布状态；`compare-replay` 比较规范化业务结果。validation lock 只允许一次不可撤销的 live 尝试，不能把重试当作新的验证机会。

仓库内的合成正式夹具可用于离线验证工具链：

```powershell
paper-search verify-run tests/fixtures/formal_run/capture
paper-search verify-run tests/fixtures/formal_run/replay
paper-search compare-replay tests/fixtures/formal_run/capture tests/fixtures/formal_run/replay
```

这些夹具证明格式和验证器行为，不代表真实数据集效果。

## Experiments and optional modules

`configs/base.yaml` 固定 `experiment: main-baseline`。可选身份为 `embedding`、`citation-expansion`、`llm-rerank`、`fixed-two-round` 和 `adaptive-evolution`；每个身份只构造其声明组件，baseline 不加载可选依赖。

可选模块的实现或离线测试通过不代表晋升。晋升需要 Gate 0–5、三次同配置 dev 比较、1,000 次 bootstrap、一次 selection-only validation 比较及单独批准；在证据不完整或阈值不通过时保持 default-off。

Embedding 的离线聚焦验证命令为：

```powershell
uv run --no-sync --no-env-file pytest tests/unit/test_embedding.py tests/unit/test_sentence_transformer.py tests/evaluation/test_embedding_benchmark.py tests/integration/test_orchestrator.py -q
```

The ablation framework is offline and injected by default. It does not call APIs or load `.env`. Selection notes remain `owner_only_provisional` until the formal evidence and promotion gates pass.

## Evidence checkpoint

截至 2026-08-03，集成源码检查点 `fcc0ff0` 的完整离线套件为 `1744 passed, 36 skipped`；Task 4 与 Task 5 的独立范围审查均为 C0/I0。可复核入口包括 [正式运行夹具](tests/fixtures/formal_run)、[双模式 E2E](tests/e2e/test_dual_mode_serve.py) 和 [统一服务进程测试](tests/integration/test_serve_process.py)。浏览器截图与无敏感信息的验收记录按策略保存在源码树外，因此 fresh clone 不应把该记录视为可独立取得的仓库制品。

## Data and access boundaries

数据契约与人工标注流程见 `data/README.md`。真实查询、gold、原始数据、人工标签、逐查询预测和 provider 原始响应不得进入 Git。快照、正式运行和验证声明也必须保存在访问受控位置，只共享经批准的安全聚合、哈希和 run ID。

不得读取、打印、搜索或复制 `.env`。只有明确授权的在线、数据准备或正式运行子进程可以加载所需凭据；文档与日志只能出现环境变量名称，不能出现其值。

当前机器证据边界和未完成项见 `docs/limitations-and-risks.md`，操作演示见 `docs/demo/demo-runbook.md`。
