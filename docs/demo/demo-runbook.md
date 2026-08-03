# 演示运行手册

## 证据选择

使用 `replay.lock.yaml` 与 snapshot manifest 已通过适用验证器的捕获。以下命令均为 replay-only，不加载 `.env`、不联系 provider、不产生 provider 成本。

仓库内工程验收首先验证合成正式对：

```powershell
uv run --no-sync --no-env-file paper-search verify-run tests/fixtures/formal_run/capture
uv run --no-sync --no-env-file paper-search verify-run tests/fixtures/formal_run/replay
uv run --no-sync --no-env-file paper-search compare-replay tests/fixtures/formal_run/capture tests/fixtures/formal_run/replay
```

预期结果：两个正式运行目录均有效，且 capture/replay 对等价。这验证证据机制，不证明真实检索质量。

Fresh clone 不包含交互可用的服务就绪捕获。进入交互章节前，授权操作员必须提供一个已验证捕获目录与访问受控的 runs root。完整仓库内 replay/服务验收运行：

```powershell
uv run --no-sync --no-env-file pytest tests/e2e/test_dual_mode_serve.py tests/integration/test_serve_process.py -q
```

E2E 测试安装 socket 与名称解析 tripwire，证明其拒绝非 loopback 目标，然后运行真实 `paper-search serve` 子进程。

## 启动统一 replay 服务

从所选 lock 期望的 artifact root 启动 loopback 上的规范化服务：

```powershell
uv run --no-sync --no-env-file paper-search serve `
  --lock <capture>/replay.lock.yaml `
  --mode replay `
  --snapshot-manifest <capture>/snapshot-manifest.json `
  --capture-output-root <runs-root> `
  --host 127.0.0.1 `
  --port 8000
```

replay 演示不要添加 `--allow-live`。保持此终端打开。

## 健康检查

在第二个终端：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health/live
Invoke-RestMethod http://127.0.0.1:8000/health/ready
```

Liveness 应报告 `live`。Readiness 应报告 `ready` 或 `degraded`、`execution_mode: replay`、绑定的 snapshot-set 身份，以及来自 `ready`、`replayed`、`degraded` 或 `failed` 的依赖状态；它不得发起 live 依赖探测。

## 浏览器演示

打开 `http://127.0.0.1:8000/`，保留默认 Replay 模式，提交一个所选快照批准的代表性查询，并验证 UI 显示：

- 选中论文 ID 与排序结果证据；
- 执行模式与快照集/时间；
- 配置哈希与每次请求的 run ID；
- 用量、停止原因、部分状态、规划器状态与回退状态；
- 依赖状态、安全警告与引文边。

再次提交同一请求。每次请求的 run ID 可以变化；规范化业务内容与稳定来源必须保持不变。

检查浏览器 console 与 network 面板：不应有 JavaScript 错误；每次提交恰好一个 `/v1/search` POST；不应有对 OpenAlex、Semantic Scholar、LLM 端点或本地文件系统路径的浏览器请求。

## 直接 API 演示

UI 与直接 API 共享同一边界：

```powershell
$body = @{
  query_id = "demo-replay-1"
  query = "<approved replay query>"
  budget_profile = "balanced"
  mode = "replay"
  include_trace = $true
} | ConvertTo-Json

$response = Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/v1/search `
  -ContentType application/json `
  -Body $body

$response | ConvertTo-Json -Depth 12
```

不得在提交的文档、截图或普通日志中放置私有查询。

## 授权 live 演示

Live 是可选的，且始终被阻塞，直到操作员分别授权目标 provider、凭据、查询类别、硬预算、捕获位置与披露边界。先前的 replay 授权不是 live 授权。

当这些前置条件满足时，验证血缘 lock 允许 live，以 `--allow-live` 启动同一服务，并对一个有界请求显式选择 `mode: live`。仅服务端标志不会让省略模式的请求变成 live。HTTP 200 之后，服务端捕获为 smoke 型目录：校验 `run.json` 的 `status: complete`、`snapshot-manifest.json` 与 `replay.lock.yaml` 存在且一致，并通过以该捕获启动 replay 服务来验证可回放性；`verify-run` 只适用于 `paper-search evaluate` 产出的正式运行目录。

只记录安全 run ID、聚合状态、哈希、有界降级码与验证结果。不得记录凭据、查询文本、原始快照、预测、gold 标签或私有路径。

## 停止与清理

返回服务终端并按 `Ctrl+C`。在 Windows 上确认端口关闭，并只检查所选 runs root 的 incomplete/lock 标记：

```powershell
Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
Get-ChildItem -LiteralPath <runs-root> -Force | Where-Object { $_.Name -match 'incomplete|\.lock$|\.lck$' }
```

预期结果：两条命令均无匹配记录。除非文档资产审查显式授权提交，否则将批准的截图与安全验收记录保存在源码控制之外。

## 当前项目状态

Replay 浏览器验收、双模式 fake-live 生命周期 E2E 与一次真实 live 浏览器捕获（2026-08-03，显式授权下完成，成本 0.001181 CNY）已验证；真实 live 暴露并修复了两个 DashScope LLM 兼容性缺陷。正式 dev/validation 捕获、指标声明、成本声明与可选模块晋升仍未运行，因为当前公开 Gate 0 仍为 blocked，且正式 dev/validation 与 Gate 6 需要各自的授权与前置证据。
