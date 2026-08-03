# 新环境部署与验收清单

使用 fresh clone 或全新 worktree。将工程验证、replay 验收与授权 live 证据保持为独立门禁。

## 运行时与依赖

- [ ] 确认 Python 3.11.x；项目支持 `>=3.11,<3.12`。
- [ ] 确认 `uv` 可用。
- [ ] 安装且仅安装一个 profile：`uv sync --locked --extra cpu`，或单独批准的 CUDA profile。
- [ ] 将裸 `uv sync` 视为仅 core 依赖，不足以完成完整验收。
- [ ] 记录命令结果与仓库修订，不记录机器特定路径或包索引凭据。

## 凭据边界

- [ ] 仅通过批准的密钥管理器核对所需变量名：`OPENALEX_API_KEY`、`SEMANTIC_SCHOLAR_API_KEY`、`LLM_API_KEY`、`LLM_BASE_URL`、`LLM_MODEL_PRIMARY`、`LLM_MODEL_FALLBACK` 与 `HF_TOKEN`。
- [ ] 绝不打印、复制、记录、提交、截图或粘贴其值。
- [ ] 离线命令保持 `--no-env-file`；这不会清除继承变量，也不强制网络隔离。
- [ ] 验收中绝不检查 `.env` 内容。

## 工程门禁

```powershell
uv run --no-sync --no-env-file python -m paper_search.health
uv run --no-sync --no-env-file pytest -q
uv run --no-sync --no-env-file ruff check .
uv run --no-sync --no-env-file mypy src
uv run --no-sync --no-env-file paper-search --help
```

- [ ] 所有命令退出 0。
- [ ] 无凭据时，凭据门控的 online 测试保持显式 skip，而非成功的 provider 检查。

## Gate 0 与数据状态

- [ ] 读取当前安全 Gate 0 报告并确认 `passed: true`，之后才可做真实 provider 或正式数据声明。
- [ ] 要求 V2 冻结 manifest、精确分区与标识符映射哈希、批准的生成定价策略、质量门策略与安全就绪证据。
- [ ] 若 Gate 0 为 blocked，停止真实证据路径并报告具体阻塞原因；不要手动修改 `data/manifest.json`。
- [ ] 将原始数据、gold、标签文件、真实查询与逐查询证据保留在 Git 与普通日志之外。

当前仓库 Gate 0 已通过（2026-08-03 r5，`data/gate0_evidence.json`）；合成夹具仍覆盖工程路径，但正式 dev/validation 与晋升仍需显式授权。

## Replay 服务门禁

- [ ] 用 `paper-search verify-run` 验证所选 capture 与 replay 制品。
- [ ] 用 `paper-search compare-replay` 验证对等价。
- [ ] 用验证的 replay lock 与 snapshot manifest 启动 `paper-search serve`，不加 `--allow-live`。
- [ ] 除非单独部署安全审查批准其他接口，否则仅绑定 loopback。
- [ ] 检查 `/health/live`、`/health/ready`、浏览器 UI 与一次直接 `/v1/search` 请求。
- [ ] 确认重复 replay 保持规范化业务结果与稳定来源。
- [ ] 确认 replay 不发起外部名称解析或 socket 连接。
- [ ] 干净停止并检查不完整制品或持有的锁。

## Live 授权门禁

三项技术授权谓词均强制；它们不是凭据，也不替代操作员的治理批准：

- [ ] 验证血缘 lock 有 `runtime_allow_live: true`；
- [ ] 操作员显式以 `--allow-live` 启动服务；
- [ ] 单次请求显式设置 `mode: live`。

- [ ] 为 provider、凭据范围、查询类别、硬预算、捕获根目录与保留策略取得单独批准。
- [ ] 确认一次请求获得隔离的 live 服务、客户端、预算与捕获会话。
- [ ] 确认成功捕获在 HTTP 200 前封存、验证并原子发布。
- [ ] 确认失败或取消的工作不可能呈现为 complete。
- [ ] 服务端 live 捕获为 smoke 型目录：检查 `run.json` 的 `status: complete`、`snapshot-manifest.json` 与 `replay.lock.yaml`，并用该捕获启动 replay 服务验证可回放；`verify-run` 只适用于 `paper-search evaluate` 产出的正式运行目录。
- [ ] 只记录安全哈希、run ID、聚合用量/成本与脱敏错误码。

## 正式 dev 与 validation 门禁

- [ ] 在冻结运行上限内运行授权 dev 捕获。
- [ ] 验证捕获，从同一快照集生成 replay，验证 replay，并比较规范化业务结果。
- [ ] 仅从完整通过的 dev 证据晋升 validation lock。
- [ ] 将 validation lock 哈希视为单一不可撤销尝试身份。
- [ ] 运行一次授权 live validation 尝试；中断或失败不授权替代尝试。
- [ ] 报告聚合结果前验证并比较 validation capture/replay。
- [ ] 将预测、失败、业务结果、快照、gold 标签与验证声明保持访问受控。

## 可选模块晋升门禁

- [ ] 整个证据生成期间保持 `configs/base.yaml` 为 `main-baseline`。
- [ ] 可选消融前要求 Gates 0–5。
- [ ] 在相同冻结输入、快照、预算与测量策略下运行三次同配置 dev 比较。
- [ ] 使用 1,000 次 bootstrap 与已提交的晋升阈值。
- [ ] 只运行批准的 selection-only validation 比较。
- [ ] 证据不完整或任一阈值不通过时保持模块 default-off。
- [ ] 改变 baseline 默认或 validation lock 前请求单独晋升决定。

## 交接记录

- [ ] 分别说明各门禁为 passed、blocked、failed 或 not run。
- [ ] 只链接访问适当的证据。
- [ ] 不把夹具成功转化为真实数据、真实 provider、质量、成本或生产就绪声明。
