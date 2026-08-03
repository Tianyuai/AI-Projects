# 当前系统架构

## 范围与证据边界

本文描述当前源码中的集成运行时。它区分已验证的离线行为与仍被阻塞或未执行的真实数据、真实网络证据，且不构成检索质量、成本或生产就绪性声明。

## 单一应用边界

`CompositionRoot` 构建规范化的 `SearchApplicationService`。同一服务边界被 smoke 运行、正式评测、FastAPI 与浏览器 UI 共用；不存在第二条仅用于评测或仅用于 UI 的检索管线。

一次请求依次经过查询分析与规划、有界多源检索、去重、硬过滤、倒数排名融合、可选命名阶段、响应转换与规范化业务结果投影。每个请求独占一个 `HardBudgetController`；预留发生在依赖工作之前，并按实际安全用量结算。

## replay 与 live 隔离

服务始终围绕经验证的 replay lock 与不可变依赖快照清单组合而成。其 replay 服务进程级绑定，不持有任何 live 依赖客户端。replay 请求只读取内容寻址的快照字节，并校验请求身份、响应哈希、清单哈希与快照集身份。

Live 是请求级作用域，要求全部三项授权：

1. replay 血缘 lock 允许 live 执行；
2. 操作员以 `--allow-live` 启动 `paper-search serve`；
3. 请求显式设置 `mode: live`。

任一授权谓词缺失即拒绝 live。被禁止的 lock 无法启动 live-capable 服务。每个已授权 live 请求构造隔离的客户端、预算状态、捕获存储与应用服务；仅当捕获已记录、快照已封存、replay 血缘已写入、证据已校验且目录已原子发布后，才暴露成功响应。失败或取消只发布失败证据，绝不发布 complete 捕获。

## provider、快照与凭据

OpenAlex、Semantic Scholar 与配置的 LLM 均通过类型化适配器访问，返回数据、安全来源、用量、延迟、缓存状态、快照引用与脱敏错误码。Live 适配器只从已授权子进程环境接收凭据；replay 适配器接收 `DependencySnapshotReader`，不能回退到网络。

快照清单包含规范化的非敏感请求身份与精确响应哈希。原始响应字节与逐查询产物保持访问受控。公开 API 错误绝不暴露异常文本、请求头、凭据、本地路径或原始依赖载荷。

## API 与浏览器 UI

`paper-search serve` 暴露：

- `GET /health/live`：进程存活；
- `GET /health/ready`：缓存的模式/快照/依赖安全状态；
- `POST /v1/search`：规范化类型化请求；
- `/` 与打包静态资源：浏览器 UI。

UI 只向 `/v1/search` 提交请求，不接受文件系统路径、任意快照选择、凭据或 provider URL。它从规范化响应渲染选中论文、证据字段、安全诊断、用量、部分/回退状态、快照时间、配置哈希与 run ID。

## 正式评测与证据

`paper-search evaluate` 将规范化服务结果适配为有序的 execution、prediction、failure、business-result、usage、metric 与 gate 记录。`FormalRunWorkspace` 先写不完整证据，校验规范化字节与绑定，再原子发布 complete 或 failed 状态。

`paper-search verify-run` 是正式运行目录的机器判定谓词，校验精确输入 lock、实验身份、可选模块标志、数据集与策略绑定、快照证据、记录顺序/基数、聚合指标与终态发布状态；它不适用于服务端 smoke 型 live 捕获目录。`paper-search compare-replay` 比较 capture 与 replay 的规范化 `BusinessResultRecord` 字节。

验证尝试按内容寻址且不可撤销。恢复必须匹配归档 lock 字节与完整 manifest 绑定；中断不授权另一次尝试。

## 实验注册表

注册表只接受以下精确身份：

| 身份 | 构造的可选行为 |
|---|---|
| `main-baseline` | 无；固定单轮 |
| `embedding` | 仅嵌入排序 |
| `citation-expansion` | 仅引文扩展 |
| `llm-rerank` | 仅约束/LLM 重排 |
| `fixed-two-round` | 仅固定两轮协调器 |
| `adaptive-evolution` | 仅自适应演化协调器 |

Baseline 规划与有界多源路由是强制组合行为，不是可选消融标志。可选 Provider/LLM 阶段共享请求预算并保留 capture/replay 引用。受保护的执行与完整性失败绝不能被降级为普通可选阶段警告。

`configs/base.yaml` 保持 `main-baseline`；实现本身不晋升任何可选身份。

## 已验证与延迟状态

离线单元/集成/E2E 覆盖、合成正式 capture/replay 验证、真实浏览器 replay 验收，以及一次在显式网络/成本授权下的真实 live 浏览器捕获均已完成；验收记录保存在源码树外。真实 live 浏览器验收暴露并修复了两个 DashScope 兼容性缺陷（`json_object` 提示词与 thinking 读超时），修复位于 LLM 客户端。当前公开 Gate 0 报告仍为 blocked，因此生产 Gate 0、正式 dev/validation 证据、全部 provider 健康的 live 浏览器验收、可测质量/成本声明与 Gate 6 晋升证据保持延迟，等待其显式授权与前置条件。

源码证据检查点为 2026-08-03 的 `fcc0ff0`。可复现仓库证据由 `tests/e2e/test_dual_mode_serve.py`、`tests/integration/test_serve_process.py` 与 `tests/fixtures/formal_run/` 实现。浏览器验收记录有意保存在源码控制之外，fresh clone 无法独立取得。
