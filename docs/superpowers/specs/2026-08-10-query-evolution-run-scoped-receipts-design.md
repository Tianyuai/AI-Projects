# Query Evolution Run-Scoped Receipts 修复设计

日期：2026-08-10
状态：待用户复核

## 1. 背景与根因

三查询 DeepSeek canary 已完成 3 次 LLM 调用并封存 3 份快照，结果为 2 条 `generated` 和 1 条 `integrity_failure`。当前 run 的账本回读为 3 条 `settled`、0 条 `reserved`，逐条用量之和与 `result.json` 的 3 次 LLM、2,541 input tokens、345 output tokens、3,877 ms 和 `0.003231` CNY 完全一致。然而 `result.json` 被写为 `canary_accounting_failed`。

根因是调用方误解了 `SQLiteBudgetLedger.report(run_id)` 的返回契约：

- `reserved` 和 `actual` 是指定 run 的聚合；
- `receipts`、`project_receipt_count` 和 `project_receipts_sha256` 则故意覆盖项目完整历史，以支持项目级 checkpoint 和审计。

Query Evolution probe 将 `report.receipts` 直接当成当前 run 的回执集合。真实账本在 canary 前已有 165 条历史回执；canary 增加 3 条后，finalizer 看见 168 条并以“数量不等于 3”为由抛出 `CanaryAccountingError`。在全新临时账本中只有 3 条回执，所以现有测试未暴露该问题。

离线最小复现已确认：账本包含 1 条历史终态回执和 3 条当前 canary 终态回执时，当前 finalizer 稳定误报 accounting failure。

## 2. 决策

采用调用方 run-scoped 过滤，不改变账本报告的项目级语义。

在 `scripts/probe_query_evolution.py` 增加单一私有 helper：读取 `ledger.report(run_id).receipts` 后，仅返回 `receipt.run_id == run_id` 的回执。所有 Query Evolution 的 run-local 数量、状态、恢复和晋级判断必须使用该 helper；项目预算、项目 checkpoint 和历史根哈希仍使用原始 `LedgerReport` 字段。

不采用以下方案：

- 修改 `LedgerReport.receipts` 为 run-scoped：会破坏既有项目历史审计契约和测试；
- 给账本增加新的公共 API：可以实现目标，但扩大核心账本接口，超出本次最小修复需要；
- 只修改 canary finalizer：会遗漏 canary 恢复、晋级判断和 full probe 恢复中的同类错误。

## 3. 目标与非目标

### 3.1 目标

1. 当前 canary run 的回执数量和终态检查只查看当前 run。
2. 历史回执不影响 canary 的恢复、失败原因或晋级判定。
3. full probe 的预留恢复路径同样只查看当前 probe run。
4. 当前封存 canary 的离线重新分类得到 `contract_canary_failed`，同时保持 `promoted=false`。
5. 保留项目级账本 checkpoint、预算上限和完整历史审计。

### 3.2 非目标

- 不修改 `SQLiteBudgetLedger.report()` 或其数据模型；
- 不重写已封存的 `result.json`、outcomes、快照或账本记录；
- 不放宽 Query Evolution schema、重复文本校验或晋级条件；
- 不修改提示词、样本选择、预算、重试或超时上限；
- 不读取 `.env`，不运行 live canary，不执行 55-query probe。

## 4. 模块与数据流

### 4.1 Run-scoped helper

helper 只承担一项职责：从项目级 receipt 历史中选择指定 run 的回执。它不计算预算、不改变账本、不恢复或终结 reservation。

调用约束：

- `ledger.report(run_id)` 报告 run 不存在时，继续保留现有 `LedgerReservationError`；
- 返回顺序保持账本创建顺序；
- 不接受 query ID、operation 或状态作为额外过滤条件；
- 调用方继续负责验证预期数量、私有 operation ID、状态和 actual usage。

### 4.2 Canary 预留与恢复

`reserve_canary_operations` 只检查当前 canary run：

- 当前 run 不存在：创建 3 个 `evolve` reservation；
- 恰有 3 个预期私有 operation ID，且全部 `reserved`、actual 为空：恢复；
- 当前 run 存在不完整、额外或终态集合：fail closed；
- 其他 run 的任何历史回执均不参与上述判断。

### 4.3 Canary 终态与晋级

`_finalize_canary_reservations` 的清理和最终验证只使用当前 canary run 的 3 条回执。晋级判断也只要求当前 run：

- 恰有 3 条回执；
- 每条为 `settled` 或 `failed`；
- 每条 actual usage 存在；
- outcomes、快照 manifest 和严格契约同时满足原有条件。

项目历史仍参与项目总成本、软停止、硬上限和 checkpoint，不会因 run 过滤而绕过预算。

### 4.4 Full probe 恢复

`reserve_probe_operations` 的恢复判断同样改用当前 probe run 的回执。历史 run 不得导致恢复失败，也不得被误恢复为当前 run。除该选择边界外，55-query probe 的 reservation 数量、operation 集合和 fail-closed 规则不变。

## 5. 错误处理与历史证据

- 当前 run 的回执缺失、数量错误、operation ID 不匹配或存在不允许的终态：保持 accounting/preflight failure；
- 项目历史中存在其他 run：正常情况，不构成当前 run accounting failure；
- 当前 run 的账本完整，但 outcome 存在 `integrity_failure`：固定 reason 为 `contract_canary_failed`；
- 当前封存 canary 不做原位修复。其 `result.json` 继续作为当时代码输出保留，诊断文档说明正确离线分类；
- 修复后不自动创建新 lock，也不获得任何 live 重跑授权。

## 6. 测试策略

按 TDD 实施，先增加会在当前代码上失败的历史账本测试：

1. 在临时账本先写入其他 run 的终态回执，再运行三条全部合格的 mocked canary；预期 `promoted=true`，当前 run 恰有 3 条终态回执，历史回执不变。
2. 在同样的历史账本前置条件下运行 2 条 `generated` 加 1 条 `integrity_failure`；预期 reason 为 `contract_canary_failed`，不是 accounting failure。
3. 先为当前 canary run 建立恰好 3 条预期的 `reserved` reservation，并同时保留其他 run 的历史回执；再次调用预留函数时必须恢复原 3 条 reservation，不创建新条目，也不受历史回执影响。
4. accounting failure 和 cancellation 清理测试加入历史回执，确认只验证当前 run 的 3 条终态。
5. full probe reservation 恢复测试加入其他 run 历史，确认只恢复当前 probe run。
6. 保留账本单元测试对 `LedgerReport.receipts` 项目完整历史的断言，证明本修复没有改变公共契约。

验证顺序：

- 先运行新增测试并观察预期 RED；
- 实现最小 helper 和调用点替换后观察 GREEN；
- 运行 Query Evolution probe 聚焦测试；
- 运行账本单元测试、Ruff、mypy 和全量离线测试；
- 对已封存 canary 的 outcomes 运行只读分类，确认 `contract_canary_failed`，不改写证据。

所有网络构造均由现有 MockTransport/guard 阻断；测试不得读取真实 `.env`。

## 7. 验收标准

- 具有历史回执的账本不再使当前 canary 误报 accounting failure；
- 当前 run 的 3 条回执仍必须全部终态且 actual 完整；
- 已封存 canary 的离线分类为 `contract_canary_failed`，仍不晋级；
- full probe 恢复不读取或恢复其他 run 的 reservation；
- `LedgerReport.receipts`、项目 checkpoint 和预算契约保持不变；
- 聚焦测试、账本测试、Ruff、mypy 和全量离线测试通过；
- 不发生 `.env` 读取、网络请求、live 重跑或历史证据改写。

## 8. 实施边界

预计只修改：

- `scripts/probe_query_evolution.py`；
- `tests/integration/test_query_evolution_probe.py`。

若测试证明必须修改核心账本 API、生产预算语义或证据 schema，应停止并重新审议设计，不在本修复中顺带扩张。
