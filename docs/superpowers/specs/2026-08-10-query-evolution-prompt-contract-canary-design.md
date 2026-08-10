# Query Evolution 提示词契约与 Canary 设计

日期：2026-08-10
状态：已批准，待实施计划

## 1. 背景与结论

首次 55 查询有界探针完成了快照封存和禁网重放，capture/replay 业务哈希一致，但 55 条结果全部为 `integrity_failure`，OpenAlex 未被调用。离线检查原始响应后确认：模型大量回显输入 envelope，或返回缺少 `strategy`、字段名错误、`no_op_reason` 非枚举及额外字段的对象。

根因位于提示词契约绑定链路，而非 OpenAlex 可用性或 Query Evolution 检索假设：

- preflight 的 `--prompt-config` 只检查文件存在，未将所选文件传入 lock 构建；
- probe runner 创建 `LiveCaptureLLMAnalyzer` 时未绑定提示词正文；
- 即使绑定，adapter 也只向 `query_analyze` 传递提示词，`query_evolve` 会退化为通用的 JSON 指令；
- 当前提示词只有高层约束和模型名称，没有向模型给出完整字段结构与枚举。

因此先修复并验证提示词契约传递，再决定是否执行完整 55 查询探针。此次修复不改变 Query Evolution 算法、严格验证器、检索合并规则或 Gate A/B/C。

## 2. 目标与非目标

### 2.1 目标

1. 让 preflight 选定的 Query Evolution 提示词成为可核验的 lock 输入。
2. 让 `query_evolve` 在线请求实际收到该提示词生成的确定性 system message。
3. 通过离线测试证明请求契约正确，再用 3 条 LLM-only canary 阻断明显的批量失败。
4. canary 全部合格后，才允许以新 lock 执行一次完整 55 查询探针。
5. 保留原失败运行、快照和账本作为不可变诊断证据。

### 2.2 非目标

- 不放宽 schema、字段集合、枚举或机械约束；
- 不增加兼容解析、LLM 修复调用、规则 fallback 或提示词自动迭代；
- canary 不调用 OpenAlex，不读取 gold、identifier map 或 availability；
- 不调整候选合并、去重、过滤、RRF 或 Gate 阈值；
- 不在本设计内进行生产集成或正式 capture 晋级。

## 3. 方案选择

采用“修复绑定链路 + 3 条契约 canary + 完整 55 条探针”。

以下方案被否决：

- 直接重跑 55 条：未先证明请求契约恢复，可能重复浪费调用；
- 放宽解析或自动修复：会掩盖真实输出质量并破坏证据完整性；
- 只做离线模拟：只能证明代码接受合成响应，不能证明真实模型遵守契约。

Canary 只承担批量运行前的熔断职责，不用于估计总体成功率，也不替代完整探针。

## 4. 模块边界

### 4.1 提示词装载器

新增一个位于 LLM 层的公共、确定性提示词装载器，供正式应用 composition 和 probe runner 共用。它负责：

- 解析 lock 绑定的 YAML 字节；
- 校验名称、版本、instructions、策略枚举和 no-op 枚举；
- 生成唯一的 system message；
- 对缺失、额外类型或无效配置 fail closed。

probe runner 不导入 application composition 的私有函数，也不维护第二份提示词拼接逻辑。

### 4.2 Lock 绑定

新 lock 使用升级后的 schema，显式保存一个提示词 artifact binding，至少包含：

- 仓库内相对路径；
- SHA-256；
- prompt name；
- prompt version。

preflight 必须使用调用方传入的 `--prompt-config`，解析后写入 binding。run 在第一条网络请求前重新读取同一路径，确认路径受仓库根目录约束、文件类型合法、哈希和 name/version 与 lock 完全一致。任一不一致均停止，且不建立网络请求。

旧 lock 和旧运行只用于历史证据，不迁移、不覆盖，也不能用于新 canary 或完整运行。

### 4.3 在线分析器

`LiveCaptureLLMAnalyzer` 在实例绑定了提示词时，将该 system message 传给该实例承载的请求，不再按 `prompt_name == "query_analyze"` 静默丢弃。调用方仍负责为不同用途创建绑定到相应 artifact 的实例。

测试必须证明：

- `query_analyze` 现有行为不变；
- `query_evolve` 收到绑定后的完整 system message；
- 未绑定时仍使用现有通用 JSON 指令；
- 请求 identity 和 snapshot 机制不因 system message 传递而绕过 prompt artifact 哈希。

### 4.4 Query Evolution 输出契约

提示词必须显式描述唯一允许的顶层对象：

```json
{
  "subqueries": [
    {
      "text": "string",
      "source_facets": ["exact payload facet"],
      "strategy": "synonym | entity_alias | facet_combination | task_decomposition"
    }
  ],
  "no_op_reason": null
}
```

并同时给出合法 no-op 结构：

```json
{
  "subqueries": [],
  "no_op_reason": "insufficient_grounded_facets | no_novel_query"
}
```

约束保持为：

- 顶层只能有 `subqueries` 和 `no_op_reason`，不得返回 `payload`、`prompt_name` 或 Markdown wrapper；
- `subqueries` 为 0 至 2 条；
- 每条只能有 `text`、`source_facets`、`strategy`；
- `source_facets` 必须逐值复制 payload 中已有 facet；
- 非空 proposal 的 `no_op_reason` 必须为 `null`；空 proposal 必须使用固定 no-op 枚举；
- 只使用 payload 中的事实与 facet，不推断 gold、标签、新 venue、新年份或无关实体。

运行时仍以现有 Pydantic schema 和机械验证器为唯一判定标准；提示词示例不构成第二套宽松 schema。

## 5. Canary 设计

### 5.1 样本选择

Canary 从新 lock 的 55 个冻结目标查询中选择 3 条，但选择过程只读取冻结 business/execution 输入。对每条查询使用正式代码构建 `QueryEvolutionContext`，再按 canonical JSON 的 UTF-8 字节长度排序：

1. 最小值；
2. 排序后的中位值；
3. 最大值。

长度相同时按 query ID 字典序打破并列。三个位置去重；在当前固定 55 条队列中必须得到恰好 3 个不同 query ID，否则 preflight fail closed。

该规则覆盖较小、典型和较大输入结构，完全可复现，且不按 gold 或模型输出挑样。

### 5.2 执行边界

Canary 使用独立 run ID、输出目录和账本预留，只建立 3 个 `evolve` 逻辑操作：

- probe/canary run ID 只允许小写字母、数字和连字符，最长 64 字符；输出目录由固定前缀和已验证 run ID 唯一推导；
- canary lock 绑定来源运行 ID、来源哈希、当前 probe code hash、提示词 binding 和账本 checkpoint；
- 固定上限为 3 个 LLM 逻辑操作、每操作最多 3 次尝试、合计最多 9 次 retry-inclusive 请求，以及 600 秒全局超时；
- 读取用户已授权的 `.env` 中必要的 LLM 凭据；
- 最多执行现有每操作重试上限；
- 写入 LLM 请求/响应快照、逐条 outcome、usage 和账本终态；
- 不建立 `search-1`/`search-2` 预留，不创建 OpenAlex client，不读取 OpenAlex 凭据；
- 不读取 gold、identifier map 或 availability；
- 不把 canary 指标写入正式 Gate 结果。

### 5.3 晋级与停止条件

只有以下条件同时满足才允许生成新的完整探针 lock：

- 3 条均有终态记录；
- 3 条均通过现有严格 schema 和机械验证；
- 每条终态只能为 `generated` 或合法 `no_op`；
- 没有 dependency、integrity、accounting 或 snapshot failure；
- 所有账本预留均为终态，实际 usage 完整结算；
- sealed snapshot manifest 可读取且与 outcome refs 一致。

任何条件失败都停止。内建请求重试耗尽后不得自动修改提示词、放宽验证、补发修复请求、更换样本或启动第二轮 canary。网络或供应商故障单独记为 dependency failure，不被误判为契约失败；是否另行运行新的 canary 需新的明确决定。

## 6. 完整探针

Canary 晋级后才执行以下一次性流程：

1. 基于当前代码、当前提示词 artifact 和当前账本 checkpoint 重建独立的完整 probe lock；
2. 运行既有只读 readiness/preflight 检查；
3. 为 55 个查询预留 `evolve`、`search-1`、`search-2` 逻辑操作；
4. 执行 live capture；
5. 封存快照并禁网 replay；
6. 比较 capture/replay 业务哈希；
7. 仅在技术完整性成立后延迟读取 gold 和 identifier map，判定 Gate A/B/C。

完整探针继续遵守 2026-08-09 有界探针设计中的预算、隐私、合并、排序、证据和 Gate 契约。Canary 不降低 Gate A 的零完整性失败要求。

## 7. 证据文件

### 7.1 Canary 私有证据

写入新的、Git 忽略的目录，例如：

`runs/_diag_query_evolution_contract_canary_<run-id>/`

只保留：

- `canary.lock.json`：来源运行与哈希、probe code hash、提示词 binding、确定性样本选择、账本 checkpoint 和 3/9/600 固定上限；
- `outcomes.jsonl`：3 条终态、proposal、usage 和 snapshot refs；
- `snapshots/`：sealed dependency snapshot；
- `result.json`：晋级/停止结论和固定 reason code。

公开或可提交的汇总不得包含 query 文本、query ID、proposal 文本、原始响应、密钥或未经清洗的 provider request ID。

### 7.2 历史证据

现有 55 条失败运行、其快照、`result.json` 和账本终态保持原样。新流程不得覆写、复用其 run ID 或将其失败记录改写为成功。

## 8. 错误处理与固定结论

- 提示词路径、哈希、name/version 或配置结构不一致：网络前停止，记 `prompt_binding_failed`；
- canary lock、来源运行、来源哈希、probe code hash、账本 checkpoint 或固定上限不一致：网络前停止，记 `canary_preflight_failed`；
- canonical request 未包含绑定 system message：离线测试失败，不允许 canary；
- canary schema/机械约束失败：记 `contract_canary_failed`，停止；
- canary 依赖失败：记 `canary_dependency_failed`，停止，不评价契约；
- canary 账本或快照失败：分别记 `canary_accounting_failed` 或 `canary_snapshot_failed`，停止；
- canary 全局超时或操作者取消：记 `canary_cancelled`，停止；
- 完整探针 Gate A 失败：Gate B/C 为 `not_evaluated`；
- Gate B 失败：否决当前 Query Evolution 检索假设，不继续提示词变体、排序调整或正式 capture；
- Gate B 通过而 Gate C 失败：只记录召回与排序瓶颈，不进入正式 capture；
- Gate C 通过：只获得申请正式 capture 的资格，仍需单独授权。

## 9. 测试策略

按 TDD 实施，测试顺序保持最小但完整：

1. 提示词装载单元测试：有效配置、完整 schema/enums、无效配置和确定性输出；
2. lock/preflight 单元或集成测试：传入的 prompt path 确实被绑定，路径和哈希不一致在网络前失败；
3. analyzer 单元测试：`query_evolve` 接收绑定 system message，现有 `query_analyze` 行为不回归；
4. canary 选择单元测试：min/median/max、并列排序、去重和 gold-blind；
5. canary 端到端模拟测试：3 条成功、schema/dependency/账本/快照/超时失败、来源/代码/固定上限/checkpoint 漂移，并断言 OpenAlex 零调用；
6. 既有 query evolution probe 聚焦测试；
7. Ruff、mypy 和全量离线测试；
8. 离线 canonical request 审计通过后，才允许一次真实 canary。

不为 canary 重复实现完整 Gate A/B/C 测试；完整 Gate 继续由既有 probe 测试覆盖。

## 10. 实施与授权边界

实施阶段可以修改代码、配置、测试和设计对应文档，但在线动作分两级：

1. 真实 3 条 LLM-only canary 需要明确授权；它包含 3 个逻辑操作，最多 9 次 retry-inclusive 请求；
2. canary 晋级后的完整 55 条 DeepSeek/OpenAlex 探针需要再次明确授权。

任何阶段都不得打印、写入快照或提交 `.env` 中的密钥。旧的未跟踪账本和 `deliverables/` 不属于本设计的修改范围。

## 11. 验收标准

设计实施完成的最低标准为：

- lock 真正绑定调用方选择的提示词 artifact；
- `query_evolve` canonical request 包含完整且可核验的 system message；
- 现有严格验证器和 Gate 契约未被放宽；
- 3 条 canary 选择确定、gold-blind、OpenAlex 零调用；
- canary lock 绑定来源、代码、提示词、账本和 3/9/600 固定上限；
- canary 失败会停止，成功才允许建立新完整 lock；
- 所有聚焦测试、Ruff、mypy 和全量离线测试通过；
- 在线结果只依据实际快照、账本和 Gate 输出报告，不以人工解释替代机器判定。
