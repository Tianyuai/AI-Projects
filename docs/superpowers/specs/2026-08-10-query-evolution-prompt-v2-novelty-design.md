# Query Evolution Prompt v2 新颖性契约设计

日期：2026-08-10
状态：设计已批准；实施计划另文编写

## 1. 背景与结论

三查询 DeepSeek canary 的真实业务结果为两条 `generated` 和一条 `integrity_failure`。账本误分类问题修复后，失败应归类为 `contract_canary_failed`，仍不满足晋级条件。

对封存快照进行只读离线诊断后确认：失败响应包含两条彼此不同的子查询，但它们分别与两个既有 seed query 在 OpenAlex canonicalization 后完全相同。模型满足了 JSON schema、字段枚举和 facet 来源要求，却没有产生新的查询。严格校验器按设计拒绝了该响应。

根因不是校验器过严，也不是账本、OpenAlex 或重试路径故障，而是 prompt 只要求“complementary”查询，没有明确告诉模型：

- 输出不得复制 `original_query` 或任一 `seed_subqueries[*].text`；
- 大小写、空白或标点变化不构成新查询；
- 无法基于现有 facets 形成合格新查询时，应返回 `no_novel_query`，而不是复述已有查询。

本设计采用单变量修订：升级 `query_evolve` prompt 至 v2，补齐新颖性与 no-op 指令；保持 schema、校验器、runner、预算和晋级门槛不变。

## 2. 目标与非目标

### 2.1 目标

1. 让真实模型明确区分“合法 JSON”与“合格的新查询”。
2. 禁止输出与 original、任一 seed 或同批已生成项在 canonicalization 后重复的查询。
3. 当没有合格新查询时，引导模型使用既有 `no_novel_query` no-op 结果。
4. 通过 prompt 版本和哈希变化使旧 lock 不能授权新的 canary。
5. 以离线测试证明新 prompt 被确定性装载，同时证明严格校验行为未被放宽。

### 2.2 非目标

- 不增加自动去重、自动修复、补发请求或规则 fallback；
- 不把重复结果静默转换为 no-op；
- 不修改 `QueryEvolutionProposal` schema、strategy/no-op 枚举或机械校验规则；
- 不修改样本选择、重试次数、超时、预算、账本、快照或晋级条件；
- 不读取 `.env`，不发送网络请求，不重建 source/canary lock，不运行 live canary 或 55-query probe；
- 不改写封存 canary 的 `result.json`、outcomes、快照或账本记录。

## 3. 方案选择

### 3.1 采用：Prompt v2 单变量修订

继续使用现有 context 字段和严格 validator，只在绑定的 prompt artifact 中加入明确的新颖性规则，并将版本升级为 `query-evolve-v2`。

该方案直接修复模型指令缺口，改动面最小，也保留真实模型不遵守契约时的可观察失败。

### 3.2 否决：新增显式 forbidden-query payload

把 original 和 seed 复制到新的禁止列表会扩张 `QueryEvolutionContext`、锁定代码哈希和测试面。现有 payload 已包含这些字段，缺少的是明确指令，不是数据，因此该方案没有必要。

### 3.3 否决：自动删除重复项或降级为 no-op

后处理可提高表面成功率，但会把模型契约失败改写为成功或 no-op，削弱证据完整性，并可能掩盖 prompt 对完整 55-query 队列仍不稳定的问题。

## 4. Prompt v2 契约

`configs/prompts/query_evolve.yaml` 保持文件名、response model、temperature、strategy 枚举和 no-op 枚举不变，将版本升级为 `query-evolve-v2`，并增加以下语义：

1. 每个生成的 `text` 必须与 `original_query`、所有 `seed_subqueries[*].text` 以及同一响应中较早的生成项不同。
2. 大小写、Unicode 表现、连续空白或标点的机械变化不构成新查询；最终机器判定唯一以 `src/paper_search/evolution/query_evolution.py` 的 `_canonical_query()` 为准，该函数先执行 NFKC 和空白折叠，再调用 `canonicalize_openalex_search_query()`。
3. 模型应在返回前检查每个候选是否满足上述新颖性要求。
4. 仍允许返回零至两条子查询。只有一条合格时返回该一条；某个候选重复时不应输出该候选；没有候选合格时返回：

```json
{"subqueries": [], "no_op_reason": "no_novel_query"}
```

`no_novel_query` 专用于已有 facts/facets 足够但无法形成新查询；`insufficient_grounded_facets` 继续用于 payload 缺少可支撑查询的 facts/facets。空 `subqueries` 必须携带其中一个既有 no-op reason，非空 `subqueries` 的 `no_op_reason` 必须为 `null`。现有 JSON 示例、字段白名单、facet 精确复制和禁止推断规则全部保留。

## 5. 模块边界与数据流

### 5.1 Prompt artifact

`configs/prompts/query_evolve.yaml` 是本次唯一修改的运行时行为 artifact。未来的新 source lock 必须绑定它的路径、`query-evolve-v2` 版本和新 SHA-256；context payload 仍由现有代码构建，不改变字段或内容。

### 5.2 Prompt 装载器

`src/paper_search/llm/prompt_artifacts.py` 对 `query_evolve` 只接受 `query-evolve-v2`。artifact 的字段结构由现有 `PromptArtifact` Pydantic schema 校验；`query_analyze` 行为不变。

### 5.3 Query Evolution validator

`src/paper_search/evolution/query_evolution.py` 不修改。它继续以当前 canonicalization 比较 original、seed 和同批生成项，并将重复项判为 `integrity_failure`。Prompt v2 不能替代或绕过该机器判定。

### 5.4 执行数据流

离线实现后的未来执行路径保持为：

1. preflight 读取并校验 prompt v2；
2. source/canary lock 绑定 prompt 路径、版本和哈希；
3. `LiveCaptureLLMAnalyzer` 将确定性 system message 与既有 context payload 发给模型；
4. validator 对模型响应执行原有严格 schema 与机械校验；
5. 合法的新查询记为 `generated`；空结果只有在携带合法 no-op reason 时记为 `no_op`；重复仍记为 `integrity_failure`。

本设计只实现支撑上述流程的离线代码与测试，不实际运行 preflight，也不创建任何新 lock。

## 6. 修改范围、历史证据与错误处理

- 实施只允许修改 `configs/prompts/query_evolve.yaml`、`src/paper_search/llm/prompt_artifacts.py`、`tests/unit/test_prompt_artifacts.py` 和 `tests/unit/test_query_evolution.py`；若旧 lock 前置失败断言无法在现有测试中表达，可同时修改 `tests/integration/test_query_evolution_probe.py`。实施计划另存于 `docs/superpowers/plans/`。
- 封存目录 `runs/_diag_query_evolution_contract-canary-20260810/` 保持原样；聚合诊断只用于只读验收输出，不新增证据文件，也不在可提交文档中写入冻结 query 文本、query ID、proposal 文本或 provider request ID。
- 历史响应在当前 validator 下仍应失败。这是预期的特征证明，不应通过修改历史数据变成成功。
- `query-evolve-v1` lock 只作为历史证据，不得用于新 canary。
- 离线 fixture 必须证明 prompt 路径、版本或哈希不匹配时，在网络调用和账本 reservation 前停止；artifact 字段结构错误则由 `PromptArtifact` 解析直接拒绝。
- 新 prompt 不能保证真实模型一定遵守契约；真实遵守情况只能由另行授权的新三查询 live canary 判断。

## 7. TDD 与离线测试策略

实施顺序保持最小但完整：

1. 先修改 prompt artifact 测试，要求 `query-evolve-v2`，并精确断言最终渲染的 system message 包含 original/seed/同批排重、机械变化不算新查询、部分成功和两个 no-op reason 的规则；观察测试在现有代码上按预期失败。
2. 修改 prompt YAML 和 `PromptArtifact` 的 query-evolve 版本约束，使测试转绿。
3. 增加或补齐 validator 特征测试，分别证明 original 重复、seed 重复和同批重复仍被拒绝；这些测试不得要求修改 validator。
4. 使用临时 lock/ledger 和网络 guard 证明 v1 或哈希不匹配在网络调用与 reservation 前失败，不创建仓库内的新 lock。
5. 运行 prompt artifact、Query Evolution 和 canary/probe 聚焦测试，再运行受影响文件 Ruff、`mypy src scripts/probe_query_evolution.py`、全量离线 pytest 和 `git diff --check`。
6. 只读复核封存失败响应，确认聚合事实仍为“两个输出分别匹配既有 seed”，且封存目录无 diff。实施前后分别记录 `data/budget_ledger.sqlite3` 的 SHA-256 并确认不变；不读取或修改 `deliverables/`。

测试不得读取真实 `.env`，不得构造真实 provider client，不得发送网络请求。若任何测试只能通过放宽 validator、改变 schema 或引入自动修复，应停止并重新审议设计。

## 8. 验收标准

- `query_evolve` artifact 版本为 `query-evolve-v2`，最终渲染的 system message 完整包含第 4 节规则；
- 临时 fixture 证明旧版本或错误哈希在网络调用和 reservation 前失效；
- original、seed 和同批重复仍被测试证明为 `integrity_failure`；
- 聚焦测试、Ruff、mypy、全量离线测试和 diff 检查通过；
- 封存目录无 diff，用户 ledger 的实施前后 SHA-256 一致；
- 没有 `.env` 读取、真实网络请求、真实 reservation、仓库内新 lock 或 live run。

## 9. 完成边界

本设计的实施阶段仅包含第 6 节列出的 prompt、装载版本约束和离线测试文件。schema、validator、预算、重试、超时、样本选择和晋级条件保持不变；不读取 `.env`，不联网，不创建仓库内 lock，也不运行 live canary。实施完成并验证通过后，重建 source/canary lock 与三查询 live canary 仍需单独决定和明确授权。
