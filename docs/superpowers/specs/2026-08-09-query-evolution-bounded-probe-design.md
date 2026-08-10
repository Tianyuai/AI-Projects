# Query Evolution 有限范围探针设计

日期：2026-08-09
状态：已复核，待实施
范围：冻结 dev 上的诊断探针，不是正式 live capture 或生产改动

## 1. 决策摘要

当前完整瓶颈诊断确认：134/134 个唯一 gold work 可在线精确解析，139 个查询–论文关联中有 125 个在现有流水线中未被检索到。后续唯一推荐方向是一个低成本、可证伪的 Query Evolution bounded probe。

本设计采用以下决策：

- 只覆盖 55 个至少存在一个“可用但未检索到”gold 关联的 dev 查询；
- 复用冻结主基线的生产 `QuerySpec`、首轮子查询和候选结果，不重新运行首轮分析或检索；
- 每个查询只允许一次 LLM 演化操作，最多生成两条新的 OpenAlex 查询；
- gold 文件、标识符映射和相关性标签只用于队列选择与封存后的评分；生成上下文不得因 gold 内容增加、删除、排序或标注任何字段。冻结检索证据中的候选标题即使碰巧对应 gold，也只作为无标签候选证据使用，不视为泄漏；
- 当前正式对照是 `main-baseline`，其中所有可选组件（包括 title candidates）均关闭；
- 探针不改动生产 `EvolutionSearchOrchestrator`、实验注册、ablation 配置、candidate lock 或正式闭环；
- Gate B 只证明检索假设产生真实信号；只有 Gate C 通过才允许申请后续正式 capture；
- 一次探针只测试这一版 LLM 生成策略，不在同轮尝试规则兜底、多个提示词或多个阈值。

## 2. 现状与问题定义

### 2.1 冻结主基线

探针绑定以下当前证据，不使用同名运行的其他历史字节：

- capture run：`dev-20260809T061903Z-9bd861e90299`；
- source Git SHA：`45ef8749210c1ec6fcbfeb9b64b911f3ea4b0d55`；
- experiment：`main-baseline`；
- config hash：`sha256:9bd861e902999e10df286a406cac4f83c38bb7c038099ab256e8c8a67f4dbd22`；
- business results：`84e801f49c040df1db566c8144df5a71ef9b08088657330d0f865f8535d6ea3f`；
- executions：`a6aaea1164f9e41d82bb889c61caf3efe1aa81eb12e4e30d88e6fc418213b99b`；
- run record：`58612fce63b519564d1e32885f0625098fdfe7a3f272ea2ab2bad9238cd84335`；
- gates：`d22e02e63d43af083fee8d226ea7c5a68bac38c15239a2935f7e00908a4428be`；
- snapshot manifest：`sha256:0f3d66f8ff9434a80094395876ea25304c894d4f77f864b01dbc4b61a830b287`；
- dev gold：`24009cf03ad069131793b9a190024e239082277bd0e48149a1efbbbb7978e215`；
- identifier map：`6ea6dbcd20a3f572d9f0dd0a0eef938ff01773db792401adf7fde1e489396e82`；
- availability evidence：`docs/evidence/gold-bottleneck-attribution-2026-08-09-doi-contract-retry3.json`，SHA-256 为 `3f445486d5cf590f3f11a51930153a45916023880e856def379e0f01d053ad04`，schema 与聚合必须继续证明 134/134 唯一 gold work 可用。

基线必须精确重建 60/60 个查询的完整有序 Top-50 序列，总输出数为 2910。之前的标题保留报告绑定另一组 business/execution 哈希和 2908 条结果，不能与本探针证据混用。

### 2.2 当前指标和损失分布

| 指标 | 当前值 |
| --- | ---: |
| 候选池 exact gold | 14 / 139 |
| Top-50 exact gold | 8 / 139 |
| 未检索到的关联 | 125 / 139 |
| Macro F1 | 0.005094687447628624 |
| Macro Recall | 0.07916666666666666 |
| Micro Recall | 0.05755395683453238 |
| Recall@5 | 0.020833333333333332 |
| Recall@10 | 0.0375 |
| Recall@20 | 0.04583333333333333 |
| 输出总数 | 2910 |

MRR 和 NDCG 不从旧标题实验复制；探针评估器必须用相同输入同时重算基线和变体。

### 2.3 旧 `fixed_two_round` 为什么不能复用

现有实现不是干净的 Query Evolution 对照，因为它：

- 在多轮包装器中使用 `rule_fallback`，没有复用生产 DeepSeek `QuerySpec`；
- 让每轮生产编排器重新分析查询，重复消耗分析预算；
- 只使用规则式约束拼接生成第二轮查询；
- 第二轮成本估计全部为零；
- 其独立实验身份与主基线组件组合不同。

因此本探针先在生产编排之外验证生成假设，Gate C 未通过前不修造正式多轮路径。

## 3. 假设与可证伪结果

### 3.1 核心假设

在生成器不读取 gold 数据源、标识符映射或相关性标签，也不根据 gold 改变 payload 的前提下，向 LLM 提供原始查询、生产 `QuerySpec`、首轮子查询和有限的无标签候选标题摘要，可以生成最多两条互补 OpenAlex 查询，找回至少一个当前属于 `not_retrieved` 的 exact gold 关联，同时保留现有候选池与 Top-50 gold。

### 3.2 结论分级

- Gate B 失败：当前 Query Evolution 假设被否决，不进入排序调整或正式 capture；
- Gate B 通过而 Gate C 失败：确认存在召回信号，但增量或最终排序不足，不进入正式 capture；
- Gate C 通过：只获得申请正式 capture 的资格，不代表生产晋级；
- 后续生产晋级仍必须满足三次运行、bootstrap 和 validation 等既有 promotion gate。

## 4. 范围与非目标

### 4.1 本探针包含

- 冻结主基线的无网络预检和精确重建；
- 55 个目标查询的单次 LLM Query Evolution；
- 最多两条新增 OpenAlex 查询；
- 新候选与冻结基线候选的离线合并、过滤、融合和排序；
- 聚合指标、预算、账本、来源和快照证据；
- 新增依赖响应的零网络重放及规范化结果比较。

### 4.2 本探针不包含

- readiness、candidate lock 重建或正式 capture/replay/compare；
- production composition、API、UI 或默认行为变更；
- `configs/ablations.yaml` 或正式实验注册变更；
- title candidates、规则式 Query Rewrite、LLM rerank 或其他模块组合；
- validation 数据集运行；
- 使用 gold 标题或标识符构造搜索词；
- 同一轮中的提示词搜索、阈值搜索或失败后的替代策略。

## 5. 数据流

### 5.1 冻结预检

在线请求前必须：

1. 验证第 2.1 节全部输入哈希、运行状态、Gate 和实验身份；
2. 验证 60 个 query set 与冻结 gold 完全一致；
3. 从 `business-results.jsonl` 读取生产 `QuerySpec` 和首轮 `SearchPlan`；
4. 从已校验快照重建 OpenAlex 候选元数据；
5. 使用现有去重、过滤、融合和排序逻辑精确重建 60/60 有序输出；
6. 要求总输出为 2910、候选池 exact gold 为 14、Top-50 exact gold 为 8；
7. 由 `preflight` 使用 gold 和已冻结 availability evidence 筛选 55 个目标查询；队列顺序严格沿冻结 60-query 顺序，只把 query ID 写入 `probe.lock.json`；
8. 计算逐逻辑操作和全批次最坏情况预算，锁定项目账本 checkpoint，并写入全部来源哈希、availability hash、队列、prompt/config/model、预算与调用上限；另对按 POSIX 路径排序的全部已跟踪 `src/**/*.py` 与 `scripts/probe_query_evolution.py` 计算 `probe_code_sha256`，每项编码为 `UTF-8 path + NUL + decimal byte length + NUL + raw bytes` 后顺序拼接；`lock_sha256` 对省略该字段后的规范化 lock JSON 计算，两个字段格式均为 `sha256:<64 lowercase hex>`；
9. `run --lock <path>` 只消费并验证现有 lock，不重新选择队列；它在前置检查、capture 和 replay 阶段不打开 availability、gold 或 identifier map，只重复冻结运行产物与 snapshot manifest 哈希、`probe_code_sha256`、60/60 有序基线、2910 总数、prompt/config/model、lock 自身哈希和账本 checkpoint 等无 gold 检查，完成全部逻辑槽预留后才允许第一条网络请求；
10. 完整技术运行在快照封存并完成禁网重放且 `capture_replay_match=matched` 后，评估层才加载命令行提供的 gold 与 identifier map，先核对 lock 中的哈希和 14/8 基线，再评分；技术失败或 replay mismatch 均不调用该 loader。任何前置检查或预留失败都在零网络请求状态停止。

基线重建的完整有序序列一致性比单纯源文件代码哈希更强；只要当前逻辑无法精确重建，就不得继续。

### 5.2 无泄漏生成上下文

每个查询的 LLM payload 只包含：

- 原始查询和生产 `QuerySpec`；
- 首轮 3–5 条子查询；
- 首轮 OpenAlex 有序流按 canonical ID 去重、但尚未做 `QuerySpec` 后过滤和 Top-50 截断时的候选总数；
- 冻结 Top-50 有序序列中最靠前的 10 个非空去重候选标题；标题先做 Unicode NFKC 和空白折叠，再按 `casefold()` 去重并保留首次出现顺序；
- 确定性 facet 序列：先放规范化后的 `QuerySpec.original_query`、`research_goal`，再按 `topics`、`methods`、`tasks`、`datasets`、`domains`、`venues`、`must_have`、`should_have` 字段顺序追加，最后按首轮子查询顺序追加 `target_constraints`；全部做 Unicode NFKC 和空白折叠，按 `casefold()` 去重并保留首次出现顺序；
- 固定生成说明和输出 schema。

不提供摘要全文、query ID、gold ID、gold 标签、命中状态、gold 缺失数量或 `SearchPlan.inherited_hard_filters`。冻结候选标题不经过 gold 比对或过滤；执行层单独持有并原样传递 `inherited_hard_filters`。

34/60 个当前 `QuerySpec` 没有结构化强约束，因此“无强约束”不得被解释为“覆盖完整”。原始查询、首轮子查询和用户陈述 facet 始终保留为生成依据。

### 5.3 提案结构和确定性校验

生成输出是严格 JSON：

- `subqueries`：0–2 个对象；
- 每个对象包含 `text`、`source_facets` 和固定枚举的 `strategy`；
- `no_op_reason` 始终是必填字段：`subqueries` 非空时必须为 `null`，为空时必须是固定枚举值。

校验规则：

- 每条生成文本先做 Unicode NFKC，再调用生产 OpenAlex 使用的同一公开查询 canonicalizer；canonicalizer 删除 `?`、`*` 并折叠空白，结果不得为空，也不得重复原始查询、首轮子查询或同批提案；
- `source_facets` 必须逐项来自输入 facet 集；
- prompt policy 禁止引入与输入 facet 无关的新实体、venue 或虚构事实；这是受记录和审计的生成约束，不伪装成无需词表或第二语义模型即可证明的确定性断言；
- 确定性校验只负责机械可判定条件：严格 schema、`source_facets` 成员关系、规范化去重、长度/字符约束，以及生成文本中显式四位年份不得冲突于冻结年份范围；
- 执行层必须把冻结 `SearchPlan.inherited_hard_filters` 原样传给 OpenAlex，并继续使用不变的 `QuerySpec` 做后过滤；生成文本不要求逐字重复这些条件；
- 查询必须满足固定长度、字符和 canonical 非空约束；
- 最多保留两条，顺序由模型输出决定，不做第二轮选择调参；
- 非法 JSON、非法字段或非法硬约束不是合法 `no_op`，而是完整性失败；
- 不使用规则式通用改写兜底，也不发起 LLM repair 调用。

### 5.4 有限在线检索

- 每个查询最多一次逻辑 LLM 生成操作；
- 每个查询最多两次逻辑 OpenAlex 搜索操作；
- 每条搜索最多返回 50 个结果；
- 使用当前 timeout、retry、pricing、credential sanitization 和 usage settlement 契约；
- 当前 retry 上限为每个逻辑操作最多 3 次 HTTP 尝试；
- 因此全批次逻辑上限为 55 次 LLM 生成和 110 次 OpenAlex 搜索，HTTP/usage 尝试的最坏上限分别为 165 和 330；
- 查询按 `probe.lock.json` 固定顺序串行执行，全批次硬耗时上限为 3600 秒；到达上限视为操作者取消，完成在途调用的 fail-close、账本终态和快照密封后停止；
- 所有重试都计入实际 usage 和账本，不能把逻辑操作数误当成实际调用数；
- 全批次维护一个只存在于 runner 内的 canonical-request memo：LLM key 绑定模型、endpoint、prompt version/name、payload 和 prompt artifact hash；OpenAlex key 绑定公开 canonicalizer 的结果、冻结 filters、limit 和 adapter。重复 key 不再次发起网络请求；它复制首次结果的 data、errors 和 snapshot provenance，生成当前槽专用的 `ProviderResult(cache_hit=True, usage=0, latency_ms=0)`，释放 request reservation，并以零实际 usage 终结持久账本 receipt，禁止重复累计首次调用的 usage；
- 一个不可恢复的生成、依赖、结算或证据失败发生后，封存当前失败并停止调度后续查询。

### 5.5 合并与评分

新增 OpenAlex 结果加入对应查询的冻结候选池后，必须复用现有：

1. canonical ID 去重；
2. 硬过滤；
3. provider 结果合并与融合；
4. 当前主基线排序；
5. Top-50 截断。

当前主基线只有 OpenAlex 单一检索来源，因此本探针必须先保留冻结首轮 OpenAlex 有序流，再按 proposal 顺序依次追加 `search-1`、`search-2` 结果，并以 canonical ID 首次出现为准去重；合并后仍作为同一个 `openalex` 来源进入现有 RRF。不得把每条新增查询伪装成新的 provider/source，也不得为新增结果设置权重或保留槽。这个固定追加规则让 Gate B 测召回、Gate C 同时暴露现有排序是否能利用召回增量，而不会混入第二个实验变量。

55 查询以外的 5 个查询直接保留冻结有序输出，并要求字节级不变。在线执行层只读取并验证 `probe.lock.json` 与冻结运行；gold 和 identifier map 路径只交给封存后的 deferred loader，网络阶段不得打开它们。

### 5.6 零网络重放

新增 LLM 和 OpenAlex 响应通过现有 dependency snapshot v2 存储。只有 55 个查询均以 `generated` 或 `no_op` 完成、账本与快照均完整时，在线探针结束后才立即：

1. 封存快照清单；
2. 使用 replay adapters 在禁网条件下重新解码；
3. 重建每查询的规范化生成、候选和 Top-50 投影；
4. 比较在线与离线规范化业务哈希；
5. 只有哈希相同才允许 Gate A 通过。

业务哈希固定覆盖一个私有 `ReplayComparableProbe`：lock hash、冻结 query 顺序，以及每查询的 query ID、终态、规范化 proposal、按 `search-1`/`search-2` 顺序排列的规范化 `Paper` 数据和稳定错误字段（provider、code、retryable）、过滤后接收/拒绝的 canonical ID 顺序、候选 canonical ID 顺序和 Top-50 canonical ID 顺序。它不包含时间、usage、headers、provider request ID、snapshot ref 或文件路径。规范化 JSON 使用排序 key、紧凑分隔符、UTF-8、`allow_nan=False` 和末尾换行。`ReplayTrace` 只含有序查询/操作、canonical request identity、snapshot refs 和终态，不含在线 proposal、Paper、候选或 Top-50；replay 只能把它用于定位证据和验证调度，`generated`/`no_op` 终态及其余业务字段必须从快照重新解析并与 trace 核对。

每查询终态只允许 `generated`、`no_op`、`integrity_failure`、`dependency_failure`、`accounting_failure`、`snapshot_failure`、`cancelled`、`not_scheduled`。首次不可恢复失败之后尚未执行的查询写为 `not_scheduled`；它们不是缺失记录。`capture_replay_match` 只允许 `matched`、`mismatched`、`not_evaluated`。技术失败运行封存现有证据，但不伪造缺失操作的 replay：`replay_business_sha256=null`、`capture_replay_match=not_evaluated`，Gate A 直接失败。Gate 状态只允许 `passed`、`failed`、`not_evaluated`：Gate A 失败时 B/C 为 `not_evaluated`，Gate B 失败时 C 为 `not_evaluated`。固定运行原因只保留 `preflight_failed`、`generation_failed`、`dependency_failed`、`accounting_failed`、`snapshot_failed`、`replay_mismatch`、`cancelled`、`gate_b_failed`、`gate_c_failed`。

终态映射固定为：LLM schema/机械约束失败是 `integrity_failure`；请求级错误或重试耗尽是 `dependency_failure`；controller、usage 或 ledger 不一致是 `accounting_failure`；snapshot 写入或密封失败是 `snapshot_failure`；全局超时或操作者取消是 `cancelled`。证据隐私或 replay 哈希问题发生在查询完成后，只作为运行原因使 Gate A 失败，不回写已经确定的查询终态。

该重放是诊断证据检查，不是正式 replay run，也不生成 candidate lock。

## 6. 模块边界

### 6.1 `configs/prompts/query_evolve.yaml`

固定：

- `version: query-evolve-v1`；
- temperature 0；
- 允许的策略和 `no_op_reason`；
- 严格 JSON 输出说明；
- `no_op_reason` 始终存在，非空提案时为 `null`；
- 最多两条查询；
- 禁止 gold、未陈述硬约束和虚构事实。

提示词说明作为 payload 中的显式字段传入，与其 SHA-256 一起进入 snapshot identity。使用独立 `query-evolve-v1` LLM client/analyzer 实例，不改变生产 `query-analyze-v1` 行为。

### 6.2 `src/paper_search/evolution/query_evolution.py`

负责：

- `QueryEvolutionContext`、`EvolutionSubquery`、`QueryEvolutionProposal` 等严格模型；
- 无 gold 的 payload 构造；
- LLM 调用协调；
- 输出解析、规范化、去重和机械可判定的硬约束校验；
- 合法 `no_op` 与失败分类；
- 生成诊断和 snapshot refs。

该模块不加载 gold、identifier map、运行目录或公开报告，也不修改现有 `EvolutionCoordinator`。

生产 OpenAlex 的查询 canonicalizer 改为公开、行为不变的纯函数；现有检索适配器和本模块共同复用它。本模块先执行 Unicode NFKC，再调用该函数，避免验证、去重与真实请求 identity 分叉。

### 6.3 `src/paper_search/evaluation/query_evolution_probe.py`

负责纯评估逻辑：

- 冻结输入和快照验证；
- 当前基线精确重建；
- 55 查询队列选择及固定分母；
- 新候选合并和不变排序；
- exact-ID、F1、Recall@K、MRR、NDCG 与 gold 保留计算；
- Gate A/B/C 判定和固定 reason codes；
- aggregate-only 结果模型与隐私校验。

网络客户端、环境变量和文件写入不进入该模块。

### 6.4 `scripts/probe_query_evolution.py`

负责薄 CLI 和副作用边界：

- `preflight`：只读检查、重建基线、使用 gold 和冻结 availability evidence 选择队列、生成带自校验哈希但不含 gold 内容的 probe lock 和最坏预算，不访问网络；
- `run --lock <path>`：要求显式 live 授权；在线执行层只加载 lock 和冻结运行，不打开 availability、gold 或 identifier map，重复无 gold 预检并核对账本 checkpoint，在第一条网络请求前为 55 个查询各自的 `evolve`、`search-1`、`search-2` 逻辑操作一次性建立持久账本预留；部分预留失败时把已经建立的预留以零实际 usage 终结并停止；完整技术运行封存快照、完成禁网重放且哈希 matched 后，再由 deferred loader 打开并核对 gold/identifier map，完成 Gate 判定；技术失败或 replay mismatch 直接判 Gate A 失败；
- 原子写入私有结果和聚合报告；
- 所有异常都结算、释放或 fail-close 当前 reservation 后退出。

脚本默认是 `preflight`，不得因为存在 `.env` 自动进入在线模式。

## 7. 指标

### 7.1 主要检索指标

- 候选池 exact gold 关联数与 recall；
- 新找回的 `not_retrieved` gold 关联数；
- 至少找回一个新关联的查询数；
- 现有 14 个候选池 gold 的逐查询保留状态；
- 新增、重复、过滤和进入 Top-50 的候选数量。

### 7.2 最终质量指标

- Top-50 exact gold 关联数；
- 现有 8 个 Top-50 gold 的逐查询保留状态；
- Macro/Micro Precision、Recall 和 F1；
- Recall@5、Recall@10、Recall@20；
- Macro MRR 和 Macro NDCG；
- 硬过滤绝对召回损失；
- 发生有序序列变化和集合变化的查询数。

### 7.3 生成与运行指标

- 合法 proposal、合法 `no_op`、schema 拒绝和硬约束拒绝数量；
- 生成查询总数、去重数量和平均每查询有效数量；
- 逻辑 LLM/OpenAlex 操作数与实际 HTTP/usage 尝试数；
- canonical-request 复用次数及其零 usage 槽数；
- LLM tokens、OpenAlex 响应率、重试、429、5xx 和 timeout；
- 实际成本、p50/p95 耗时、预算截断和账本检查点；
- capture/replay 规范化业务哈希一致性。

## 8. Gate 与晋级

### 8.1 Gate A：证据完整性

必须全部满足：

- 预检中的运行、哈希、60/60 重建、2910 输出和基线指标完全一致；
- 55/55 查询都有唯一记录；完整运行时全部为合法 `generated` 或 `no_op`，失败运行则必须包含当前失败终态及其后的 `not_scheduled`；
- 无未解决的生成、OpenAlex 或 snapshot 失败；
- 在线与零网络重放的规范化业务哈希一致；
- `integrity_failure=0`、`provenance_failure=0`、`unaccounted_usage_failure=0`；
- 所有 reservation 均有唯一结算、释放或 fail-closed 终态；
- 每个已调度逻辑操作的 request controller 终态与对应持久账本 receipt 一一匹配，未调度的预留以零实际 usage 终结；
- 逻辑操作和实际尝试均不超过第 5.4 节上限；
- 私有产物不逸出 `runs/`，公开结果通过 aggregate-only 检查。

OpenAlex 页面中可保留有效数据的 `invalid_work` 是已记录的数据质量警告，不属于未解决依赖失败；请求级错误、重试耗尽、snapshot 或结算失败仍使 Gate A 失败。Gate A 失败时，Gate B/C 固定为 `not_evaluated`。

### 8.2 Gate B：检索假设成立

必须全部满足：

- 候选池 exact gold 高于 14；
- 至少找回 1 个原属于 `not_retrieved` 的 gold 关联；
- 原有 14 个候选池 gold 全部逐查询保留；
- gold-isolation provenance 断言通过：capture 接口不接受 availability/gold/identifier map，网络阶段未打开它们，payload 没有 gold/标签字段，且候选标题未经过 gold 驱动的增加、删除、排序或标注。

Gate B 通过只表示机制有真实召回信号。

### 8.3 Gate C：允许申请正式 capture

必须全部满足：

- Gate A、Gate B 通过；
- Top-50 exact gold 高于 8；
- 原有 8 个 Top-50 gold 全部逐查询保留；
- Macro F1 delta 相对当前基线至少为 `0.01`；
- Macro/Micro Recall、Recall@5/10/20、MRR、NDCG 均不下降；
- 硬过滤绝对召回损失不增加；
- 生产预算估计非零且在 balanced 限额内。

生产预算估计按操作类型分别统计已调度逻辑操作；一个逻辑操作的样本值包含其全部重试 usage，未调度的零 usage 槽不进入分布。每个 usage 维度取“探针最大实际值”和“向上取整的 p95 × 1.2”中的较大值，不得再次使用零估计。

## 9. 停止条件

### 9.1 请求前停止

以下任一条件触发零请求停止：

- lock 自身哈希、availability/input 哈希、运行身份或 snapshot manifest 不一致；
- 60/60 有序基线或 2910 总数无法精确重建；
- 55 查询队列或固定分母不一致；
- prompt/config hash 不一致；
- 项目账本无法预留最坏情况预算；
- live 授权缺失。

### 9.2 执行中停止

完成当前失败封存后停止调度新查询：

- 非法 LLM schema、非法硬约束或缺失合法 no-op；
- 依赖重试耗尽；
- reservation、usage 或 ledger 不一致；
- 超过逻辑操作或实际尝试硬上限；
- 达到 3600 秒全批次硬耗时上限；
- snapshot 写入、密封或来源验证失败；
- 操作者取消。

### 9.3 完成后停止

- Gate B 失败：记录否决，不做提示词变体、排序改动或正式 capture；
- Gate B 通过但 Gate C 失败：只记录召回与排序瓶颈，不正式 capture；
- Gate C 通过：停止在 `capture_candidate`，等待单独授权；
- 不在本探针中自动重建锁、运行 readiness 或启动正式闭环。

不使用中途 gold 得分做质量早停；如果没有技术失败，固定 55 查询必须按锁定顺序完成，保持可比较分母。

## 10. 预算与凭据

- 每查询一套独立 request budget controller；其中一个 LLM reservation 覆盖最多 3 次 usage 尝试，两个 OpenAlex reservation 各覆盖最多 3 次 usage 尝试，分别低于 balanced 的 5 次 LLM 和 48 次搜索上限；
- 持久项目账本与 request controller 按逻辑操作一一对应。`run` 在第一条网络请求前为全批次最多 165 个逻辑操作槽位完成预留；合法 `no_op` 或提前停止留下的未调度槽位，以零实际 usage 终结，不记为外部调用；
- 适配器完成结算后，使用现有 `finalize_controller_actual` 将同一终态和实际 usage 写入对应账本 receipt；适配器已返回的可计量 provider 错误保持 settled usage，取消或未能可靠结算的在途调用才 fail-closed；
- 全批次受 3600 秒硬耗时上限约束；账本 reservation TTL 必须大于该上限并留有 300 秒收尾余量，即固定为 3900 秒；每查询 request controller 使用 balanced 的 800 秒硬耗时上限；
- 每个逻辑操作的成本估计不得超过 `0.30 CNY` request hard cap，所有预留之和不得超过 dev run cap 和项目 soft/hard cap；任一条件不满足即零请求停止；
- 运行前输出只含聚合最坏调用与成本，不输出凭据；
- 在线探针仍需单独授权；
- 获准后只从 `D:\AI Projects\Projects\.env` 临时加载必要的 `LLM_API_KEY` 和连续编号的 `OPENALEX_API_KEY...`；
- 不读取或覆盖模型、base URL 等环境配置，不打印、写入快照或提交密钥；
- DeepSeek 请求继续使用 `thinking: disabled`。

## 11. 证据文件

### 11.1 私有诊断产物

写入已被 Git 忽略的：

`runs/_diag_query_evolution_<run-id>/`

只保留：

- `probe.lock.json`：lock 自身哈希、probe code hash、全部来源与 availability 哈希、冻结顺序的 55 查询 ID、prompt/config/model、预算、账本 checkpoint 和调用上限；
- `outcomes.jsonl`：55 个逐查询终态，以及可用时的 proposal、检索、候选、Top-50、usage 和 snapshot refs；失败后未调度查询也必须有 `not_scheduled` 记录；
- `snapshots/`：dependency snapshot v2 清单及响应；
- `result.json`：聚合指标、可空 replay 业务哈希、`matched`/`mismatched`/`not_evaluated`、三态 Gate、固定运行原因和建议。

### 11.2 可提交聚合证据

- `docs/evidence/query-evolution-probe-<date>.json`；
- `docs/query-evolution-probe-<date>.md`。

聚合 JSON 固定包含：

- schema、假设和版本；
- 来源运行、输入和 prompt/config 哈希；
- 固定分母与完整性状态；
- 生成、检索、排序、usage 和重放聚合；
- Gate A/B/C、固定 reason codes；
- `stop` 或 `capture_candidate` 建议。

公开证据不得包含 query 文本、query ID、生成查询、论文标题、gold ID、候选 ID、原始响应、密钥或未清洗 provider request ID。

## 12. 测试策略

### 12.1 单元测试

新增 `tests/unit/test_query_evolution.py`，覆盖：

- 0–2 条提案、合法 no-op 和严格 schema；
- 空强约束 QuerySpec 的上下文构造；
- 重复、空白、超长和非法字符查询；
- Unicode NFKC 后复用 OpenAlex canonicalizer，覆盖 `foo?`/`foo` 重复和 `*` 规范化为空；
- 显式年份冲突拒绝、payload 排除 `inherited_hard_filters`，以及实体/venue 相关性仅作为 prompt policy 而非伪确定性断言；
- `source_facets` 必须来自输入；
- facet、候选计数和 Top-10 标题的固定构造顺序；生成器模型和序列化 payload 不存在 gold、标签或 `inherited_hard_filters` 字段；
- `no_op_reason` 始终存在，非空提案为 `null`；
- 逻辑操作与重试尝试上限；
- 非零预算估计；
- Gate A/B/C 所有边界值和 reason codes。

### 12.2 离线评估测试

新增 `tests/evaluation/test_query_evolution_probe.py`，覆盖：

- 合成 sealed run 的精确基线重建；
- 输入哈希、snapshot 路径、清单或有序序列不一致时 fail closed；
- availability hash、lock 自身哈希、冻结队列顺序和 post-seal gold hash 不一致时 fail closed；
- 新候选复用现有去重、过滤、融合和排序；
- 冻结 OpenAlex 顺序不变、新结果按 proposal 顺序追加、canonical ID 首次出现优先，且不得拆成虚构来源；
- 候选池与 Top-50 gold 逐查询保留；
- 非目标 5 查询字节级不变；
- MRR、NDCG 和 delta 计算；
- aggregate-only schema 拒绝受限明细。

### 12.3 模拟端到端测试

新增 `tests/integration/test_query_evolution_probe.py`，使用 `httpx.MockTransport` 覆盖：

- 正常 LLM 两条查询与 OpenAlex 成功；
- runner 向每条新增搜索原样传递冻结 `inherited_hard_filters`，并继续使用不变的 `QuerySpec` 后过滤；
- 相同 LLM/OpenAlex canonical request 只访问网络一次，后续逻辑槽复用 snapshot refs 并以零 usage 终结；
- 合法 no-op；
- LLM 非法 JSON 或非法硬约束；
- OpenAlex 部分成功页；
- `invalid_work` 保留有效数据并只记警告，请求级失败仍阻断 Gate A；
- 429 后成功、5xx/timeout 重试耗尽；
- 预算不足、usage 结算不一致和 fail-close；
- 全批次预留的部分失败回滚、未调度零 usage 终态、3900 秒账本 TTL 和 3600 秒全局取消；
- 完整技术运行在 snapshot capture 后禁网 replay；技术失败运行封存证据但不构造缺失操作的 replay/hash；
- 在线与重放规范化业务哈希完全一致；
- `ReplayComparableProbe` 字段集合固定，时间、usage、request ID 和 snapshot refs 不参与哈希；失败后的查询全部写为 `not_scheduled`，Gate 使用固定三态和最小运行原因枚举。

自动测试不得读取 `.env` 或访问网络。

### 12.4 验证顺序

1. 先写失败测试，再实现最小功能；
2. 运行 unit/evaluation/integration 聚焦测试；
3. 运行 Ruff 和 mypy；
4. 运行全量离线测试；
5. 执行只读 `preflight`，验证当前真实 60/60 基线和最坏预算；
6. 经单独授权后只执行一次 `run`；
7. 自动完成快照 replay、业务哈希比较和 Gate 判定；
8. 只提交 aggregate-only 证据和决策文档。

## 13. Gate C 通过后的后续边界

Gate C 通过后另行设计生产集成，至少需要：

- 让正式多轮路径复用一次生产 query analysis，而不是重新分析；
- 将已验证生成器适配到 `EvolutionCoordinator`；
- 使用探针真实 p95/maximum 推导非零 round estimate；
- 新增独立、单变量的实验身份；
- 重新构建 candidate lock，运行 readiness；
- 经单独授权执行正式 capture → verify → replay → compare；
- 按既有三次运行、bootstrap 和 validation promotion gate 决定是否晋级。

这些工作均不属于本设计的探针实施范围。
