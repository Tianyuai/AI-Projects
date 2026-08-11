# 封存查询重组离线对照设计

日期：2026-08-11
状态：设计完成，待用户复核
范围：只读封存 dev 数据上的一次性诊断；不修改生产检索行为

## 1. 决策摘要

本实验只回答一个问题：Prompt v2 已经封存的基线与两条扩展查询结果中，是否存在被当前“按查询槽依次追加”方式淹没、但可由固定且无 Gold 泄漏的合并方法恢复到 Top-50 的信号。

采用三个预注册方案：

1. `append_v2`：精确复现当前追加顺序，作为内部基准；
2. `round_robin_slots`：所有基线查询槽和扩展查询槽按结果名次轮询交织；
3. `rrf_slots_k60`：每个查询槽作为一条独立排名流，使用现有 RRF、固定 `k=60` 融合。

三个方案共享同一候选集合、同一 `QuerySpec`、同一硬过滤、同一 Top-50 截断和同一 verified identifier map。方案构造阶段不得读取 Gold；三个方案一次性全部运行，看到评分后不得增加变体、修改参数或重跑。

该实验是排除性诊断，不是生产晋级实验。即使某个重组方案优于 `append_v2`，只要不能达到本设计的充分性门槛，下一步仍转向重新建立干净的 title-informed 多查询检索基线。

## 2. 背景与诊断问题

2026-08-11 verified-identifier rescore 在同一语义金标下得到：

| 来源 | 检索到的 Gold | Top-50 Gold | 未检索到 | 排名在 Top-50 外 |
| --- | ---: | ---: | ---: | ---: |
| 当前正式基线 2026-08-10 | 40/143 | 17/143 | 103 | 23 |
| 正式基线 2026-08-09 | 41/143 | 19/143 | 102 | 22 |
| 旧版 title 2026-08-05 | 49/143 | 30/143 | 94 | 19 |
| Prompt v2 封存探针 | 42/143 | 19/143 | 101 | 23 |

这组证据说明：

- 当前正式基线的主要问题是检索覆盖不足，不是硬过滤；四个来源的 `filtered_out` 都为 0；
- Prompt v2 只比其封存源多找回 1 个 verified Gold 关联，并没有增加 Top-50 命中；
- Prompt v2 候选池仍有 23 个 Gold 排在 Top-50 外，需用一次受控实验确认现有合并顺序是否掩盖了信号；
- 旧版 title 的 30/143 是外部充分性基准，但其来源运行不同，不能作为本实验的内部基线或与 Prompt v2 原始槽位混合。

因此本实验不回答“哪种新查询文本最好”，也不声称全新 retrieval-query 方法的召回效果。没有对应快照的新查询不能离线评估。

## 3. 方案权衡

### 3.1 采用：三个固定重组方案的一次性离线对照

优点：零网络、零预算、可重复；能够隔离“已有查询结果如何合并”这一变量；可以给出继续重组或停止重组的明确结论。

限制：不能增加候选池覆盖，最多利用已经封存的 42 个 Gold；结果只适用于该封存 probe，不能直接证明生产收益。

### 3.2 暂不采用：继续修改 Prompt v2

此前多次提示词修订最终只增加 1 个 verified 候选 Gold，Top-50 增益为 0。继续修改需要新的 DeepSeek/OpenAlex capture，也会再次混合查询质量与合并质量两个变量。应先完成本次零成本归因。

### 3.3 暂不采用：直接重建 title-informed live 基线

旧版结果支持该方向，但新方法需要新的查询和对应 OpenAlex 响应，无法纯离线验证。只有本实验结束并按停止条件归档结论后，才单独设计有限 capture；本设计不授权该 capture。

## 4. 范围与非目标

### 4.1 范围

- 只读验证 Prompt v2 封存 lock、result、outcomes、snapshot manifest 及所有被引用快照；
- 只读验证其绑定的 2026-08-09 正式源运行；
- 先加载并验证 2026-08-11 verified identifier generation，再允许读取实验源；
- 构造三个固定候选顺序，应用现有硬过滤并截取 Top-50；
- 使用 143 个 verified Gold 关联计算聚合指标和逐阶段流失；
- 发布一份规范 JSON 和一份由 JSON 直接渲染的 Markdown；
- 对正式产物执行既有公共 JSON/Markdown 隐私扫描。

### 4.2 非目标

- 不读取 `.env`，不连接 DeepSeek、OpenAlex 或其他网络服务；
- 不读取或修改预算账本，不创建 reservation/receipt；
- 不生成新查询，不修改 prompt、查询解析、硬过滤、正式 fusion 或生产配置；
- 不修改封存 run、快照、lock、outcomes 或既有 rescore 证据；
- 不做权重网格、RRF `k` 搜索、Top-k 搜索、按查询选择最佳方案或 Gold 驱动调参；
- 不运行 readiness、live capture、replay、compare 或 production promotion；
- 不把此次结果表述为新检索方法的在线效果。

## 5. 固定输入与来源绑定

### 5.1 Verified identifier generation

沿用 `scripts.rescore_identifier_semantics.build_fixed_report()` 的验证顺序和固定路径：

- `docs/evidence/identifier-map-semantic-audit-2026-08-11.json`；
- `data/dev/gold.jsonl`；
- `data/annotation_work/identifier_semantics/identity-evidence.json`；
- `data/annotation_work/identifier_semantics/snapshots/snapshot-manifest.json`；
- `data/annotation_work/identifier_semantics/relation-audit.v2.json`；
- `data/annotation_work/identifier_semantics/dev-identifier-map.semantic-v2.json`。

任何一项哈希、审计状态或关系契约失败，都必须在读取实验源之前停止。

### 5.2 封存 Prompt v2 源

唯一允许的实验目录是：

`runs/_diag_query_evolution_query-evolution-prompt-v2-full-20260810`

必须继续验证：

- `probe.lock.json` 的自哈希、代码哈希、prompt 版本/哈希和固定 query 顺序；
- lock 绑定的 `source_run_id=dev-20260809T061903Z-9bd861e90299` 及 source hashes；
- `result.json` 的 capture/replay 匹配状态和业务哈希；
- `outcomes.jsonl` 的完整顺序、终态、无错误搜索和 outcome hash；
- snapshot manifest 的哈希、snapshot set ID，以及每个引用响应可读且匹配请求身份。

Prompt v2 的原始新增结果只允许与其绑定的 2026-08-09 源槽位组合。2026-08-10 正式基线和 2026-08-05 title 来源只作为报告中的外部聚合基准，不参与候选重组。

### 5.3 外部聚合基准

唯一允许的外部基准文件是：

`docs/evidence/identifier-map-semantic-rescore-2026-08-11.json`

加载器必须验证其规范 JSON、公共隐私契约、`schema_version=identifier-semantic-rescore-v2`、`status=passed`，并要求其中六个 `generation_hashes` 与第 5.1 节本次实际加载的 generation 完全一致。随后只读取固定标签 `formal_baseline_2026_08_10` 的 17/143 和 `legacy_title_2026_08_05` 的 30/143 聚合值。标签缺失、顺序漂移、数值漂移或 generation 不一致均属于完整性失败；不得从 HANDOFF、Markdown 或人工输入替代这些值。

## 6. 三个预注册方案

所有方案先把每个查询的 `baseline_results` 和最多两个 Prompt v2 `additions` 视为可排序的有序槽位。每个槽位内部顺序保持封存顺序；Paper 数据不修改；canonical ID 首次入选后去重。

封存 execution 的 `retrieved_paper_ids` 和 `post_filter_paper_ids` 是基线逐阶段流的权威来源，不能由可排序槽位反推。每个方案的 retrieved 流固定为“封存 retrieved 流 + additions canonical IDs”的稳定并集；post-filter 流固定为“封存 post-filter 流 + additions 经现有硬过滤后接受的 canonical IDs”的稳定并集。三个方案只能改变这些已接受候选进入 selected Top-50 的顺序，不能删除、增加或重新分类 retrieved/post-filter ID。

### 6.1 `append_v2`

严格调用现有 `merge_probe_results()` / `project_openalex_stream()` 行为：依次遍历全部基线槽，再遍历 `search-1`、`search-2`。它必须精确复现 verified rescore 中 Prompt v2 的结果：

- `not_retrieved=101`；
- `filtered_out=0`；
- `ranked_outside_top50=23`；
- `selected_top50=19`。

任一数值不一致都属于输入或实现完整性失败，整个实验停止，不解释其他方案。

### 6.2 `round_robin_slots`

按固定槽位顺序：所有基线槽在前，随后 `search-1`、`search-2`。从 rank 1 开始，每轮依次取每个非空槽位的当前 rank Paper，按 canonical ID 首次出现去重，直到所有槽位耗尽。随后应用不变的硬过滤并截取前 50。

该方案不给扩展槽额外权重，也不保留配额；它只消除“整个早期槽位永久压住后续槽位”的块状追加效应。

### 6.3 `rrf_slots_k60`

每个槽位作为独立排名流，使用现有 `fuse_provider_results(..., method="rrf", rrf_k=60)`。槽位键采用零填充固定名称以保证排序稳定；同一 Paper 在多个槽出现时累加 RRF 分数；最终按现有融合器的 `(-score, canonical_id)` 规则排序。随后只保留通过同一硬过滤的前 50。

`k=60` 是项目现有默认值，不允许改动或搜索。该方案检验跨查询槽重复出现的共识信号能否改善排序。

## 7. 模块边界

### 7.1 纯评估模块

新增 `src/paper_search/evaluation/query_recomposition.py`，只包含：

- 固定方案枚举和 Pydantic 报告模型；
- `compose_append()`、`compose_round_robin()`、`compose_rrf()` 三个无 I/O 纯函数；
- 对单查询继承封存 retrieved/post-filter 流、合并 additions、应用固定排序和 Top-50 的投影函数；
- verified identifier 评分、保留性比较、结果分类和报告构建函数。

该模块不得导入网络客户端、dotenv、账本或文件路径。

### 7.2 固定源与发布脚本

修改 `scripts/rescore_identifier_semantics.py`，只把现有 Prompt v2 封存验证流程提取为公共只读 `load_verified_probe_materials()`；现有 `load_probe_source()` 改为消费该返回值，输出和 rescore schema 必须保持不变。对应回归测试继续放在 `tests/scripts/test_rescore_identifier_semantics.py`。不得借此重构其他来源适配器。

新增 `scripts/analyze_sealed_query_recomposition.py`，负责：

- 固定路径与固定命令 `run`、`render-markdown`；
- 复用 identifier generation 验证器和 Prompt v2 封存源验证逻辑；
- 把已验证的原始槽位交给纯评估模块；
- 规范序列化、隐私扫描、禁止覆盖发布和 Markdown 恢复。

CLI 不接受输入路径、输出路径、网络、环境文件、账本、方案、权重或阈值参数。

### 7.3 测试与证据

- 新增 `tests/evaluation/test_query_recomposition.py`；
- 新增 `tests/scripts/test_analyze_sealed_query_recomposition.py`；
- 正式 JSON：`docs/evidence/sealed-query-recomposition-offline-2026-08-11.json`；
- 正式 Markdown：`docs/sealed-query-recomposition-offline-2026-08-11.md`。

不修改 HANDOFF、路线图和历史报告；是否更新这些文件留到用户审阅实验结果之后。

## 8. 数据流与隔离

执行顺序固定：

1. 验证 verified identifier generation；失败即停止，且不得读取 probe 源；
2. 读取 Gold 顺序并锁定 60 个 query ID；
3. 验证 Prompt v2 lock、source run、result、outcomes、manifest 和引用快照；
4. 重建绑定的 2026-08-09 基线可排序槽位、权威 retrieved/post-filter 流与 Prompt v2 新增槽位；
5. 在不读取 Gold 的组合层一次性生成三个方案；
6. 断言三个方案的 retrieved canonical ID 集合逐查询完全一致；
7. 断言 `append_v2` 精确复现 101/0/23/19；
8. 验证既有规范 rescore JSON 与本次 generation 哈希一致，再读取其中 17/143 与 30/143 的固定外部聚合基准；
9. 构造聚合报告，先扫描 JSON 和 Markdown，再进行禁止覆盖发布；
10. 若 JSON 已成功而 Markdown 发布失败，只允许从规范 JSON 恢复 Markdown，不得重跑组合和评分。

Gold 和 identifier map 只进入评分层，不进入三个组合函数。测试必须用陷阱对象证明组合函数无法访问 Gold。

## 9. 指标、门槛与结论分类

每个方案报告：

- `true_positive_count`、macro F1、macro recall、micro recall、MRR、NDCG；
- `not_retrieved`、`filtered_out`、`ranked_outside_top50`、`selected_top50`；
- 是否保留 `append_v2` 的全部 selected verified Gold 关联；
- retrieved/post-filter 集合是否与 `append_v2` 相同。

不发布逐查询记录、query ID、paper ID、标题、查询文本、请求 ID、响应或 snapshot 路径。

### 9.1 完整性门槛

以下条件全部满足才允许解释结果：

- verified identifier generation 通过；
- 封存 probe capture/replay matched，所有绑定和快照验证通过；
- `append_v2` 精确复现 101/0/23/19；
- 三个方案逐查询 retrieved 集合和 post-filter 集合完全一致；
- 所有指标有限，逐阶段数量守恒，总数均为 143；
- 公共 JSON 和 Markdown 隐私扫描通过。

### 9.2 信号门槛

一个重组方案只有同时满足以下条件，才记为 `usable_recomposition_signal`：

- `selected_top50 > 19`；
- 保留 `append_v2` 的全部 selected verified Gold；
- macro F1、macro recall、MRR、NDCG 均不低于 `append_v2`；
- `filtered_out=0`，且 retrieved/post-filter 集合不变。

### 9.3 充分性门槛

只有在满足信号门槛且 `selected_top50 >= 30` 时，结果才记为 `legacy_benchmark_met`。30 是旧版 title 的外部效果基准，不代表生产晋级。

### 9.4 固定结论

四种结论只适用于 generation、源绑定、capture/replay 和快照均已验证通过之后的实验结果。任何输入验证、隐私扫描或发布前置条件失败都不生成实验报告，CLI 使用固定安全错误退出，停止并请求人工检查，不得重跑正式命令。

- 输入有效但 `append_v2` 复现、集合恒等或阶段守恒失败：`integrity_failure`，可发布聚合失败结论，随后停止并请求人工检查；
- 没有方案达到信号门槛：`no_usable_recomposition_signal`，停止查询重组，下一步设计 title-informed 检索；
- 有信号但未达到 30：`signal_insufficient`，记录合并层存在可用信号，但不继续调参，下一步仍设计 title-informed 检索；
- 达到 30：`legacy_benchmark_met`，允许单独设计生产等价整合验证，但本实验仍不修改生产代码。

## 10. 证据与发布契约

JSON schema 固定为 `sealed-query-recomposition-offline-v1`，仅包含：输入哈希、三个固定方案名、聚合指标、守恒/保留布尔值、外部基准数值、最终分类和固定 reason codes。

规范 JSON 使用 UTF-8、排序 key、紧凑分隔符、`allow_nan=False` 和单个末尾换行。Markdown 只从已验证报告模型渲染，不重新计算指标。

发布前必须先完成两种隐私扫描，再写任何目标文件。两个目标只要有一个已存在，`run` 就失败且不覆盖。写入使用同目录临时文件、落盘同步和 no-replace 发布。异常信息对外只输出固定安全文本，不输出路径、标识符或私有内容。

## 11. 测试策略

实施必须遵循 TDD：

1. 先用小型槽位 fixture 证明现有代码缺少三个固定组合接口；
2. 分别验证追加、轮询和 RRF 的精确顺序、重复 ID、空槽、单槽和稳定 tie-break；
3. 验证三个方案候选集合一致，只有顺序允许不同；
4. 验证 Gold 陷阱对象在组合阶段从不被读取；
5. 验证评分守恒、基线 Gold 保留、四种固定结论和 19/30 边界；
6. 验证 generation 失败时 probe 文件零读取；
7. 验证 probe/source/hash/replay/snapshot 任一漂移均在评分和写入前停止；
8. 验证 CLI 无路径、网络、环境、账本、变体和参数选项；
9. 验证规范 JSON、Markdown 恢复、双扫描先于写入、no-overwrite 和安全错误文本；
10. 运行相关 Query Evolution、semantic rescore、Gate 回归测试，再运行 Ruff、mypy 和全量离线 pytest。

正式执行前必须确认两个目标文件均不存在。正式 `run` 只执行一次；若发生 JSON 已发布但 Markdown 未发布，只允许 `render-markdown` 恢复。

## 12. 验收标准

- 三个且仅三个预注册方案被构造和评分；
- 没有网络、`.env`、账本、reservation、live capture 或封存证据改写；
- Prompt v2 只与其 2026-08-09 绑定源组合，其他来源只作外部聚合基准；
- `append_v2` 在 verified identifier 语义下精确复现 101/0/23/19；
- 三个方案候选集合相同，指标守恒，总 Gold 为 143；
- 结果只能落入第 9.4 节四种结论之一，且停止条件自动执行；
- 正式证据聚合、规范、不可覆盖、隐私安全且可从 JSON 恢复 Markdown；
- 全部相关测试、全量离线测试、Ruff、mypy 和 `git diff --check` 通过；
- 实验结束后不自动继续调参、修改生产检索或发起 capture。
