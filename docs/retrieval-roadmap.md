# 检索提升路线

更新于 2026-08-11。目标是在冻结 dev 上提高宏平均 F1，同时保持 capture/replay 证据链可复现。

## 当前判断

- 当前代码上的可复现闭环基线 macro F1 为 `0.0038000670`，macro recall 为 `0.0541666667`，micro recall 为 `0.0431654676`；三项均低于 2026-08-09 闭环，因此只固化为运行基线，不视为质量提升，召回仍是主要瓶颈。
- 标题候选部分成功页修复使候选池 exact gold 从 19 增至 20，但最终 Top-50 仍为 13；既定离线排序变体无一满足晋级条件。
- Citation、Topic、Embedding、普通 Query Rewrite 和既有 LLM Query Variants 已被实测否决，详见 `experiment-decisions.md`。
- Gold 精确可用性 v2 的旧聚合探针为 132/134 个唯一 work `available`、2 个 DOI `integrity_failure`；该报告及对应 JSON 是历史 evidence，保持不改。新的 DOI 契约重跑证据见 `docs/evidence/gold-bottleneck-attribution-2026-08-09-doi-contract-retry3.json`：134/134 个唯一 work `available`、0 个完整性失败，`diagnostic_complete=true`。其历史推荐 `retrieval_query_evolution_probe` 已执行，最新结论见 Phase 8–9。Task 1 的离线 `httpx.MockTransport` 契约测试仍保留。

## Phase 0：建立当前代码基线（已完成）

已完成 readiness → dev capture → verify → replay → compare：

- capture：`runs/dev-20260810T104256Z-d9e89476d484`；
- replay：`runs/dev-20260810T123542Z-e097e1aa48b2`；
- 两轮正式验证均为 `valid: true`、`issue_codes: []`；Gate passed、`provenance_failures=0`、业务结果 `equivalent: true`；
- 闭环源提交为 `d70f1c5c3e84e36c7ddb07ffb42c2e65a8ccd5f1`，快照 manifest 为 `sha256:70332aaaa93604dacdebf18f6d0225e6d25dd85c0c0e9dc088a474bb742c753a`；
- capture 实际用量为 60 次 LLM、274 次 OpenAlex、`0.031221` CNY；正式 ledger 已推进至 2049 条。旧 candidate lock 和 readiness 均不得复用。

基线必须同时满足：

- quality gate passed；
- `provenance_failures=0`；
- capture 与 replay 业务结果一致。

该 Gate 证明证据链有效，不代表达到 macro F1 目标。任何新在线实验仍须先取得独立离线晋级证据。

## Phase 1：两个必要诊断

### 1. Gold 精确可用性（已完成；历史推荐已执行）

使用 DOI、arXiv ID 和 OpenAlex ID 做只读精确反查，只输出聚合原因。禁止把 gold 标识符转换成检索查询。

现有 P0 探针测量的是生成标题能否搜到 gold，不等同于 gold 是否存在于 OpenAlex。

本次固定输入为 60 个查询、143 条原始 gold 标识、139 个归一化查询–论文关联和 134 个唯一 work。新的契约重跑实际使用 135 次 HTTP 尝试（硬上限 402），134 个唯一 work 全部 `available`，0 个 `exact_not_found`，0 个完整性失败。新证据见 `docs/evidence/gold-bottleneck-attribution-2026-08-09-doi-contract-retry3.json` 及对应 Markdown；旧报告和 JSON 仍作为历史记录保留。

DOI exact-endpoint acceptance contract 已完成离线固化：请求使用规范化 DOI 时，HTTP 200 加有效 OpenAlex Work ID 即为 `available`；响应顶层 DOI 缺失、不同或不可解析不改变该结论。响应缺失或无法解析 Work ID 仍按既有完整性原因失败。请求使用 OpenAlex-ID 时，响应 Work ID 仍必须与请求 ID 严格匹配。该契约由固定合成 `httpx.MockTransport` 测试覆盖，且不改变生产检索、报告 schema、隐私或账本规则。

诊断已完整闭环：`available_not_retrieved_dominant`，125 个关联未被检索到，6 个在 Top-50 外，8 个已选入 Top-50，推荐方向为 `retrieval_query_evolution_probe`。该阶段当时只允许设计并执行 bounded probe；后续执行结果见 Phase 5–8。

### 2. 标题候选流失（已完成）

逐阶段统计 exact gold：

1. 生成标题；
2. OpenAlex 标题验证结果；
3. 合并候选池；
4. RRF 排序池；
5. 最终 `selected_paper_ids`。

详见 `title-candidate-stage-loss-2026-08-09.md`。硬过滤无 exact-gold 流失；已观测到的可操作点是部分成功响应的整页丢弃和最终 Top-50 排序/截断。

## Phase 2：标题候选保留与输出选择（已完成）

- 已保留部分成功 OpenAlex 响应中的有效论文，同时保留 `invalid_work` 诊断和用量账本。
- 已在同一冻结 dev 上精确重建历史 Top-50，并比较修复后 RRF、标题权重 1.25/1.5/2.0/3.0 和标题保留槽 1/2/3/5/10。
- 修复新增 57 篇合格候选和 1 个候选池 exact gold，但未增加最终 Top-50 gold；没有变体提高 macro F1 并同时通过全部护栏。

因此只保留正确性修复，不修改生产排序，不重建候选锁，不进入 live capture。详见 `title-retention-offline-2026-08-09.md`。标题数量或新排序仅在出现实质不同、可证伪且低成本证据为正的假设时重开。

## Phase 3：Query Evolution 条件实验

当前 `fixed_two_round` 不可直接启用，因为它：

- 使用规则兜底而不是生产 DeepSeek `QuerySpec`；
- 在实验身份下关闭标题候选；
- 第二轮预算估计为零。

该阶段的生产集成设想已被后续 bounded-probe 路径取代。exact-ID recall 提升且不损失已有命中只足以通过检索 Gate B；正式 capture 还必须满足 Phase 9 的 Top-50、macro F1、排序非回退和预算护栏。

## 后续条件项

- Query Type：仅在分类型误差分析显示稳定差异后实施；
- Selector/LLM rerank：仅在召回明显提升后实施；
- 新数据源：仅在 Gold 精确可用性诊断证明 OpenAlex 覆盖不足后引入。

每个晋升改动必须是单变量实验，使用独立配置、锁和 capture/replay/compare 证据。

## Phase 4: Query Evolution bounded probe（离线实现完成）

- 已加入严格的 query-evolution 契约、仅聚合评估，以及 offline-first bounded runner。
- preflight 已针对冻结 run 通过，按原始顺序选出 55 个查询，并重建基线 `60/2910/14/8`。
- 锁定上限为 55/110 logical operations、165/330 attempts、3600 秒全局超时和 3900 秒 ledger TTL。
- 专项测试、全量离线测试、Ruff、mypy 与 diff 检查通过：专项 `29 passed`；全量 `1954 passed, 36 skipped`。
- 锁文件为 `runs/_diag_query_evolution_preflight/probe.lock.json`；历史 evidence 与 `runs/candidate.lock.yaml` 未改变。
- 本阶段未执行 live capture、replay、compare、readiness、候选锁重建、validation、网络请求、`.env` 读取或 ledger reservation；其后获批的 bounded `run` 结果见 Phase 5。

## Phase 5：Query Evolution bounded probe 实际结果

- 已执行 55 个锁定查询；capture 与零网络 replay 的业务 hash 一致。
- 生成并封存 55 个 LLM 快照与 55 条结果；由于 55 个 LLM 输出均未通过 `query-evolution-proposal-v1` 严格契约，OpenAlex 阶段按 fail-closed 规则未启动。
- Gate A `failed`，Gate B/C `not_evaluated`；账本 165 个槽位全部终态，无遗留 reservation。
- 本次结果是 Query Evolution 输出契约的负向诊断，不支持晋级或排名改动。随后只修订并验证 proposal 输出契约，没有原样重跑；后续见 Phase 6–8。

## Phase 6：Query Evolution prompt-contract canary 准备（离线完成）

- 已固化 prompt artifact、live probe 绑定、source/canary lock、三查询选择和 LLM-only bounded canary runner；Task 4 的账本终态缺陷已补测修复并通过独立复审。
- 离线质量门：专项 `106 passed`；全量 `1984 passed, 36 skipped`；`mypy src scripts/probe_query_evolution.py` 为 0 errors；`git diff --check` 通过。全仓库 Ruff 仅受未触碰的 `deliverables/project-docs/edit_docx.py` 中既有 F401 阻塞。
- source lock 为 `runs/_locks/query_evolution_contract-v2-source-20260810/probe.lock.json`，canary lock 为 `runs/_locks/query_evolution_contract-20260810/canary.lock.json`。canary 固定 3 个确定性查询、3 个 logical operations、最多 9 次 LLM attempts 和 600 秒全局超时；ledger checkpoint 为 `sha256:0d3774553fc1bf7b67ba2794ed9d73522112463d63965cff8283083c082a3adc`。
- 本阶段只运行离线 preflight，没有读取 `.env`、没有 reservation、没有 OpenAlex provider/client、没有网络请求，也没有创建 canary run directory；历史失败 evidence 与用户未跟踪文件保持不变。
- 该阶段当时仅允许对 canary lock 单独授权三查询 DeepSeek live canary；后续授权结果见 Phase 7。该历史 lock 不得再次使用。

## Phase 7：Query Evolution 三查询 canary 结果（未晋级）

- 已按 canary lock 执行一次三查询 DeepSeek live canary，证据目录为 `runs/_diag_query_evolution_contract-canary-20260810/`。
- 实际调用为 3 次 LLM、0 次 OpenAlex/search；结果为 2 条 `generated`、1 条 `integrity_failure`。失败原因是 canonicalization 后出现重复子查询文本。
- `result.json` 的 `promoted=false`，固定 reason 为 `canary_accounting_failed`；聚合用量为 3 次 LLM、2,541 input tokens、345 output tokens、3,877 ms、`0.003231` CNY。
- 账本离线回读确认 3/3 回执均为 `settled`、actual 完整、无遗留 `reserved`，且回执用量与 aggregate usage 一致。因此 result reason 与账本事实之间存在需解释的诊断不一致。
- 该结果不支持晋级、修改排名或直接复用原锁。后续离线 reason/accounting 诊断形成了独立 Prompt-v2 假设与新锁，结果见 Phase 8；原 canary 仍不得重跑。

## Phase 8：Prompt-v2 全量探针与 Gate 契约修复（已完成）

- 独立锁定的 55-query Prompt-v2 探针完成 30 条 `generated` 和 25 条 `no_op`；capture/replay matched，所有账本回执均为终态，且无 integrity、provenance、accounting 或 snapshot failure。
- 密封历史 `result.json` 保持原样，其 Gate A `failed` 来自评估契约缺陷，而不是在线运行不完整：55-query 执行数被错误地与 60-query 冻结指标分母比较，Gate B 也错误复用了 selected/Top-50 流。
- 契约修复已在 `3b4e94a`、`00168cb` 完成并合并至 `main` 的 `d70f1c5`。离线重算结果为 Gate A `passed`、Gate B `passed`、Gate C `failed`：unique retrieved gold 14 → 15，Top-50 gold 8 → 8。
- 结论是 Query Evolution 已产生 1 个真实召回增量，但现有合并/排序/截断没有把它转化为 Top-50 或 macro F1 增益。不得重写旧 evidence、原样重跑 live probe，或直接进入生产 capture。

## Phase 9：当前基线固化与下一步

- 2026-08-10 的正式 `capture → verify → replay → compare` 已完整通过；权威运行身份与指标见“当前判断”和 Phase 0。
- 下一步只使用已密封 evidence 做新增 retrieved gold 的逐阶段流失诊断，定位它在 canonical merge、RRF、过滤或 Top-50 截断中的具体位置；同时解释当前 baseline 相对上一闭环的召回下降，不发起新网络调用。
- 只有一个单变量离线方案能保留既有 gold、提高 Top-50 gold 和 macro F1，并通过 hard-filter、排序非回退和预算护栏时，才建立独立配置与候选锁，刷新 readiness 并申请 live capture。
- Selector/LLM rerank 只能在上述诊断证明候选池有稳定可利用召回后考虑；validation 仍保持单独授权且不可撤销。

## Phase 10：sealed query recomposition 离线对照（已完成并封存）

- 已在 Prompt-v2 封存槽位上固定比较且仅比较 `append_v2`、`round_robin_slots` 和 `rrf_slots_k60`，正式命令只执行一次，没有网络、`.env`、ledger、readiness、candidate lock 或 live capture。
- `append_v2` 精确复现 `101/0/23/19`；round-robin selected Gold 为 24；RRF selected Gold 为 25。三种方法的 retrieved/post-filter 集合相同，所有阶段均守恒为 143。
- round-robin 未保留 append 的全部 selected Gold；RRF 满足可用信号门槛，但 25 仍低于旧版 title 基准 30。因此正式结论为 `signal_insufficient`，不是生产晋级。
- 该结果否定继续在同一候选集上调重组顺序、RRF 参数或增加变体。主要损失仍是 `not_retrieved=101`，继续只改排序无法解决主体召回缺口。
- 正式证据：`docs/evidence/sealed-query-recomposition-offline-2026-08-11.json` 和 `docs/sealed-query-recomposition-offline-2026-08-11.md`。禁止覆盖、重跑或从其结果反向挑选新变体。

## Phase 11：title-informed 多查询干净基线（下一阶段，仅设计）

下一步不是继续修改现有 recomposition，而是建立独立的检索基线。首次工作只产出设计文档，不发起网络请求。

设计必须预注册：

- 输入边界：仅从 reference title 提取主题实体、方法、任务和数据集线索；不得将 Gold ID、Gold 标题命中结果或评分反馈送入查询构造。
- 固定查询族：少量、互补、可解释，并在评分前整体冻结；禁止逐查询挑选、结果驱动追加和参数网格。
- 公平对照：继续使用 verified identifier 语义、143 个 Gold 关联分母以及固定的 17、19、25、30 外部聚合基准；历史证据只读。
- 首要指标：`not_retrieved` 必须相对 101 明显下降；随后才评估 selected Top-50、macro F1、macro recall、MRR、NDCG，以及既有 Gold 保留和 hard-filter 非回退。
- 晋级边界：只有检索覆盖和 Top-50 同时产生预注册的实质改善，且不损失既有命中，才允许单独设计 production-equivalent 集成或有限 live canary。具体数值门槛必须在任何实验运行前写入设计并复核。
- 停止边界：若干净查询基线不能降低 `not_retrieved`，停止查询工程，转向 OpenAlex/其他数据源覆盖、identifier mapping 或 Gold/reference 数据诊断；若召回提高但 Top-50 不提高，再单独研究保留/排序，不把两类改动混在同一实验。
- 禁止事项：当前阶段不修改生产代码、不重建锁、不刷新 readiness、不运行 live capture/replay/compare、不读取 `.env` 或 ledger，也不自动采用参考文献中的方法。

Phase 11 的推荐顺序只有三步：先写正式设计；再做矛盾、遗漏、冗余自审并确定指标/门槛/停止条件；用户批准后再撰写精简实施计划。未经批准不执行实验。
