# 检索提升路线

更新于 2026-08-09。目标是在冻结 dev 上提高宏平均 F1，同时保持 capture/replay 证据链可复现。

## 当前判断

- 已闭环基线宏 F1 为 `0.0050946874`，micro recall 为 `0.0575539568`，召回是主要瓶颈。
- 标题候选部分成功页修复使候选池 exact gold 从 19 增至 20，但最终 Top-50 仍为 13；既定离线排序变体无一满足晋级条件。
- Citation、Topic、Embedding、普通 Query Rewrite 和既有 LLM Query Variants 已被实测否决，详见 `experiment-decisions.md`。
- Gold 精确可用性 v2 的旧聚合探针为 132/134 个唯一 work `available`、2 个 DOI `integrity_failure`；该报告及对应 JSON 是历史 evidence，保持不改。新的 DOI 契约重跑证据见 `docs/evidence/gold-bottleneck-attribution-2026-08-09-doi-contract-retry3.json`：134/134 个唯一 work `available`、0 个完整性失败，`diagnostic_complete=true`，推荐方向为 `retrieval_query_evolution_probe`。Task 1 的离线 `httpx.MockTransport` 契约测试仍保留。

## Phase 0：建立干净基线（已完成）

已完成 readiness → dev capture → verify → replay → compare：

- capture：`runs/dev-20260809T061903Z-9bd861e90299`；
- replay：`runs/dev-20260809T063333Z-6897d295a3c8`；
- Gate passed、`provenance_failures=0`、业务结果 `equivalent: true`。

基线必须同时满足：

- quality gate passed；
- `provenance_failures=0`；
- capture 与 replay 业务结果一致。

在此之前不运行新的全量在线实验。

## Phase 1：两个必要诊断

### 1. Gold 精确可用性（已完成，推荐 Query Evolution bounded probe）

使用 DOI、arXiv ID 和 OpenAlex ID 做只读精确反查，只输出聚合原因。禁止把 gold 标识符转换成检索查询。

现有 P0 探针测量的是生成标题能否搜到 gold，不等同于 gold 是否存在于 OpenAlex。

本次固定输入为 60 个查询、143 条原始 gold 标识、139 个归一化查询–论文关联和 134 个唯一 work。新的契约重跑实际使用 135 次 HTTP 尝试（硬上限 402），134 个唯一 work 全部 `available`，0 个 `exact_not_found`，0 个完整性失败。新证据见 `docs/evidence/gold-bottleneck-attribution-2026-08-09-doi-contract-retry3.json` 及对应 Markdown；旧报告和 JSON 仍作为历史记录保留。

DOI exact-endpoint acceptance contract 已完成离线固化：请求使用规范化 DOI 时，HTTP 200 加有效 OpenAlex Work ID 即为 `available`；响应顶层 DOI 缺失、不同或不可解析不改变该结论。响应缺失或无法解析 Work ID 仍按既有完整性原因失败。请求使用 OpenAlex-ID 时，响应 Work ID 仍必须与请求 ID 严格匹配。该契约由固定合成 `httpx.MockTransport` 测试覆盖，且不改变生产检索、报告 schema、隐私或账本规则。

诊断已完整闭环：`available_not_retrieved_dominant`，125 个关联未被检索到，6 个在 Top-50 外，8 个已选入 Top-50，推荐方向为 `retrieval_query_evolution_probe`。下一步只设计并执行 bounded probe，不直接重建锁或进入 live capture。

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

重新实验前必须让它复用生产查询分析、组合已选标题候选基线，并用真实调用推导非零预算。规则版与 LLM 版一次只测一个；只有 exact-ID recall 提升且不损失已有命中时才进入正式 capture。

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
- 本阶段未执行 live capture、replay、compare、readiness、候选锁重建、validation、网络请求、`.env` 读取或 ledger reservation；下一步是对该锁单独授权 bounded `run`。

## Phase 5：Query Evolution bounded probe 实际结果

- 已执行 55 个锁定查询；capture 与零网络 replay 的业务 hash 一致。
- 生成并封存 55 个 LLM 快照与 55 条结果；由于 55 个 LLM 输出均未通过 `query-evolution-proposal-v1` 严格契约，OpenAlex 阶段按 fail-closed 规则未启动。
- Gate A `failed`，Gate B/C `not_evaluated`；账本 165 个槽位全部终态，无遗留 reservation。
- 本次结果是 Query Evolution 输出契约的负向诊断，不支持晋级或排名改动。下一步应先修订/验证 proposal 输出契约，再以独立锁运行变体；不得原样重跑。

## Phase 6：Query Evolution prompt-contract canary 准备（离线完成）

- 已固化 prompt artifact、live probe 绑定、source/canary lock、三查询选择和 LLM-only bounded canary runner；Task 4 的账本终态缺陷已补测修复并通过独立复审。
- 离线质量门：专项 `106 passed`；全量 `1984 passed, 36 skipped`；`mypy src scripts/probe_query_evolution.py` 为 0 errors；`git diff --check` 通过。全仓库 Ruff 仅受未触碰的 `deliverables/project-docs/edit_docx.py` 中既有 F401 阻塞。
- source lock 为 `runs/_locks/query_evolution_contract-v2-source-20260810/probe.lock.json`，canary lock 为 `runs/_locks/query_evolution_contract-20260810/canary.lock.json`。canary 固定 3 个确定性查询、3 个 logical operations、最多 9 次 LLM attempts 和 600 秒全局超时；ledger checkpoint 为 `sha256:0d3774553fc1bf7b67ba2794ed9d73522112463d63965cff8283083c082a3adc`。
- 本阶段只运行离线 preflight，没有读取 `.env`、没有 reservation、没有 OpenAlex provider/client、没有网络请求，也没有创建 canary run directory；历史失败 evidence 与用户未跟踪文件保持不变。
- 下一步仅是对该 canary lock 单独授权三查询 DeepSeek live canary。未获授权前不得执行 `canary-run`；更不得直接执行完整 55-query probe。canary 若未晋级，按固定失败原因停止，不得原样重跑。
