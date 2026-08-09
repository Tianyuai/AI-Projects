# 检索提升路线

更新于 2026-08-09。目标是在冻结 dev 上提高宏平均 F1，同时保持 capture/replay 证据链可复现。

## 当前判断

- 已闭环基线宏 F1 为 `0.0050946874`，micro recall 为 `0.0575539568`，召回是主要瓶颈。
- 标题候选部分成功页修复使候选池 exact gold 从 19 增至 20，但最终 Top-50 仍为 13；既定离线排序变体无一满足晋级条件。
- Citation、Topic、Embedding、普通 Query Rewrite 和既有 LLM Query Variants 已被实测否决，详见 `experiment-decisions.md`。

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

### 1. Gold 精确可用性

使用 DOI、arXiv ID 和 OpenAlex ID 做只读精确反查，只输出聚合原因。禁止把 gold 标识符转换成检索查询。

现有 P0 探针测量的是生成标题能否搜到 gold，不等同于 gold 是否存在于 OpenAlex。

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
