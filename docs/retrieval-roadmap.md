# 检索提升路线

更新于 2026-08-09。目标是在冻结 dev 上提高宏平均 F1，同时保持 capture/replay 证据链可复现。

## 当前判断

- 已闭环基线宏 F1 为 `0.0050946874`，micro recall 为 `0.0575539568`，召回是主要瓶颈。
- 标题候选同轮诊断显示：标题响应验证到 13 个 exact gold，12 个进入合并/RRF 池，11 个进入最终 Top-50。
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

## Phase 2：标题候选保留与输出选择（下一阶段）

- 先保留部分成功 OpenAlex 响应中的有效论文，不因单个 `invalid_work` 丢弃整页；
- 在同一冻结 dev 上离线比较标题来源保留与排序策略；
- 标题数量 10/20 仅在预算能覆盖所有验证请求时比较，不再运行未封闭的全量尝试；
- 离线搜索 `K ∈ {10,20,30,50}`；
- 离线搜索阈值 `0.45–0.75`，步长 `0.05`；
- 以宏平均 F1 选择，Precision、Recall、Recall@K、MRR、NDCG 作为护栏。

离线无增量的变体不进入 live capture。选定组合在 validation 前冻结。

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
