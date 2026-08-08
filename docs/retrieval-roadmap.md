# 检索提升路线

更新于 2026-08-08。目标是在冻结 dev 上提高宏平均 F1，同时保持 capture/replay 证据链可复现。

## 当前判断

- 当前宏 F1 约 0.006，51/60 查询零命中，召回是主要瓶颈。
- 标题候选是唯一已有正向召回信号；不同实验显示候选池与最终命中存在潜在落差，但尚不能直接归因。
- Citation、Topic、Embedding、普通 Query Rewrite 和既有 LLM Query Variants 已被实测否决，详见 `experiment-decisions.md`。

## Phase 0：建立干净基线

清理完成并重建下一版锁后，执行 readiness → dev capture → verify → replay → compare。

基线必须同时满足：

- quality gate passed；
- `provenance_failures=0`；
- capture 与 replay 业务结果一致。

在此之前不运行新的全量在线实验。

## Phase 1：两个必要诊断

### 1. Gold 精确可用性

使用 DOI、arXiv ID 和 OpenAlex ID 做只读精确反查，只输出聚合原因。禁止把 gold 标识符转换成检索查询。

现有 P0 探针测量的是生成标题能否搜到 gold，不等同于 gold 是否存在于 OpenAlex。

### 2. 标题候选流失

逐阶段统计 exact gold：

1. 生成标题；
2. OpenAlex 标题验证结果；
3. 合并候选池；
4. RRF 排序池；
5. 最终 `selected_paper_ids`。

下一项实现工作选择流失最大的阶段。

## Phase 2：标题候选保留与输出选择

- 在同一冻结 dev 上对比 10 与 20 个标题；
- 检查标题验证排名、融合贡献和最终截断；
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
