# OpenAlex 部分成功与标题 Top-50 保留设计

日期：2026-08-09

## 目标

按顺序完成两项工作：

1. 修复 title-candidates 阶段因单个 OpenAlex `invalid_work` 而丢弃整页有效论文的问题；
2. 基于已封存的同轮候选做 Top-50 保留/排序离线对照，找到可证伪的下一个候选方案。

本轮不运行 live capture，不修改生产排序，不增加标题数量，不引入新数据源。

## 已确认根因

OpenAlex provider 的 decoder 会跳过无效 Work，并同时返回：

- `data`：同页中已成功归一化的 papers；
- `errors`：`invalid_work` 等结构化诊断。

`LLMTitleCandidateStage.recall` 当前在 `search.errors` 非空时无条件 `continue`，因此没有使用同一响应中的有效 `search.data`。历史快照已观测到 1 个 exact gold 因此未进入合并池。

问题位于标题阶段的消费策略，不在 OpenAlex decoder、硬过滤或 RRF。

## 修复设计

保持 provider 语义不变：不吞掉、降级或改写 `invalid_work`。标题阶段对每次搜索按以下规则处理：

- 始终保留 provider diagnostic，包括 errors 和 snapshot refs；
- `search.errors` 非空时仍记录该次异常；
- `search.data` 非空时，继续对有效 papers 去重并纳入 title-candidates 来源；
- 只有该次搜索没有任何有效 paper 时，才将它视为完全失败；
- 最终状态仍由是否存在有效 papers 决定，不改变预算结算、快照或安全诊断契约。

先增加回归测试，使用同时含有效 paper 和 `invalid_work` 的 provider result。测试必须在修复前因有效 paper 被丢弃而失败，修复后验证 paper、diagnostic、usage 和状态均正确。

## 离线重建 Gate

标题对照使用 `runs/dev-20260805T035209Z-7af4b103f6cc` 的封存 LLM/OpenAlex 快照、execution 和 business result。该 run 只用于离线诊断，不作为当前正式基线。

诊断程序必须先：

1. 按快照顺序重建 baseline OpenAlex 与 title-candidates 来源列表；
2. 使用历史语义重建标准 RRF；
3. 仅保留 `post_filter_paper_ids` 中的候选；
4. 与封存 `selected_paper_ids` 逐查询、逐顺序比较。

已有只读试算能够精确复现 60/60 条 Top-50 序列，总结果均为 2,908。正式诊断脚本仍须把该结果作为 fail-closed Gate；任何不一致都停止变体评估。

## Top-50 对照设计

首先生成“部分成功修复 + 标准 RRF”结果，单独量化 bug fix 本身带来的变化。然后只比较两类确定性策略：

### 1. 加权 RRF

保持 RRF 分母和排名公式，只调整标题来源权重：

```text
score = Σ provider_weight / (60 + source_rank)
```

OpenAlex baseline 权重固定为 `1.0`，title-candidates 权重比较 `{1.0, 1.25, 1.5, 2.0, 3.0}`。不使用现有 `method="weighted"`，因为它的 `weight / rank` 会同时更换融合公式，无法单独归因来源权重。

### 2. 标题席位保留

比较保证 Top-50 中至少有 `{1, 2, 3, 5, 10}` 个 title-candidates 来源候选。如当前 Top-50 不足，按标题来源排名提升已通过硬过滤的候选，并从末尾移除最低排名的非标题候选。不将未通过硬过滤的 paper 强制加入输出。

不组合搜索权重和席位参数，避免小样本网格过拟合。由于当前没有可校准的相关性分数，不执行 `0.45–0.75` 阈值搜索。

## 指标与决策

所有变体都与同轮历史基线比较：

- macro F1 `0.0081353423`；
- macro recall `0.0947222222`。

报告同时输出 macro/micro precision、recall 与 F1、Recall@5/10/20、MRR、NDCG、exact-gold 总数和有命中查询数。候选方案只在以下条件全部满足时标记为可晋升：

1. macro F1 高于同轮历史基线；
2. 每个查询在历史 Top-50 中已命中的 exact gold 仍全部保留；
3. Recall@5/10/20、MRR 和 NDCG 不低于历史基线。

若无变体通过，结论为“不修改生产排序”，不选择最不差参数充当正向结果。

## 产物与安全边界

- 修复代码与回归测试；
- 可重复执行的离线诊断脚本；
- 仅含聚合数据和参数决策的 JSON 产物与 Markdown 报告；
- 必要的交接、路线图与实验决策更新。

不输出冻结查询文本、gold ID、生成标题、响应正文、密钥或 request ID。不修改 `data/`、`runs/_diag_*` 历史产物或当前 ledger。

## 验证

1. 回归测试先 RED 后 GREEN；
2. title-candidates、OpenAlex、fusion、orchestrator 相关测试通过；
3. 离线重建 Gate 为 60/60 序列完全一致；
4. 诊断 JSON 由 schema 验证，报告数字由产物重算核对；
5. 全量 pytest 与项目既有静态检查通过；
6. 提交前确认只包含本任务文件，未跟踪账本与 `deliverables/` 保持不变。
