# 检索实验决策记录

更新于 2026-08-09。本文件只记录聚合指标，不包含冻结查询文本。

## 使用规则

- 已否决方法默认不再消耗正式额度。
- 重新开启必须提出与旧实验实质不同、可证伪的假设。
- 新假设先通过离线或小规模探针，再申请正式 capture。

## 决策

| 方法 | 既有证据 | 决策 | 重新开启条件 |
| --- | --- | --- | --- |
| Citation Expansion | 10 条零命中查询、20 篇 gold，Recall@50 为 0；见 `runs/_diag_citation_expansion_probe_20260804T150231Z.json` | 否决 | 候选种子、图数据源或扩展机制实质变化，且新探针为正 |
| Topic Retrieval | top-50 命中为 0；完美 topic ceiling 仅 1 篇位于 rank 180 | 否决 | topic 映射或索引机制实质变化，且新 ceiling probe 为正 |
| Embedding Reranking | gold top-50 从 13 降至 6，F1 从 0.0081 降至 0.0044 | 否决 | 候选池、表示或训练目标改变，且离线 F1 超过原排序 |
| Query Rewrite | W1–W5 未救回零命中查询 | 否决 | 生成器获得旧实验没有的新证据 |
| LLM Query Variants | gold top-50 从 13 降至 8–10，并增加大量请求 | 否决 | 小规模 exact-ID recall 提升且不损失已有命中 |
| Title Candidates | 同轮正式运行中，标题响应验证 13 个 exact gold，12 个进入合并/RRF 池，11 个进入 Top-50；独立 20-title P0 联合池为 41 篇/24 查询 | 有条件继续 | 先修复部分成功页丢弃并离线改善 Top-50 保留；预算未封闭前不增加标题数量 |

## 当前结论

召回仍是主要瓶颈。同轮诊断已排除硬过滤；下一实现目标是保留部分成功 OpenAlex 页的有效结果，并用已封存候选优化 Top-50 选择。Gold 精确可用性报告仍是引入新数据源前的必要证据。
