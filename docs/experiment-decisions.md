# 检索实验决策记录

更新于 2026-08-08。本文件只记录聚合指标，不包含冻结查询文本。

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
| Title Candidates | 独立 P0 标题探针的联合池覆盖 41 篇 gold、24 个查询；尚不是与正式最终输出的同阶段对照 | 继续 | 先做同一 run 的阶段流失诊断，不先增加标题数量 |

## 当前结论

召回仍是主要瓶颈，但下一步不是继续增加检索模块，而是先确认 gold 的 OpenAlex 可用性，并用同一 run 验证标题候选是否以及在哪个阶段流失。
