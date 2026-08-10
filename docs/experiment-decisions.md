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
| Title Candidates | 部分成功页修复使候选池 exact gold 由 19 增至 20，但修复后 RRF 的 Top-50 仍为 13；权重 1.25 仅改善早期排序指标，权重 1.5–3.0 会丢失既有 gold，保留槽 1/2/3/5/10 无增量 | 保留修复；排序不晋级 | 只有实质不同的选择策略离线提高 macro F1、保留已有 gold 且排序护栏不回退时才重开 |

## 当前结论

召回仍是主要瓶颈。新的 DOI 契约重跑确认 134/134 个唯一 work 可用、0 个完整性失败；关联级损失为 125 个未被检索到、6 个排名在 Top-50 外、8 个已选入 Top-50。诊断完整，当前唯一推荐方向为 `retrieval_query_evolution_probe`。下一步应先提出可证伪的 Query Evolution 假设并执行低成本 bounded probe；在离线指标、gold 保留、排序护栏和预算检查通过前，不进入正式 live capture。
