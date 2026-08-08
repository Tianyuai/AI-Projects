# paper-search 项目交接

更新于 2026-08-09。权威工作区：`D:\AI Projects\.worktrees\week3`。

## 1. 项目目标

VivaAI 参加第八届中国研究生人工智能创新大赛赛题三，构建复杂学术查询的论文搜索与推荐系统。内部目标是冻结 dev 宏平均 F1 ≥ 0.30；当前约 0.006。

正式评测使用 live capture → verify → 零网络 replay → compare。基线必须 gate passed、`provenance_failures=0`，且 capture/replay 业务结果一致。

## 2. 当前状态

- DeepSeek `deepseek-v4-flash` 查询解析已验证 60/60。
- 标题候选默认生成 20 个标题并经 OpenAlex 验证。
- 最近四轮正式 capture 均因 OpenAlex 限流或额度问题 gate failed；已完成查询的 replay 可干净复现。
- 51/60 查询零命中，最终共命中约 10 篇 gold；召回是主要瓶颈。
- 全量测试最近记录为 1856 passed / 36 skipped。
- 当前候选锁已重建并绑定提交 `c427541670e2523f8556a0d204eae964198ef9b1`；readiness 已确认 LLM、OpenAlex、Semantic Scholar 均为 ready。

## 3. 活跃文件

- `README.md`：运行入口；
- `PRD.md`：固定产品与评测契约；
- `docs/retrieval-roadmap.md`：当前提升顺序；
- `docs/experiment-decisions.md`：已验证和已否决实验；
- `configs/title_candidates.yaml`：当前正式实验配置；
- `runs/candidate.lock.yaml`：本地候选锁；
- `data/dev/gold.jsonl`：冻结 dev gold。

## 4. 已否决尝试

Citation Expansion、Topic Retrieval、Embedding Reranking、普通 Query Rewrite 和既有 LLM Query Variants 均已有负向实测。除非方法或输入发生实质变化并先通过低成本探针，否则不要重复。

标题候选是唯一已有正向召回信号，但当前数据不是同一 run 的阶段对照。后续先验证候选是否在 OpenAlex 验证、融合或最终输出阶段流失。

## 5. 下一步

1. 在单独授权后使用当前候选锁跑 dev capture → verify → replay → compare；
2. 完成 Gold 精确可用性和标题候选流失诊断；
3. 优先优化标题候选保留与输出选择；
4. Query Evolution 仅在重做生产分析组合和预算估计后进行小规模探针。

## 6. 锁状态

旧 v21 锁绑定 `c22abf9`，已因源码 SHA 不匹配而废弃。当前候选锁为 v22，绑定提交 `c427541670e2523f8556a0d204eae964198ef9b1` 和最新 ledger checkpoint；readiness 已通过。capture、validation 和任何新的在线实验仍需要单独授权。

## 7. 环境与红线

- 完整测试环境：`D:\AI Projects\Projects\.venv`；
- 密钥位于 `D:\AI Projects\Projects\.env`，不得读取、打印或提交；
- 正式命令只加载 `LLM_API_KEY`、`OPENALEX_API_KEY`（含 `_2.._7`）和 `SEMANTIC_SCHOLAR_API_KEY`；
- 不加载 `LLM_BASE_URL`、`LLM_MODEL_PRIMARY`、`LLM_MODEL_FALLBACK`；
- OpenAlex key 必须从裸名 `OPENALEX_API_KEY` 开始连续编号；余额低于一次 search 成本才轮换，余额充足的 429 按每秒限流退避；
- DeepSeek 请求必须保留 `thinking: disabled`；
- 每次正式 run 都会推进 ledger，重建锁前必须重新读取 `project_checkpoint()`；
- `c22abf9` 的 reservation elapsed 软处理和标题阶段降级不可回退；
- 不删除 `runs/`、`_diag_*` 或 `data/`，不修改 `data/manifest.json`；
- capture 与 replay 之间不得提交代码；
- 不在聊天或公开文档中写入冻结查询文本；
- validation 不可撤销，必须单独授权。
