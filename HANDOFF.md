# paper-search 项目交接

更新于 2026-08-09。权威工作区：`D:\AI Projects\.worktrees\week3`。

## 1. 项目目标

VivaAI 参加第八届中国研究生人工智能创新大赛赛题三，构建复杂学术查询的论文搜索与推荐系统。内部目标是冻结 dev 宏平均 F1 ≥ 0.30；当前已闭环基线为 `0.0050946874`。

正式评测使用 live capture → verify → 零网络 replay → compare。基线必须 gate passed、`provenance_failures=0`，且 capture/replay 业务结果一致。

## 2. 当前状态

- 通过的 live capture：`runs/dev-20260809T061903Z-9bd861e90299`；通过的零网络 replay：`runs/dev-20260809T063333Z-6897d295a3c8`。
- 两轮均为 `formal_valid: true`、`quality_passed: true`、Gate `passed`；业务结果比较 `equivalent: true`。
- 查询分析与可解析检索响应均为 60/60；60 条全部使用 primary planner，无 fallback。
- 基线 macro F1 `0.0050946874`、macro recall `0.0791666667`、micro recall `0.0575539568`；召回仍是主要瓶颈。
- 闭环源提交为 `45ef8749210c1ec6fcbfeb9b64b911f3ea4b0d55`；修复链为 `a720bbe` 与 `45ef874`。
- 标题部分成功页修复已完成；封存离线对照精确重建 60/60 查询和 2,908 个 Top-50 结果。候选池 exact gold 从 19 增至 20，但最终仍为 13，没有排序变体晋级。
- 本轮完整环境验证为 1886 passed / 36 skipped；Ruff 全量通过。mypy 仍有 15 个既有错误，位于本轮未修改的 `query/parser.py`、`application/readiness.py`、`retrieval/snapshot_adapters.py` 和 `llm/snapshot_adapters.py`。
- live capture 已将项目 ledger 推进到 1984 条，根哈希为 `sha256:3267b17bdff93676ad2d0f793257559f5549ad11aa4ff29ed097d11bbc60495f`。

## 3. 活跃文件

- `README.md`：运行入口；
- `PRD.md`：固定产品与评测契约；
- `docs/retrieval-roadmap.md`：当前提升顺序；
- `docs/experiment-decisions.md`：已验证和已否决实验；
- `docs/title-candidate-stage-loss-2026-08-09.md`：标题候选同轮逐阶段流失诊断；
- `docs/title-retention-offline-2026-08-09.md`：部分成功页修复及 Top-50 离线保留决策；
- `docs/quality-gate-root-cause-2026-08-09.md`：Gate 失败根因与已验证闭环；
- `configs/title_candidates.yaml`：当前正式实验配置；
- `runs/candidate.lock.yaml`：本地候选锁；
- `data/dev/gold.jsonl`：冻结 dev gold。

## 4. 已否决尝试

Citation Expansion、Topic Retrieval、Embedding Reranking、普通 Query Rewrite 和既有 LLM Query Variants 均已有负向实测。除非方法或输入发生实质变化并先通过低成本探针，否则不要重复。

标题候选是唯一已有正向召回信号。部分成功页错误丢弃已修复：15 个含错误响应恢复 80 篇有效论文，57 篇成为新增合格候选，候选池多覆盖 1 个 exact gold。修复后标准 RRF 仍只有 13 个 Top-50 exact gold；权重与保留槽离线变体均未提高 macro F1，因此不得重复或进入 live capture。

本轮实现提交从 `3fabf6d` 到 `70c9c3c`；设计与实施计划提交为 `5a92f2d`、`c5d05bb`。离线分析未发起网络请求，也未修改候选锁或 ledger。

## 5. 下一步

1. 完成 Gold 精确可用性的聚合报告，再决定是否需要新数据源；
2. 若继续研究 Top-50 选择，只接受与现有权重/保留槽实质不同的离线假设；
3. 只有离线 macro F1 提升、保留已有 gold 且排序护栏不回退时，才重建锁、运行 readiness 并申请新的 live capture。

## 6. 锁状态

当前 `runs/candidate.lock.yaml` 绑定提交 `45ef8749210c1ec6fcbfeb9b64b911f3ea4b0d55`，但其 ledger checkpoint 仍为 1924 条。通过的 live capture 已将 ledger 推进到 1984 条，因此该锁只是已用基线证据，不得用于新 live run。新在线实验前必须重建锁并重跑 readiness；capture、validation 和任何新的在线实验仍需单独授权。

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
