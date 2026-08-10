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
- 15 个既有 mypy 错误已通过只涉及类型收窄和局部命名的最小修复清零；相关 85 个回归测试通过，业务行为未扩张。
- 最新全量验证为 1923 passed / 36 skipped / 1 环境失败；进程级 Git ownership 兼容设置已消除 clone 失败，剩余失败是 Windows 当前 GBK locale 解码 `uv build` UTF-8 输出的问题，不涉及本次文档，已登记为本次诊断的 packaging 平台环境豁免，不能据此声称全量完全通过。Git 已跟踪 Python 文件 Ruff 全部通过，`mypy src scripts/analyze_gold_bottlenecks.py` 为 0 errors。
- 最新项目 ledger 为 1989 条，根哈希为 `sha256:7e7e63a2e2ed0587d68af5a64792c1ce68949d9c49f7da714d7f14f5b47544f9`。
- 两次 DOI 契约 availability rerun 曾在第 1 次 HTTP 尝试后失败；网络恢复后第三次获批重试成功，新的证据见 `docs/evidence/gold-bottleneck-attribution-2026-08-09-doi-contract-retry3.json` 与对应 Markdown：134/134 个唯一 work `available`、0 个完整性失败，`diagnostic_complete=true`，推荐方向为 `retrieval_query_evolution_probe`。旧历史 evidence 保持不改。
- Gold 精确可用性 v2 的旧聚合诊断固定输入为 60/143/139/134，唯一 work 为 132 `available`、2 个 DOI `integrity_failure`；该报告是历史 evidence，保持不改。Task 1 已实现 DOI exact-endpoint acceptance contract，并由离线 `httpx.MockTransport` 合成测试覆盖：HTTP 200 且响应包含有效 OpenAlex Work ID 即为 `available`，顶层 DOI 缺失、不同或不可解析均不影响；OpenAlex-ID 请求仍严格要求规范化 ID 匹配。新的契约重跑结果已单独写入 retry3 evidence。

## 3. 活跃文件

- `README.md`：运行入口；
- `PRD.md`：固定产品与评测契约；
- `docs/retrieval-roadmap.md`：当前提升顺序；
- `docs/experiment-decisions.md`：已验证和已否决实验；
- `docs/title-candidate-stage-loss-2026-08-09.md`：标题候选同轮逐阶段流失诊断；
- `docs/title-retention-offline-2026-08-09.md`：部分成功页修复及 Top-50 离线保留决策；
- `docs/quality-gate-root-cause-2026-08-09.md`：Gate 失败根因与已验证闭环；
- `docs/gold-bottleneck-attribution-2026-08-09.md`：Gold 精确可用性与流水线瓶颈聚合诊断；
- `docs/evidence/gold-bottleneck-attribution-2026-08-09.json`：同一诊断的固定 schema 证据；
- `docs/gold-bottleneck-attribution-2026-08-09-doi-contract-retry3.md`：新 DOI 契约重跑的完整瓶颈诊断；
- `docs/evidence/gold-bottleneck-attribution-2026-08-09-doi-contract-retry3.json`：新 DOI 契约重跑的固定 schema 证据；
- `configs/title_candidates.yaml`：当前正式实验配置；
- `runs/candidate.lock.yaml`：本地候选锁；
- `data/dev/gold.jsonl`：冻结 dev gold。

## 4. 已否决尝试

Citation Expansion、Topic Retrieval、Embedding Reranking、普通 Query Rewrite 和既有 LLM Query Variants 均已有负向实测。除非方法或输入发生实质变化并先通过低成本探针，否则不要重复。

标题候选是唯一已有正向召回信号。部分成功页错误丢弃已修复：15 个含错误响应恢复 80 篇有效论文，57 篇成为新增合格候选，候选池多覆盖 1 个 exact gold。修复后标准 RRF 仍只有 13 个 Top-50 exact gold；权重与保留槽离线变体均未提高 macro F1，因此不得重复或进入 live capture。

本轮实现提交从 `3fabf6d` 到 `70c9c3c`；设计与实施计划提交为 `5a92f2d`、`c5d05bb`。离线分析未发起网络请求，也未修改候选锁或 ledger。

最新稳定化提交为 `437ba0d`、`44d7aab`、`ff0ea28`、`1dfac84`：增加 v2 完整性失败聚合、唯一诊断账本运行 ID，清零原有 15 个目标 mypy 错误，并补齐嵌套 schema、计数守恒和写盘重读校验。本轮只执行一次获准的 exact-ID 在线诊断，没有运行 readiness、capture、replay、compare 或 validation，也没有修改候选锁。

## 5. 下一步

1. DOI exact-endpoint acceptance contract 已离线决定并由固定合成夹具覆盖；新的在线重跑已确认 134/134 个唯一 work 可用，完整性瓶颈已排除。
2. 当前唯一推荐方向为 `retrieval_query_evolution_probe`：先设计并执行有明确假设、低成本、可回滚的 bounded probe，不直接进入 live capture。
3. 新方向若要进入正式 capture，仍需先通过离线指标、保留已有 gold、排序护栏和预算检查，并另行授权 readiness/capture。

## 6. 锁状态

当前 `runs/candidate.lock.yaml` 绑定提交 `45ef8749210c1ec6fcbfeb9b64b911f3ea4b0d55`，但其 ledger checkpoint 仍为 1924 条。最新诊断已将 ledger 推进到 1989 条，因此该锁只是已用基线证据，不得用于新 live run。新在线实验前必须重建锁并重跑 readiness；capture、validation 和任何新的在线实验仍需单独授权。

## 7. 环境与红线

- 完整测试环境：`D:\AI Projects\Projects\.venv`；
- 密钥位于 `D:\AI Projects\Projects\.env`，默认不得读取、打印或提交；只有明确授权的单次诊断可将必要 key 临时注入当前进程，执行后不持久化；
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

## 8. Query Evolution bounded probe implementation (2026-08-10)

- Implementation commits: `a3d2cd9`, `b8cce03`, `a522d5d`, and `47012d3`.
- Offline verification: focused tests `29 passed`; full suite `1954 passed, 36 skipped`; Ruff and mypy passed; `git diff --check` passed.
- Zero-network preflight: `preflight_complete=true`; 55 queued queries in frozen order; baseline `60/2910/14/8` reconstructed.
- Preflight lock: `runs/_diag_query_evolution_preflight/probe.lock.json`; `probe_run_id=query-evolution-preflight`; future run directory `runs/_diag_query_evolution_query-evolution-preflight`.
- Limits: 55/110 logical operations, 165/330 attempt caps, 3600-second global timeout, 3900-second ledger TTL.
- Evidence hashes: availability `sha256:3f445486d5cf590f3f11a51930153a45916023880e856def379e0f01d053ad04`; probe code `sha256:07ce27806bd93a73ac4a8d499c3ca0e3ded83fbf4f7c702880e8ff6ba54a29d2`; lock `sha256:dc261edb560915f1907149f371e2266be54fb328a3c2019076da226ea96117d2`.
- This phase did not read `.env`, make reservations, rebuild the candidate lock, run readiness, or execute live capture/replay/compare/validation. The live bounded run remains separately authorized work.
- User-owned untracked `data/budget_ledger.sqlite3` and `deliverables/` were preserved.

## 9. Query Evolution live bounded probe result (2026-08-10)

- Executed with `probe.lock.live6.json`; output is `runs/_diag_query_evolution_query-evolution-preflight/`.
- Capture and zero-network replay matched: `sha256:037e4f6cb6758061f1ab4e22fd586162277feb9c291ce3dc5ab22216c14edef7`.
- 55 LLM snapshots and 55 outcomes were sealed; OpenAlex was correctly not called because every LLM proposal failed the strict `query-evolution-proposal-v1` integrity contract.
- Gate result: Gate A `failed`, Gate B/C `not_evaluated`; all 55 terminals were `integrity_failure`.
- Ledger closed all 165 slots: 55 settled LLM slots and 110 zero-usage failed search slots; actual usage was 55 LLM calls, 29,772 input tokens, 16,212 output tokens, cost `0.062196` CNY; no reserved slots remain.
- This is a valid negative probe result, not evidence for a ranking change. Do not rerun the same prompt unchanged; first diagnose or revise the proposal-output contract, then create a new separately locked variant.

## 10. Query Evolution prompt-contract canary preparation (2026-08-10)

- The offline implementation is complete in commits `7c46689`, `53c82ac`, `151c2c4`, `f3d2174`, `b52292b`, `55ec90b`, and the smoke-fixture compatibility fix `380bd29`.
- Task 4 initially exposed a real accounting defect: cancellation and accounting failure could leave canary receipts reserved. The finalizer and regression tests now force all three receipts to terminal state; independent re-review approved the fix.
- Offline verification: focused contract/canary suite `106 passed`; full suite `1984 passed, 36 skipped`; `mypy src scripts/probe_query_evolution.py` reports 0 errors across 96 source files; `git diff --check` passed.
- Full-repository Ruff is not green only because the untouched user-owned `deliverables/project-docs/edit_docx.py` has a pre-existing unused `shutil` import. It was not modified.
- Offline source lock: `runs/_locks/query_evolution_contract-v2-source-20260810/probe.lock.json`; embedded lock hash `sha256:68801ce497cb2f409eafaa18588c08233dcb5bac8bc407873b81ff0fa95f8a74`; physical file hash `sha256:ac3413e0ecdec73495487acfff2d000434bc6f4b64c76158b6f8f1fcc882193e`.
- Offline canary lock: `runs/_locks/query_evolution_contract-20260810/canary.lock.json`; embedded lock hash `sha256:72579944c679ef03d42d0dd8771019aad070da2c036cb54344bf931852392df5`; physical file hash `sha256:44b580e3fa865825e72147f003a27650c205ab0ae08077b9191b41a092613cb5`.
- The canary lock selects exactly 3 deterministic query IDs and fixes 3 logical operations, 9 LLM attempts, and a 600-second global timeout. Its ledger checkpoint is `sha256:0d3774553fc1bf7b67ba2794ed9d73522112463d63965cff8283083c082a3adc`.
- This preparation ran without reading `.env`, making ledger reservations, constructing an OpenAlex provider, or sending network requests. It created only the two offline lock files.
- Next action is a separately authorized three-query DeepSeek canary using the exact canary lock. Do not run it, and do not run the full 55-query probe, until the user grants live authorization. If the canary is not promoted, stop on its fixed failure reason and do not rerun unchanged.

## 11. Query Evolution three-query canary result (2026-08-10)

- The authorized canary used `runs/_locks/query_evolution_contract-20260810/canary.lock.json` and wrote evidence to `runs/_diag_query_evolution_contract-canary-20260810/`.
- It made exactly 3 DeepSeek LLM calls and 0 OpenAlex/search calls. The evidence contains 3 LLM snapshots and no OpenAlex snapshot.
- Outcomes were 2 `generated` and 1 `integrity_failure`; the failed query was rejected for duplicate subquery text after canonicalization. The canary result is `promoted=false` with fixed reason `canary_accounting_failed`.
- Aggregate usage was 3 LLM calls, 2,541 input tokens, 345 output tokens, 3,877 ms, and `0.003231` CNY. Ledger readback shows 3/3 receipts `settled`, actual usage present, no remaining `reserved` receipt, and the per-receipt sums match the aggregate usage.
- Snapshot manifest hash is `sha256:a3582e00ac3ebe24dcc78539e60a7fe843e52eefac85675c861e51be38fb4729`; snapshot set ID is `sha256:eefba557af57d4c33a9671412346ac9b36d522bae4ee65787a0bd378fcdd5fcb`.
- The mismatch between the fixed result reason (`canary_accounting_failed`) and the terminal, numerically consistent ledger readback is now a diagnostic item. It must be explained before any new live run; this result does not support promotion, prompt editing, or a rerun.
- Apply the stop rule: do not rerun this canary unchanged and do not execute the full 55-query probe. The next work is offline diagnosis of the reason classification/accounting path, followed by a new independently reviewed lock only if a changed hypothesis is justified.
