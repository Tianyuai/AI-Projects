# paper-search 项目交接

更新于 2026-08-12。权威工作区：`D:\AI Projects\.worktrees\week3`。

## 1. 项目目标

VivaAI 参加第八届中国研究生人工智能创新大赛赛题三，构建复杂学术查询的论文搜索与推荐系统。内部目标是冻结 dev 宏平均 F1 ≥ 0.30；当前代码上的可复现闭环基线为 `0.0038000670`，尚未达到目标，也没有超过上一闭环的 `0.0050946874`。

正式评测使用 live capture → verify → 零网络 replay → compare。基线必须 gate passed、`provenance_failures=0`，且 capture/replay 业务结果一致。

## 2. 当前状态

- 当前权威 live capture：`runs/dev-20260810T104256Z-d9e89476d484`；零网络 replay：`runs/dev-20260810T123542Z-e097e1aa48b2`。
- capture 与 replay 均经正式 `verify-run` 验证，`valid: true`、`issue_codes: []`；两轮均为 `formal_valid: true`、`quality_passed: true`、Gate `passed`，业务结果比较 `equivalent: true`。
- 查询分析与可解析检索响应均为 60/60；60 条全部使用 primary planner，无 fallback。
- 当前基线 macro F1 `0.0038000670`、macro recall `0.0541666667`、micro recall `0.0431654676`；相对上一闭环分别下降约 25.4%、31.6% 和 25.0%，因此本轮只固化可复现性，不声称质量提升，召回仍是主要瓶颈。
- 当前闭环源提交为 `d70f1c5c3e84e36c7ddb07ffb42c2e65a8ccd5f1`。快照集 ID 为 `sha256:8364a7eb7d90b89ceed25a158a414df083732664ae6e10fa5337fc11aca8ba2c`，manifest 哈希为 `sha256:70332aaaa93604dacdebf18f6d0225e6d25dd85c0c0e9dc088a474bb742c753a`。
- capture 封存 334 个响应，60 条本轮回执全部 `settled`，0 failures、0 integrity/provenance/sanitization/unaccounted-usage failures；实际用量为 60 次 LLM、274 次 OpenAlex、19,459 tokens、`0.031221` CNY。
- 标题部分成功页修复已完成；封存离线对照精确重建 60/60 查询和 2,908 个 Top-50 结果。候选池 exact gold 从 19 增至 20，但最终仍为 13，没有排序变体晋级。
- 标题实验的 19/20/13 与 Query Evolution 的 14/15/8 来自不同历史运行、投影阶段和关联去重口径，不得直接互换；标题报告绑定 `dev-20260805T035209Z-7af4b103f6cc` 及其 business/execution 哈希，共 2,908 条结果，Query Evolution 绑定 `dev-20260809T061903Z-9bd861e90299` 的另一组哈希，共 2,910 条结果。当前 Query Evolution 结论统一使用修订后的 unique resolved association 口径。
- 15 个既有 mypy 错误已通过只涉及类型收窄和局部命名的最小修复清零；相关 85 个回归测试通过，业务行为未扩张。
- 当前 `main` 合并后的最新离线验证为 2003 passed / 39 skipped / 1 online deselected；排除用户未跟踪 `deliverables/` 后的 tracked Ruff 通过，`mypy src scripts/probe_query_evolution.py` 对 96 个文件为 0 errors。
- 最新正式项目 ledger 为 2049 条，根哈希为 `sha256:172d5c8d8946e1d1d44e738a33cd3bbb778354a48500908816b2f4742c938e91`。
- 两次 DOI 契约 availability rerun 曾在第 1 次 HTTP 尝试后失败；网络恢复后第三次获批重试成功，新的证据见 `docs/evidence/gold-bottleneck-attribution-2026-08-09-doi-contract-retry3.json` 与对应 Markdown：134/134 个唯一 work `available`、0 个完整性失败，`diagnostic_complete=true`。该证据当时推荐的 `retrieval_query_evolution_probe` 已执行，当前结论见第 12 节。旧历史 evidence 保持不改。
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
- `runs/candidate.lock.yaml`：本轮已使用、因 ledger 推进而失效的本地候选锁；
- `runs/dev-20260810T104256Z-d9e89476d484/`：当前权威 live capture、密封快照与 replay lock；
- `runs/dev-20260810T123542Z-e097e1aa48b2/`：当前权威零网络 replay；
- `data/dev/gold.jsonl`：冻结 dev gold。

## 4. 已否决尝试

Citation Expansion、Topic Retrieval、Embedding Reranking、普通 Query Rewrite 和既有 LLM Query Variants 均已有负向实测。除非方法或输入发生实质变化并先通过低成本探针，否则不要重复。

标题候选是唯一已有正向召回信号。部分成功页错误丢弃已修复：15 个含错误响应恢复 80 篇有效论文，57 篇成为新增合格候选，候选池多覆盖 1 个 exact gold。修复后标准 RRF 仍只有 13 个 Top-50 exact gold；权重与保留槽离线变体均未提高 macro F1，因此不得重复或进入 live capture。

该历史标题阶段的实现提交从 `3fabf6d` 到 `70c9c3c`；设计与实施计划提交为 `5a92f2d`、`c5d05bb`。离线分析未发起网络请求，也未修改候选锁或 ledger。

该历史稳定化阶段的提交为 `437ba0d`、`44d7aab`、`ff0ea28`、`1dfac84`：增加 v2 完整性失败聚合、唯一诊断账本运行 ID，清零原有 15 个目标 mypy 错误，并补齐嵌套 schema、计数守恒和写盘重读校验。该阶段只执行一次获准的 exact-ID 在线诊断，没有运行 readiness、capture、replay、compare 或 validation，也没有修改候选锁。

## 5. 下一步

1. 当前正式 baseline、verified identifier rescore 和 sealed query recomposition 均已闭环；不得把 Gate passed 或局部排序增益解释为达到竞赛目标。
2. 本次重组结论是 `signal_insufficient`：append/round-robin/RRF 的 selected Gold 分别为 19/24/25，均未达到旧版 title 基准 30，且主要损失仍为 `not_retrieved=101/143`。不得重跑或继续调重组参数。
3. 下一步只设计独立的 `title-informed` 多查询检索干净基线，先预注册固定输入、查询族、指标、晋级门槛和停止条件；不联网、不修改生产检索。详细边界见第 13 节和路线图 Phase 11。

## 6. 锁状态

当前 `runs/candidate.lock.yaml` 绑定提交 `d70f1c5c3e84e36c7ddb07ffb42c2e65a8ccd5f1` 和 1989 条 ledger checkpoint，已被本轮 capture 使用。正式 ledger 现为 2049 条，因此该锁已失效，不得再次用于 live run；2026-08-10 的 readiness 也已过期。`runs/dev-20260810T104256Z-d9e89476d484/replay.lock.yaml` 仅用于该密封快照，已成功完成 replay。任何新 live run 都必须先按最新 checkpoint 重建独立锁并刷新 readiness。

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
- This phase did not read `.env`, make reservations, rebuild the candidate lock, run readiness, or execute live capture/replay/compare/validation. At that checkpoint, the live bounded run remained separately authorized; its later result is recorded below.
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
- At that checkpoint, the next action was a separately authorized three-query DeepSeek canary using the exact canary lock. The later authorized result and the independently revised Prompt-v2 path are recorded below; this historical lock must not be reused.

## 11. Query Evolution three-query canary result (2026-08-10)

- The authorized canary used `runs/_locks/query_evolution_contract-20260810/canary.lock.json` and wrote evidence to `runs/_diag_query_evolution_contract-canary-20260810/`.
- It made exactly 3 DeepSeek LLM calls and 0 OpenAlex/search calls. The evidence contains 3 LLM snapshots and no OpenAlex snapshot.
- Outcomes were 2 `generated` and 1 `integrity_failure`; the failed query was rejected for duplicate subquery text after canonicalization. The canary result is `promoted=false` with fixed reason `canary_accounting_failed`.
- Aggregate usage was 3 LLM calls, 2,541 input tokens, 345 output tokens, 3,877 ms, and `0.003231` CNY. Ledger readback shows 3/3 receipts `settled`, actual usage present, no remaining `reserved` receipt, and the per-receipt sums match the aggregate usage.
- Snapshot manifest hash is `sha256:a3582e00ac3ebe24dcc78539e60a7fe843e52eefac85675c861e51be38fb4729`; snapshot set ID is `sha256:eefba557af57d4c33a9671412346ac9b36d522bae4ee65787a0bd378fcdd5fcb`.
- At that checkpoint, the fixed reason (`canary_accounting_failed`) disagreed with the terminal, numerically consistent ledger readback and blocked promotion. Subsequent offline work produced the independently reviewed Prompt-v2 path recorded below; this historical mismatch is not the current blocker and the old canary still must not be reused.
- The stop rule prohibited rerunning this canary unchanged. Subsequent offline diagnosis produced an independently reviewed Prompt-v2 hypothesis and lock; its full-probe result is recorded below. The historical canary lock remains non-reusable.

## 12. Query Evolution Prompt-v2 full probe and Gate repair (2026-08-10)

- The independently locked Prompt-v2 probe completed all 55 queries with 30 `generated` and 25 `no_op` outcomes. Capture and zero-network replay matched; all receipts reached terminal state, with no integrity, provenance, accounting, or snapshot failure.
- The sealed historical `result.json` remains unchanged and reports Gate A `failed`. That conclusion was caused by two evaluation-contract defects: comparing 55 executed queries with the 60-query frozen metric denominator, and using selected/Top-50 IDs as the Gate-B retrieval stream.
- Commits `3b4e94a` and `00168cb` repaired the contracts without changing ranking behavior: Gate A now checks locked versus terminal execution counts; Gate B uses full retrieved IDs; Gate C continues to use selected Top-50 IDs; canonical gold associations are counted once.
- Offline recomputation from the sealed evidence is Gate A `passed`, Gate B `passed`, Gate C `failed`: unique retrieved gold improves 14 → 15, while Top-50 gold remains 8 → 8. This proves a small retrieval signal but no usable ranking/output gain, so the probe does not qualify for formal production capture.
- The repair is merged into `main` at `d70f1c5`. Do not rewrite historical evidence or rerun the same live probe; diagnose why the added gold does not survive Top-50 offline first.

## 13. Verified identifier rescore 与 sealed query recomposition 封存结论（2026-08-11）

- Verified identifier rescore 已完成并发布：当前正式基线 selected verified Gold 为 `17/143`，旧版 title 外部基准为 `30/143`。对应证据为 `docs/evidence/identifier-map-semantic-rescore-2026-08-11.json` 与 `docs/identifier-map-semantic-rescore-2026-08-11.md`。
- 为判断 Prompt-v2 的封存槽位中是否仍有可利用排序信号，已预注册并执行一次且仅一次零网络 `append_v2 / round_robin_slots / rrf_slots_k60` 对照。正式证据为 `docs/evidence/sealed-query-recomposition-offline-2026-08-11.json` 与 `docs/sealed-query-recomposition-offline-2026-08-11.md`。
- 完整性基准精确复现：`append_v2` 的阶段计数为 `not_retrieved=101 / filtered_out=0 / ranked_outside_top50=23 / selected_top50=19`，总数守恒为 143，三种方法的 retrieved/post-filter 集合一致。
- `round_robin_slots` 得到 24 个 selected Gold，但没有保留 append 的全部已选 Gold，因此不满足可用信号门槛；`rrf_slots_k60` 得到 25 个 selected Gold，保留 append 命中且指标不回退，被判定为可用重组信号。
- RRF 的 25 仍低于旧版 title 基准 30，正式结论固定为 `signal_insufficient`，reason code 为 `usable_signal_below_legacy_benchmark`。这证明现有封存候选中存在一定排序信号，但仅靠重组无法恢复旧版水平；主要瓶颈仍是 `not_retrieved=101/143`。
- 停止条件已经触发：不得重跑本次正式命令，不得继续增加重组变体、调 RRF 参数或按查询挑选方法，也不得据此重建 candidate lock、刷新 readiness 或启动 live capture。
- 下一阶段的唯一建议是建立独立、干净的 `title-informed` 多查询检索基线：先写设计并预注册固定查询构造、指标、晋级门槛和停止条件；设计阶段只使用现有封存材料，不联网、不修改生产检索、不读取 `.env` 或 ledger。
- 新基线的首要判据必须针对检索覆盖，而非先调排序：在相同 verified identifier 语义和 143 个 Gold 关联分母下，首先证明 `not_retrieved` 明显低于 101，再评估 Top-50 是否达到或超过外部基准 30。未降低 `not_retrieved` 时立即停止查询工程，转向数据源覆盖、标识符映射或 Gold/reference 输入诊断。
- 相关实现、证据与验证器已经合并并推送到 `main`，实现与证据链截至 `2a538fc3643a1309ee8bf247039ac6639173ddc8`；本节的 HANDOFF/路线图封存提交位于其后。最终离线验证在含完整 sealed 材料的 week3 工作区为 `2233 passed, 35 skipped, 1 online deselected`，Ruff 与 mypy 均通过。
- 主工作区缺少 Git 忽略的 sealed run、Gold 和 identifier-map 本地材料，因此在那里运行全量测试会有 58 个环境性失败；同一提交已在 week3 工作区通过完整离线测试。主工作区不依赖这些私有材料的相关测试为 `102 passed`。
- `HANDOFF.md` 与 `docs/retrieval-roadmap.md` 已由 `cabf67c` 提交；当前 week3 只剩三项用户未跟踪路径：`data/budget_ledger.sqlite3`、`deliverables/` 和 `docs/evidence/identifier-map-semantic-audit-2026-08-10.json`，均不得清理、覆盖或误提交。

## 14. Scheme B 模块化候选召回离线验收（2026-08-12）

- 离线验收已在 `D:\AI Projects\.worktrees\week3` 完成：四个模块组分别为 `70`、`42`、`23`、`68` passed；`tests/recall_experiments` 为 `217 passed`；全仓库 `pytest -m "not online" -q` 为 `2452 passed, 35 skipped, 1 deselected`。跳过项仅是 Windows 平台不支持 symlink 或 POSIX inode 语义的已标注测试。
- 质量门结果：对 `248` 个 tracked Python 文件的 Ruff 为 `All checks passed!`；`mypy src scripts` 为 `135 source files`、`0` errors；`git diff --check` exit `0`。
- 模块化能力已验证：文本、标题和引用扩展三个检索 handler 可独立注册/替换且相互不导入；候选池是版本化、保留 provenance、Gold-independent 的完整去重投影；runner 仅编排注入接口，不含方法 ID、action-type 分支、provider 构造器、历史常量或排序指标；Phase-1 报告模型不含 Precision/F1/MRR/NDCG/Top-K 字段。
- 离线安全边界已验证：novel action 在 snapshot replay 中以 `snapshot_unavailable` fail-closed，不会回退到 live；LLM `initial`/`repair` 均有独立预算 reservation 和终态处理；本次没有读取 `.env`、联网、构造 live provider、写 ledger 或执行 live Scheme B repeat。
- 历史证据仍为精确受限结论：Query Rewrite、LLM Query Variants 与 Title Candidates 是 `aggregate_only`；Query Evolution 的 action/provider snapshot bytes 可验，但缺少独立绑定的 identifier-map/per-query Gold 命中，故为 `not_comparable`；Citation Expansion 为 `insufficient_historical_evidence`。Scheme B 总结仍为 `insufficient_historical_evidence`，不能给出 overall `passed`。
- 5 个 historical bindings 的 9 个 source hashes 加 1 个 Gold association hash（共 `10/10`）与当前 bytes、Task 0 inventory 及 Task 0 review binding 完全一致；其中 1 个路径 tracked、9 个由 `.gitignore` 覆盖。Oracle Gold-document catalog 仍为 `oracle_catalog_blocked`，未授权前不得执行 Oracle/DeepSeek 或任何 live comparison。
- 用户路径保持本轮开始时的未跟踪状态且未写入：`data/budget_ledger.sqlite3` SHA-256 `56f7f6871a6c642223786900462e81c8a877824a94bd952dfd2b395268956750`、`docs/evidence/identifier-map-semantic-audit-2026-08-10.json` SHA-256 `79e0c02ab59f8e61d243d8f5d73a2d713c6cd1d846aff61c6f2a3aa90ea03c8c`、`deliverables/`（2059 files）tree SHA-256 `6aa105b4bda3482fe8bb64a7097b6fc73d92531c3be29010ddc1a3ef2ae8a6b1`。

## 15. Scheme B 可验证 live runtime（2026-08-12）

- 已补齐一次性的 live 基础设施缺口：现有 OpenAlex、Semantic Scholar 与 DeepSeek capture 对象从其真实冻结执行配置提供只读身份，方案 B 的 budgeted adapters 绑定同一实际 `ActualCostPricer` 与 `HardBudgetController`，组合层不接受调用方自报身份。
- LLM 与检索请求使用的 endpoint、model、dependency、adapter 和 identity 来自同一个冻结且带完整性校验的 transport config；相似域名、userinfo、query/fragment/port、非法 endpoint/operation、对象篡改和 exact-surface duck object 都会在 provider dispatch 前终止。
- 价格策略、完整预算、`formal_live=True`、reservation TTL 与控制器版本进入运行身份；未知实际成本仍 fail-closed。CLI 继续要求 recipe 的 live backend 与运行时 `--allow-live` 双重满足。
- Prompt/搜索槽位方法今后只改 Prompt/YAML；预生成搜索词走现有 manual-action；新生成或检索方法只实现并注册对应模块。禁止新增方法专属 runner、候选池、evaluator、产物格式、比较命令或独立脚本。
- 离线 `httpx.MockTransport` 已验证合法 runtime 可进入公共生成/检索链路，未授权时零 provider calls；snapshot/replay、manual/fixed actions、严格 execution identity 和历史结论均保持兼容。独立 whole-feature 复审为 Critical 0 / Important 0 / Minor 0。
- 本阶段没有读取 `.env`、没有调用 DeepSeek/OpenAlex/Semantic Scholar、没有写 ledger，也没有产生方案 B 性能结果。下一步仍是单独授权的 Query Evolution 三查询 canary：使用已有冻结 binding 和公共 runner，封存一次 live capture 后立即离线 exact replay，不再写新方法框架。
