# 第八届中国研究生人工智能创新大赛项目文档：项目总结草稿

> 本稿依据 `D:\AI Projects\赛题指南\附件2：第八届中国研究生人工智能创新大赛项目文档模版.pdf` 的四级结构整理。
> 当前内容是工程事实总结，不代表已获得真实 Gate、网络、成本或公开发布证据。

## 封面信息

- 项目名称：离线优先、可复现、可审计的学术论文检索与评测系统
- 版本号码：Week 1–4 integrated baseline / Phase 4 Task 4 review-fix pending acceptance
- 文档日期：2026.08.03
- 团队名称：待补充
- 参赛组别：待补充

## 记录更改历史

| 序号 | 更改原因 | 版本 | 作者 | 更改日期 | 备注 |
|---|---|---|---|---|---|
| 1 | Week 1–4 集成工程与 Phase 3 formal evidence 完成 | e076c8f | Codex | 2026.08.03 | Phase 3-C 最终复核 PASS，C0/I0/M0；真实 E3/E4 延后 |
| 2 | Phase 4 Task 1 API/router 完成 | d125da8 + c029f5c | Codex | 2026.08.03 | typed errors、readiness、三重 live authorization、atomic publication |
| 3 | Phase 4 Task 2 UI/API 与发行包完成 | a8b2108 + 8f59777 | Codex | 2026.08.03 | 浏览器 UI 走 `/v1/search`；wheel/sdist 含 Python source 与 UI assets |
| 4 | Phase 4 Task 3 serve 生命周期完成 | a5928bc..d663da0 | Codex | 2026.08.03 | lineage、TOCTOU、SIGTERM、真实 fake-live cleanup；C0/I0/M0 |
| 5 | Phase 4 Task 4 实验模块初版 | 8ce02a8 | Codex | 2026.08.03 | registry/async stages/evolution 初版；复核发现 C1/I3 |
| 6 | Phase 4 Task 4 review-fix 实现提交 | ddf9972 | Codex | 2026.08.03 | production composition、typed failure、snapshot provenance、取消清理已补齐；待同一审查者完整范围复核 |

## 1 项目概况

### 1.1 背景和基础

学术论文检索同时面临多源异构、结果可解释性不足、外部服务不可重复、预算不可控和评测证据难以审计等问题。本项目以离线优先为原则，构建统一的论文检索、证据记录、冻结数据、预算账本、formal evaluation、API/UI 和可选实验模块。

已有基础包括：

- Python 3.11+ 工程与严格领域模型、配置/锁文件和依赖快照体系；
- OpenAlex、Semantic Scholar、LLM 等依赖的可验证 snapshot/capture/replay 适配器；
- V2 frozen manifest、identifier-map、safe path、symlink/TOCTOU 防护；
- 分层 request/run/project ledger、reservation/receipt、硬预算和 cancellation cleanup；
- 统一 SearchApplicationService、FastAPI API、浏览器 UI 和 `paper-search serve` 生命周期；
- formal capture、verify-run、compare-replay、Gate measures 和审计报告。

### 1.2 场景和价值

适用场景包括：科研人员按主题检索论文、企业研发团队进行技术情报初筛、教育/科研机构对检索系统做可复现实验、以及对检索结果进行引用证据和预算审计。

项目价值不以未验证的线上效果数字表述，而体现在工程可审计性：同一 frozen input 与 snapshot 可重复 replay；每次外部依赖调用具有请求身份、响应哈希、缓存/快照引用和 usage 账本；失败、取消、预算耗尽和发布状态具有明确边界；API、CLI、UI 和 evaluation 共享同一应用服务。

### 1.3 所需支持

离线工程验证只需要本地 Python/uv、测试运行时、SQLite/文件系统和 deterministic fake transport。后续若要完成真实 Gate/E3/E4，需要明确授权、受控网络/供应商凭证、一次性预算上限、人工验收和安全的 artifact 存储；当前没有进行真实网络、真实成本或公开状态操作。

## 2 项目规划

### 2.1 整体目标

参赛期间目标是交付一个可展示的 replay-first 原型系统：

1. 通过 `paper-search smoke/evaluate/verify-run/compare-replay/serve` 使用统一服务；
2. 通过浏览器 UI 提交 canonical `SearchRequest`，展示论文、证据、来源、usage、partial/fallback、snapshot/run/config provenance；
3. 用 frozen V2 数据、snapshot bytes、完整账本和 formal measures 支撑可复现实验；
4. 让 citation expansion、LLM rerank、embedding 和多轮 evolution 以明确 experiment identity、default-off 和预算约束接入；
5. 在真实 Gate 未授权或证据不完整时保持 replay/default-off，不虚构线上结果。

当前工程状态：Phase 3-C 已通过；Phase 4 Task 1–3 已通过；Phase 4 Task 4 的 review-fix 已提交并完成本地 focused/adjacent/static 验证，但尚未获得同一审查者对完整 commit range 的最终 C0/I0 复核。

### 2.2 技术创新点

- **证据先于结论**：formal validation 不信任 run 内自报的 cost、retrieved IDs、filter IDs、relation hash 或 prompt identity，而是从冻结输入、verified lock、snapshot bytes、provider decoder 和账本重算；
- **全链路可复现**：live capture、replay lock、snapshot manifest、response bytes、LLM prompt artifact、project ledger checkpoint 形成绑定链；
- **安全预算控制**：reservation→execute→settle/fail 的顺序、硬上限、project history、取消清理和 fail-closed mismatch 统一处理；
- **统一服务边界**：CLI、FastAPI、UI、smoke/evaluate 和 serve 复用 SearchApplicationService，不维护 UI-only/evaluation-only pipeline；
- **实验身份隔离**：baseline 默认不构造 optional stage，named experiment 通过 registry 选择 embedding/citation expansion/LLM rerank/fixed-two-round/adaptive-evolution，并共享 executor、budget、snapshot 和 evaluation path；
- **可审计的 Gate measures**：retrieval、hard-filter、fabrication、provenance、sanitization、unaccounted usage 等指标必须从 decoded evidence 计算，缺证据时保持 unavailable 或 fail closed。

## 3 实施方案

### 3.1 技术可行性分析

数据来源采用冻结 V2 manifest、partition、identifier-map 和 deterministic dependency snapshots；不依赖实时网络即可运行主 baseline、replay、formal verify/compare 和大部分 API/UI/实验测试。OpenAlex/Semantic Scholar/LLM 适配器均有 canonical request identity、response hash、safe headers 和 snapshot reader。

算力需求以 CPU、本地磁盘和小规模 deterministic tests 为主。真实 provider、真实 Gate 和大规模模型推理不是当前工程完成条件，必须在授权后单独执行；因此文档中不应把离线测试通过写成真实线上性能结论。

### 3.2 技术细节

系统分层如下：

- **Domain/lock/config**：严格 Pydantic contracts、RuntimeConfig、InputLock/ReplayLock/ValidationLock、prompt artifact SHA；
- **Storage/snapshot**：DependencySnapshotManifestV2、safe relative path、descriptor-stable reads、provider-specific decoder；
- **Application service**：统一 SearchApplicationService 与 CompositionRoot，负责 baseline/replay/live 组合；
- **Pipeline**：one-round executor、budget controller、optional experiment components、EvolutionCoordinator；
- **Evidence**：CaptureSession、atomic artifact publication、usage ledger、formal audit measures、verify-run/compare-replay；
- **Delivery**：typed FastAPI routing、cached readiness、browser static assets、`paper-search serve`。

Task 4 review-fix 已覆盖以下问题，但验收边界仍需复核：

1. `RuntimeConfig.experiment` 已接入 validated registry、CompositionRoot、request-scoped service/orchestrator，并为 fixed-two/adaptive 接入共享 `EvolutionCoordinator`；
2. optional stage 仅将明确的 availability failure 降级，Budget/Config/Snapshot/Integrity/Adapter 等受保护错误继续传播或 fail-closed；
3. `asyncio.CancelledError` 路径会终止 active reservation 并清理客户端；
4. citation/rerank snapshot refs 已进入 diagnostics、`OrchestratorResult`、`SearchExecutionResult` 和 evaluation adapter；
5. 本地证据：249 个 focused/adjacent 测试通过，10 个 smoke/serve/formal-path 测试通过，Ruff、mypy、diff-check 通过；这些不等同于同一审查者的最终 C0/I0。

### 3.3 计划和分工

| 阶段 | 状态 | 主要工作 |
|---|---|---|
| Phase 1–2 | 完成 | freeze、locks、snapshot、ledger、artifact、budget、replay 基础 |
| Phase 3-A/B/C | C 已完成 | formal runner、validator、Gate measures、provenance/sanitization/账本验证 |
| Phase 4 Task 1 | 完成 | typed API errors/readiness/router、三重 live authorization |
| Phase 4 Task 2 | 完成 | canonical browser UI、`/v1/search`、wheel/sdist assets |
| Phase 4 Task 3 | 完成 | serve lifecycle、lineage、TOCTOU、SIGTERM、capture cleanup |
| Phase 4 Task 4 | 实现已提交，待最终复核 | experiment registry、async optional stages、production/evolution wiring、typed failure/cancellation/provenance 修复；下一步是同一审查者完整范围 C0/I0 复核 |
| Phase 4 Task 5–8 | 待开始/需授权 | dual-mode E2E、浏览器 Gate 5、文档收敛、Gate 6 ablations/promotion |

团队分工建议：实现者负责代码与定向测试；复核者按完整 commit range 检查 C/I/M；项目负责人确认授权、真实 Gate、文本事实和最终提交范围。

## 4 参考资料

- 第八届中国研究生人工智能创新大赛项目文档模板：用户提供的附件 PDF。
- 项目 Phase 4 计划：`docs/superpowers/plans/2026-07-30-week1-4-phase4-api-ui-experiments.md`。
- Phase 3-C 最终复核：`.superpowers/sdd/p3c-review-05824ad..e076c8f.md`。
- Phase 4 Task 1/2/3 报告：`.superpowers/sdd/phase4-task-1-report.md`、`.superpowers/sdd/phase4-task-2-report.md`、`.superpowers/sdd/phase4-task-3-review-package-final.md`。
- 关键提交：`e076c8f`、`d125da8`、`c029f5c`、`a8b2108`、`8f59777`、`d663da0`、`8ce02a8`、`ddf9972`。

## 当前写作边界

- 可以写：架构、工程创新、离线可行性、replay/API/UI/实验设计、已通过的测试事实。
- 不可以写成已完成：真实 Gate 通过、真实 provider 成本、线上性能、浏览器 Gate 5 验收、Gate 6 promotion、公开发布状态。
- “可行”“支持”“设计为”与“已验证”必须区分；所有数字只引用对应测试/报告，不外推到真实线上效果。
