# 第八届中国研究生人工智能创新大赛项目文档：项目总结草稿

> 本稿依据 `D:\AI Projects\赛题指南\附件2：第八届中国研究生人工智能创新大赛项目文档模版.pdf` 的四级结构整理（封面、目录、记录更改历史、1 项目概况、2 项目规划、3 实施方案、4 参考资料）。
> 当前内容是工程事实总结，不代表已获得真实 Gate、网络、成本或公开发布证据。

## 封面信息

- 项目名称：离线优先、可复现、可审计的学术论文检索与评测系统
- 版本号码：Week 1–4 integrated baseline / Phase 4 Task 4 review-fix pending acceptance
- 文档日期：2026.08.03
- 团队名称：待补充
- 参赛组别：待补充

## 目录

1. 项目概况
   - 1.1 背景和基础
   - 1.2 场景和价值
   - 1.3 所需支持
2. 项目规划
   - 2.1 整体目标
   - 2.2 技术创新点
3. 实施方案
   - 3.1 技术可行性分析
   - 3.2 技术细节
   - 3.3 计划和分工
4. 参考资料

## 记录更改历史

| 序号 | 更改原因 | 版本 | 作者 | 更改日期 | 备注 |
|---|---|---|---|---|---|
| 1 | Week 1–4 集成工程与 Phase 3 formal evidence 完成 | e076c8f | Codex | 2026.08.03 | Phase 3-C 最终复核 PASS，C0/I0/M0；真实 E3/E4 延后 |
| 2 | Phase 4 Task 1 API/router 完成 | d125da8 + c029f5c | Codex | 2026.08.03 | typed errors、readiness、三重 live authorization、atomic publication |
| 3 | Phase 4 Task 2 UI/API 与发行包完成 | a8b2108 + 8f59777 | Codex | 2026.08.03 | 浏览器 UI 走 `/v1/search`；wheel/sdist 含 Python source 与 UI assets |
| 4 | Phase 4 Task 3 serve 生命周期完成 | a5928bc..d663da0 | Codex | 2026.08.03 | lineage、TOCTOU、SIGTERM、真实 fake-live cleanup；C0/I0/M0 |
| 5 | Phase 4 Task 4 实验模块初版 | 8ce02a8 | Codex | 2026.08.03 | registry/async stages/evolution 初版；复核发现 C1/I3 |
| 6 | Phase 4 Task 4 review-fix 实现提交 | ddf9972 | Codex | 2026.08.03 | production composition、typed failure、snapshot provenance、取消清理已补齐；待同一审查者完整范围复核 |
| 7 | 按大赛文档模板（附件2）核对并完善项目总结 | 03eec8e+（工作树） | Claude Code | 2026.08.03 | 增加目录、技术对比调研与预期指标、国内外相关项目、潜在价值、竞赛时间点说明与外部参考资料；按 PRD 第 1–4 周 × Task 1–14 重写 3.3 计划并补入数据模型、预算配置、评测契约与自适应演化创新点；未改变工程事实表述 |

## 1 项目概况

### 1.1 背景和基础

本项目针对大赛赛题三「科研场景下复杂学术查询的智能论文搜索与推荐」（官方评分重点为 F1 指标，权重 70%）：系统需对自然语言描述的复杂学术查询自动完成查询理解、约束识别、分解与改写，基于大模型的自主搜索及迭代策略调整，候选论文过滤和细粒度综合排序，以及结果归纳、列表展示和论文关系展示。

学术论文检索同时面临多源异构、结果可解释性不足、外部服务不可重复、预算不可控和评测证据难以审计等问题：通用对话式检索结果不可重放，官方平台接口的版本、限流与计费不可控，自建检索库又缺少调用级证据链。本项目从这一痛点出发，以离线优先为原则，构建统一的论文检索、证据记录、冻结数据、预算账本、formal evaluation、API/UI 和可选实验模块。

国内外相关项目：本项目的数据基础来自国外公开学术元数据平台 OpenAlex 与 Semantic Scholar（大规模开放的研究元数据与引用接口）；通用对话式/聚合式检索系统交互便捷，但结果随在线数据与模型版本漂移、不可逐次重放，且接口的版本、限流与计费不可控（对比见 1.2）。本项目不重建任何数据源，而是构建位于上游数据源与用户之间的可重放、可审计、预算可控的检索与评测工程层。

已有基础包括：

- Python 3.11+ 工程与严格领域模型、配置/锁文件和依赖快照体系；
- OpenAlex、Semantic Scholar、LLM 等依赖的可验证 snapshot/capture/replay 适配器；
- V2 frozen manifest、identifier-map、safe path、symlink/TOCTOU 防护；
- 分层 request/run/project ledger、reservation/receipt、硬预算和 cancellation cleanup；
- 统一 SearchApplicationService、FastAPI API、浏览器 UI 和 `paper-search serve` 生命周期；
- formal capture、verify-run、compare-replay、Gate measures 和审计报告。

团队构成：待补充（参赛团队信息未定稿，涉及成员分工见 3.3）。

### 1.2 场景和价值

适用场景包括：科研人员按主题检索论文、企业研发团队进行技术情报初筛、教育/科研机构对检索系统做可复现实验、以及对检索结果进行引用证据和预算审计。

潜在价值：学术检索是科研情报、企业技术调研与可复现评测等服务链路的入口环节，该类服务的需求随科研产出的持续增长而稳定存在。本项目的差异化价值在于以工程手段解决「结果不可重放、成本不可解释、证据不可审计」三类共性问题。市场规模与商业回报未做独立的公开数据核实，本稿不给出未经核实的定量数字。

对比性分析（设计层面的定性判断，尚未进行真实线上基准测评）：

| 对比维度 | 常见方案 | 本项目设计 |
|---|---|---|
| 结果可复现 | 直接调在线接口，结果随版本/限流漂移 | frozen manifest + snapshot capture/replay，同一输入可确定性重放 |
| 成本可控 | 无逐次预算记账，账单事后不可解释 | reservation→execute→settle、硬预算上限、usage 账本 |
| 证据可审计 | 仅返回 top-k 列表，无调用级证据 | 请求身份、响应哈希、prompt artifact、source lineage、Gate measures |
| 评测可信 | 依赖 run 自报结果 | formal verify-run 从冻结输入与 decoded evidence 重算指标 |
| 系统一致性 | UI/评测/API 各自独立 pipeline | 统一 SearchApplicationService 与 CompositionRoot |

项目价值不以未验证的线上效果数字表述，而体现在工程可审计性：同一 frozen input 与 snapshot 可重复 replay；每次外部依赖调用具有请求身份、响应哈希、缓存/快照引用和 usage 账本；失败、取消、预算耗尽和发布状态具有明确边界；API、CLI、UI 和 evaluation 共享同一应用服务。

### 1.3 所需支持

- 算力与硬件：本地 CPU（Python 3.11+ 与 uv 运行环境）、磁盘（冻结数据与 snapshot）、SQLite/文件系统即可；当前 optional LLM/embedding 阶段保持 default-off，GPU 不构成完成条件；
- 软件与数据：OpenAlex、Semantic Scholar 等公开学术元数据及其已冻结的 V2 manifest 与 deterministic snapshot；
- 培训与人力：大赛规则与文档模板解读、代码复核与提交规范培训；团队分工见 3.3；
- 条件性授权：若要推进真实 Gate/在线评测（E3/E4 等），需要受控网络、供应商凭证、一次性预算上限、人工验收与安全的 artifact 存储；当前未进行真实网络、真实成本或公开状态操作。

## 2 项目规划

### 2.1 整体目标

参赛期间目标是交付一个可展示的 replay-first 原型系统：

1. 通过 `paper-search smoke/evaluate/verify-run/compare-replay/serve` 使用统一服务；
2. 通过浏览器 UI 提交 canonical `SearchRequest`，展示论文、证据、来源、usage、partial/fallback、snapshot/run/config provenance；
3. 用 frozen V2 数据、snapshot bytes、完整账本和 formal measures 支撑可复现实验；
4. 让 citation expansion、LLM rerank、embedding 和多轮 evolution 以明确 experiment identity、default-off 和预算约束接入；
5. 在真实 Gate 未授权或证据不完整时保持 replay/default-off，不虚构线上结果。

行业初步验证属于后续目标：真实 provider/成本/公开状态操作需单独授权，当前阶段以原型展示与离线形式化验证为主。

产品目标（源自 PRD 第 4.1 节）：构建可独立批量评测的端到端论文搜索系统；在固定开发集上显著优于「原始查询直接调用单一搜索 API」的基础基线；支持复杂查询的结构化解析、子查询生成、多源召回、引文扩展和细粒度排序；对每个查询记录搜索轨迹、候选变化、调用次数、Token 与延迟；输出高度/部分相关论文列表及真实论文关系；提供一个可通过消融实验验证的创新点——预算感知的自适应查询演化。非目标：不自建全量学术搜索引擎、不训练大语言模型或大型 Embedding 模型、不无限深度遍历引文图、前端状态不参与评分、不生成文献综述正文。内部工程目标（非官方分数承诺）：主配置在开发集宏平均 F1 不低于 0.30 且相对「原始查询 + OpenAlex」基线绝对提升至少 0.03，验证集相对基线提升至少 0.02。

当前工程状态：Phase 3-C 已通过；Phase 4 Task 1–3 已通过；Phase 4 Task 4 的 review-fix 已提交并完成本地 focused/adjacent/static 验证，但尚未获得同一审查者对完整 commit range 的最终 C0/I0 复核。按 PRD 的任务映射与逐项状态见 3.3。

### 2.2 技术创新点

关键技术对比调研见 1.2 表格，核心创新点如下：

- **证据先于结论**：formal validation 不信任 run 内自报的 cost、retrieved IDs、filter IDs、relation hash 或 prompt identity，而是从冻结输入、verified lock、snapshot bytes、provider decoder 和账本重算；
- **全链路可复现**：live capture、replay lock、snapshot manifest、response bytes、LLM prompt artifact、project ledger checkpoint 形成绑定链；
- **安全预算控制**：reservation→execute→settle/fail 的顺序、硬上限、project history、取消清理和 fail-closed mismatch 统一处理；
- **统一服务边界**：CLI、FastAPI、UI、smoke/evaluate 和 serve 复用 SearchApplicationService，不维护 UI-only/evaluation-only pipeline；
- **实验身份隔离**：baseline 默认不构造 optional stage，named experiment 通过 registry 选择 embedding/citation expansion/LLM rerank/fixed-two-round/adaptive-evolution，并共享 executor、budget、snapshot 和 evaluation path；
- **预算感知的自适应查询演化（PRD 第 12 节定义的创新点）**：CoverageAnalyzer 统计每个强约束的候选覆盖情况，对未覆盖/低覆盖约束生成定向下一轮查询，先估算新增查询的调用与 Token 成本再执行，边际收益低于阈值或预算不足即停止，并与固定一轮、固定两轮做同预算同数据对照；进入主配置必须同时满足「稳定提升 + 全部泛化门槛」和以下任一收益条件——A）宏平均 F1 绝对提升至少 0.02；B）InternalScore 提升至少 0.02 且 F1 下降不超过 0.005、失败率增幅不超过 1 个百分点；未达门槛时使用完整 baseline 参赛，不强行启用创新模块；
- **可审计的 Gate measures**：retrieval、hard-filter、fabrication、provenance、sanitization、unaccounted usage 等指标必须从 decoded evidence 计算，缺证据时保持 unavailable 或 fail closed。

## 3 实施方案

### 3.1 技术可行性分析

数据来源采用冻结 V2 manifest、partition、identifier-map 和 deterministic dependency snapshots；不依赖实时网络即可运行主 baseline、replay、formal verify/compare 和大部分 API/UI/实验测试。评测数据（PRD 第 14.1 节）：首选 Hugging Face `CarlanLark/pasa-dataset`，开发/验证/模拟测试按 60/30/50 条分层划分并分别冻结、分开报告；备用方案为 `allenai/asta-bench`（v0.3.1 PaperFindingBench），两数据源不得混入同一 F1 分区；另有 24 条覆盖七类查询及中英文改写、不参与 F1 调参的压力集。行业知识获取：元数据来自 OpenAlex/Semantic Scholar 等公开学术 API 的确定性 snapshot，检索与评测规则沉淀为可读的 schema、适配器与 formal measures，不依赖单一供应商实时可达性。OpenAlex/Semantic Scholar/LLM 适配器均有 canonical request identity、response hash、safe headers 和 snapshot reader。

预算可行性与成本控制（PRD 第 7.1 节）：`budget_low.yaml` 固定 10000 tokens、0.10 元、最多精排 12 篇；`budget_balanced.yaml` 固定 24000 tokens、0.30 元、最多精排 30 篇；四周 API 总预算硬上限 200 元，达到 160 元时停止大规模开发集重复实验，只保留验证与最终复现额度；程序启动时若 Token/费用上限缺失则拒绝进入使用 LLM 的主配置，离线测试使用模拟 LLM 不消耗真实 Token。

算力需求以 CPU、本地磁盘和小规模 deterministic tests 为主，人力上已形成"实现—复核—负责人授权"的分工（见 3.3），足以支撑离线工程与形式化验证。真实 provider、真实 Gate 和大规模模型推理不是当前工程完成条件，必须在授权后单独执行；因此文档中不应把离线测试通过写成真实线上性能结论。

### 3.2 技术细节

系统分层如下：

- **Domain/lock/config**：严格 Pydantic contracts、RuntimeConfig、InputLock/ReplayLock/ValidationLock、prompt artifact SHA；
- **Storage/snapshot**：DependencySnapshotManifestV2、safe relative path、descriptor-stable reads、provider-specific decoder；
- **Application service**：统一 SearchApplicationService 与 CompositionRoot，负责 baseline/replay/live 组合；
- **Pipeline**：one-round executor、budget controller、optional experiment components、EvolutionCoordinator；
- **Evidence**：CaptureSession、atomic artifact publication、usage ledger、formal audit measures、verify-run/compare-replay；
- **Delivery**：typed FastAPI routing、cached readiness、browser static assets、`paper-search serve`。

关键流程设计：canonical `SearchRequest` → 预算 reservation →（可选）live 组合（受三重 live authorization 约束）→ provider/LLM 异步执行 → 快照与证据原子发布（publication-before-200）→ usage settle → 结构化结果与 diagnostics；失败、取消、预算耗尽收敛为 typed errors 与清理路径（任务取消会终止 active reservation 并清理客户端与 artifact）。

核心数据模型与预算默认值（PRD 第 7.1 节）：`QuerySpec`（original_query、research_goal、topics/methods/tasks/datasets/domains、year 范围、venues、must_have/should_have/exclusions、ambiguities）→ 固定 3–6 个 `SubQuery` 组成的 `SearchPlan` → 多源 `Paper`（canonical_id 与 provider ID 并存）→ 带 `CandidateEvidence`（匹配子查询/约束、过滤原因、各类分数、scoring_version）的 `RankedPaper` → `StructuredSearchResponse`（`selected_paper_ids` 是自动 F1 评分的唯一预测集合，high/partial 分组只用于解释展示）。`SearchBudget` 默认：搜索 API 目标 8/上限 12 次、LLM 目标 3/上限 5 次、最大 2 轮、最大 6 个子查询、精排候选 30 篇、输出 50 篇、引文种子目标 1/上限 2、软截止 80 秒/硬终止 90 秒；引文边必须通过 `ResolvedCitationEdge`（canonical 映射 + source_edge_hash）才进入最终响应。

评测契约（PRD 第 14 节，官方评分器发布前使用固定内部契约）：预测集合先按 `final_score` 阈值与 Top-K 唯一生成再去重评分；主指标为逐查询 F1 宏平均，微平均 F1、P/R/Recall@K 为辅助；金标准与预测均为空时该查询记为 1，仅一方为空记为 0；开发集只做一次网格选择（K∈{10,20,30,50}、阈值 0.45–0.75）后冻结，不在验证/测试集继续调参。内部近似得分 `InternalScore = 0.70×宏平均F1 + 0.20×EfficiencyScore + 0.10×StructuredOutputScore`，只用于内部比较。结构化验收线：Schema 合法率 100%、论文 ID/链接可验证率 ≥99%、理由完整率 ≥95%、展示引文边可验证率 100%、虚构论文或关系数量为 0。

Task 4 review-fix 已覆盖以下问题，但验收边界仍需复核：

1. `RuntimeConfig.experiment` 已接入 validated registry、CompositionRoot、request-scoped service/orchestrator，并为 fixed-two/adaptive 接入共享 `EvolutionCoordinator`；
2. optional stage 仅将明确的 availability failure 降级，Budget/Config/Snapshot/Integrity/Adapter 等受保护错误继续传播或 fail-closed；
3. `asyncio.CancelledError` 路径会终止 active reservation 并清理客户端；
4. citation/rerank snapshot refs 已进入 diagnostics、`OrchestratorResult`、`SearchExecutionResult` 和 evaluation adapter。

预期技术指标（设计保证，已在离线工程测试中验证；不代表真实线上性能）：

- 确定性：同一 frozen input 与 snapshot 可重复 replay（compare-replay）；
- 预算：硬上限 + fail-closed，无超支路径；
- 证据完整性：每项外部依赖调用具备请求身份、响应哈希与账本记录；
- 发布一致性：HTTP 200 之前完成 artifact 原子发布；
- 测试基线：Phase 3-C 主门禁 157 passed/2 skipped、Ruff/mypy/diff-check 通过；Task 4 review-fix 本地 249 个 focused/adjacent 测试与 10 个 smoke/serve/formal-path 测试通过；这些不等同于同一审查者的最终 C0/I0。

### 3.3 计划和分工

本项目按 PRD（四周研发总计划，PRD 第 12 节）组织为「第 1–4 周 × Task 1–14」：每周有明确阶段闸门，每个任务固定执行顺序「先写测试 → 确认失败 → 最小实现 → 通过测试 → 小规模真实数据检查 → 提交」。工程执行层使用 Phase 1–4 / Task 1–8 编号，两者对应关系为：PRD Task 1–8（第 1–2 周）对应 Phase 1–3 与 Phase 4 Task 1–3；PRD Task 9/10/12（第 3–4 周高级模块与创新点）的工程骨架由 Phase 4 Task 4 以 default-off 实验身份接入。下表状态只列有证据支持的事实：

| 周 | PRD Task | 内容（验收产物） | 对应工程交付 | 状态 |
|---|---|---|---|---|
| 第 1 周 | 1 | 项目骨架、配置、领域模型、硬预算计数器 | Phase 1–2 | 完成 |
| 第 1 周 | 2 | 数据集适配与评测指标：P/R/F1/Recall@K、official adapter、60/30/50 分区、24 条压力集 | Phase 3 评测基础 | 工程完成；真人标签、gold 与正式冻结待完成（`waiting_for_human_label_freeze`） |
| 第 1 周 | 3 | OpenAlex 检索、SQLite 缓存、不可变快照与 SHA-256 manifest、归一化 | Phase 2 | 完成（含独立复核包） |
| 第 1 周 | 4 | 去重、硬约束过滤、词法排序、评测 runner | Phase 2–3 | 工程完成；第 1 周闸门受标注数据阻塞 |
| 第 2 周 | 5 | LLM 客户端、查询解析/规划（验收：结构化成功率 99%、强约束抽取 Recall 90%） | Phase 2–3（prompt 绑定与 artifact SHA） | 工程完成；量化指标验收待标注数据，未宣称达成 |
| 第 2 周 | 6 | Semantic Scholar 多源融合（RRF 与加权融合均可配置） | Phase 2 | 完成 |
| 第 2 周 | 7 | 预算控制（预留/结算、软硬截止、持久化）与最小编排器 | Phase 2（budget/ledger/orchestrator） | 完成 |
| 第 2 周 | 8 | `POST /v1/search`、健康检查、baseline 集成与可复现预测文件 | Phase 3 + Phase 4 Task 1 | 完成（第 2 周闸门达成） |
| 第 3 周 | 9 | Embedding、引文扩展、LLM 精排（每项可独立关闭，消融通过才进主配置） | Phase 4 Task 4（registry，default-off） | 实现已提交待最终复核；真实消融待标注数据 |
| 第 3 周 | 10 | 实验记录、bootstrap 置信区间、消融框架、参数选择 | Phase 4 Task 4（experiments/ablations/statistics） | 框架实现已提交待最终复核；真实消融待标注数据 |
| 第 3 周 | 11 | 稳定性（429/5xx/超时）、无缓存批量评测、泛化测试、最小展示 | Phase 4 Task 2/3 | 部分完成：稳定性与浏览器 UI 完成；真实泛化评测（E3/E4）延后 |
| 第 4 周 | 12 | 预算感知的自适应查询演化（创新点，见 2.2） | Phase 4 Task 4（evolution/coordinator 等） | 实现已提交待最终复核；同预算同数据对照需授权与标注数据 |
| 第 4 周 | 13 | 最终验证与版本冻结（依赖、模型、Prompt、阈值、随机种子） | data-freeze V2 计划 | 部分：冻结 manifest/ID 已固定；最终验证 deferred |
| 第 4 周 | 14 | 展示、文档、答辩 | 本文档 | 文档撰写中；演示与答辩未开始 |
| — | Phase 4 Task 5–8 | dual-mode E2E、浏览器 Gate 5、文档收敛、Gate 6 ablations/promotion | Phase 4 后续 | 待开始/需授权 |

阶段闸门现状：第 1 周闸门未闭合（真人标签、gold、正式冻结缺失）；第 2 周闸门达成（预算内合法结构化结果与可复现预测文件）；第 3 周闸门部分达成（配置与消融框架齐全，≥3 组核心消融依赖标注数据）；第 4 周竞赛交付完成定义（PRD 第 15.5 节）未全部满足。

推进顺序：先完成文档交付，再按「Phase 4 Task 4 完整范围复核 → 定向修复 → Task 5–8」推进工程，真实 Gate/线上操作以授权为前提。推进节奏以大赛官方时间节点（报名—初赛—决赛）为里程碑，官方具体日期以大赛通知为准，提交前由团队负责人核对。

团队分工建议：实现者负责代码与定向测试；复核者按完整 commit range 检查 C/I/M；项目负责人确认授权、真实 Gate、文本事实和最终提交范围。

## 4 参考资料

- 第八届中国研究生人工智能创新大赛项目文档模板：`附件2：第八届中国研究生人工智能创新大赛项目文档模版.pdf`（用户提供）。
- 项目 PRD（四周研发总计划，含 Task 1–14、评测方案、验收标准）：`PRD.md`。
- 项目 Phase 4 计划：`docs/superpowers/plans/2026-07-30-week1-4-phase4-api-ui-experiments.md`。
- Week 3/4 模块计划：`docs/superpowers/plans/2026-07-28-week3-task9-embedding-ranking.md`、`2026-07-28-week3-task10-experimentation.md`、`2026-07-28-task10-experiment-ablation.md`、`2026-07-29-data-freeze-v2.md`。
- Phase 3-C 最终复核：`.superpowers/sdd/p3c-review-05824ad..e076c8f.md`。
- Phase 4 Task 1/2/3/4 报告：`.superpowers/sdd/phase4-task-1-report.md`、`.superpowers/sdd/phase4-task-2-report.md`、`.superpowers/sdd/phase4-task-3-review-package-final.md`、`.superpowers/sdd/phase4-task-4-report.md`。
- 关键提交：`e076c8f`、`d125da8`、`c029f5c`、`a8b2108`、`8f59777`、`d663da0`、`8ce02a8`、`ddf9972`。
- OpenAlex 文档：https://docs.openalex.org（公开学术元数据 API）。
- Semantic Scholar API：https://www.semanticscholar.org/product/api。
- 技术栈参考：FastAPI（https://fastapi.tiangolo.com）、Pydantic（https://docs.pydantic.dev）、uv（https://docs.astral.sh/uv）。

## 当前写作边界

- 可以写：架构、工程创新、离线可行性、replay/API/UI/实验设计、已通过的测试事实。
- 不可以写成已完成：真实 Gate 通过、真实 provider 成本、线上性能、浏览器 Gate 5 验收、Gate 6 promotion、公开发布状态。
- “可行”“支持”“设计为”与“已验证”必须区分；所有数字只引用对应测试/报告，不外推到真实线上效果。
