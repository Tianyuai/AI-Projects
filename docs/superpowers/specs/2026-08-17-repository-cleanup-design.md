# GitHub 仓库主线清理设计

**日期：** 2026-08-17

**远端：** `https://github.com/Tianyuai/AI-Projects.git`

**批准方案：** 依赖驱动整理；保留提交历史，不重写历史

## 1. 目标

把 GitHub 当前目录收敛为两条仍然有效的能力：

1. 已验证的论文检索端到端系统；
2. 当前用于单模块逐步优化的候选召回框架。

清理后，队友从 `main` 拉取即可找到唯一运行入口、唯一架构说明和仍受支持的测试，不再把历史实验、已否决生产模块或一次性诊断工具误认为当前主线。

## 2. 已确认现状

- 远端 `main` 当前指向 `2a538fc3643a1309ee8bf247039ac6639173ddc8`。
- 当前稳定提交 `6323244` 是远端 `main` 的后代，领先 59 个提交，因此可以无强推地更新 `main`。
- 初次盘点时当前工作区有 102 项未提交或未跟踪内容；设计提交后已增至 103 项，说明模块优化仍在继续。用户已确认这些内容属于正在进行的单模块优化工作。
- 实施前必须重新生成受保护路径、状态和内容哈希清单。清单内工作必须保持原样，不得由本次清理暂存、修改、删除或推送到稳定 `main`；实施期间由用户或其他任务产生的新变化也不得被回滚。
- GitHub 仍有 10 个旧远端开发分支。用户已批准在新 `main` 验证完成后删除这些分支，但保留其提交历史，不执行历史重写。

## 3. 非目标与安全边界

- 不改写 Git 历史，不使用强推，不压缩或重建仓库历史。
- 不读取、打印或提交 `.env` 和任何凭据。
- 不提交本地 `runs/`、`deliverables/`、预算账本、私有训练数据或 provider 原始响应。
- 不执行 live provider 调用、正式 validation 或产生外部成本的运行。
- 不把当前未提交的学习、语义路由或 evidence-steered 优化并入稳定主线。
- 不以文件名、提交日期或“默认关闭”为唯一删除依据；删除必须同时满足已否决、当前入口不可达或已由公共框架替代，并通过依赖检查验证。

## 4. 采用的整理方式

采用依赖驱动整理，而不是只删文档，也不创建无历史的新仓库。

清理从已提交的稳定 `HEAD` 创建独立 `codex/repository-cleanup` 分支和隔离工作树。所有修改、测试、提交和推送都在隔离工作树完成。当前脏工作树继续承载未完成的模块优化，清理流程不对其执行 stash、reset、checkout 或自动格式化。

## 5. 最终支持面

### 5.1 生产端到端主链

保留以下命令及其直接依赖：

- `paper-search smoke`
- `paper-search serve`
- `paper-search evaluate`
- `paper-search verify-run`
- `paper-search compare-replay`
- `paper-search ranking-metrics`

端到端数据流保持为：查询分析与规划 → OpenAlex/Semantic Scholar 有界检索 → 标准化与去重 → 硬过滤 → RRF/词法融合 → 结构化结果 → API/UI 或正式评测产物。

保留预算控制、价格策略、依赖快照、capture/replay、运行锁、正式验证、API、UI、健康检查和打包能力。

### 5.2 当前模块优化框架

保留 `paper-search recall` 的通用工作流：冻结输入、生成上下文、校验动作、运行 recipe、canary 和当前产物比较。保留 `src/paper_search/recall_experiments/` 中仍为通用接口、候选池、身份、预算、快照、runner、evaluator、live runtime 和 provider adapter 提供服务的代码。

历史证据库存、旧方法回放和旧 prompt 变体不自动视为当前框架。它们若只服务已结束实验，应删除；如果某个共享适配器仍被当前 recall runtime 使用，则迁移到明确的公共模块后再删除旧外壳。

### 5.3 兼容边界

历史正式运行夹具仍需被 `verify-run` 和 `compare-replay` 读取。为此可以保留只读且固定为 `false` 的旧证据字段，但这些字段不得继续构造或启用已否决模块。

旧 experiment 名称应在配置加载阶段被明确拒绝。生产组合层只构造当前 `main-baseline`；候选召回优化通过独立 `recall` 边界进入，不再伪装成生产 experiment flag。

## 6. 删除与收敛范围

### 6.1 已否决的生产执行路径

从生产配置、实验注册表、组合根和编排器移除以下执行入口：

- `embedding`
- `citation-expansion` 的旧生产阶段
- `llm-rerank`
- 旧 `title-candidates` 生产阶段
- `fixed-two-round`
- `adaptive-evolution`

随后删除只为这些入口服务的源码、配置和测试，包括旧 embedding/sentence-transformer 排序、旧约束重排、旧 Query Evolution、旧标题候选生产实现、消融矩阵及专属 benchmark/probe。

`ranking.fusion`、`ranking.lexical` 等主链能力继续保留。citation provider 适配代码若仍被当前 recall 框架引用，则迁入 recall 或共享 provider 边界，不能随旧生产 citation stage 一并删除。

### 6.2 一次性研究工具与历史材料

删除已完成结论的一次性脚本及其专属测试，例如历史 gold 瓶颈分析、sealed recomposition、identifier rescore/rebuild、旧 Query Evolution probe。仍为冻结数据复现或当前公共入口直接调用的工具必须保留或转换为正式 CLI 后才能删除。

删除以下当前目录噪声：

- `.superpowers/sdd/` 的旧审查报告；
- `docs/superpowers/specs/` 与 `docs/superpowers/plans/` 的历史过程文档；
- 已被决策记录吸收的逐次实验报告和对应公开聚合 evidence；
- 旧聊天恢复提示、旧阶段交接和重复的部署说明；
- `configs/recall_experiments/historical/` 及仅支持历史库存命令的代码；
- 已被当前 canonical recipe 替代的 prompt/recipe 变体。

本设计和后续实施计划会在实施完成前保留以供审阅；最终清理提交可以将它们从当前目录移除，Git 历史仍保留完整过程。

### 6.3 文档收敛

最终协作文档只保留职责明确的入口：

- `README.md`：安装、命令和最短验证路径；
- `PRD.md`：当前产品目标、范围和验收标准，删除过期四周计划与旧方案清单；
- `HANDOFF.md`：当前状态、受保护工作和下一步，不再追加历史流水账；
- `docs/architecture/current-system.md`：当前真实模块边界和数据流；
- `docs/TEAMMATE_ONBOARDING.md`：队友拉取、环境准备、分支和测试约定；
- `docs/experiment-decisions.md`：已否决方法的简明结论和重新开启条件；
- `docs/limitations-and-risks.md`：当前限制与安全边界。

演示、答辩或比赛交付文档只有仍被当前协作流程直接引用时才保留；本地未跟踪 `deliverables/` 不进入远端。

## 7. 实施顺序

1. 从当前稳定提交建立隔离清理分支，记录受保护工作区路径、状态、内容哈希和远端分支指针。
2. 先用测试固定当前生产主链和 recall 公共接口。
3. 从组合根和编排器移除已否决执行入口，再删除不可达实现。
4. 迁移 recall 仍需的共享 provider/模型能力，保证模块优化框架不依赖旧生产实验外壳。
5. 删除对应配置、历史命令、一次性脚本、专属测试和过程文档。
6. 重写当前入口文档，并增加仓库结构与支持命令的自动检查。
7. 运行聚焦测试、全量离线测试、Ruff、mypy、打包和 Git 一致性检查。
8. 只在全部门禁通过后提交清理分支，并将其无强推更新到远端 `main`。
9. 再次读取远端分支指针；仅删除已记录且仍未变化的旧远端分支。若任一分支在清理期间移动，停止删除该分支并报告。

## 8. 验证门槛

清理被视为成功必须同时满足：

- 实施前记录的受保护状态清单在内容和状态上未被本次流程改变；实施期间外部新增变化被识别为外部变化而非本次清理结果；
- `paper-search --help` 只展示受支持命令；
- production config 无法启用已否决 experiment；
- replay `smoke`、API/UI、正式 evaluate/verify/compare 和双模式 E2E 测试通过；
- `tests/recall_experiments` 中保留的当前公共框架测试通过；
- 全量非 online pytest 通过，允许的 skip 必须与平台或显式凭据门控一致；
- Ruff、mypy、`git diff --check` 和构建/打包测试通过；
- 代码和活跃文档中不存在被删除模块的可执行入口或陈旧路径；
- 远端 `main` 的更新是 fast-forward，不使用 `--force`；
- 删除远端分支前后均记录精确 SHA，且分支未在清理期间发生变化。

## 9. 回滚与故障处理

清理提交按“主链收敛、模块迁移、历史材料删除、文档更新”分层，便于在推送前逐项复核。推送前出现任何回归时只修正隔离清理分支，不改动受保护工作区。

远端 `main` 更新后若发现问题，使用普通 revert 提交恢复，不重写历史。旧远端分支删除后仍可通过其已记录 SHA 和 Git 历史恢复；若 SHA 不再可达，则在删除前创建本地只读记录并停止操作，而不是冒险删除。
