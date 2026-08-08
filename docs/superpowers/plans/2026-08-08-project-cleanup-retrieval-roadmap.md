# Project Cleanup and Retrieval Roadmap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 精简本地项目工作区，统一活跃文件与目录名称，并将已有实验结论整合为一份不会重复无效尝试的检索提升路线。

**Architecture:** 本计划只处理本地文件清理、目录重命名和 Markdown 文档整合。正式运行证据与冻结数据保持不变；未来的锁重建、live capture 和检索代码改造不属于本计划。

**Tech Stack:** PowerShell、Git、Markdown、ripgrep

## Global Constraints

- 权威工作区固定为 `D:\AI Projects\.worktrees\week3`。
- 不读取或打印 `.env`，不调用外部 provider，不运行 validation 或 live capture。
- 不删除或修改 `runs/`、任何 `_diag_*`、`data/`、`.venv/`。
- 所有递归删除使用已批准的字面路径；删除前解析绝对路径并确认其位于权威工作区内。
- `deliverables/` 保持本地未跟踪状态，不执行 `git add deliverables`。
- 不修改检索实现、配置、预算、定价和质量门。
- 当前 v21 锁绑定 `c22abf9`，已经因设计提交改变 HEAD；本计划不得用它运行正式 capture。

---

### Task 1: 删除批准的冗余文件并补充忽略规则

**Files:**
- Modify: `.gitignore:1`
- Delete: `docs/PROJECT_HANDOFF_TASK4.md`
- Delete local/untracked: approved cache, output, old deliverable, and old plan paths listed below

**Interfaces:**
- Consumes: 设计文档中批准的精确删除清单
- Produces: 不含旧缓存、旧提交包和过时 Task 4 交接文档的工作区；后续工具目录不会重新污染 `git status`

- [ ] **Step 1: 验证删除目标与保护目录**

Run:

```powershell
$workspaceRoot = [IO.Path]::GetFullPath('D:\AI Projects\.worktrees\week3')
$deleteTargets = @(
  '.mypy_cache',
  '.ruff_cache',
  '.uv-cache',
  '.pdf-check',
  '.sheet-build',
  '.superpowers\sdd',
  '.gate0-report.json',
  'outputs\annotation_status_20260729',
  'deliverables\初赛提交包_20260805',
  'deliverables\VivaAI_材料交接包_20260805.zip',
  'docs\PROJECT_HANDOFF_TASK4.md',
  'docs\superpowers\plans\2026-07-28-task10-experiment-ablation.md',
  'docs\superpowers\plans\2026-07-28-week3-task10-experimentation.md',
  'docs\superpowers\plans\2026-07-28-week3-task9-embedding-ranking.md',
  'docs\superpowers\plans\2026-07-29-data-freeze-v2.md'
)
$protectedTargets = @('runs', 'data', '.venv')
foreach ($relative in $deleteTargets) {
  $resolved = [IO.Path]::GetFullPath((Join-Path $workspaceRoot $relative))
  if (-not $resolved.StartsWith($workspaceRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Target escaped workspace: $resolved"
  }
  [PSCustomObject]@{ Relative = $relative; Absolute = $resolved; Exists = Test-Path -LiteralPath $resolved }
}
foreach ($relative in $protectedTargets) {
  $resolved = Join-Path $workspaceRoot $relative
  if (-not (Test-Path -LiteralPath $resolved)) { throw "Protected path missing: $resolved" }
}
```

Expected: all delete targets resolve below the workspace; `runs`, `data`, and `.venv` exist.

- [ ] **Step 2: 删除精确目标**

Run:

```powershell
$workspaceRoot = [IO.Path]::GetFullPath('D:\AI Projects\.worktrees\week3')
$deleteTargets = @(
  '.mypy_cache',
  '.ruff_cache',
  '.uv-cache',
  '.pdf-check',
  '.sheet-build',
  '.superpowers\sdd',
  '.gate0-report.json',
  'outputs\annotation_status_20260729',
  'deliverables\初赛提交包_20260805',
  'deliverables\VivaAI_材料交接包_20260805.zip',
  'docs\PROJECT_HANDOFF_TASK4.md',
  'docs\superpowers\plans\2026-07-28-task10-experiment-ablation.md',
  'docs\superpowers\plans\2026-07-28-week3-task10-experimentation.md',
  'docs\superpowers\plans\2026-07-28-week3-task9-embedding-ranking.md',
  'docs\superpowers\plans\2026-07-29-data-freeze-v2.md'
)
foreach ($relative in $deleteTargets) {
  $resolved = [IO.Path]::GetFullPath((Join-Path $workspaceRoot $relative))
  if (-not $resolved.StartsWith($workspaceRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Target escaped workspace: $resolved"
  }
  if (Test-Path -LiteralPath $resolved) {
    Remove-Item -LiteralPath $resolved -Recurse -Force
  }
}
```

Expected: command exits 0; no protected path appears in `$deleteTargets`.

- [ ] **Step 3: 更新 `.gitignore`**

Apply:

```diff
 # Python virtual environments and caches
 .venv/
 __pycache__/
 *.py[cod]
 .pytest_cache/
 .ruff_cache/
 .mypy_cache/
 .coverage
 htmlcov/
+
+# Local build and agent work products
+.uv-cache/
+.pdf-check/
+.sheet-build/
+.superpowers/
```

- [ ] **Step 4: 验证清理结果**

Run:

```powershell
$deleted = @(
  '.mypy_cache', '.ruff_cache', '.uv-cache', '.pdf-check', '.sheet-build',
  '.superpowers\sdd', '.gate0-report.json', 'outputs\annotation_status_20260729',
  'deliverables\初赛提交包_20260805', 'deliverables\VivaAI_材料交接包_20260805.zip',
  'docs\PROJECT_HANDOFF_TASK4.md',
  'docs\superpowers\plans\2026-07-28-task10-experiment-ablation.md',
  'docs\superpowers\plans\2026-07-28-week3-task10-experimentation.md',
  'docs\superpowers\plans\2026-07-28-week3-task9-embedding-ranking.md',
  'docs\superpowers\plans\2026-07-29-data-freeze-v2.md'
)
$remaining = $deleted | Where-Object { Test-Path -LiteralPath $_ }
if ($remaining) { throw "Approved targets still exist: $($remaining -join ', ')" }
foreach ($protected in @('runs', 'data', '.venv')) {
  if (-not (Test-Path -LiteralPath $protected)) { throw "Protected path missing: $protected" }
}
git status --short
```

Expected: no approved deletion target remains; protected paths exist; Git shows the tracked deletion of `docs/PROJECT_HANDOFF_TASK4.md` and the `.gitignore` modification.

- [ ] **Step 5: 提交 tracked 清理规则**

Run:

```powershell
git add -- .gitignore docs/PROJECT_HANDOFF_TASK4.md
git commit -m "chore: remove obsolete project handoff artifacts"
```

Expected: one commit containing only `.gitignore` and the tracked Task 4 handoff deletion. Untracked deliverables are not staged.

---

### Task 2: 简化当前交付物目录并修正内部路径

**Files:**
- Move: `deliverables/初赛提交包_20260806` → `deliverables/submission`
- Move: `deliverables/演示包_20260806` → `deliverables/demo`
- Move: `deliverables/项目文档_20260806` → `deliverables/project-docs`
- Modify local/untracked: `deliverables/submission/README.md`
- Modify local/untracked: `deliverables/demo/README.md`
- Modify local/untracked: `deliverables/demo/start_demo.ps1:6`
- Create local/untracked: `deliverables/project-docs/README.md`

**Interfaces:**
- Consumes: Task 1 保留下来的 2026-08-06 交付物
- Produces: 三个简洁目录名和可继续运行的 replay 演示脚本

- [ ] **Step 1: 验证重命名源与目标**

Run:

```powershell
$workspaceRoot = [IO.Path]::GetFullPath('D:\AI Projects\.worktrees\week3')
$moves = @(
  @('deliverables\初赛提交包_20260806', 'deliverables\submission'),
  @('deliverables\演示包_20260806', 'deliverables\demo'),
  @('deliverables\项目文档_20260806', 'deliverables\project-docs')
)
foreach ($pair in $moves) {
  $source = [IO.Path]::GetFullPath((Join-Path $workspaceRoot $pair[0]))
  $destination = [IO.Path]::GetFullPath((Join-Path $workspaceRoot $pair[1]))
  if (-not (Test-Path -LiteralPath $source)) { throw "Missing source: $source" }
  if (Test-Path -LiteralPath $destination) { throw "Destination already exists: $destination" }
  if (-not $source.StartsWith($workspaceRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Source escaped workspace: $source"
  }
  if (-not $destination.StartsWith($workspaceRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Destination escaped workspace: $destination"
  }
}
```

Expected: all three sources exist and all three destinations are absent.

- [ ] **Step 2: 执行字面路径重命名**

Run:

```powershell
Move-Item -LiteralPath 'deliverables\初赛提交包_20260806' -Destination 'deliverables\submission'
Move-Item -LiteralPath 'deliverables\演示包_20260806' -Destination 'deliverables\demo'
Move-Item -LiteralPath 'deliverables\项目文档_20260806' -Destination 'deliverables\project-docs'
```

Expected: the three new directories exist and the old directory names are absent.

- [ ] **Step 3: 更新交付物版本说明与演示脚本**

Apply:

```diff
--- deliverables/submission/README.md
+++ deliverables/submission/README.md
@@
 ## 命名与版本
 
 - 团队：VivaAI
 - 项目：科研场景下复杂学术查询的智能论文搜索与推荐
+- 包版本：2026-08-06
 - 代码版本：`c6a4b5f`（见 `code/COMMIT.txt`）
--- deliverables/demo/README.md
+++ deliverables/demo/README.md
@@
 # 演示视频录制包（给队友）
 
 本包用于录制初赛项目视频（Replay UI 演示），已实测可运行。
+
+包版本：2026-08-06。
--- deliverables/demo/start_demo.ps1
+++ deliverables/demo/start_demo.ps1
@@
-    --lock 'D:\AI Projects\.worktrees\week3\deliverables\演示包_20260806\replay_demo.lock.yaml' `
+    --lock 'D:\AI Projects\.worktrees\week3\deliverables\demo\replay_demo.lock.yaml' `
```

Create `deliverables/project-docs/README.md` with:

```markdown
# 项目文档源文件

版本：2026-08-06。

本目录保存项目文档的可编辑源文件、配图和生成脚本：

- `第八届中国研究生人工智能创新大赛_项目文档_优化版.docx`：优化后的项目文档；
- `项目文档_工作副本.docx`：编辑工作副本；
- `figures/`：文档配图与生成脚本；
- `edit_docx.py`、`replace_images.py`：文档处理脚本。

正式评测证据仍以工作区 `runs/` 中的密封运行记录为准。
```

- [ ] **Step 4: 验证新目录和硬编码路径**

Run:

```powershell
foreach ($path in @('deliverables\submission', 'deliverables\demo', 'deliverables\project-docs')) {
  if (-not (Test-Path -LiteralPath $path)) { throw "Renamed deliverable missing: $path" }
}
foreach ($path in @('deliverables\初赛提交包_20260806', 'deliverables\演示包_20260806', 'deliverables\项目文档_20260806')) {
  if (Test-Path -LiteralPath $path) { throw "Old deliverable path remains: $path" }
}
$stale = rg -n "初赛提交包_20260806|演示包_20260806|项目文档_20260806" deliverables
$rgCode = $LASTEXITCODE
if ($rgCode -eq 0) { throw "Stale deliverable path found: $stale" }
if ($rgCode -ne 1) { throw "rg failed with exit code $rgCode" }
```

Expected: existence checks pass; `rg` returns no match. Do not stage `deliverables/`.

---
### Task 3: 合并实验结论并重写活跃路线文档

**Files:**
- Move/Replace: `docs/improvement-plan-2026-08-07.md` → `docs/retrieval-roadmap.md`
- Create: `docs/experiment-decisions.md`
- Replace: `HANDOFF.md`
- Modify: `README.md:80-87`
- Delete local/untracked: `academic_retrieval_v3_optimization_plan.md`

**Interfaces:**
- Consumes: 设计文档、旧 v3 计划、现行 improvement plan 和聚合诊断证据
- Produces: 唯一活跃路线图、实验决策记录和简洁交接入口

- [ ] **Step 1: 将活跃路线图移到新路径**

Run:

```powershell
if (-not (Test-Path -LiteralPath 'docs\improvement-plan-2026-08-07.md')) {
  throw 'Source roadmap is missing'
}
if (Test-Path -LiteralPath 'docs\retrieval-roadmap.md') {
  throw 'Destination roadmap already exists'
}
Move-Item -LiteralPath 'docs\improvement-plan-2026-08-07.md' -Destination 'docs\retrieval-roadmap.md'
```

Expected: `docs/retrieval-roadmap.md` exists and the old path is absent.

- [ ] **Step 2: 创建实验决策记录**

Replace `docs/experiment-decisions.md` with:

```markdown
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
| Title Candidates | 唯一正向召回信号；联合池覆盖 41 篇 gold、24 个查询 | 继续 | 优先诊断候选从验证、融合到最终输出的流失，不先增加标题数量 |

## 当前结论

召回仍是主要瓶颈，但下一步不是继续增加检索模块，而是先确认 gold 的 OpenAlex 可用性，并定位标题候选在现有流水线中的流失位置。
```

- [ ] **Step 3: 重写检索路线图**

Replace `docs/retrieval-roadmap.md` with:

```markdown
# 检索提升路线

更新于 2026-08-08。目标是在冻结 dev 上提高宏平均 F1，同时保持 capture/replay 证据链可复现。

## 当前判断

- 当前宏 F1 约 0.006，51/60 查询零命中，召回是主要瓶颈。
- 标题候选是唯一已有正向召回信号，但候选联合池到最终输出存在明显流失。
- Citation、Topic、Embedding、普通 Query Rewrite 和既有 LLM Query Variants 已被实测否决，详见 `experiment-decisions.md`。

## Phase 0：建立干净基线

清理完成并重建下一版锁后，执行 readiness → dev capture → verify → replay → compare。

基线必须同时满足：

- quality gate passed；
- `provenance_failures=0`；
- capture 与 replay 业务结果一致。

在此之前不运行新的全量在线实验。

## Phase 1：两个必要诊断

### 1. Gold 精确可用性

使用 DOI、arXiv ID 和 OpenAlex ID 做只读精确反查，只输出聚合原因。禁止把 gold 标识符转换成检索查询。

现有 P0 探针测量的是生成标题能否搜到 gold，不等同于 gold 是否存在于 OpenAlex。

### 2. 标题候选流失

逐阶段统计 exact gold：

1. 生成标题；
2. OpenAlex 标题验证结果；
3. 合并候选池；
4. RRF 排序池；
5. 最终 `selected_paper_ids`。

下一项实现工作选择流失最大的阶段。

## Phase 2：标题候选保留与输出选择

- 在同一冻结 dev 上对比 10 与 20 个标题；
- 检查标题验证排名、融合贡献和最终截断；
- 离线搜索 `K ∈ {10,20,30,50}`；
- 离线搜索阈值 `0.45–0.75`，步长 `0.05`；
- 以宏平均 F1 选择，Precision、Recall、Recall@K、MRR、NDCG 作为护栏。

离线无增量的变体不进入 live capture。选定组合在 validation 前冻结。

## Phase 3：Query Evolution 条件实验

当前 `fixed_two_round` 不可直接启用，因为它：

- 使用规则兜底而不是生产 DeepSeek `QuerySpec`；
- 在实验身份下关闭标题候选；
- 第二轮预算估计为零。

重新实验前必须让它复用生产查询分析、组合已选标题候选基线，并用真实调用推导非零预算。规则版与 LLM 版一次只测一个；只有 exact-ID recall 提升且不损失已有命中时才进入正式 capture。

## 后续条件项

- Query Type：仅在分类型误差分析显示稳定差异后实施；
- Selector/LLM rerank：仅在召回明显提升后实施；
- 新数据源：仅在 Gold 精确可用性诊断证明 OpenAlex 覆盖不足后引入。

每个晋升改动必须是单变量实验，使用独立配置、锁和 capture/replay/compare 证据。
```

- [ ] **Step 4: 重写 `HANDOFF.md`**

Replace `HANDOFF.md` with:

```markdown
# paper-search 项目交接

更新于 2026-08-08。权威工作区：`D:\AI Projects\.worktrees\week3`。

## 1. 项目目标

VivaAI 参加第八届中国研究生人工智能创新大赛赛题三，构建复杂学术查询的论文搜索与推荐系统。内部目标是冻结 dev 宏平均 F1 ≥ 0.30；当前约 0.006。

正式评测使用 live capture → verify → 零网络 replay → compare。基线必须 gate passed、`provenance_failures=0`，且 capture/replay 业务结果一致。

## 2. 当前状态

- DeepSeek `deepseek-v4-flash` 查询解析已验证 60/60。
- 标题候选默认生成 20 个标题并经 OpenAlex 验证。
- 最近四轮正式 capture 均因 OpenAlex 限流或额度问题 gate failed；已完成查询的 replay 可干净复现。
- 51/60 查询零命中，最终共命中约 10 篇 gold；召回是主要瓶颈。
- 全量测试最近记录为 1856 passed / 36 skipped。

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

标题候选是唯一已有正向召回信号。后续先定位候选在 OpenAlex 验证、融合和最终输出之间的流失。

## 5. 下一步

1. 本次清理完成后，基于最终提交和最新 ledger checkpoint 重建下一版候选锁；
2. 在单独授权后跑 readiness → dev capture → verify → replay → compare；
3. 完成 Gold 精确可用性和标题候选流失诊断；
4. 优先优化标题候选保留与输出选择；
5. Query Evolution 仅在重做生产分析组合和预算估计后进行小规模探针。

## 6. 锁状态

旧 v21 锁绑定 `c22abf9`。设计与清理提交已经改变 HEAD，因此 v21 不得继续用于正式 capture。重建锁和 live capture 都需要单独授权，本次清理不执行。

## 7. 环境与红线

- 完整测试环境：`D:\AI Projects\Projects\.venv`；
- 密钥位于 `D:\AI Projects\Projects\.env`，不得读取、打印或提交；
- 正式命令只加载 `LLM_API_KEY`、`OPENALEX_API_KEY`（含 `_2.._7`）和 `SEMANTIC_SCHOLAR_API_KEY`；
- 不加载 `LLM_BASE_URL`、`LLM_MODEL_PRIMARY`、`LLM_MODEL_FALLBACK`；
- 不删除 `runs/`、`_diag_*` 或 `data/`，不修改 `data/manifest.json`；
- capture 与 replay 之间不得提交代码；
- 不在聊天或公开文档中写入冻结查询文本；
- validation 不可撤销，必须单独授权。
```

- [ ] **Step 5: 更新 README 的活跃文档入口和标题数量**

Apply:

```diff
--- README.md
+++ README.md
@@
 `configs/base.yaml` 固定 `experiment: main-baseline`。可选身份为 `embedding`、`citation-expansion`、
 `llm-rerank`、`title-candidates`、`fixed-two-round` 和 `adaptive-evolution`；每个身份只构造其
 声明组件，baseline 不加载可选依赖。`title-candidates` 使用 `configs/title_candidates.yaml`
-（LLM 生成 10 个候选论文标题，经 OpenAlex 验证后并入候选池；编排器按 `max_output_papers=50`
+（LLM 生成 20 个候选论文标题，经 OpenAlex 验证后并入候选池；编排器按 `max_output_papers=50`
 截断最终输出）。
 
 可选模块的实现或离线测试通过不代表晋升。晋升需要 Gate 0–5、三次同配置 dev 比较、1,000 次 bootstrap、一次 selection-only validation 比较及单独批准；在证据不完整或阈值不通过时保持 default-off。
+
+当前项目状态见 `HANDOFF.md`；检索提升顺序见 `docs/retrieval-roadmap.md`；已完成实验的继续/停止决策见 `docs/experiment-decisions.md`。
```

- [ ] **Step 6: 删除已合并的旧 v3 计划**

Run:

```powershell
$workspaceRoot = [IO.Path]::GetFullPath('D:\AI Projects\.worktrees\week3')
$oldPlan = [IO.Path]::GetFullPath((Join-Path $workspaceRoot 'academic_retrieval_v3_optimization_plan.md'))
if (-not $oldPlan.StartsWith($workspaceRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
  throw "Target escaped workspace: $oldPlan"
}
if (Test-Path -LiteralPath $oldPlan) {
  Remove-Item -LiteralPath $oldPlan -Force
}
```

Expected: old root plan is absent; its experiment conclusions exist in `docs/experiment-decisions.md`.

- [ ] **Step 7: 验证文档一致性**

Run:

```powershell
if (-not (Test-Path -LiteralPath 'docs\retrieval-roadmap.md')) { throw 'Roadmap missing' }
if (-not (Test-Path -LiteralPath 'docs\experiment-decisions.md')) { throw 'Decision record missing' }
if (-not (Test-Path -LiteralPath 'HANDOFF.md')) { throw 'Handoff missing' }
$stale = rg -n "improvement-plan-2026-08-07|academic_retrieval_v3_optimization_plan" HANDOFF.md README.md docs/retrieval-roadmap.md docs/experiment-decisions.md
$rgCode = $LASTEXITCODE
if ($rgCode -eq 0) { throw "Stale document path found: $stale" }
if ($rgCode -ne 1) { throw "rg failed with exit code $rgCode" }
rg -n "Citation Expansion|Topic Retrieval|Embedding|Query Evolution|Title Candidates" docs/retrieval-roadmap.md docs/experiment-decisions.md HANDOFF.md
```

Expected: first `rg` returns no match; second `rg` shows the same continue/stop status across the three active documents.

- [ ] **Step 8: 提交活跃文档**

Run:

```powershell
git add -- README.md HANDOFF.md docs/retrieval-roadmap.md docs/experiment-decisions.md
git commit -m "docs: consolidate retrieval roadmap and experiment decisions"
```

Expected: commit includes only the four active documents. `deliverables/` remains unstaged.

---

### Task 4: 最终一致性检查

**Files:**
- Verify only: repository and local deliverables

**Interfaces:**
- Consumes: Tasks 1–3
- Produces: 可交接的清理结果；不创建新文件、不运行网络评测

- [ ] **Step 1: 检查保护路径与新目录**

Run:

```powershell
foreach ($path in @(
  'runs', 'data', '.venv',
  'deliverables\submission', 'deliverables\demo', 'deliverables\project-docs',
  'HANDOFF.md', 'docs\retrieval-roadmap.md', 'docs\experiment-decisions.md'
)) {
  if (-not (Test-Path -LiteralPath $path)) { throw "Required path missing: $path" }
}
```

Expected: command exits 0.

- [ ] **Step 2: 检查旧活跃路径引用**

Run as one command:

```powershell
$stale = rg -n --hidden --glob '!.git/**' --glob '!runs/**' --glob '!data/**' --glob '!docs/superpowers/specs/2026-08-08-project-cleanup-retrieval-roadmap-design.md' --glob '!docs/superpowers/plans/2026-08-08-project-cleanup-retrieval-roadmap.md' "improvement-plan-2026-08-07|academic_retrieval_v3_optimization_plan|初赛提交包_20260806|演示包_20260806|项目文档_20260806|PROJECT_HANDOFF_TASK4" .
$rgCode = $LASTEXITCODE
if ($rgCode -eq 0) { throw "Stale active path found: $stale" }
if ($rgCode -ne 1) { throw "rg failed with exit code $rgCode" }
```

Expected: no match.

- [ ] **Step 3: 检查 Git 差异与暂存边界**

Run:

```powershell
git diff --check
git status --short
```

Expected: `git diff --check` exits 0. `git status --short` may show only `?? deliverables/` because deliverables intentionally remain local and untracked; no deleted cache、旧计划或 `outputs/` 条目残留。

- [ ] **Step 4: 记录锁交接状态**

Run:

```powershell
rg -n "v21|c22abf9|不得继续用于正式 capture|单独授权|重建下一版锁" HANDOFF.md docs/retrieval-roadmap.md
```

Expected: `HANDOFF.md` explicitly marks v21 unusable; the roadmap requires lock renewal before the next baseline. Do not rebuild the lock in this plan.

