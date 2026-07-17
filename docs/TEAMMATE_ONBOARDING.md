# Week 1 协作者立即执行任务清单

> 工作分支：`codex/week1-collaboration`
>
> 当前状态：Task 1–4 工程部分已完成；正式数据冻结、真实 baseline、人工审计和 Week 1 gate 尚未完成。
>
> 最后更新：2026-07-17

## 1. 当前目标

协作者现在不继续开发检索算法，而是准备和审核真实、完整、可复现的数据证据。当前仓库没有正式 `data/manifest.json`，因此不得运行或宣称正式 Week 1 baseline，也不得宣布 Week 1 gate 通过。

执行顺序：环境与访问检查 → 固定数据准备 → 90 条类型/领域标记 → 40 条约束标注 → 固定 20 条独立双标 → 正式数据冻结 → 真实 baseline → 模糊去重审计 → Recall/失败率/成本审核 → Week 1 gate。

## 2. 安全红线

- 不读取、打印、搜索、解析、复制或提交任何 `.env` 内容。
- 真实查询、PaSa 原始文件、gold、人工标签、受限工作包及含受限文本的截图或日志不得进入 Git。
- 不提交 API Key、Token、Authorization Header 或带凭据的 URL。
- 不得仅把 manifest 的 `status` 改成 `frozen` 就宣称数据冻结。
- 不得修改 gold、随机种子、dataset revision、split ID 或抽样算法来改善指标。
- 不得使用 LLM 代替人工标注；双标完成前双方不得交换答案。
- 不得自行修改评测、去重、过滤或排序算法；发现问题先提交证据。
- 不得修改、格式化、暂存或提交 `docs/superpowers/specs/2026-07-15-task2-evaluation-design.md`。
- 未经主负责人批准，不得创建 PR、合并、强制推送、删除 worktree 或清理分支。

## 3. 工作包 0：环境与访问就绪

### 输入

- Git、Python 3.11、uv 和仓库访问权限；
- 你自己的 Hugging Face 只读账号及 `HF_TOKEN`；
- 你自己的 OpenAlex key（环境变量名为 `OPENALEX_API_KEY`），正式 baseline 前必须可由运行子进程加载；
- 已接受 `CarlanLark/pasa-dataset` gated dataset 条款。

### 操作

首次克隆使用：

```powershell
git clone https://github.com/Tianyuai/AI-Projects.git
Set-Location AI-Projects
git fetch origin
git switch --track origin/codex/week1-collaboration
```

已经有本地仓库时，先确认工作树清洁，再使用：

```powershell
git status --short --branch
git fetch origin
git show-ref --verify --quiet refs/heads/codex/week1-collaboration
if ($LASTEXITCODE -eq 0) {
  git switch codex/week1-collaboration
} else {
  git switch --track origin/codex/week1-collaboration
}
git branch --set-upstream-to=origin/codex/week1-collaboration codex/week1-collaboration
git pull --ff-only
```

两种情况随后都执行：

```powershell
uv sync --all-groups
uv run --no-env-file pytest -q
uv run --no-env-file ruff check .
uv run --no-env-file mypy src
```

这里的 `--no-env-file` 只表示“不会主动加载 `.env`”，不会清除进程中已经存在的环境变量，也不是网络隔离。`uv sync --all-groups` 可能访问包索引；需要真正离线时，必须预先准备依赖缓存，并由主负责人提供断网环境或防火墙验证方式。

数据准备子进程可以通过 `--env-file` 加载凭据，但不要打开或输出 `.env`：

```powershell
uv run --env-file .env python scripts/prepare_task2_data.py --output-root data
```

### 验收

- [ ] 分支为 `codex/week1-collaboration`；
- [ ] Python 为 3.11.x，依赖安装成功；
- [ ] 不主动加载 `.env` 的 pytest、Ruff、mypy 通过；
- [ ] PaSa 访问已获批；
- [ ] `OPENALEX_API_KEY` 已由本人申请并可供正式运行子进程加载；未配置时将阻塞工作包 6；
- [ ] 汇报中没有密钥、本机用户名或私有绝对路径。

若 PaSa 不可访问，不得自行换源。由主负责人决定是否启用唯一备用 `allenai/asta-bench` 标签 `v0.3.1`，并记录原因；两种来源不得混合构造同一 F1 分区。

## 4. 工作包 1：固定数据准备与身份核对

### 固定输入

- 数据源：`CarlanLark/pasa-dataset`；
- revision：`232428b0c867268c3b8ded90db4d98c1b30501d6`；
- 随机种子：`20260714`；
- 抽样算法：`answer-count-largest-remainder-v1`；
- 源文件：`AutoScholarQuery/dev.jsonl`、`AutoScholarQuery/test.jsonl`、`RealScholarQuery/test.jsonl`。

### 操作与验收

- [ ] 运行数据准备脚本，确认解析到的 revision 完全一致；
- [ ] 核对三个源文件 SHA-256、许可、访问条件和下载日期；
- [ ] 核对开发/验证/模拟测试数量为 60/30/50；
- [ ] 核对 `dev.ids.json`、`validation.ids.json`、`simulated_test.ids.json`；
- [ ] 核对 `constraint_annotation.ids.json` 为 40 条；
- [ ] 核对 `overlap_annotation.ids.json` 为 20 条；
- [ ] 所有 query ID 均为非空唯一字符串；
- [ ] 相同输入重跑得到相同 ID 顺序和哈希；
- [ ] Git 状态不包含 raw、gold、真实查询或标注文件。

运行前先用 `git check-ignore` 验证 `data/raw/`、`data/dev/gold.jsonl`、`data/validation/gold.jsonl`、`data/simulated_test/` 和 `data/annotation_work/` 均被忽略。受限输出只保存在访问受控的本地目录或团队批准的端到端加密存储中；交接时只在 Git/普通聊天中记录 SHA-256，不发送正文或公开链接。

`data/splits/*.ids.json` 中只含 query ID 的清单属于可提交安全元数据；包含查询文本、gold 或人工答案的标注工作包仍是受限文件。

完成后等待主负责人明确发布：

```text
Task 2 标注工作包 v1 已冻结
```

没有这条通知，不得开始正式标注。

## 5. 工作包 2：90 条查询类型与领域标记

### 输入

冻结的开发集 60 条、验证集 30 条、冻结版 `data/annotation_guide.md` 和团队约定的稳定 `annotator` 代号。

### 每条必填

- `query_id`：保持原值；
- `query_type`：只使用指南允许的标签；
- `domain`：只使用指南规定的格式；
- `annotator`：使用稳定代号。

无法归类时不要发明标签。将 query ID、候选解释和疑问写入私密问题清单，统一口径后再继续。

### 验收

- [ ] 90 条全部完成；
- [ ] 无缺失、无新增、无重复、无 query ID 修改；
- [ ] 字段通过标注 Schema；
- [ ] 疑难项均有统一结论；
- [ ] 标注文件通过私密渠道保存，未进入 Git。

## 6. 工作包 3：固定 40 条查询约束标注

由 `constraint_annotation.ids.json` 指定，不得自行挑选。固定字段为：

```text
query_id
research_goal
must_have
should_have
exclusions
year_from
year_to
venues
query_type
domain
annotator
```

### 规则

- `research_goal` 描述真实研究目标，不写成关键词堆积；
- `must_have` 只放缺失后就不应入选的硬约束；
- `should_have` 放提高相关性但不一票否决的条件；
- `exclusions` 只记录原查询明确排除的内容；
- 年份和 venue 仅在原查询明确给出时填写；
- 不根据个人常识补造查询没有表达的条件；
- 不确定项按指南记录并进入问题清单。

### 验收

- [ ] 恰好 40 条且与冻结 ID 清单一致；
- [ ] 字段名、类型、年份区间和集合字段通过 Schema；
- [ ] 集合中没有空字符串；
- [ ] query ID 无缺失、重复、新增或修改；
- [ ] 工作包和答案未提交到 Git。

## 7. 工作包 4：固定 20 条独立双标与一致性

这 20 条包含在上述 40 条中，不是额外 20 条。

1. 确认双方使用相同 ID 清单、工作包版本和指南版本；
2. 协作者与主负责人独立完成；
3. 双方完成前不得交换答案或使用 LLM 代标；
4. 两人都报告完成后才交换文件哈希并计算一致性；
5. 对 `query_type`、`domain` 等关键离散字段计算 Cohen's kappa；
6. 任一关键字段低于 `0.80` 时，先修订指南，再独立重标分歧样本；重标后的分歧样本只用于裁决，不得冒充新的独立 kappa；
7. 若初始 kappa 未达标，必须在查看初始分歧明细或开展裁决前冻结一组新的独立重叠 ID；一致性工具此时只披露聚合 kappa，再使用新样本重新验证；
8. 一致性达标后，主负责人按固定 ID 清单额外复核 10 条，所有问题必须有结论和责任人。

### 验收

- [ ] 固定 20 条由双方独立完成；
- [ ] 每个关键离散字段 kappa 不低于 0.80；
- [ ] 未达门槛项已完成指南修订和分歧重标；
- [ ] 初始 kappa 未达标时，已用新冻结的独立重叠样本重新验证，未用已知分歧样本抬高 kappa；
- [ ] 额外 10 条复核全部闭环；
- [ ] 最终标注通过私密渠道交接并核对 SHA-256。

## 8. 工作包 5：gold、ID 清单与 manifest 正式冻结

### 开始条件

工作包 1–4 完成；gold labels 已由真人确认完整；主负责人已决定各分区 `zero_answer_policy` 为 `reject` 或 `allow`。

### manifest 必核字段

```text
status
revision
count
gold_path
gold_sha256
ids_path
ids_sha256
labels_complete
zero_answer_policy
```

### 冻结验收

- [ ] `revision` 非空且与固定来源一致；
- [ ] 每个 `count` 为正整数；
- [ ] `gold_path`、`ids_path` 是限制在 `data/` 下的相对路径；
- [ ] `gold_sha256`、`ids_sha256` 是精确字节的 `sha256:<64 hex>`；
- [ ] gold JSONL 非空且恰有 `count` 条；
- [ ] ID 清单恰有 `count` 个非空唯一字符串；
- [ ] gold query ID 与 ID 清单顺序完全一致；
- [ ] `labels_complete: true`；
- [ ] `zero_answer_policy` 明确为 `reject` 或 `allow`；
- [ ] `reject` 下每条 gold 至少有一个 relevant paper ID；
- [ ] `allow` 下零答案记录已人工确认且策略进入正式运行身份；
- [ ] 两人交叉核对后，由主负责人批准 `status: frozen`。

只有全部通过后才能冻结。任一内容变化都必须产生新哈希并重新审核，不能覆盖旧证据。旧 manifest 和审核汇总按 `data-freeze-<revision>-v<N>` 命名保存在团队批准的证据目录；受限 gold 仍只保存在私密存储。

若主负责人批准切换到备用数据源，原工作包立即失效，必须生成新的数据身份和工作包版本，并重新发布明确的“标注工作包 v<N> 已冻结”通知。

## 9. 工作包 6：冻结数据上的真实 Week 1 baseline

### 开始条件

正式 manifest 已真实冻结；开发集 gold/ID/哈希验证通过；运行进程能以授权方式加载 OpenAlex key；输出目录不会覆盖不同身份的正式产物。

由主负责人运行，协作者负责记录和审核：

```powershell
uv run --env-file .env python -m paper_search.evaluation.runner `
  --config configs/base.yaml `
  --split dev `
  --output experiments/baseline-week1
```

### 必审产物

`predictions.jsonl`、`metrics.json`、`usage.json`、`run.json`、`deduplication.jsonl`、`filtering.jsonl`、`snapshot_manifest.json` 和 `snapshots/`。

### 验收

- [ ] Git SHA、split、gold/ID 哈希和 manifest identity 正确；
- [ ] scoring version、去重阈值、过滤规则和 penalty 参数已记录；
- [ ] 缓存响应均绑定到同目录不可变快照；
- [ ] 快照哈希与原始字节一致；
- [ ] 每个查询都有预测或明确失败；
- [ ] 逐查询与 aggregate 指标、调用量和延迟齐全；
- [ ] 没有密钥、私有路径或受限原文泄露；
- [ ] 没有部分写入或覆盖旧正式证据。

## 10. 工作包 7：模糊去重人工抽样审计

1. 从正式 `deduplication.jsonl` 中选取 `match_rule == "fuzzy_title"` 的合并；
2. 模糊合并不超过 200 条时全部审计；超过 200 条时按种子 `20260714` 确定性抽取 200 条，并在查看判断结果前冻结抽样算法、样本 ID 清单和清单 SHA-256；
3. 核对标题、年份、作者和稳定 ID；
4. 标记 `correct_merge`、`false_merge` 或 `needs_adjudication`；
5. 两人复核全部 false merge 和 needs adjudication；
6. 汇总样本数、正确数、错误数、裁决数和准确率。

### 验收

- [ ] 每条判断有最小证据和审计人；
- [ ] 模糊去重人工准确率不低于 `98%`；
- [ ] 低于 98% 时停止 gate，只提交失败案例，不放宽口径；
- [ ] 可提交报告不含受限查询或原始响应正文。

## 11. 工作包 8：Recall、失败率、成本与产物审核

对相同冻结查询和相同原始候选，分别统计：原始候选、去重后、硬过滤后、词法排序与 Top-K 后。`0.02` 验收线应用于“去重输入候选 → 硬过滤接受池”的累计 Recall 绝对下降；Top-K 截断损失单独报告，不混入该门槛。

### 质量验收

- [ ] 计算各阶段 relevant paper 保留数和 Recall；
- [ ] 分别报告去重、过滤和截断造成的 Recall 绝对变化；
- [ ] 过滤/初筛后 Recall 绝对下降不超过 `0.02`；
- [ ] 每个 relevant paper 损失都有 query ID、paper ID、阶段和原因；
- [ ] 不通过修改 gold 或冻结状态掩盖损失。

### 运行验收

- [ ] API 调用数、缓存命中、P50、P95 和失败率齐全；
- [ ] 硬失败率与显式部分结果比例分开报告；
- [ ] 未知费用保持 `null/None`，不伪造为 0；
- [ ] 指标、用量和快照 manifest 相互引用一致；
- [ ] 异常和失败案例已分类。

任一指标不达标时，交给主负责人按严格 TDD 修复；协作者不自行改算法。

Week 1 对成本没有额外数值门槛；“成本通过审核”只表示已知费用按真实值记录、未知费用保持 `null/None`、调用量和延迟齐全且没有超出配置硬预算。

## 12. Week 1 阶段闸门进入条件

只有以下条件全部满足，主负责人才能判定 gate：

- [ ] Task 1–4 工程基线有本次 gate 审核后新运行的聚焦测试、全量 pytest、Ruff 和 mypy 证据；
- [ ] revision、抽样算法、随机种子和分区身份已冻结；
- [ ] gold/ID 非空、数量一致、ID 唯一且顺序一致；
- [ ] 精确字节哈希与 manifest 一致；
- [ ] `labels_complete: true` 且零答案策略明确；
- [ ] 冻结开发集真实 baseline 生成完整不可变产物；
- [ ] 可以从查询得到去重论文列表并计算 F1；Week 1 gate 只要求 baseline 可计算，不设置最低 F1，通过不代表达到最终 PRD 质量目标；
- [ ] 模糊去重人工准确率至少 98%；
- [ ] Recall 损失证据齐全，过滤/初筛绝对下降不超过 0.02；
- [ ] 调用量、延迟、失败率、成本和快照证据通过审核；
- [ ] `git diff --check` 通过；`git diff --name-only <Task4基线>...HEAD` 不包含受保护或无关文件；凭据模式扫描无命中；
- [ ] 独立审查记录审查人、时间、审查范围和 Critical/Important/Minor 结论，所有有效问题已关闭。

任何一项未完成，状态必须写“阻塞”或“未通过”。Week 1 gate 未通过前不开始关系图。

## 13. Git 交付边界

### 可以提交

- 代码、测试和安全文档；
- revision、SHA-256、数量、随机种子和抽样算法；
- 只含 query ID 的安全清单；
- 不含真实查询、gold、人工答案或响应正文的汇总指标和审计统计。

### 禁止提交

- `.env` 或任何凭据；
- PaSa 原始 JSONL；
- 开发、验证或模拟测试 gold；
- 真实查询文本工作包；
- 任一成员的人工标注；
- 含受限内容的截图、日志、缓存数据库、压缩包或临时文件。

## 14. 汇报模板

### 环境和固定数据就绪

```text
阶段：环境与固定数据核对完成
分支：codex/week1-collaboration
Python：3.11.x
pytest/Ruff/mypy：通过
PaSa 访问：已获批/阻塞
revision：<SHA>
分区数量：dev=60, validation=30, simulated_test=50
约束/重叠清单：40/20
受限文件进入 Git：否
阻塞：<无或匿名化说明>
```

### 标注工作包完成

```text
阶段：工作包 <2/3/4> 完成
输入身份：<指南版本和 ID 清单哈希>
记录数量：<数量>
Schema：通过/失败
缺失/重复/新增 ID：0/0/0
疑难项：<数量>
受限文件进入 Git：否
下一步：<等待批准或下一工作包>
```

### 双标一致性

```text
阶段：20 条独立双标完成
指南版本：<版本>
完成前交换答案：否
query_type kappa：<数值>
domain kappa：<数值>
需重标样本：<数量>
额外复核 10 条：完成/待完成
```

### 数据冻结申请

```text
阶段：申请正式冻结
revision：<SHA>
count：dev=60, validation=30, simulated_test=50
gold/ID 哈希：已逐项复核
ID 唯一且顺序一致：是
labels_complete：true
zero_answer_policy：<reject/allow>
双人复核：完成
未解决问题：<无或列表>
```

### Week 1 gate 审核

```text
阶段：Week 1 gate 待判定
run identity / Git SHA：<值>
macro F1 / Recall：<值>
模糊去重准确率：<值>
过滤/初筛 Recall 绝对损失：<值>
硬失败率 / 部分结果率：<值>/<值>
API 调用 / P50 / P95 / 成本：<值>
快照与哈希：通过/失败
秘密与范围检查：通过/失败
独立审查：通过/失败
建议：通过/阻塞/修复后重审
```

## 15. 现在立即执行

1. 拉取 `codex/week1-collaboration`；
2. 完成工作包 0 的不主动加载 `.env` 的环境验证；
3. 确认自己的 PaSa 只读访问；
4. 完成工作包 1，核对 revision、哈希和 60/30/50、40、20；
5. 回复主负责人“环境与固定数据核对完成”；
6. 等待“Task 2 标注工作包 v1 已冻结”；
7. 收到通知后依次完成 90 条、40 条和固定 20 条人工工作；
8. 正式冻结批准前，不运行 baseline，不修改 manifest 状态。
