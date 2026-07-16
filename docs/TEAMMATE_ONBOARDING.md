# 协作者快速接入与 Task 2 工作指南

> 适用对象：第一次加入项目、此前只看得到 GitHub 仓库的协作者。
>
> 当前协作分支：`codex/task2-evaluation`
>
> 最后更新：2026-07-16

## 1. 先了解三件事

1. 这是一个面向复杂学术查询的智能论文搜索与推荐项目；主负责人负责架构、后端、评测和算法，协作者当前重点负责 Task 2 的数据质检与人工标注。
2. Task 2 的工程部分正在按 TDD 实现。编写本文档时，评测模型与论文 ID 规范化已经完成，对应提交为 `f701565`，全量测试为 66 项。
3. PaSa 是受限数据集。真实查询、金标准和人工标注文件不得提交到 GitHub；GitHub 只保存代码、安全文档、哈希、ID 清单和不含受限文本的汇总信息。

建议按以下顺序阅读：

1. 本文档；
2. [`PRD.md`](../PRD.md) 中的“Task 2：数据集适配和评测指标”；
3. [`Task 2 设计文档`](superpowers/specs/2026-07-15-task2-evaluation-design.md)；
4. [`Task 2 实施计划`](superpowers/plans/2026-07-16-task2-evaluation-implementation.md)。

## 2. API、`.env` 和软件环境

一句话说明：请安装 Git、Python 3.11 和 uv，使用自己的 OpenAlex、Semantic Scholar、DashScope 和 Hugging Face 只读凭据，复制 `.env.example` 为本地 `.env` 后填写，最后执行 `uv sync --all-groups`；不得复制主负责人的密钥、`.venv` 或整个本地项目目录。

### 2.1 需要自行申请的凭据

| 本地环境变量 | 用途 | 当前 Task 2 是否必需 |
|---|---|---|
| `HF_TOKEN` | 访问已接受条款的 PaSa gated dataset | 是 |
| `OPENALEX_API_KEY` | 后续 OpenAlex 检索 | 当前人工标注不必调用 |
| `SEMANTIC_SCHOLAR_API_KEY` | 后续 Semantic Scholar 检索 | 当前人工标注不必调用 |
| `LLM_API_KEY` | DashScope 兼容接口 | 当前人工标注不得使用 LLM 代写 |
| `LLM_BASE_URL` | DashScope 兼容接口地址 | 环境联调需要 |
| `LLM_MODEL_PRIMARY` | 主模型名称 | 使用项目当前配置，不自行猜测 |
| `LLM_MODEL_FALLBACK` | 备用模型名称 | 使用项目当前配置，不自行猜测 |

注意：当前分支的 `.env.example` 还没有 `HF_TOKEN=`，它会在 Task 2 数据准备阶段补上。在此之前，请只在你自己的 `.env` 末尾手动加入：

```dotenv
HF_TOKEN=
```

真实值只写在 `.env` 中。不要将密钥粘贴到 Issue、提交记录、日志、截图或聊天消息中。

### 2.2 首次拉取和验证

```powershell
git clone https://github.com/Tianyuai/AI-Projects.git
Set-Location AI-Projects
git fetch origin
git switch --track origin/codex/task2-evaluation
Copy-Item .env.example .env
uv sync --all-groups
uv run pytest -q
uv run ruff check .
uv run mypy src
```

如果本地已经有同名分支，使用：

```powershell
git switch codex/task2-evaluation
git pull --ff-only
```

环境验收要求：Python 为 3.11.x，`uv sync --all-groups` 成功，pytest、Ruff 和 mypy 全部通过。若失败，请把命令、错误信息和系统版本发给主负责人，但先删除其中可能出现的本机用户名、绝对私有路径和凭据。

## 3. 你现在应当做什么

在人工标注工作包生成前，先完成以下工作：

- [ ] 克隆仓库并切换到 `codex/task2-evaluation`；
- [ ] 完成 Python 3.11、uv 和项目依赖安装；
- [ ] 使用自己的账号申请四类 API/数据访问权限并配置本地 `.env`；
- [ ] 在 Hugging Face 页面接受 `CarlanLark/pasa-dataset` 的访问条款；
- [ ] 运行 pytest、Ruff 和 mypy，确认环境可复现；
- [ ] 阅读本文档、Task 2 设计和实施计划；
- [ ] 等待主负责人明确发出“Task 2 标注工作包 v1 已冻结”的开始信号。

在开始信号发出前，不要自行选择样本、修改 split ID、复制网络上的 PaSa 数据，或提前创建自己的标注格式。

## 4. Task 2 的目标和你的职责边界

Task 2 要建立可复现的离线评测体系，包括数据适配、ID 规范化、Precision/Recall/F1、Recall@5/10/20、固定抽样、数据冻结和人工一致性检查。

主负责人负责：

- JSONL 契约、ID 规范化和映射；
- PaSa 与预测格式适配器；
- 指标内核和评测命令；
- 固定 revision、抽样、哈希、manifest 和覆盖保护；
- 标注 Schema、校验工具、Cohen's kappa 和最终复核。

协作者负责：

- 开发集与验证集的查询类型和领域标记；
- 固定 40 条开发查询的约束标注；
- 与主负责人对其中固定 20 条进行相互独立的双人标注；
- 按标注指南修正分歧样本；
- 从全新环境走查数据准备说明，记录复现问题；
- 整理不包含受限文本的进度、缺失项和问题汇总。

你不负责修改评测算法、金标准、抽样种子、数据 revision 或 split ID。发现问题时先记录并提交给主负责人，不要直接“修正”公开或冻结标签。

## 5. 人工标注的开始条件

只有以下条件全部满足后，才开始正式标注：

- 主负责人明确宣布“Task 2 标注工作包 v1 已冻结”；
- `data/manifest.json` 已记录 PaSa revision、文件哈希、随机种子和抽样算法；
- `data/splits/dev.ids.json` 与 `data/splits/validation.ids.json` 已冻结；
- `data/annotation_guide.md` 已明确 query type、domain 和各约束字段的口径；
- 本地 `data/annotation_work/` 工作包已生成并通过 Schema 校验；
- 双方确认固定的 20 条重叠标注 query ID，但没有查看对方答案。

推荐每位成员使用自己的 Hugging Face 账号和 Token，通过同一 revision 在本地复现工作包。确需传递工作包时，只能使用双方约定的私密渠道，不能上传到 GitHub。

## 6. 具体工作包

### 工作包 A：90 条查询类型与领域标记

标记范围：

- 开发集 60 条；
- 验证集 30 条；
- 合计 90 条。

每条记录至少填写：

- `query_id`：保持工作包原值，不得修改；
- `query_type`：只能使用 `data/annotation_guide.md` 允许的标签；
- `domain`：只能使用指南约定的格式；
- `annotator`：使用团队约定的稳定代号。

遇到无法归类的查询时，不自行发明新标签。把 `query_id`、候选标签和疑问写入单独的问题清单，由双方统一修改指南后再继续。

### 工作包 B：40 条开发查询约束标注

这 40 条由冻结的 ID 清单指定，不是由协作者自行挑选。每条记录字段固定为：

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

填写原则：

- `research_goal`：概括用户真正要解决的研究问题，不改写成搜索关键词堆积；
- `must_have`：缺失任一项就不应入选的硬约束；
- `should_have`：提高相关性但不构成一票否决的软约束；
- `exclusions`：明确需要排除的方法、主题、文献类型或条件；
- `year_from`、`year_to`：只在原查询明确要求时填写，边界包含在内；
- `venues`：只记录原查询明确给出的 venue；
- 不确定时按指南记录，不根据个人领域常识补造用户没有表达的条件。

以下示例是项目原创的格式示例，不来自 PaSa：

```json
{"query_id":"example-q1","research_goal":"查找用于长文档检索的高效注意力方法","must_have":["长文档检索"],"should_have":["稀疏注意力"],"exclusions":["仅图像任务"],"year_from":2020,"year_to":null,"venues":[],"query_type":"method","domain":"information-retrieval","annotator":"member-b"}
```

实际允许的 `query_type` 和 `domain` 值以冻结版 `data/annotation_guide.md` 为准，不能从示例推断完整标签集合。

### 工作包 C：固定 20 条双人独立标注

这 20 条包含在协作者的 40 条中，不是额外增加 20 条。流程如下：

1. 双方确认相同的 20 个 query ID 和相同指南版本；
2. 协作者与主负责人分别独立填写；
3. 双方完成前不交换、不查看对方答案，也不使用 LLM 生成标签；
4. 完成后由主负责人运行一致性工具；
5. 对 `query_type`、`domain` 等关键离散字段计算 Cohen's kappa；
6. 任一关键字段低于 `0.80` 时，先修订指南，再重新标注分歧样本；
7. 一致性达标后，由主负责人额外复核 10 条。

### 工作包 D：复现和数据质检

工程脚本完成后，协作者在自己的全新环境执行数据准备与测试命令，检查：

- 文档是否足以完成安装和数据准备；
- 固定 revision 和样本数是否一致；
- 同一输入重复运行是否得到相同 ID 清单和哈希；
- 受限数据和标注文件是否被 Git 正确忽略；
- 错误信息是否会泄露 Token、本机用户名或完整 `.env`；
- 指南中是否存在两种合理但冲突的解释。

问题汇总可以提交到 GitHub，但只能包含命令、匿名化错误、哈希、数量和改进建议，不得包含真实查询、金标准或标注内容。

## 7. 交付物与验收标准

协作者完成 Task 2 配合工作的标准：

- [ ] 90 条记录全部具有合法的 `query_type`、`domain` 和 `annotator`；
- [ ] 固定 40 条查询约束记录全部填写，字段名和类型通过 Schema 校验；
- [ ] 固定 20 条重叠样本在未查看对方答案的情况下独立完成；
- [ ] query ID 无缺失、无新增、无重复、无修改；
- [ ] 年份区间合法，集合字段没有空字符串；
- [ ] 所有疑难项已进入问题清单并得到统一口径；
- [ ] 关键离散字段 Cohen's kappa 不低于 `0.80`，或已按流程修订指南并重标；
- [ ] 主负责人额外复核的 10 条已完成问题闭环；
- [ ] 真实标注文件通过私密渠道交接，没有进入 Git 历史；
- [ ] GitHub 中只提交不含受限文本的复现问题或汇总信息。

人工标签未完成和冻结前，Task 2 状态必须保持：

```text
waiting_for_human_label_freeze
```

## 8. Git 和数据安全规则

### 可以提交到 GitHub

- 代码、测试和安全文档；
- `data/README.md`、标注指南和不含受限文本的说明；
- 数据 revision、SHA-256、样本数量、随机种子和抽样算法版本；
- 只含 query ID 的安全 split 清单；
- 不含真实查询文本的统计汇总和匿名化问题记录。

### 禁止提交到 GitHub

- `.env`、API Key、Token、Authorization Header；
- PaSa 原始 JSONL；
- 开发/验证/模拟测试金标准；
- 真实查询文本工作包；
- 任一成员的人工标注 JSONL；
- 包含受限内容的截图、日志、压缩包或临时文件。

开始修改安全文档或测试前，从协作分支新建自己的分支：

```powershell
git switch codex/task2-evaluation
git pull --ff-only
git switch -c collab/task2-data-qa
```

完成后先运行测试，再向 `codex/task2-evaluation` 提交 Pull Request。只有人工标注或受限数据变化时，不需要也不允许通过 Git 提交；使用私密交接渠道并由主负责人记录哈希和冻结状态。

## 9. 主负责人的主线能否继续

可以继续，而且不应等待协作者学习完成后才推进核心工程。

### 当前可以继续的工作

- 严格 JSONL 读写、文件哈希和 ID Map；
- PaSa/预测格式适配器；
- Precision、Recall、F1、Recall@5/10/20 和宏/微平均；
- 评测 CLI、确定性抽样、冻结保护和 manifest；
- 标注 Schema、校验工具和 Cohen's kappa 代码；
- 使用合成 fixture 的检索、去重、排序和后端开发；
- 压力集、错误处理、安全检查和自动化测试。

### 必须等待协作者交付后才能完成的工作

- 将 Task 2 标记为全部完成；
- 冻结最终人工标签和一致性结果；
- 验收强约束抽取 Recall；
- 计算按 query type/domain 分组的正式指标；
- 用人工标签完成最终失败归因或模型方案选择；
- 声称开发/验证数据工作已经全部验收。

主线开发期间使用固定接口、合成 fixture 和冻结 ID 清单。不要用未冻结人工标签调参，也不要提前查看或反复运行模拟测试集标签。

## 10. 推荐的协作节奏

协作者只需在以下节点主动报告：

1. `环境就绪`：附 pytest、Ruff、mypy 的结果，不附密钥；
2. `工作包已复现`：附 revision、manifest/hash 和行数，不附查询文本；
3. `20 条独立标注完成`：只报告完成状态，双方此时才能交换答案；
4. `40 条约束标注完成`：附 Schema 校验结果和疑难项数量；
5. `90 条类型/领域标记完成`：附缺失/重复检查结果；
6. `分歧重标完成`：附各关键字段 kappa 和指南版本；
7. `私密交接完成`：双方核对文件哈希，主负责人更新冻结状态。

## 11. 当前立即行动清单

如果你今天刚加入，请只做以下五件事：

1. 拉取 `codex/task2-evaluation`；
2. 配置自己的 API 和 `.env`；
3. 安装依赖并跑通全部检查；
4. 阅读本文档、Task 2 设计和实施计划；
5. 回复主负责人：`环境就绪，等待 Task 2 标注工作包 v1 冻结通知`。

在收到冻结通知后，再开始 90 条类型/领域标记和 40 条约束标注。
