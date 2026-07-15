# Task 2 数据集适配和评测指标设计

**日期：** 2026-07-15

**状态：** 已批准，待实施

**负责人：** 主负责人

**分支：** `codex/task2-evaluation`

## 1. 背景与目标

Task 2 建立项目统一的离线评测契约，使后续检索、排序和预算优化都能用同一条命令计算可复现指标。实现必须适配 PaSa 数据集，同时为赛事未来可能发布的官方 Schema 保留独立适配边界。

本任务的目标是：

1. 定义严格、可复用的 JSONL 金标准与预测格式；
2. 统一 DOI、arXiv、OpenAlex、Semantic Scholar 和标题标识；
3. 实现 PaSa 数据适配和固定内部评测适配器；
4. 实现 Precision、Recall、F1、Recall@5/10/20、宏平均和微平均；
5. 提供一条命令完成静态预测文件评测；
6. 固定数据 revision、文件哈希、随机种子、抽样规则和访问条件；
7. 生成协作者可直接填写的人工标注工作包和一致性检查工具；
8. 不将受限数据、真实标签、Token 或私钥提交到 Git。

## 2. 非目标

本任务不下载 PaSa 论文数据库、训练集或模型，不实现在线检索，不训练模型，不用 LLM 伪造人工独立标注，也不提前假设赛事一定会发布官方 Schema。

人工标注尚未完成时，工程部分可以验收，但 Task 2 的项目状态必须标记为“等待人工标签冻结”，不能宣称全部数据工作已完成。

## 3. 已确认的外部数据

数据仓库：`CarlanLark/pasa-dataset`

固定 revision：`232428b0c867268c3b8ded90db4d98c1b30501d6`

访问方式：Hugging Face gated dataset，脚本使用本地 `HF_TOKEN` 只读访问

许可证：CC BY-NC-SA 4.0

只下载以下三个文件：

| 文件 | 官方行数 | 用途 |
|---|---:|---|
| `AutoScholarQuery/dev.jsonl` | 1000 | 抽取 60 条开发样本 |
| `AutoScholarQuery/test.jsonl` | 1000 | 抽取 30 条验证样本 |
| `RealScholarQuery/test.jsonl` | 50 | 全部作为模拟测试样本 |

三个文件的字段均为 `qid`、`question`、`answer`、`answer_arxiv_id`、`source_meta`。当前 `source_meta` 仅提供 `published_time`，不包含查询类型或领域，因此自动抽样不能虚称按人工标签分层。

## 4. 总体架构

新增 `paper_search.evaluation` 包：

```text
src/paper_search/evaluation/
├── __init__.py
├── dataset.py
├── official_adapter.py
├── metrics.py
└── annotation.py
```

模块职责：

- `dataset.py`：内部记录模型、JSONL 读写、标识归一化、ID 映射、哈希和确定性抽样；
- `official_adapter.py`：PaSa 源格式、内部格式、固定预测格式之间的转换；
- `metrics.py`：纯指标函数、聚合结果和命令行入口；
- `annotation.py`：标注工作包校验与 Cohen's kappa 计算；
- `scripts/prepare_task2_data.py`：下载、验证、抽样、冻结和生成工作包的编排脚本。

核心指标不依赖网络。下载与准备步骤和评分步骤严格分离。

## 5. 内部数据模型

所有模型沿用 Task 1 的 Pydantic 约定：`extra="forbid"`、不可变、字符串去空白、集合字段显式定义。

### 5.1 EvaluationQuery

```text
query_id: 非空字符串
query: 非空字符串
relevant_paper_ids: 非重复的规范化论文 ID 列表
metadata: JSON 可序列化字典
```

`relevant_paper_ids` 在模型边界完成规范化并拒绝重复规范 ID。允许为空，以覆盖未来官方数据中的无答案查询和边界测试。

### 5.2 PredictionRecord

```text
query_id: 非空字符串
predicted_paper_ids: 按排名排列的论文 ID 列表
```

预测去重在评分入口执行，保留第一次出现的位置，从而维持 Recall@K 的排名语义。

### 5.3 IdentifierMap

可选 ID 映射文件保存 `alias -> canonical_id`。加载时先规范化两端，再检测：

- 同一 alias 映射到多个 canonical ID；
- 映射链产生环；
- 空 ID 或不支持的命名空间。

发生冲突时立即失败，不静默选择任一映射。

## 6. 论文标识归一化

内部统一使用带命名空间的字符串：

```text
doi:10.xxxx/xxxx
arxiv:2501.10120
openalex:W123456
s2:<semantic-scholar-paper-id>
title:<normalized-title>
```

规则如下：

- DOI：去掉 `doi:`、`https://doi.org/`、`http://doi.org/` 和 `http://dx.doi.org/` 前缀，转小写；
- arXiv：去掉 `arXiv:`、abs/pdf URL、`.pdf` 和末尾版本号 `vN`，保留基础 ID；
- OpenAlex：去掉 OpenAlex URL 或命名空间，统一为大写 `W` 加数字；
- Semantic Scholar：去掉论文 URL 或命名空间，保留稳定 paper ID，命名空间统一为 `s2:`；
- 标题：执行 Unicode NFKC、大小写折叠、标点转空格、连续空白折叠；
- 标题只用于缺少正式 ID 时的内部诊断后备，不替代赛事官方 ID。

无法识别且没有显式类型提示的裸字符串必须报错，避免把标题误当 Provider ID。

## 7. OfficialEvaluationAdapter

在赛事官方评分器尚未发布时，适配器采用 PRD 14.0 的固定内部契约：

1. PaSa `qid` 映射到 `query_id`；
2. `question` 映射到 `query`；
3. 优先使用 `answer_arxiv_id` 生成 `arxiv:` 金标准；
4. 仅当正式 ID 缺失时，才从 `answer` 生成 `title:` 后备标识；
5. `source_meta` 原样放入 metadata，并增加 source、split 和 dataset revision；
6. 预测文件固定为 `query_id` 与 `selected_paper_ids`；
7. 预测字段、查询 ID、去重和空答案行为用固定 fixture 验证。

未来赛事发布官方 Schema 时，增加新的适配器实现和官方样例测试。指标内核与现有数据模型不因外部 Schema 改动而重写。官方契约优先于内部契约，但内部结果保留用于回归比较。

## 8. 数据准备与冻结

数据准备流程：

```text
读取 HF_TOKEN
→ 按固定 revision 下载三个 JSONL
→ 校验 HTTP 状态、行数和 SHA-256
→ 转换为 EvaluationQuery
→ 按固定种子进行确定性抽样
→ 原子写入本地 gold、ID 清单、manifest 和工作包
```

### 8.1 抽样

随机种子固定为 `20260714`。

- 开发集：从 AutoScholarQuery dev 抽取 60 条；
- 验证集：从 AutoScholarQuery test 抽取 30 条；
- 模拟测试集：RealScholarQuery test 全部 50 条。

开发集和验证集按标准答案数量分层：`1`、`2–3`、`4–7`、`8+`。按各层在源数据中的占比分配目标数量，使用最大余数法处理取整，层内使用固定种子洗牌。若某层不足，脚本停止并报告，不静默退化为普通随机抽样。

查询类型与领域在样本确定后由协作者人工补标。manifest 必须明确记录自动分层依据是答案数量，而不是人工查询类别。

### 8.2 冻结与覆盖保护

首次成功准备后写入 `data/manifest.json`，记录：

- repo ID、revision、下载日期和访问条件；
- 每个源文件的相对路径、行数、字节数和 SHA-256；
- 每个目标分区的数量、ID 清单哈希、抽样算法版本和随机种子；
- 许可证标识与原始仓库 URL。

已存在且哈希匹配时允许幂等重跑；内容不匹配时拒绝覆盖。需要更换 revision 时创建新的冻结版本并显式更新 manifest，不原地篡改旧结果。

## 9. 数据目录与 Git 策略

```text
data/
├── README.md
├── manifest.json
├── annotation_guide.md
├── splits/
│   ├── dev.ids.json
│   ├── validation.ids.json
│   └── simulated_test.ids.json
├── stress/
│   └── queries.jsonl
├── raw/                    # gitignored
├── dev/gold.jsonl          # gitignored
├── validation/gold.jsonl   # gitignored
├── simulated_test/         # gitignored
└── annotation_work/        # gitignored
```

仓库提交代码、文档、manifest、哈希、随机种子、ID 清单和项目原创压力查询。受限原始数据、真实查询文本工作包和标签不提交。每位队员使用自己的 Hugging Face 账号和 Token 按相同 revision 复现。

`.env.example` 增加空的 `HF_TOKEN=`，真实值只保存在已忽略的 `.env`。

## 10. 指标契约

评分前执行以下步骤：

1. 读取并验证所有 gold 和 prediction 记录；
2. 拒绝重复 query ID；
3. 规范化论文 ID 并应用可选 ID 映射；
4. gold 作为集合；prediction 去重并保留首次排名；
5. prediction 中未知 query ID 视为输入错误；
6. gold 中缺失 prediction 的查询按空预测处理。

单查询规则：

- gold 和 prediction 都为空：Precision、Recall、F1、Recall@K 均为 1；
- 只有一方为空：上述指标均为 0；
- 否则按 TP、FP、FN 计算 Precision、Recall 和 F1；
- Recall@K 为前 K 个去重预测命中的 gold 数量除以 gold 数量。

聚合结果：

- 主指标：逐查询 F1 的宏平均；
- 辅助指标：宏平均 Precision、Recall、Recall@5/10/20；
- 微平均：先汇总 TP、FP、FN，再计算 Precision、Recall、F1；
- 输出逐查询指标和命中的规范化 ID，支持错误分析。

## 11. 命令行接口

评分命令：

```powershell
python -m paper_search.evaluation.metrics \
  --gold data/dev/gold.jsonl \
  --pred tests/fixtures/predictions.jsonl \
  --out experiments/smoke/metrics.json
```

可选参数：`--id-map`。输出使用 UTF-8、排序键和稳定字段顺序，并通过临时文件加原子替换写入。

输出结构：

```text
contract_version
input_hashes
summary
per_query
```

`summary` 至少包含 macro precision/recall/F1、micro precision/recall/F1、Recall@5/10/20、查询数和缺失预测数。

## 12. 人工标注工作流

协作者任务：

1. 为 60 条开发和 30 条验证查询标注 query type 与 domain；
2. 为固定的 40 条开发查询填写约束标注；
3. 字段固定为 `query_id, research_goal, must_have, should_have, exclusions, year_from, year_to, venues, query_type, domain, annotator`；
4. 协作者完成 40 条，主负责人独立标注其中相同的 20 条；
5. 主负责人额外复核 10 条。

`annotation.py` 对关键离散字段计算 Cohen's kappa。任一关键字段低于 0.80 时，先修改 `data/annotation_guide.md`，再重新标注分歧样本。工具只计算真实提交的两份标注，不生成或补写人工答案。

## 13. 压力集

项目原创 24 条压力查询，不参与 F1 调参。覆盖 PRD 七类查询：主题、方法、数据集、时间/venue、组合约束、关系和排除；同时覆盖中英文表达、语义等价改写、长查询、歧义和缺失元数据。

压力集只包含查询与测试标签，不复制 PaSa 受限内容。后续报告改写前后的预测集合 Jaccard、F1 和失败原因。

## 14. 错误处理与安全

以下情况立即失败并给出可定位错误：

- `HF_TOKEN` 缺失、无权访问或被撤销；
- 下载 revision 与固定 revision 不一致；
- 文件行数、SHA-256 或 Schema 不匹配；
- JSONL 非法、字段缺失、未知字段或重复 query ID；
- ID 格式非法、映射冲突或映射环；
- 抽样层不足；
- 输出目录中已有不一致的冻结文件。

错误信息不得包含 Token、Authorization Header 或完整 `.env` 内容。manifest 不记录本机绝对路径、用户名或 Token。

## 15. 测试策略

严格采用 TDD：每个行为先写失败测试并确认失败原因，再实现最小代码。

测试覆盖：

- DOI、arXiv、OpenAlex、Semantic Scholar 和标题归一化；
- 非法和歧义 ID；
- JSONL 非法行、字段缺失、未知字段和重复 query ID；
- PaSa 源格式转换与标题后备；
- ID 映射、冲突和环；
- 零预测、零 gold、双方为空和重复预测；
- Recall@5/10/20 的去重排名语义；
- 未知预测 query ID 和缺失预测；
- 固定 fixture 的单查询、宏平均和微平均；
- CLI 成功输出与非法输入失败；
- 确定性分层抽样、分层不足和幂等冻结；
- 标注 Schema 和 Cohen's kappa。

核心测试完全离线。真实下载测试标记为 online，不进入默认单元测试。

## 16. 验收标准

工程验收：

```powershell
uv run pytest tests/evaluation -v
uv run python -m paper_search.evaluation.metrics --gold data/dev/gold.jsonl --pred tests/fixtures/predictions.jsonl --out experiments/smoke/metrics.json
uv run pytest -q
uv run ruff check .
uv run mypy src
```

验收结果必须满足：

- 评测测试、全量测试、Ruff 和 mypy 全部通过；
- CLI 退出码为 0，并生成宏平均 F1 与逐查询结果；
- 同一输入重复运行输出指标一致；
- manifest 能定位 revision、文件哈希、抽样算法和随机种子；
- Git 中不存在受限数据、真实 Token、私钥或人工工作包；
- 人工标签未冻结前，状态明确记录为待办。

## 17. 团队交付

主负责人交付评测内核、适配器、数据准备、冻结、manifest、测试、标注指南和复核工具。协作者交付查询类型/领域标记和约束标注。主负责人不等待协作者学习进度才完成核心工程，但人工标签冻结仍是 Task 2 的独立阶段闸门。
