# Task 2 数据与协作者工作流

当前状态：`waiting_for_human_label_freeze`

本目录保存 Task 2 可复现评测所需的安全元数据、固定 ID 清单、标注指南和项目原创压力查询。PaSa 原始数据、真实查询、金标准与人工标注受访问条件约束，不进入 Git。

## 固定数据源

- 仓库：`CarlanLark/pasa-dataset`
- Revision：`232428b0c867268c3b8ded90db4d98c1b30501d6`
- 许可：CC BY-NC-SA 4.0
- 访问：Hugging Face gated dataset；每位成员使用自己的只读 `HF_TOKEN`
- 随机种子：`20260714`
- 抽样算法：`answer-count-largest-remainder-v1`

脚本只请求以下文件：

- `AutoScholarQuery/dev.jsonl`
- `AutoScholarQuery/test.jsonl`
- `RealScholarQuery/test.jsonl`

## 数据准备

先在自己的 Hugging Face 账号接受数据条款，将 Token 仅写入本地环境配置，然后执行：

```powershell
uv sync --all-groups
uv run --env-file .env python scripts/prepare_task2_data.py --output-root data
```

脚本验证固定 revision、Schema 和官方行数，随后生成：

- 开发集 60 条、验证集 30 条、模拟测试集 50 条；
- `data/manifest.json` 与各分区 ID 清单；
- 私密的类型/领域和约束标注源文件；
- 固定 40 条约束标注 ID 与固定 20 条双人重叠 ID。

所有文件使用确定性排序、SHA-256 和冻结写入。完全相同的重跑允许通过；任一已有文件内容不同则拒绝覆盖。

## 评测命令

使用仓库内的非受限合成 fixture 做冒烟测试：

```powershell
uv run python -m paper_search.evaluation.metrics `
  --gold tests/fixtures/evaluation/gold.jsonl `
  --pred tests/fixtures/evaluation/predictions.jsonl `
  --out experiments/smoke/metrics.json
```

对本地开发集评测时，将 `--gold` 指向 `data/dev/gold.jsonl`，将 `--pred` 指向符合固定预测 Schema 的本地文件。不要提交受限输入或由其生成的逐查询内容。

## Git 边界

| 路径 | Git 状态 | 内容 |
|---|---|---|
| `data/manifest.example.json` | 提交 | Manifest 结构示例 |
| `data/manifest.json` | 可提交 | Revision、哈希、数量、算法和状态；不得含查询文本 |
| `data/splits/` | 可提交 | 只含固定 query ID 的清单 |
| `data/stress/queries.jsonl` | 提交 | 24 条项目原创压力查询 |
| `data/annotation_guide.md` | 提交 | 人工标注口径 |
| `data/raw/` | 忽略 | PaSa 原始文件 |
| `data/dev/gold.jsonl` | 忽略 | 受限开发集金标准 |
| `data/validation/gold.jsonl` | 忽略 | 受限验证集金标准 |
| `data/simulated_test/` | 忽略 | 模拟测试集内容 |
| `data/annotation_work/` | 忽略 | 真实查询源文件与双方标注 |

## 协作者开始流程

1. 切换到 `codex/task2-evaluation`，执行 `uv sync --all-groups`；
2. 运行全量 pytest、Ruff 和 mypy，报告环境结果但不附密钥；
3. 使用自己的账号运行数据准备命令；
4. 核对 manifest 中的 revision、三个源文件哈希、分区数量和工作包数量；
5. 核对 `data/splits/constraint_annotation.ids.json` 为 40 条、`overlap_annotation.ids.json` 为 20 条；
6. 等待主负责人发出“Task 2 标注工作包 v1 已冻结”通知；
7. 两位成员按 `data/annotation_guide.md` 独立完成固定重叠样本，完成前不交换答案。

工程脚本和合成测试通过不代表人工标签已经完成。90 条类型/领域标记、40 条约束标注、固定 20 条独立双标及最终 kappa 必须由真人完成后，状态才能从 `waiting_for_human_label_freeze` 更新。
