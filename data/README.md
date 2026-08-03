# Task 2 数据与协作者工作流

当前状态：`frozen`（V2 冻结已核准；Gate 0 r5 `passed: true`，安全报告见 `data/gate0_evidence.json`）

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
uv sync --locked --extra cpu
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

## 已发布的标注输入与预冻结校验

“Task 2 标注工作包 v1 已冻结（仅标注输入）”通知已经发布。它只固定标注输入；在正式冻结完成前 manifest 必须保持 `waiting_for_human_label_freeze`。当前仓库已于 2026-08-03 由 Gate 0 r5 核准为 V2 frozen。

协作者可在不运行正式 freeze audit 的情况下分别校验 90 条类型/领域和 40 条约束文件；主负责人用相同入口校验独立 20 条 overlap 文件：

```powershell
uv run --no-sync --no-env-file python -m paper_search.evaluation.annotation --kind type-domain --labels data/annotation_work/type_domain_labels.jsonl --ids data/splits/type_domain_annotation.ids.json
uv run --no-sync --no-env-file python -m paper_search.evaluation.annotation --kind constraints --labels data/annotation_work/constraint_labels.jsonl --ids data/splits/constraint_annotation.ids.json
uv run --no-sync --no-env-file python -m paper_search.evaluation.annotation --kind constraints --labels data/annotation_work/overlap_labels.jsonl --ids data/splits/overlap_annotation.ids.json
```

成功后只通过受控渠道交换记录数量与精确字节 SHA-256，不交换逐条答案。该校验不计算 kappa、不修改 manifest，也不能替代正式冻结审核。
## Frozen partition contract

Do not change the manifest status to `frozen` until human labeling and the data freeze are complete. Each frozen partition must contain:

- a positive integer `count`;
- confined relative `gold_path` and `ids_path` values under `data/`;
- exact-byte `gold_sha256` and `ids_sha256` values using the `sha256:<hex>` form;
- `labels_complete: true`;
- `zero_answer_policy` set explicitly to `reject` or `allow`.

The manifest must retain a nonempty dataset `revision`. Gold JSONL must be nonempty, contain exactly `count` records, and list query IDs in exactly the same order as the unique nonempty strings in the JSON ID list. Under `reject`, every gold record needs at least one relevant paper ID. Under `allow`, zero-answer records are permitted and the policy becomes part of formal run identity.
## 冻结审核与批准

不要手工把 `status` 改为 `frozen`。团队通过私密渠道准备三份真人标注文件：协作者交付类型/领域与约束标注，主负责人交付独立 overlap 标注。文件路径、正文和标注人答案不得写入 Git、普通聊天或审核报告。主负责人先运行只读审核，并为每个 manifest 分区显式选择 `reject` 或 `allow`；工具会拒绝遗漏、重复、未知分区和非法策略，不存在全局默认值：

```powershell
uv run --no-sync --no-env-file python -m paper_search.evaluation.freeze `
  --data-root data `
  --type-domain-labels <private-type-domain-labels.jsonl> `
  --constraint-labels <private-constraint-labels.jsonl> `
  --overlap-labels <private-overlap-labels.jsonl> `
  --zero-answer-policy dev=reject `
  --zero-answer-policy validation=reject `
  --zero-answer-policy simulated_test=allow
```

以上命令只向 stdout 输出安全 JSON 摘要，不写 report 或 manifest。摘要只包含数量、精确字节 SHA-256、分区身份和聚合 kappa；不包含查询、paper ID、标注正文、标注人答案或私密路径。

审核通过且策略经主负责人确认后，主负责人使用完全相同的输入增加显式批准参数：

```powershell
uv run --no-sync --no-env-file python -m paper_search.evaluation.freeze `
  --data-root data `
  --type-domain-labels <private-type-domain-labels.jsonl> `
  --constraint-labels <private-constraint-labels.jsonl> `
  --overlap-labels <private-overlap-labels.jsonl> `
  --zero-answer-policy dev=reject `
  --zero-answer-policy validation=reject `
  --zero-answer-policy simulated_test=allow `
  --approve `
  --report data/freeze_reports/data-freeze-232428b0-v1.json
```

`--report` 必须位于 `data/freeze_reports/` 下。程序先完整写入安全 report，再通过受保护的同目录原子替换把 manifest 从审核时的精确 prepared 字节转换为 frozen 字节。若 manifest 在审核后变化，批准失败。若 report 已完整写入但 manifest 替换失败，该 report 是可复用的孤立证据，不代表已经冻结；当前 `data/manifest.json` 的 `status: frozen`、report 路径及 report SHA-256 才是唯一权威状态。

孤立 report 重试时使用完全相同的私密文件、逐分区策略、`--report` 路径和完整批准命令；程序只复用字节完全一致的 report，任何差异都会拒绝。三份私密文件应保存在仓库外的访问受控目录，文件名不要包含查询或人员身份；仅在本地受控终端运行，按团队保留策略清理，不复制到 CI、普通日志或共享命令记录。

冻结命令不自动暂存或提交文件。真实 gold、原始数据和三份人工标签始终不得进入 Git；`data/manifest.json`、安全 report 和 ID 清单是否进入后续证据提交，必须由主负责人另行执行范围与隐私审查。

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

1. 切换到 `codex/week1-collaboration`，执行 `uv sync --locked --extra cpu`；
2. 运行全量 pytest、Ruff 和 mypy，报告环境结果但不附密钥；
3. 使用自己的账号运行数据准备命令；
4. 核对 manifest 中的 revision、三个源文件哈希、分区数量和工作包数量；
5. 核对 `data/splits/constraint_annotation.ids.json` 为 40 条、`overlap_annotation.ids.json` 为 20 条；
6. 核对已经发布的“Task 2 标注工作包 v1 已冻结（仅标注输入）”通知；该通知只固定标注输入；
7. 使用 `domain-labels-v1` 和上述安全校验命令完成 90/40/20 工作流，两位成员完成前不交换答案。

工程脚本和合成测试通过不代表人工标签已经完成。90 条类型/领域标记、40 条约束标注、固定 20 条独立双标及最终 kappa 必须由真人完成后，状态才能从 `waiting_for_human_label_freeze` 更新。
