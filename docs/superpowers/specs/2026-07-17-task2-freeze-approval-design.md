# Task 2 数据冻结审核与显式批准设计

**日期：** 2026-07-17
**分支：** `codex/week1-collaboration`
**状态：** 已批准，待实施计划
**范围：** 主负责人在真人标注完成后，对 Task 2 数据执行安全审核，并完成一次性正式冻结

## 1. 背景

现有 `scripts/prepare_task2_data.py` 已固定 PaSa revision、源文件、抽样种子和抽样算法，并生成开发集、验证集、模拟测试集、ID 清单和私密标注源文件。准备阶段 manifest 的状态固定为 `waiting_for_human_label_freeze`。

现有 Week 1 runner 已能严格消费 `status == "frozen"` 的 manifest，并验证分区 count、gold/ID 路径、精确字节 SHA-256、标签完成状态、零答案策略、ID 唯一性、gold/ID 顺序和 Git SHA。它不负责证明真人标注已经完成，也不应负责修改 manifest。

当前缺口是准备阶段与正式 runner 之间没有受控的冻结审批边界。不能依赖人工直接编辑 manifest，也不能仅修改 `status` 宣称冻结完成。

## 2. 目标

建立一个独立、可测试、可审计的冻结审核与批准组件：

1. 验证准备 manifest 和固定数据身份未变化；
2. 验证 90 条类型/领域标记、40 条约束标注和固定20 条独立双标；
3. 计算关键字段 Cohen's kappa，并强制 `0.80` 门槛；
4. 要求主负责人按分区显式指定 `zero_answer_policy`；
5. 生成不含受限文本的安全审核报告；
6. 仅在显式批准且所有检查通过时，执行一次性原子状态转换；
7. 保持现有正式 runner 为只读消费者。

## 3. 非目标

- 不下载数据，不读取 `.env`，不访问网络；
- 不修改 gold、split ID、抽样种子、dataset revision 或人工标签；
- 不自动裁决标注分歧；
- 不运行 OpenAlex baseline；
- 不实现模糊去重、Recall、成本或最终 Week 1 gate 汇总；
- 不提交任何真实查询、gold 或人工标注文件；
- 不修改受保护的 Task 2 设计文档。

## 4. 总体架构

新增 `paper_search.evaluation.freeze` 模块，分为纯审核、冻结计划和受控写入三层：

```text
prepared manifest + gold/ID + private annotation files + explicit policies
  -> audit_freeze_candidate()
  -> FreezeAuditReport + FrozenManifestPlan
  -> approve_freeze(expected_prepared_sha256, approve=True)
  -> atomic manifest transition + safe freeze report
  -> existing Week 1 runner
```

审核逻辑不读取环境变量、不访问网络、不直接写文件。CLI 负责读取明确传入的路径、调用纯函数、显示安全错误，并在 `--approve` 时调用受控写入函数。

## 5. 数据模型

### 5.1 类型与领域标记

新增最小冻结模型：

```python
class TypeDomainAnnotationRecord(DomainModel):
    query_id: NonEmptyStr
    query_type: QueryType
    domain: DomainLabel
    annotator: NonEmptyStr
```

该模型只接受90 条工作包所需字段，拒绝未知字段。约束标注继续复用现有 `AnnotationRecord`。

### 5.2 分区策略

CLI 必须为准备 manifest 中的每个分区恰好提供一个策略：

```python
ZeroAnswerPolicy = Literal["reject", "allow"]
```

策略不得由 gold 内容自动推断。缺失、重复、未知分区或非法策略均拒绝审核。

### 5.3 审核报告

安全报告至少包含：

```python
class PartitionFreezeAudit(DomainModel):
    count: PositiveInt
    gold_path: NonEmptyStr
    gold_sha256: NonEmptyStr
    ids_path: NonEmptyStr
    ids_sha256: NonEmptyStr
    zero_answer_policy: ZeroAnswerPolicy
    labels_complete: Literal[True]


class FreezeAuditReport(DomainModel):
    prepared_manifest_sha256: NonEmptyStr
    dataset_revision: NonEmptyStr
    source_file_count: PositiveInt
    type_domain_count: PositiveInt
    type_domain_sha256: NonEmptyStr
    constraint_count: PositiveInt
    constraint_sha256: NonEmptyStr
    overlap_count: PositiveInt
    overlap_sha256: NonEmptyStr
    agreement: AgreementReport
    partitions: dict[str, PartitionFreezeAudit]
    approval_requested: bool
```

报告不得包含查询、相关论文 ID、标注正文、标注人答案、分歧明细、凭据、本机绝对路径或原始响应。三份人工标签文件只记录精确字节 SHA-256，不记录路径或内容。

## 6. 审核流程

### 6.1 准备 manifest 身份

审核要求：

- manifest 是 UTF-8 JSON 对象；
- `status == "waiting_for_human_label_freeze"`；
- repo ID、revision、随机种子和抽样算法与准备阶段固定常量一致；
- `source_files`、`partitions` 和 `work_packages` 结构完整；
- 当前 manifest 精确字节 SHA-256 在审核开始时记录。

源文件逐项核对相对 `raw_path`、row count、byte count 和 SHA-256。所有路径必须限制在 `data/` 根目录下。

### 6.2 分区审核

对准备 manifest 中的每个分区：

1. gold 和 ID 路径必须限制在 `data/` 下；
2. gold JSONL 必须非空并通过 `EvaluationQuery` Schema；
3. ID JSON 必须是非空唯一字符串列表；
4. gold 记录数和 ID 数量必须等于 manifest count；
5. gold query ID 顺序必须与 ID 清单完全一致；
6. 重新计算 gold 和 ID 精确字节 SHA-256；
7. `reject` 策略下每条 gold 至少有一个 relevant paper ID；
8. `allow` 策略允许零答案，但策略进入正式 manifest 和运行身份。

准备 manifest 尚未声明 `gold_sha256` 时，审核器从精确字节计算；已经声明时必须完全匹配，不允许静默覆盖。

### 6.3 工作包身份

审核以下安全 ID 清单和私密源文件身份：

- 类型/领域工作包：90 条；
- 约束工作包：40 条；
- 重叠清单：20 条；
- 工作包 source/ID 文件的精确字节哈希必须与准备 manifest 一致；
- overlap ID 必须是 constraint ID 的子集；
- constraint ID 必须是 dev ID 的子集；
- type/domain ID 必须恰好覆盖 dev 与 validation ID 的并集。

### 6.4 真人标签

CLI 明确接收三个私密文件路径：

- `--type-domain-labels`：90 条 `TypeDomainAnnotationRecord`；
- `--constraint-labels`：协作者完成的40 条 `AnnotationRecord`；
- `--overlap-labels`：主负责人独立完成的20 条 `AnnotationRecord`。

审核要求：

- 标签 ID 与对应冻结 ID 清单集合和数量完全一致；
- 每个文件内 query ID 唯一；
- `annotator` 非空；
- overlap 文件恰好覆盖固定20 条；
- 从 constraint 标签中取相同20 条，与 overlap 标签对齐；
- 对 `query_type` 和 `domain` 分别计算 kappa；
- 任一字段 kappa 低于 `0.80` 时拒绝冻结；
- 安全错误只指出字段和门槛状态，不输出具体 query ID 或双方答案。

若初始 kappa 未达标，指南修订和新独立重叠样本由人工流程处理；本组件不自动重抽样或重新计算已知分歧样本。

## 7. CLI

命令入口：

```powershell
uv run --no-env-file python -m paper_search.evaluation.freeze `
  --data-root data `
  --type-domain-labels <private-path> `
  --constraint-labels <private-path> `
  --overlap-labels <private-path> `
  --zero-answer-policy dev=reject `
  --zero-answer-policy validation=reject `
  --zero-answer-policy simulated_test=allow
```

默认只审核，不写文件。成功时向标准输出写安全 JSON 摘要；失败时退出码为 `2`，标准错误只显示固定安全消息。

显式批准增加：

```powershell
  --approve `
  --report data/freeze_reports/data-freeze-<revision>-v1.json
```

`--approve` 必须与显式 `--report` 同时出现。报告路径必须限制在 `data/freeze_reports/` 下。CLI 不接受模糊的全局默认策略。

## 8. 一次性状态转换

批准阶段采用 compare-and-swap：

1. 审核开始时读取并记录 prepared manifest 精确字节和 SHA-256；
2. 所有私密文件和数据完成验证后，再次读取 manifest；
3. 若字节或 SHA-256 与审核开始时不同，拒绝写入；
4. 生成确定性 frozen manifest 字节和安全 report 字节；
5. 对 manifest、report 及目标父目录做写入前 preflight；
6. report 通过临时文件和原子替换完整写入；
7. 再次确认 manifest 仍为审核时的精确字节后，通过临时文件和原子替换完成一次性转换；
8. frozen manifest 是冻结状态的唯一权威证据；完整 report 单独存在不代表已经冻结；
9. manifest 原子替换失败时允许留下内容完整但非权威的孤立 report，下次相同审核可幂等复用；不得留下截断 report、临时文件或部分 manifest。

frozen manifest 在原分区字段上增加：

```text
gold_sha256
ids_sha256
labels_complete: true
zero_answer_policy
```

顶层增加冻结证据：

```text
status: frozen
prepared_manifest_sha256
freeze_report_path
freeze_report_sha256
```

冻结报告记录 `approval_requested: true`。它只证明审核输入和批准意图；是否已经冻结必须读取当前 manifest 的状态、report 路径和 report SHA-256。报告不记录时间戳、用户名或本机路径，以保持确定性和隐私。

已冻结时：

- 完全相同的 policies、三份标签文件哈希和报告内容允许幂等审核，不重写字节；
- 任何不同内容均拒绝覆盖；
- 不支持回退到 waiting 状态。

## 9. 错误处理与安全

- 所有用户可见错误使用固定消息，不回显原始异常；
- 标签 Schema 错误不输出完整记录；
- 路径错误不输出私密绝对路径；
- 不序列化查询、gold 正文、paper ID 或标注内容到安全报告；
- 不读取 `.env`，不访问网络；
- CLI 使用 `--no-env-file` 只表示不主动加载 `.env`，不宣称清除已有进程环境；
- 审核只读阶段不修改任何文件；
- `--approve` 是唯一授权写入冻结状态的入口。

## 10. TDD 测试策略

按以下批次严格执行 RED → GREEN → 重构：

1. `TypeDomainAnnotationRecord` 的合法、未知字段和非法标签测试；
2. prepared manifest 固定身份、路径逃逸、源文件哈希和结构测试；
3. 分区 count、gold/ID 哈希、唯一性、顺序和零答案策略测试；
4. 90/40/20 工作包覆盖与哈希测试；
5. kappa 通过、低于门槛、缺失/重复 ID 和安全错误测试；
6. policy 缺失、重复、未知分区和显式策略测试；
7. audit-only 不写文件测试；
8. `--approve` compare-and-swap、TOCTOU、原子写入、幂等和拒绝覆盖测试；
9. sentinel 凭据、查询和私密路径不进入 stdout、stderr、manifest 或 report 的安全测试；
10. 与现有 Week 1 runner 的 frozen manifest 集成测试。

所有测试使用合成 fixture，不读取真实 PaSa 数据或人工标签。

## 11. 文件范围

预计修改：

```text
src/paper_search/evaluation/annotation.py
src/paper_search/evaluation/freeze.py
src/paper_search/evaluation/__init__.py
tests/evaluation/test_annotation.py
tests/evaluation/test_freeze.py
tests/integration/test_week1_pipeline.py
data/README.md
docs/TEAMMATE_ONBOARDING.md
```

若实现无需修改集成测试或导出文件，则保持不动。不得修改受保护的 Task 2 设计文档，也不得提交 `data/manifest.json`、私密标签或真实 gold。

## 12. 验收标准

- audit-only 能验证完整合成 90/40/20 工作包且不写文件；
- 每个分区必须显式指定 `reject` 或 `allow`；
- 任一 Schema、ID、哈希、顺序、kappa 或策略问题都会阻止冻结；
- `--approve` 只允许从精确匹配的 waiting manifest 一次性转为 frozen；
- frozen manifest 可被现有 Week 1 runner 接受；
- 安全报告不含受限正文、paper ID、标签答案、凭据或绝对路径；
- 不同内容不能覆盖 frozen manifest 或正式报告；
- 聚焦测试、全量 pytest、Ruff、mypy、`git diff --check`、范围检查、秘密扫描和独立审查全部通过。
