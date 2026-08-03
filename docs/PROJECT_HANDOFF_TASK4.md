# 赛题三项目上下文交接：从 Task 4 继续

> 用途：在新的 Codex 工作区或对话中恢复项目上下文。本文档不包含任何 API Key、Token 或受限数据。
>
> 更新日期：2026-07-16

## 1. 项目目标与人员背景

- 项目用于参加研究生人工智能竞赛的赛题三，核心方向是资源受限的学术论文检索、排序、证据组织和可复现评测。
- 参赛人数为 2 人，周期约 1 个月。
- 主负责人负责架构、后端、评测、检索和模型主线；另一位队员仍在学习阶段，优先承担数据辅助、文档和后续前端展示。
- 主负责人已学习 Python 深度学习基础、李沐 Transformer、Hugging Face 基础、RAG 原论文和《Hands-On Large Language Models》。
- 当前电脑 GPU 为 NVIDIA GeForce RTX 3050 Ti Laptop GPU，显存 4 GB；必要时可以借用其他 GPU。
- API 预算有限，因此设计强调缓存、预算控制、可复现实验和低成本基线。

## 2. 正式目录与工作区

### 当前应打开的 Codex 工作区

```text
<task2-worktree>
```

这是链接 Git worktree，当前分支为 `codex/task2-evaluation`。

不要从 `<scratch-root>` 继续开发。该目录只曾用于 Codex 跨盘写入的临时 staging，目前没有项目文件。

### 其他目录

- `<projects-root>`：Git 主检出目录，分支为 `main`；共用虚拟环境和 `.env` 位于这里。
- `<competition-guides-root>`：保存赛题说明等原始资料。
- `<uv-executable>`：项目使用的 uv 可执行文件。

## 3. Git 与远端

- GitHub：`Tianyuai/AI-Projects`
- Remote：`ssh://git@ssh.github.com:443/Tianyuai/AI-Projects.git`
- 当前功能分支：`codex/task2-evaluation`
- 打开新工作区后不要重新创建仓库或 worktree，也不要在 `main` 上直接实现 Task 4。

最近关键提交：

```text
5d92f73 docs: mark task 3 adapter work complete
3ff4247 feat: adapt PaSa and ranked prediction records
63e819e feat: add strict evaluation file contracts
30a3e31 docs: add teammate task 2 onboarding guide
f701565 feat: add evaluation models and identifier normalization
```

## 4. 环境与密钥状态

- Python 3.11、uv、项目虚拟环境和依赖已经配置。
- PyTorch 位于 `<projects-root>\.venv`；依赖由 `pyproject.toml` 和 `uv.lock` 锁定。
- 已配置或申请 DashScope、OpenAlex、Semantic Scholar 和 Hugging Face。
- `.env` 位于 `<projects-root>`，不得读取、输出或提交其中的密钥值。
- 安全提醒：全局 Codex 配置中曾有一个 `ANTHROPIC_AUTH_TOKEN` 在诊断输出中暴露。必须在服务商侧撤销并重新生成；本文档不记录其值。

## 5. 已完成工作

### 主要文档

- 总计划：`PRD.md`
- Task 2 设计：`docs/superpowers/specs/2026-07-15-task2-evaluation-design.md`
- Task 2 实施计划：`docs/superpowers/plans/2026-07-16-task2-evaluation-implementation.md`
- 队友接入说明：`docs/TEAMMATE_ONBOARDING.md`

### 项目骨架

- 项目配置、领域模型和预算控制已经完成并验证。

### 评测实施计划 Task 1

- 已实现 `EvaluationQuery`、`PredictionRecord`。
- 已实现 DOI、arXiv、OpenAlex、Semantic Scholar 和标题标识规范化。
- 模型严格、冻结，并拒绝未知字段。

### 评测实施计划 Task 2

- 严格 UTF-8 JSONL 逐行读取、行号错误和重复 `query_id` 检查。
- 错误消息脱敏、确定性原子 JSONL 写入和 `sha256:<digest>` 文件哈希。
- Identifier Map 的规范化冲突、原始重复键、映射链和循环检查。

### 评测实施计划 Task 3

- 已实现严格 `PaSaRecord`。
- `qid` 映射到 `query_id`，`question` 映射到 `query`。
- 正式 arXiv ID 优先，只有对应位置缺失 ID 时才使用标题后备。
- 保存 dataset revision、source、split 和 `source_meta`。
- 已实现固定预测格式 `InternalPredictionRecord` 和预测转换。
- 预测重复项保留到后续评分阶段。

## 6. 当前验证基线

最近一次已提交快照验证：

```text
Task 3 focused tests: 7 passed
Full pytest: 97 passed
Ruff: passed
mypy: passed (10 source files)
```

新工作区启动后重新运行：

```powershell
git branch --show-current
git status -sb
$env:UV_PROJECT_ENVIRONMENT='<projects-root>\.venv'
& '<uv-executable>' run --no-sync pytest -q
& '<uv-executable>' run --no-sync ruff check .
& '<uv-executable>' run --no-sync mypy src
```

预期分支为 `codex/task2-evaluation`，pytest 为 97 项通过，Ruff 和 mypy 无错误。

## 7. 已知但不要误处理的状态

`docs/superpowers/specs/2026-07-15-task2-evaluation-design.md` 可能显示为 modified，但内容 diff 为空，原因是 Windows LF/CRLF 元数据差异。

- 不要暂存、恢复或修改该文件；
- 不要把它混入 Task 4 提交；
- 提交时显式列出 Task 4 文件。

## 8. 下一步：实施计划 Task 4

Task 4 名称：`Per-query metrics and aggregate evaluator`。

计划文件：`docs/superpowers/plans/2026-07-16-task2-evaluation-implementation.md`。

目标文件：

```text
src/paper_search/evaluation/metrics.py
tests/evaluation/test_metrics.py
```

主要接口：

```python
deduplicate_ranked(values: Sequence[str]) -> list[str]
score_query(gold: Sequence[str], predicted: Sequence[str]) -> QueryMetrics
evaluate(
    gold: Sequence[EvaluationQuery],
    predictions: Sequence[PredictionRecord],
    *,
    id_map: IdentifierMap | None = None,
) -> EvaluationResult
```

必须实现：

1. Precision、Recall、F1、Recall@5/10/20；
2. gold 与 prediction 都为空时所有指标为 1；只有一方为空时为 0；
3. 预测按首次出现去重并保持排名；
4. 命中 ID 按预测顺序返回；
5. 宏平均是逐查询指标的算术平均；
6. 微平均通过汇总 TP、FP、FN 后计算；
7. gold 缺失 prediction 时按空预测处理；
8. prediction 出现未知 `query_id` 时立即报错；
9. 可选 `IdentifierMap` 在集合比较前应用于所有 gold 和 prediction ID；
10. 定义冻结的 `QueryMetrics`、`MetricSummary` 和 `EvaluationResult`。

输入与错误边界：

- `gold` 是 `EvaluationQuery` 序列，`predictions` 是 `PredictionRecord` 序列；
- gold 或 prediction 内部出现重复 `query_id` 时立即报错；
- prediction 中出现 gold 不认识的 `query_id` 时抛出可定位的 `ValueError`；
- gold 查询没有 prediction 时不报错，按空预测参与宏/微平均；
- 空的 gold 集合必须有明确行为，不得产生除零错误或 NaN；具体返回契约在写第一轮测试时固定。

注意：实施计划中的示例 `doi:10.1/a` 不是合法 DOI。编写测试时使用例如 `doi:10.1000/a` 的合法形式，不要放宽 DOI 规范化规则。

建议严格按 TDD 分两轮：

1. 空集合、排名去重、Recall@K：RED → GREEN；
2. 宏/微平均、缺失预测、未知查询、ID Map：RED → GREEN。

计划提交信息：`feat: add reproducible paper retrieval metrics`。

## 9. 新 Codex 任务的首条提示词

在打开 `<task2-worktree>` 后，可直接发送：

```text
请先阅读 docs/PROJECT_HANDOFF_TASK4.md、PRD.md、
docs/superpowers/specs/2026-07-15-task2-evaluation-design.md 和
docs/superpowers/plans/2026-07-16-task2-evaluation-implementation.md。
核对当前分支为 codex/task2-evaluation，并重新运行交接文档中的基线验证。
不要处理设计文档的换行元数据修改，不要读取或输出任何 .env 密钥。
验证通过后，严格按照实施计划和 TDD 继续 Task 4：
Per-query metrics and aggregate evaluator。
完成后运行聚焦测试、全量 pytest、Ruff 和 mypy，进行独立代码审查，
只提交 Task 4 相关文件并推送到现有功能分支。
```

## 10. 上文缺失是否影响后续

不会造成实质阻塞。后续开发的可靠信息源优先级为：

1. 当前 Git 提交和测试；
2. PRD、设计文档和实施计划；
3. 本交接文档；
4. 聊天历史。

聊天压缩可能省略中间交流，但不会删除已经提交的代码、测试、计划、Git 历史和本地配置。新任务只要读取上述文件并验证基线，就可以安全继续。
