# Task 4 去重、硬过滤、词法排序与评测 Runner 设计

**日期：** 2026-07-16
**分支：** `codex/task4-week1-retrieval`
**状态：** 对话设计已批准，待书面设计复核
**范围：** 最新 `PRD.md` 第一周 Task 4，以及协作者最简列表页

## 1. 背景与目标

Task 3 已交付 OpenAlex 检索、统一 `Paper`、SQLite 缓存和不可变响应快照。Task 2 已交付逐查询与聚合指标，但真实 dev 标注尚未冻结。

Task 4 在这些契约之上建立可解释、可重复的第一周候选处理链路：

```text
OpenAlex Paper
  -> 确定性去重
  -> 硬约束过滤与不确定性标记
  -> 关键词覆盖和 BM25 排序
  -> 预测记录与逐查询/聚合指标
  -> 可复现运行产物与最简列表页
```

本轮工程验收使用固定测试夹具和注入的假 Provider。真实 dev baseline 与第一周阶段闸门必须等待协作者冻结标注后再运行，不能用合成夹具替代或提前宣称通过。

## 2. 范围

### 2.1 本轮实现

- DOI、跨源 ID、规范化标题和保守模糊标题去重；
- 每次合并的匹配层级、代表记录和成员记录审计信息；
- 年份、venue、排除条件、撤稿和稳定 ID 硬过滤；
- 缺失或含糊字段的不确定性原因和确定性降权；
- 关键词覆盖率、BM25 和组合词法分数；
- 从查询到预测、指标、用量、延迟和快照引用的评测 runner；
- `--config`、`--split`、`--output` CLI；
- 查询框和论文列表的最简只读 UI；
- 固定夹具、单元测试、集成测试和安全测试。

### 2.2 明确不实现

- Embedding、LLM 精排和细粒度相关性判断；
- 引文网络扩展；
- Semantic Scholar 在线 Provider；
- 人工标注工作流或对未冻结数据的修改；
- 关系图和完整产品 UI；
- 把真实查询、受限原始数据、凭据或 `.env` 写入仓库；
- 在 Task 4 分支内合并其他分支。

## 3. 总体设计选择

采用“分层纯函数 + 薄 runner”。去重、过滤和词法排序不访问网络、不读文件、不读取环境变量；它们只接收显式输入并返回冻结的领域结果。runner 负责 I/O、Provider 调用、预算、计时、快照导出和指标编排。UI 只调用应用服务，不复制算法。

未采用单一有状态 Pipeline 类，因为它会把匹配规则、过滤审计、排序和外部 I/O 耦合，降低 TDD 的故障定位能力。未采用配置驱动规则引擎，因为第一周规则已经明确，提前抽象会扩大交付面。

## 4. 去重

### 4.1 文件与接口

`src/paper_search/processing/deduplicate.py` 提供冻结结果模型和纯函数：

```python
class MergeDecision(DomainModel):
    representative_id: NonEmptyStr
    member_ids: list[NonEmptyStr]
    match_rule: Literal["doi", "external_id", "exact_title", "fuzzy_title"]
    match_value: NonEmptyStr


class DeduplicationResult(DomainModel):
    papers: list[Paper]
    decisions: list[MergeDecision]


def deduplicate_papers(
    papers: Sequence[Paper],
    *,
    id_map: IdentifierMap | None = None,
    fuzzy_title_threshold: float = 0.98,
) -> DeduplicationResult: ...
```

`papers` 保持首次候选顺序作为稳定的最终并列依据。`id_map` 复用 Task 2 的 canonical identifier 解析契约；未提供时只比较记录自身携带的规范化标识符。

### 4.2 匹配优先级

按以下顺序建立合并关系，命中高优先级后不降级改写其原因：

1. 合法规范化 DOI 完全相同；
2. DOI、OpenAlex ID、Semantic Scholar ID 经 `IdentifierMap` 解析后存在共同 terminal ID；
3. `normalize_title()` 后标题完全相同；
4. 规范化标题相似度不低于 `0.98`，且出版年份相同、至少一个规范化作者姓氏相同。

模糊规则中，年份或作者任一侧缺失时不合并。阈值是实现默认值和运行产物的一部分，后续只能版本化修改。

### 4.3 聚类和字段合并

匹配关系使用确定性并查集形成传递闭包。代表记录按以下质量元组从高到低选择，再以首次输入位置决胜：

```text
有 DOI、稳定外部 ID 数量、有摘要、作者数量、有年份、有 venue、引用数存在
```

标量字段取代表记录值；代表缺失时按成员输入顺序补第一个非空值。`sources` 和作者列表去重并保留首次出现顺序。标识符发生冲突时保留代表值，并在决定记录中保留全部成员 ID；不静默生成新的不受支持标识符。

输出论文顺序按各簇首次输入位置排列。单成员簇出现在 `papers`，但不生成 `MergeDecision`。

## 5. 硬过滤与不确定性降权

### 5.1 文件与接口

`src/paper_search/processing/filter.py` 提供：

```python
class AcceptedPaper(DomainModel):
    paper: Paper
    uncertainty_reasons: list[NonEmptyStr]
    score_multiplier: float


class RejectedPaper(DomainModel):
    paper: Paper
    reason_code: NonEmptyStr
    reason: NonEmptyStr


class FilterResult(DomainModel):
    accepted: list[AcceptedPaper]
    rejected: list[RejectedPaper]


def apply_hard_filters(
    papers: Sequence[Paper],
    query: QuerySpec,
) -> FilterResult: ...
```

`score_multiplier` 范围为 `[0, 1]`。每个不确定性原因乘以固定 `0.9`，最低截断为 `0.7`。原因去重且按规则定义顺序返回。

### 5.2 明确拒绝规则

规则按下列顺序执行，第一条命中原因作为审计原因：

1. `is_retracted is True` -> `retracted`；
2. 不存在 DOI、OpenAlex、Semantic Scholar 或 arXiv 稳定 ID -> `missing_stable_id`；
3. 已知年份早于 `year_from` 或晚于 `year_to` -> `year_out_of_range`；
4. 已知 venue 与显式 `query.venues` 全部不匹配 -> `venue_mismatch`；
5. 标题或摘要命中显式 `query.exclusions` -> `excluded_term`。

领域 `Paper` 已要求非空标题；若绕过模型进入非法标题，应在模型边界失败，不在过滤器内静默修复。

### 5.3 不确定性规则

- 查询有年份约束而论文年份缺失 -> `missing_year`；
- 查询有 venue 约束而论文 venue 缺失 -> `missing_venue`；
- `is_retracted is None` -> `unknown_retraction_status`；
- 查询有排除条件而论文摘要缺失 -> `missing_abstract_for_exclusion`。

缺失字段不触发硬删除。每个接受或拒绝结果都携带可测试、可序列化的原因。

## 6. 关键词覆盖与 BM25

### 6.1 文件与接口

`src/paper_search/ranking/lexical.py` 提供：

```python
class LexicalScore(DomainModel):
    paper: Paper
    bm25_score: float
    normalized_bm25: float
    keyword_coverage: float
    uncertainty_multiplier: float
    final_score: float


def rank_lexically(
    query: QuerySpec,
    candidates: Sequence[AcceptedPaper],
) -> list[LexicalScore]: ...
```

### 6.2 文本与分词

- 查询文本由 `original_query`、`topics`、`methods`、`tasks`、`datasets`、`domains`、`must_have` 和 `should_have` 依次连接；
- 文档文本为标题加摘要；
- Unicode NFKC、casefold 后提取 Unicode 字母和数字 token；
- 空摘要合法，空候选返回空列表；
- BM25 使用已锁定依赖 `rank-bm25`，不下载模型、不访问网络。

### 6.3 分数组合

- `keyword_coverage`：去重查询 token 中出现在文档 token 集合的比例；
- `bm25_score`：保留原始 BM25 分数用于审计；
- `normalized_bm25`：在本次候选集合内做稳定 min-max 归一化；全部相等时为 `0`；
- `final_score = (0.7 * normalized_bm25 + 0.3 * keyword_coverage) * uncertainty_multiplier`；
- 按 `final_score`、`keyword_coverage`、`bm25_score` 降序，再按原候选顺序和 canonical ID 升序稳定决胜。

权重和分词版本写入 runner 的运行产物。第一周真实 baseline 后如需调参，必须产生新 scoring version，不能改写旧结果。

## 7. 评测 Runner

### 7.1 文件与边界

`src/paper_search/evaluation/runner.py` 包含可注入应用服务与 CLI。核心函数接收显式 Provider/search callable、时钟和输出目录，使集成测试不访问网络。

CLI 契约：

```powershell
uv run python -m paper_search.evaluation.runner `
  --config configs/base.yaml `
  --split dev `
  --output experiments/baseline-week1
```

默认读取 `data/<split>/gold.jsonl`。不存在、manifest 状态不是已冻结，或缺少真实标签时立即失败并给出安全错误；不得退回合成数据后仍标记为 dev baseline。

runner 通过 `load_runtime_config(..., env_file=None)` 只接收进程环境中的凭据，不主动打开 `.env`。调用方如需 dotenv，应由 `uv --env-file` 或 PowerShell 预先注入。任何运行产物、异常和日志都不得包含凭据。

### 7.2 单查询流程

1. 从 `EvaluationQuery` 构造最小 `QuerySpec`；后续 Task 5 可替换为正式解析器；
2. 为本查询创建预算 reservation；
3. 调用 OpenAlex Provider，单查询失败被记录但不终止整个 split；
4. 对候选执行去重、硬过滤和词法排序；
5. 截断到 `budget.max_output_papers`；
6. 生成 `PredictionRecord`；
7. 累加实际 Provider 调用数和端到端延迟；
8. 收集本次实际引用的 cache keys。

全部查询完成后调用 Task 2 `evaluate()` 生成逐查询、宏平均和微平均指标。runner 不复制指标公式。

### 7.3 输出产物

输出目录使用 UTF-8、稳定排序、原子写入，并拒绝覆盖内容不同的既有正式产物：

```text
experiments/baseline-week1/
├── predictions.jsonl
├── metrics.json
├── usage.json
├── run.json
├── deduplication.jsonl
├── filtering.jsonl
├── snapshot_manifest.json
└── snapshots/
```

- `predictions.jsonl`：按 gold 查询顺序的排名 ID；
- `metrics.json`：现有 `EvaluationResult`、输入哈希和 metrics contract version；
- `usage.json`：逐查询及总调用数、延迟和错误分类；
- `run.json`：git SHA、公开 config hash、split、scoring version、规则参数、输入文件哈希和 snapshot manifest 相对路径；
- `deduplication.jsonl`、`filtering.jsonl`：不含原始秘密的逐查询审计结果；
- `snapshot_manifest.json` 和 `snapshots/`：复用 Task 3 不可变快照契约，并在写指标前调用 `validate_snapshot_manifest()`。

正式指标必须引用同一输出目录内已验证的 manifest。没有成功响应时仍输出合法的空 manifest 或明确失败，不伪造快照引用。

## 8. 最简 UI

`src/paper_search/ui/app.py` 使用现有 FastAPI 依赖提供最小应用：

- `GET /` 返回包含查询框的 HTML；
- `POST /search` 调用注入的搜索应用服务；
- 列表显示标题、作者、年份、venue、来源和词法分数；
- 可展开显示去重规则、过滤/不确定性原因和分数组成；
- HTML 转义所有外部文本；
- 不在 UI 模块实现去重、过滤或排序算法；
- Provider 缺失或失败时显示安全错误，不显示 URL 凭据、header 或原始异常。

UI 测试注入假服务，不调用在线 Provider。UI 不承担真实 dev baseline 的阶段闸门。

## 9. 错误处理与安全

- 用户输入错误使用可定位的 `ValueError` 或 CLI 退出码 `2`；
- 单查询 Provider 错误进入 usage/error 记录并产生空预测，其余查询继续；
- 输出目录冲突、manifest 校验失败、输入哈希变化或冻结状态无效时整次正式运行失败；
- 不记录 API key、Authorization header、完整请求 URL、`.env` 内容、本机用户名或绝对工作路径；
- 测试使用 sentinel 凭据验证所有序列化产物和错误输出均不包含该值；
- 真实查询和受限标签遵循 Task 2 数据边界，不提交仓库。

## 10. TDD 与测试策略

实施按以下红—绿—重构批次推进，每个批次先提交失败测试证据，再写最小实现：

1. DOI、跨源 ID、精确标题、模糊标题、非合并边界、传递聚类和稳定字段合并；
2. 每条硬过滤原因、不确定字段、乘数截断和稳定顺序；
3. tokenizer、关键词覆盖、BM25、相同分数、空候选和不确定性降权；
4. 假 Provider 的多查询 runner、单查询失败隔离、指标、用量、快照绑定、原子写入和覆盖保护；
5. UI 查询框、结果列表、HTML 转义、算法边界和安全错误；
6. CLI 固定夹具集成测试与真实 dev 未冻结时的明确失败测试。

计划新增：

```text
tests/unit/test_deduplicate.py
tests/unit/test_filter.py
tests/unit/test_lexical.py
tests/evaluation/test_runner.py
tests/integration/test_week1_pipeline.py
tests/ui/test_app.py
tests/fixtures/week1/
```

验收命令统一使用 `uv run --no-sync --no-env-file`，除非显式运行真实在线 baseline。最终至少运行聚焦测试、全量 pytest、`ruff check .`、`mypy src`、`git diff --check` 和独立代码审查。

## 11. 验收边界

### 11.1 本轮可以判定完成

- 四级去重和全部过滤原因有固定测试；
- 词法排序确定、可解释且可复现；
- 假 Provider 端到端可从查询产生去重列表并计算 F1；
- CLI 契约、产物契约和快照引用契约均有集成测试；
- 最简 UI 不复制算法且通过安全测试；
- 聚焦测试、全量 pytest、Ruff、mypy 和独立审查通过；
- 仅 Task 4 文件提交并推送独立分支。

### 11.2 本轮不能判定完成

以下事项保持未完成，直到协作者冻结 Task 2 真实数据：

- 对真实 dev split 执行第一次 baseline；
- 证明过滤后 Recall 绝对下降不超过 `0.02`；
- 对模糊标题去重进行人工抽查并证明准确率不低于 `98%`；
- 冻结 dev、validation 和抽样脚本；
- 宣布第一周阶段闸门通过。

数据冻结后只补跑正式命令、验证不可变产物并记录证据；若指标不达标，在 Task 4 分支继续修复，不启动关系图。

## 12. Git 与交付

- 基线为 Task 3 已验收提交 `6e9fad4`；
- 工作分支为 `codex/task4-week1-retrieval`；
- 保留并忽略用户已有的 `docs/superpowers/specs/2026-07-15-task2-evaluation-design.md` 换行元数据修改；
- 不读取或提交 `.env`；
- 设计、计划、测试、实现和安全夹具按 TDD 批次提交；
- 完成后只推送 Task 4 分支，不创建合并提交，不在用户授权前合并。
