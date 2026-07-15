# 科研场景复杂学术查询智能论文搜索与推荐系统 PRD

> 文档状态：已确认，可进入实施  
> 制定日期：2026-07-14  
> 实施周期：4 周  
> 团队规模：2 人  
> 核心策略：API 驱动、分层检索、先完成可评测 baseline，再实现预算感知的查询演化

## 1. 文档目的

本文档定义赛题三“科研场景下复杂学术查询的智能论文搜索与推荐”的产品目标、系统边界、功能需求、技术架构、实施步骤、验收标准、模型落地门槛和后续优化路径。

本文档同时是四周研发总计划。实施过程中，任何新增模型、Agent 或检索策略都必须通过固定开发集上的对照实验进入主配置，不能仅凭主观效果或单个案例判断。

## 2. 项目背景与约束

### 2.1 赛题要求

系统需要针对自然语言描述的复杂学术查询，自动完成：

1. 查询理解、约束识别、查询分解与改写；
2. 基于大模型的自主搜索及迭代策略调整；
3. 候选论文过滤和细粒度综合排序；
4. 搜索结果归纳、列表展示和论文关系展示。

官方评分重点为：

| 指标 | 权重 | 对应系统能力 |
|---|---:|---|
| F1 Score | 70% | 查询理解、召回、过滤、引文扩展、排序 |
| 运行效率 | 20% | API 调用、Token、延迟、缓存、停止条件 |
| 结构化回复 | 10% | 论文分组、匹配依据、链接、关系图 |

最终成绩中，自动评分占 60%，专家评分占 40%；专家评分关注创新性、落地可行性和算法泛化性。

### 2.2 团队与资源约束

- 团队共 2 人；主负责人承担算法、后端、评测和技术决策，协作者承担数据质检、测试、实验整理、前端和展示。
- 文档中的“主负责人”指成员 A（你），“协作者”指成员 B（队友）；开始实施时在项目看板中用真实姓名替换角色显示名，不影响代码和接口。
- 开发周期为 1 个月，最后 2–3 天必须冻结功能，仅用于复现、修复和材料整理。
- 当前设备为 NVIDIA GeForce RTX 3050 Ti Laptop GPU，显存 4GB。
- 可以临时借用更强 GPU，但借用 GPU 不能成为系统在线运行的必要条件。
- LLM API 预算有限；所有实验必须同时记录质量和成本。
- 首版只使用论文标题、摘要和元数据，不把全文下载与解析放入关键路径。

## 3. 赛题任务的本质

本项目不是普通关键词搜索、普通 RAG 问答或以聊天为中心的 Agent。其本质是：

> 在有限调用成本和时间预算内，将包含多维约束的自然语言学术问题转化为可执行的检索计划，通过多路召回、候选判断和迭代搜索，返回尽可能完整且准确的论文集合。

系统的主循环为：

```text
理解查询
  → 生成检索计划
  → 获取候选论文
  → 判断约束覆盖与新增收益
  → 扩展检索或停止
  → 精排、分组和结构化输出
```

核心研究问题依次是：

1. 如何识别主题、方法、任务、数据集、领域、年份、venue 和排除条件；
2. 如何拆分复杂查询，同时不丢失原始强约束；
3. 如何融合关键词搜索、语义相似和引文网络；
4. 如何在高召回与低噪声之间平衡；
5. 如何判断继续检索的边际收益是否值得新增调用；
6. 如何保证每篇论文、每条链接和每条关系均可追溯、不可虚构。

## 4. 产品目标与非目标

### 4.1 产品目标

- 构建可独立批量评测的端到端论文搜索系统。
- 在固定开发集上显著优于“原始查询直接调用单一搜索 API”的基础基线。
- 支持复杂查询的结构化解析、子查询生成、多源召回、引文扩展和细粒度排序。
- 对每个查询记录搜索轨迹、候选变化、调用次数、Token 和延迟。
- 输出高度相关、部分相关论文列表及真实的论文关系。
- 提供一个可通过消融实验验证的创新点：预算感知的自适应查询演化。

### 4.2 非目标

- 不自建全量学术搜索引擎。
- 不在四周主线中从零训练大语言模型或大型 Embedding 模型。
- 不进行无限深度的引文图遍历。
- 不默认下载和解析所有论文全文。
- 不把多 Agent 数量当作创新性指标。
- 不让前端状态参与算法评测或改变最终论文集合。
- 不承诺生成文献综述正文；赛题核心交付是论文集合、匹配依据和结构化关系。

## 5. 用户与核心场景

### 5.1 目标用户

- 需要针对细粒度问题寻找论文的研究生和科研人员；
- 需要建立系统性检索集合的算法研究者；
- 需要评估论文搜索系统效果的竞赛评委。

### 5.2 代表性查询类型

1. 主题型：寻找研究某一现象或问题的论文；
2. 方法型：寻找采用特定模型、训练方法或推理策略的论文；
3. 数据集型：寻找在指定数据集上评估特定任务的论文；
4. 时间/venue 型：限定年份、会议或期刊；
5. 组合约束型：同时包含主题、方法、数据和时间等条件；
6. 关系型：寻找某项工作的前置研究、后续工作或同路线研究；
7. 排除型：明确排除某种方法、领域或论文类型。

## 6. 总体架构

### 6.1 架构原则

- 单体模块化：四周内使用一个 Python 项目，不引入微服务部署负担。
- 接口隔离：LLM、Embedding、Reranker 和搜索源均通过统一接口接入。
- 评测优先：任何功能先定义数据记录和评测方式，再实现业务逻辑。
- 分层降本：规则和轻量模型先处理，昂贵 LLM 只处理少量关键候选。
- 可回放：缓存原始响应和中间结果，使实验可复现。
- 可降级：某个 API 或模型不可用时，系统仍能返回基础结果。

### 6.2 数据流

```text
NaturalLanguageQuery
  ↓ QueryParser
QuerySpec
  ↓ QueryPlanner
SearchPlan / SubQueries
  ↘ BudgetController（所有外部调用前预留、调用后结算）
  ↓ OpenAlexProvider + SemanticScholarProvider
RawPapers
  ↓ Normalizer + Deduplicator + HardFilter
CandidatePool
  ↓ LexicalRanker + EmbeddingRanker
SeedPapers
  ↓ CitationExpander
ExpandedCandidatePool
  ↓ ConstraintReranker
RankedPapers
  ↓ CoverageAnalyzer
Stop or Next SearchPlan
  ↓ ResultAssembler
StructuredSearchResponse
```

### 6.3 核心组件

| 组件 | 单一职责 | 失败时降级 |
|---|---|---|
| QueryParser | 从原始查询抽取结构化约束 | 规则抽取 + 原始查询 |
| QueryPlanner | 生成 3–6 个有目标的子查询 | 使用原始查询和关键词组合 |
| SearchProvider | 从学术 API 获取论文 | 切换备用源或读取缓存 |
| Normalizer | 统一字段和 ID | 保留原始字段并标记缺失 |
| Deduplicator | DOI、ID 和标题去重 | 只做 DOI/精确标题去重 |
| HardFilter | 执行年份、venue、排除条件 | 对不确定条件改为降权 |
| InitialRanker | 关键词和向量低成本初排 | 退化为 API relevance 顺序 |
| CitationExpander | 对高相关种子执行一跳扩展 | 跳过扩展 |
| ConstraintReranker | 对少量候选逐约束判断 | 使用本地分数融合 |
| BudgetController | 控制调用、Token、迭代和时间 | 达到硬上限立即停止 |
| ResultAssembler | 生成分组列表、解释和关系图 | 返回最小论文列表 |
| EvaluationRunner | 批量评测、消融和实验记录 | 单样本运行并保存错误 |

## 7. 核心数据模型与接口

`src/paper_search/domain/models.py` 第一行使用 `from __future__ import annotations`，因此下列类型可以按模块职责组织并使用前向引用；最终实现仍需通过 Pydantic 模型重建测试。

### 7.1 数据对象

`QuerySpec` 至少包含：

```python
class QuerySpec(BaseModel):
    original_query: str
    research_goal: str
    topics: list[str]
    methods: list[str]
    tasks: list[str]
    datasets: list[str]
    domains: list[str]
    year_from: int | None
    year_to: int | None
    venues: list[str]
    must_have: list[str]
    should_have: list[str]
    exclusions: list[str]
    ambiguities: list[str]
```

检索计划固定为：

```python
class SubQuery(BaseModel):
    query_id: str
    text: str
    query_type: Literal["exact", "expanded", "decomposed"]
    target_constraints: list[str]
    priority: int
    provider_hint: Literal["openalex", "semantic_scholar", "either"]

class SearchPlan(BaseModel):
    subqueries: list[SubQuery]
    inherited_hard_filters: dict[str, object]
    rationale: str
```

`Paper` 至少包含：

```python
class Paper(BaseModel):
    canonical_id: str
    title: str
    abstract: str | None
    authors: list[str]
    publication_year: int | None
    venue: str | None
    doi: str | None
    openalex_id: str | None
    semantic_scholar_id: str | None
    url: str | None
    citation_count: int | None
    reference_ids: list[ProviderPaperId]
    cited_by_ids: list[ProviderPaperId]
    is_retracted: bool | None
    sources: list[str]
```

引文相关 ID 和关系边必须保留来源：

```python
class ProviderPaperId(BaseModel):
    provider: Literal["openalex", "semantic_scholar"]
    value: str

class CitationEdge(BaseModel):
    provider: Literal["openalex", "semantic_scholar"]
    citing_provider_id: ProviderPaperId
    cited_provider_id: ProviderPaperId
    citing_canonical_id: str | None = None
    cited_canonical_id: str | None = None

class ResolvedCitationEdge(BaseModel):
    provider: Literal["openalex", "semantic_scholar"]
    citing_canonical_id: str
    cited_canonical_id: str
    source_edge_hash: str
```

`CitationEdge` 仅用于保存 Provider 原始边；`ResolvedCitationEdge` 才能进入最终响应。去重后必须将原始 Provider ID 映射为 `canonical_id`，验证顶层 provider 与两个端点来源一致，并保存原始边哈希。只有两端均成功映射到最终论文节点的边才允许转换；其余边只保留在实验日志中。

`CandidateEvidence` 保存论文被找到、过滤和评分的全过程：

```python
class CandidateEvidence(BaseModel):
    paper_id: str
    matched_subqueries: list[str]
    matched_constraints: list[str]
    unmatched_constraints: list[str]
    filter_reasons: list[str]
    lexical_score: float
    embedding_score: float
    rerank_score: float | None
    constraint_coverage: float
    source_agreement: float
    authority_score: float
    recency_score: float
    final_score: float
    scoring_version: str
    relevance_level: Literal["high", "partial", "irrelevant"]

class RankedPaper(BaseModel):
    paper: Paper
    evidence: CandidateEvidence
```

最终响应模型固定为：

```python
class StructuredSearchResponse(BaseModel):
    query_id: str
    query_analysis: QueryAnalysisResult
    selected_paper_ids: list[str]
    high_relevance: list[RankedPaper]
    partial_relevance: list[RankedPaper]
    citation_edges: list[ResolvedCitationEdge]
    search_trace: list[dict]
    usage: UsageActual
    stop_reason: str
    is_partial: bool
    warnings: list[str]
    config_hash: str
    git_sha: str
```

`selected_paper_ids` 是自动 F1 评分的唯一预测集合；高度相关和部分相关分组只用于解释和展示，不能隐式改变该集合。

`SearchBudget` 的默认值：

```python
class SearchBudget(BaseModel):
    max_search_api_calls: int = 12
    target_search_api_calls: int = 8
    max_llm_calls: int = 5
    target_llm_calls: int = 3
    max_iterations: int = 2
    max_subqueries: int = 6
    max_rerank_candidates: int = 30
    max_output_papers: int = 50
    max_citation_seeds: int = 2
    target_citation_seeds: int = 1
    max_elapsed_seconds: int = 90
    soft_deadline_seconds: int = 80
    max_total_tokens: int
    max_cost_cny: float
```

`budget_low.yaml` 固定 `max_total_tokens=10000`、`max_cost_cny=0.10`、最多精排 12 篇；`budget_balanced.yaml` 固定 `max_total_tokens=24000`、`max_cost_cny=0.30`、最多精排 30 篇。项目四周 API 总预算硬上限为 200 元，达到 160 元时停止大规模开发集重复实验，只保留验证和最终复现额度。程序启动时若 Token 或费用上限缺失则拒绝进入使用 LLM 的主配置；离线测试配置使用模拟 LLM，不消耗真实 Token。

默认搜索调用分配为：OpenAlex 主召回 3–6 次、Semantic Scholar 补充召回 1–2 次、引文扩展 2–4 次。系统不对所有子查询无差别调用两个来源；第二来源仅处理高优先级子查询或用于弥补当前未覆盖约束。平衡配置默认扩展 1 篇种子，硬上限 2 篇。一次实际 HTTP 请求计一次调用；分页、重试、引用和被引请求均分别计次，缓存命中计零次外部调用。

默认 LLM 调用分配为：第 1 次合并完成查询解析和检索计划，平衡配置第 2–3 次分别精排不超过 15 篇，低预算配置第 2 次精排不超过 12 篇。JSON 修复、自适应下一轮查询或精排重试只能使用剩余应急额度。达到硬上限后必须使用已有分数完成结果，不允许继续调用。

查询分析输入最多 1000 tokens、结构化输出最多 1800 tokens。精排时每篇“标题 + 摘要 + 元数据”最多 400 tokens，低预算单批最多 12 篇且输出最多 2000 tokens，平衡配置每批最多 15 篇且输出最多 2500 tokens。发起调用前根据实际 tokenizer 估算总量；若剩余预算不足，则依次缩减候选数、缩短摘要到 250 tokens、跳过精排，不能超额调用。超出部分按摘要末尾截断，并记录截断标记。

### 7.2 模块接口

```python
class LLMClient(Protocol):
    async def generate_json(
        self, *, prompt_name: str, payload: dict, reservation: BudgetReservation
    ) -> ProviderResult[dict]: ...

class SearchProvider(Protocol):
    async def search(
        self, query: str, filters: dict, limit: int, reservation: BudgetReservation
    ) -> ProviderResult[list[Paper]]: ...
    async def references(
        self, paper_id: ProviderPaperId, limit: int, reservation: BudgetReservation
    ) -> ProviderResult[CitationExpansion]: ...
    async def citations(
        self, paper_id: ProviderPaperId, limit: int, reservation: BudgetReservation
    ) -> ProviderResult[CitationExpansion]: ...

class Ranker(Protocol):
    async def rank(
        self, query: QuerySpec, papers: list[Paper]
    ) -> list[CandidateEvidence]: ...

class BudgetController(Protocol):
    def reserve(self, action: str, estimate: UsageEstimate) -> BudgetReservation: ...
    def settle(self, reservation: BudgetReservation, actual: UsageActual) -> None: ...
    def should_stop(self, coverage: dict, new_relevant_count: int) -> bool: ...
```

`ProviderResult[T]` 必须同时携带 `data`、`usage`、`provenance`、`cache_hit`、`latency_ms` 和 `errors`。`UsageEstimate/UsageActual` 分别记录 HTTP 调用数、LLM 调用数、输入/输出 Token、预计/实际费用和耗时。并发调用必须先原子预留预算，完成后结算；预留失败则不得发起调用。BudgetController 是 Orchestrator 的横切依赖，不是管线末端的普通步骤。

`UsageEstimate/UsageActual` 的字段统一为 `search_api_calls`、`llm_calls`、`input_tokens`、`output_tokens`、`cost_cny`、`elapsed_ms`；未知费用必须为 `None`，不能填 0。`ProviderResult.provenance` 至少记录 provider、endpoint、模型 ID、请求时间和响应哈希。

`CitationExpansion` 包含 `papers: list[Paper]` 和 `raw_edges: list[CitationEdge]`；`BudgetReservation` 包含唯一 ID、动作、预留的六维用量和过期时间；错误对象统一为 `ErrorDetail(code, message, retryable, provider, request_id)`。Provider 内部每次分页或重试前都必须从传入 reservation 切分一个子预留；余额不足时停止分页/重试并返回部分数据。

```python
class UsageEstimate(BaseModel):
    search_api_calls: int = 0
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_cny: float | None = None
    elapsed_ms: int = 0

class UsageActual(UsageEstimate):
    pass

class BudgetReservation(BaseModel):
    reservation_id: str
    action: str
    reserved: UsageEstimate
    expires_at: datetime

class ErrorDetail(BaseModel):
    code: str
    message: str
    retryable: bool
    provider: str
    request_id: str | None

class CitationExpansion(BaseModel):
    papers: list[Paper]
    raw_edges: list[CitationEdge]

T = TypeVar("T")

class ProviderResult(BaseModel, Generic[T]):
    data: T
    usage: UsageActual
    provenance: dict[str, str]
    cache_hit: bool
    latency_ms: int
    errors: list[ErrorDetail]
```

查询解析和规划在模块职责上分开，但默认通过同一个 `query_analyze` Prompt 和一次模型调用完成：

```python
class QueryAnalysisResult(BaseModel):
    query_spec: QuerySpec
    search_plan: SearchPlan
```

Parser 负责校验和降级，Planner 负责对子查询排序、裁剪和强约束继承检查。

## 8. 功能需求

### FR-01 查询理解与约束抽取

1. 输入非空自然语言查询；最大长度由配置控制。
2. 使用低温度 LLM 生成 `QuerySpec`。
3. 对 JSON Schema 进行严格校验。
4. 首次校验失败时执行一次 JSON 修复；再次失败后降级到规则解析。
5. 年份、数据集、venue 和显式排除条件视为强约束，不得在改写中丢失。
6. 对无法判断的表达写入 `ambiguities`，不得擅自补全为事实。
7. 保存原始模型输出、解析结果、模型版本、Prompt 版本和 Token。

验收：结构化成功率不低于 99%；人工标注集上的强约束抽取召回率不低于 90%。

### FR-02 查询分解、改写与扩展

1. 生成精确查询、扩展查询和拆分查询。
2. 每个子查询必须说明目标约束和来源。
3. 默认生成 3–6 个子查询，超出预算时按优先级截断。
4. 扩展可以使用缩写、全称、同义词和领域术语。
5. 子查询必须继承适用的年份、venue 和排除条件。
6. 对单一明确查询允许跳过分解。

验收：启用查询规划后，开发集 F1 相对原始查询基线有稳定提升，且调用增长在预算内。

本文档中的“稳定提升”统一定义为：同一配置独立运行 3 次后，宏平均 F1 增量的中位数至少为 `+0.01`，且按查询做 1000 次 bootstrap 得到的 95% 置信区间下界不低于 `-0.005`；验证集宏平均 F1 不得下降超过 `0.01`。未达到该门槛的模块完成消融即可验收，但不得进入主配置。

### FR-03 多源学术检索

1. OpenAlex 为主搜索源，Semantic Scholar 为补充搜索及引文源。
2. 使用 `httpx.AsyncClient`，设置连接和读取超时。
3. 对 429 和 5xx 使用带抖动的指数退避。
4. 所有响应写入 SQLite 缓存，缓存键包含 provider、endpoint、参数和版本。
5. 使用 `select/fields` 只获取需要字段，降低响应大小。
6. 每篇论文记录来源和命中的子查询。
7. API 密钥只从环境变量读取，不写入日志和仓库。

成功搜索响应缓存 7 天，论文详情和已确认引文关系缓存 30 天；超时和 5xx 不缓存，429 仅缓存退避截止时间 60 秒。每个子查询默认最多获取 50 条，原始候选总量硬上限 300，标准化去重后硬上限 200。

验收：正常网络下非空有效查询的召回响应率不低于 95%；单个请求失败不终止批量评测。

### FR-04 标准化、去重与硬过滤

去重优先级：DOI → 外部 ID 映射 → 规范化标题 → 标题高相似且作者、年份一致。

硬过滤包括：

- 明确年份范围；
- 明确 venue；
- 明确排除条件；
- 撤稿标记；
- 无标题或无法形成稳定 ID 的记录。

当字段缺失或约束存在歧义时，不直接删除论文，而是标记为“不确定”并在排序中降权。

验收：DOI 相同论文 100% 合并；标题模糊去重人工抽查准确率不低于 98%；每个删除动作都有原因。

### FR-05 低成本初筛

初筛采用三级漏斗：

1. 硬约束过滤；
2. BM25/关键词覆盖评分；
3. `BAAI/bge-small-en-v1.5` 对查询与“标题 + 摘要”计算向量相似度。

默认在 CPU 运行 Embedding，GPU 仅作为配置项。该模型为 33M 参数、384 维向量、最大 512 tokens，适合当前设备。若下载或运行失败，退化为关键词和 API relevance 排序。

验收：压缩到精排候选集合后，开发集 Recall 的绝对下降不超过 0.02；本地峰值显存不超过 3.2GB。

### FR-06 引文网络扩展

1. 从初排前列默认选择 1 篇、最多 2 篇种子论文。
2. 分别获取参考文献和被引论文，执行一跳扩展。
3. 每个扩展结果记录种子、方向和来源。
4. 新候选重新经过标准化、去重、硬过滤和初筛。
5. 对每个种子和全局候选数量设置上限。
6. 引文扩展不能绕过预算控制器。

验收：完成开启/关闭扩展的消融；只有满足“稳定提升”定义时才进入主配置，否则按查询类型关闭或作为负结果记录。

### FR-07 细粒度相关性判断

只对最多 30 篇候选调用 LLM 或本地 Reranker。判断维度：

- 主题匹配；
- 方法匹配；
- 任务、数据集和领域匹配；
- 年份和 venue 匹配；
- 对研究目标的直接贡献；
- 是核心论文还是仅作背景。

模型输出 0–4 分相关性、已满足约束、未满足约束、证据和简短理由。理由必须基于检索到的标题、摘要和元数据，不能补写全文事实。

验收：人工标注论文对上的相关/不相关判断准确率目标不低于 85%；相对不使用精排的配置有可复现 F1 提升。

### FR-08 融合排序

首版最终分数：

```text
final_score =
0.30 × constraint_coverage
+ 0.25 × rerank_relevance
+ 0.20 × embedding_similarity
+ 0.10 × lexical_score
+ 0.05 × source_agreement
+ 0.05 × authority_score
+ 0.05 × recency_score
```

若缺少 LLM 精排分数，则将其权重按比例分配给约束覆盖、向量和关键词分数。所有分数先归一化到 `[0, 1]`。权威性和时效性合计不得超过 10%，防止高引用但不相关的论文压过真正相关论文。

阈值由开发集确定，但主配置必须固定并版本化。列表至少分为高度相关、部分相关两组；不相关论文不进入最终结果，但保留在实验记录中。

### FR-09 预算控制与停止策略

停止条件满足任意一项即触发：

- 达到搜索 API、LLM、Token 或时间硬上限；
- 达到最大 2 轮迭代；
- 连续一轮没有新增高相关论文；
- 强约束覆盖已稳定且候选集合变化低于配置阈值；
- 预计新增调用超出剩余预算。

预算控制器必须记录每次动作的预计成本和实际成本。任何模块不得直接绕过控制器调用外部 API。

### FR-10 结构化结果

输出包含：

- 原始查询和结构化理解；
- 高度相关和部分相关论文；
- 标题、作者、年份、venue、DOI/真实链接；
- 匹配与未匹配约束；
- 入选理由；
- 引文关系边；
- 子查询和搜索过程摘要；
- API 调用数、LLM 调用数、Token、延迟和缓存命中。

关系图中的节点只允许使用真实候选论文，边只允许使用搜索 API 返回的引用关系。

HTTP 契约固定为：

```python
class SearchRequest(BaseModel):
    query_id: str
    query: str
    budget_profile: Literal["low", "balanced"] = "balanced"
    include_trace: bool = True

POST /v1/search -> StructuredSearchResponse
GET /health/live -> {"status": "ok"}
GET /health/ready -> {"status": "ready" | "degraded", "providers": {...}}
```

预测文件使用 UTF-8 JSONL，每行固定为：

```json
{"query_id":"q001","selected_paper_ids":["doi:10.x/y"],"config_hash":"sha256...","git_sha":"...","run_id":"..."}
```

不得将解释文本或未归一化标题写入 `selected_paper_ids`。

### FR-11 前端展示

Streamlit 前端包含：

1. 查询输入和预算模式选择；
2. 高度相关/部分相关论文列表；
3. 年份、venue、相关性筛选；
4. 论文详情和约束匹配；
5. 引文关系图（Should，只有核心 F1 baseline 和无缓存效率达标后实施）；
6. 搜索轨迹与成本面板；
7. API 降级、超时和部分结果提示。

前端只能调用后端公开接口，不复制检索和排序逻辑。

### FR-12 实验评测

1. 支持单条、数据集和批量运行；
2. 计算 Precision、Recall、F1 和 Recall@K；
3. 记录调用数、Token、端到端延迟和失败率；
4. 保存查询、标准答案、预测、阶段候选和配置快照；
5. 支持通过配置关闭查询分解、第二搜索源、引文扩展、Embedding 和 LLM 精排；
6. 同一实验不得混用不同 Prompt 或模型版本；
7. 实验目录必须包含代码提交 SHA；没有提交时标记为 `dirty`，不得作为最终报告结果。

## 9. 非功能需求

### NFR-01 可靠性

- 单个外部请求失败不终止批量任务；
- 所有外部调用必须有超时、重试上限和错误分类；
- 批量评测支持断点续跑；
- 部分结果必须显式标记，不得伪装成完整结果。

### NFR-02 可复现性

- Python、依赖、模型、Prompt、配置和随机种子均固定；
- 原始 API 响应可缓存回放；
- 每个正式实验把实际使用的原始 API 响应复制到 `experiments/<run_id>/snapshots/`，生成包含文件 SHA-256、provider、endpoint、请求参数和响应时间的 `snapshot_manifest.json`；正式快照不受缓存 TTL 清理影响。
- 最终结果记录 Git SHA 和配置哈希；
- 开发集、验证集、模拟测试集严格分离。

### NFR-03 性能与成本

- 搜索 API 目标不超过 8 次，硬上限 12 次；
- LLM 目标不超过 3 次，硬上限 5 次；
- 无缓存 P50 目标不超过 30 秒，P95 不超过 80 秒；80 秒触发软截止并组装部分结果，85 秒前开始返回，90 秒为硬终止；
- 有缓存重复查询 P50 目标不超过 8 秒；
- 每次对比实验必须报告 F1 与成本变化。

### NFR-04 安全与合规

- 密钥仅放在 `.env`，提交 `.env.example`；
- 日志不得包含密钥；
- 记录并核对模型、数据集和 API 的许可；
- 不绕过学术 API 的限流或服务条款；
- 不输出无法从数据源验证的论文标识。

## 10. 模型与技术选型

### 10.1 默认技术栈

- Python 3.11；
- FastAPI + Pydantic；
- httpx；
- SQLite；
- rank-bm25；
- sentence-transformers；
- NetworkX；
- Streamlit；
- pytest + pytest-asyncio；
- Ruff + mypy；
- uv 依赖与虚拟环境管理；
- YAML 配置和 Prompt 文件。

### 10.2 生成模型策略

- 使用 OpenAI-compatible 的 `LLMClient` 适配国内可用、支持 JSON/Schema 输出的 API。
- 默认候选服务使用阿里云百炼 OpenAI-compatible endpoint `https://dashscope.aliyuncs.com/compatible-mode/v1`；候选 A 为 `qwen3.6-flash`，候选 B 为 `qwen3.7-plus`。第 0 天必须确认两个模型均可调用；若账号区域实际模型 ID 不同，只允许在 `data/manifest.json` 记录服务端返回的精确 ID 后替换，不能使用模糊别名。
- 第 8–9 天在同一 40 条约束标注集上盲测候选 A/B，以强约束 Recall 优先、JSON 成功率其次、平均成本和 P95 延迟再次的顺序选主模型；强约束 Recall 相差小于 0.01 时选择成本更低者。另一模型成为备用。
- 不在业务代码中写死厂商模型名；最终提交配置必须记录真实模型 ID、服务商、调用日期和 Prompt 版本。
- 温度默认为 0；结构化失败只允许修复一次。
- 不把即将停用的 API 模型别名写入最终配置。

### 10.3 本地模型策略

- 默认 Embedding：`BAAI/bge-small-en-v1.5`；
- 可选本地 Reranker：`cross-encoder/ms-marco-MiniLM-L6-v2`；
- 两者均需在 CPU 和当前 4GB GPU 上做峰值内存、吞吐和效果测试；
- 本地 Reranker 只有在 F1 有稳定收益且 P95 延迟仍满足目标时进入主配置；
- 借用高显存 GPU 后可尝试微调 Reranker，但线上推理仍需可在可获得环境运行。

### 10.4 模型进入主配置的门槛

1. 满足本文档“稳定提升”定义；
2. 完成 3 次独立运行及 bootstrap 置信区间；
3. 验证集宏平均 F1 下降不超过 0.01，任一主要查询类型下降不超过 0.02；
4. 每增加 1000 tokens 或 1 次外部调用都报告配对 F1 增量；增量不为正时不得无条件启用；
5. 结构化成功率和失败率满足要求；
6. 能在目标环境稳定运行；
7. 权重或 API 许可允许竞赛使用；
8. 模型、参数、Prompt 和随机种子可复现。

## 11. 建议代码结构

```text
.
├── PRD.md
├── README.md
├── pyproject.toml
├── uv.lock
├── .env.example
├── configs/
│   ├── base.yaml
│   ├── budget_low.yaml
│   ├── budget_balanced.yaml
│   ├── ablations.yaml
│   └── prompts/
│       ├── query_analyze.yaml
│       └── constraint_rerank.yaml
├── data/
│   ├── README.md
│   ├── manifest.json
│   ├── provider_readiness.json
│   ├── annotation_guide.md
│   ├── dev/
│   ├── validation/
│   └── simulated_test/
├── src/paper_search/
│   ├── config.py
│   ├── domain/models.py
│   ├── llm/client.py
│   ├── query/parser.py
│   ├── query/planner.py
│   ├── retrieval/base.py
│   ├── retrieval/openalex.py
│   ├── retrieval/semantic_scholar.py
│   ├── processing/normalize.py
│   ├── processing/deduplicate.py
│   ├── processing/filter.py
│   ├── ranking/lexical.py
│   ├── ranking/embedding.py
│   ├── ranking/rerank.py
│   ├── ranking/fusion.py
│   ├── graph/citation_expand.py
│   ├── control/budget.py
│   ├── control/coverage.py
│   ├── pipeline/orchestrator.py
│   ├── storage/cache.py
│   ├── storage/experiment.py
│   ├── evaluation/dataset.py
│   ├── evaluation/official_adapter.py
│   ├── evaluation/metrics.py
│   ├── evaluation/runner.py
│   ├── api/app.py
│   └── ui/app.py
├── tests/
│   ├── fixtures/
│   ├── unit/
│   ├── integration/
│   └── evaluation/
└── experiments/
    └── README.md
```

## 12. 四周实施计划

所有任务执行顺序均为：先写测试 → 确认失败 → 最小实现 → 通过测试 → 小规模真实数据检查 → 提交。每个任务结束后必须留下可独立运行的交付物。

四周按 28 个日历日规划，主负责人每天保证核心开发时段，协作者按任务包交付。第 25–28 天为冻结期，不开发新算法功能。

### 开工前检查（第 0 天，不占核心研发日）

- 当前仓库只有本 PRD 和 Git 元数据，没有现有代码需要兼容。
- 安装 Git、Python 3.11 和 uv；依次运行 `git --version`、`python --version`、`uv --version`。项目初始化后统一使用 `uv sync --all-groups`，所有 Python 命令均用 `uv run` 执行。
- 创建 `.env.example`，只包含变量名：`OPENALEX_API_KEY`、`SEMANTIC_SCHOLAR_API_KEY`、`LLM_API_KEY`、`LLM_BASE_URL`、`LLM_MODEL_PRIMARY`、`LLM_MODEL_FALLBACK`。
- 确认 `qwen3.6-flash` 和 `qwen3.7-plus` 均能通过默认 endpoint 返回结构化响应；只在第 8–9 天根据盲测选主备顺序。
- 申请并验证 OpenAlex API key；平衡配置还必须验证 Semantic Scholar API key。低预算配置允许没有 Semantic Scholar key，并自动关闭第二来源。
- 用 PowerShell 分别调用 OpenAlex `/works?search=transformer&per_page=1`、Semantic Scholar `/graph/v1/paper/search?query=transformer&limit=1` 和 LLM `/chat/completions`；将状态码、速率限制响应头和时间写入 `data/provider_readiness.json`，不得写入密钥。
- 配置哈希统一使用“按键排序后的 UTF-8 JSON”计算 SHA-256。
- 年份合法范围统一为 1900 至“当前年份 + 1”；区间起点不得大于终点。
- `budget_low.yaml` 和 `budget_balanced.yaml` 使用第 7 节规定的 Token、调用和时间上限。

### 第 1 周：最小检索闭环和评测基础

#### Task 1：项目骨架、配置和领域模型（第 1 天）

**主负责人文件：** `pyproject.toml`、`src/paper_search/config.py`、`src/paper_search/domain/models.py`、`src/paper_search/control/budget.py`  
**测试文件：** `tests/unit/test_config.py`、`tests/unit/test_models.py`、`tests/unit/test_budget.py`

- [ ] 创建 Python 3.11 项目和锁文件，加入 FastAPI、Pydantic、httpx、pytest、Ruff、mypy。
- [ ] 实现 `QuerySpec`、`Paper`、`CandidateEvidence`、`SearchBudget` 和最终响应模型。
- [ ] 实现配置加载、环境变量覆盖和配置哈希。
- [ ] 先实现所有外部模块共用的 `reserve/settle` 硬预算计数器；Task 7 再增加软截止、持久化和覆盖停止逻辑。
- [ ] 测试非法年份、空标题、负预算、缺失 Token 上限和未知配置字段。
- [ ] 运行 `uv sync --all-groups`，预期生成/更新 `uv.lock` 且退出码为 0。
- [ ] 运行 `uv run pytest tests/unit/test_config.py tests/unit/test_models.py tests/unit/test_budget.py -v`，预期全部通过。
- [ ] 运行 `uv run ruff check .` 和 `uv run mypy src`，预期无错误。

**验收产物：** 领域模型可被后续模块直接导入；配置错误在启动阶段被发现。

#### Task 2：数据集适配和评测指标（第 2 天启动，第 7 天冻结）

**主负责人文件：** `src/paper_search/evaluation/dataset.py`、`src/paper_search/evaluation/official_adapter.py`、`src/paper_search/evaluation/metrics.py`  
**协作者文件：** `data/README.md`、开发集样例  
**测试文件：** `tests/evaluation/test_metrics.py`、`tests/evaluation/test_dataset.py`

- [ ] 定义包含 `query_id`、`query`、`relevant_paper_ids` 和元数据的 JSONL 格式。
- [ ] 实现 DOI、OpenAlex ID、Semantic Scholar ID 和标题的答案归一化。
- [ ] 实现 `OfficialEvaluationAdapter`；官方评分器未发布时使用第 14.0 节契约，并通过固定 fixture 验证预测文件字段、ID、去重和空答案行为。
- [ ] 实现 Precision、Recall、F1、Recall@5、Recall@10、Recall@20。
- [ ] 为零预测、零标准答案、重复预测和 ID 映射编写边界测试。
- [ ] 按第 14.1 节准备 60 条开发、30 条验证和 50 条模拟测试样本，记录 dataset revision、文件哈希、访问条件、抽样脚本和随机种子。
- [ ] 协作者在第 7 天前完成开发/验证样本的查询类型与领域标记；模拟测试集只保存 ID 清单，第 14 天冻结后不再查看标签。
- [ ] 建立 24 条覆盖七类查询及中英文改写的压力集；该压力集只用于鲁棒性，不参与 F1 调参。
- [ ] 协作者在第 7 天前交付 40 条查询约束标注 JSONL，字段固定为 `query_id, research_goal, must_have, should_have, exclusions, year_from, year_to, venues, query_type, domain, annotator`；主负责人复核 10 条。
- [ ] 两人独立标注同一批 20 条并计算一致性；关键离散字段 Cohen's kappa 低于 0.80 时，先修订 `data/annotation_guide.md`，再重标分歧样本。
- [ ] 运行 `uv run pytest tests/evaluation -v`，预期全部通过。
- [ ] 运行 `uv run python -m paper_search.evaluation.metrics --gold data/dev/gold.jsonl --pred tests/fixtures/predictions.jsonl --out experiments/smoke/metrics.json`，预期退出码 0 并生成宏平均 F1 与逐查询结果。

**验收产物：** 一条命令可以对静态预测文件计算指标；数据格式有清晰说明。

#### Task 3：OpenAlex 检索、缓存和标准化（第 3–4 天）

**主负责人文件：** `retrieval/openalex.py`、`storage/cache.py`、`processing/normalize.py`  
**测试文件：** `tests/unit/test_openalex.py`、`tests/unit/test_cache.py`、`tests/unit/test_normalize.py`

- [ ] 用固定 JSON fixture 编写成功、空结果、429、5xx、超时和缺失摘要测试。
- [ ] 实现 OpenAlex `search`、字段选择、年份过滤和分页上限。
- [ ] 实现 SQLite 缓存、TTL、缓存键和响应元数据。
- [ ] 实现从缓存导出不可变实验快照及 SHA-256 manifest；同一 run 的指标只能引用该 manifest 中的响应。
- [ ] 实现 OpenAlex 响应到统一 `Paper` 的映射。
- [ ] 实现限次重试和错误分类，不允许无限重试。
- [ ] 使用 3 条真实查询做冒烟测试，保存原始响应快照。
- [ ] 运行 `uv run pytest -m "not online" tests/unit/test_openalex.py tests/unit/test_cache.py tests/unit/test_normalize.py -v`；fixture 测试必须通过。
- [ ] 运行 `uv run pytest -m online tests/integration/test_openalex_live.py -v`；只有已配置 key 时执行，结果写入 `experiments/smoke/provider.json`。

**验收产物：** 输入查询可得到统一论文对象；断网时可回放缓存 fixture。

#### Task 4：去重、基础过滤、初始排序和评测命令（第 5–7 天）

**主负责人文件：** `processing/deduplicate.py`、`processing/filter.py`、`ranking/lexical.py`、`evaluation/runner.py`  
**协作者文件：** `src/paper_search/ui/app.py` 的最简列表页  
**测试文件：** 对应 unit 和 integration 测试

- [ ] 为 DOI、跨源 ID、精确标题和模糊标题去重编写测试。
- [ ] 实现硬约束过滤和“不确定字段降权”规则。
- [ ] 实现关键词覆盖与 BM25 排序。
- [ ] 实现 `uv run python -m paper_search.evaluation.runner --config configs/base.yaml --split dev --output experiments/baseline-week1`。
- [ ] 输出逐查询指标、宏平均指标、调用数和延迟。
- [ ] 协作者实现查询框和论文列表，不实现算法逻辑。
- [ ] 运行全部测试并对开发集完成第一次基线评测。
- [ ] 第 7 天冻结开发集、验证集和抽样脚本；后续新增样本单独进入压力集，不回写已冻结分区。

**第 1 周阶段闸门：** 可以从查询得到去重论文列表并计算 F1；未达到则第 2 周前两天继续修复，不开发关系图。

### 第 2 周：完整端到端 baseline

#### Task 5：LLM 客户端、查询解析和规划（第 8–9 天）

**主负责人文件：** `llm/client.py`、`query/parser.py`、`query/planner.py`、`configs/prompts/query_analyze.yaml`  
**测试文件：** `tests/unit/test_llm_client.py`、`test_query_parser.py`、`test_query_planner.py`

- [ ] 用模拟 LLM 编写合法 JSON、非法 JSON、空响应、超时和修复失败测试。
- [ ] 实现统一 OpenAI-compatible 客户端、Token 记录和模型元数据。
- [ ] 实现一次调用返回 `QueryAnalysisResult`，再由 Parser 校验、Planner 裁剪；实现一次修复和规则降级。
- [ ] 实现 3–6 个带目标约束的子查询。
- [ ] 使用人工标注的约束集计算强约束抽取 Recall。
- [ ] 对两个候选 API 在同一 20 条样本上盲测，冻结主模型和备用模型配置。
- [ ] 比较“原始查询”与“查询规划”对 Recall、F1、成本的影响。

**验收产物：** 查询理解结构化成功率达到 99%，强约束抽取 Recall 达到 90%。

#### Task 6：Semantic Scholar 与多源融合（第 10–11 天）

**主负责人文件：** `retrieval/semantic_scholar.py`、`retrieval/base.py`、`ranking/fusion.py`  
**测试文件：** `tests/unit/test_semantic_scholar.py`、`tests/unit/test_fusion.py`

- [ ] 为搜索、批量详情、引用、参考文献、限流和缺失条目编写 fixture 测试。
- [ ] 实现 Semantic Scholar Provider。
- [ ] 合并两个来源，保留来源一致性特征。
- [ ] 实现 Reciprocal Rank Fusion 和当前加权融合，两者均可配置。
- [ ] 验证第二搜索源对 Recall、F1 和调用量的增量。

**验收产物：** 任一搜索源失败时仍返回另一来源结果；跨源论文正确合并。

#### Task 7：预算控制和最小编排器（第 12 天）

**主负责人文件：** `control/budget.py`、`pipeline/orchestrator.py`、`storage/experiment.py`  
**测试文件：** `tests/unit/test_budget.py`、`tests/integration/test_orchestrator.py`

- [ ] 为 API、LLM、Token、费用、软截止和硬截止写失败测试。
- [ ] 在 Task 1 的硬预算计数器上增加并发原子预留、费用核算、软截止、预留过期和持久化。
- [ ] 实现只包含“查询分析 → 多源召回 → 去重过滤 → 关键词融合”的最小 Orchestrator。
- [ ] 记录每步 `ProviderResult`、配置哈希、Prompt 版本和停止原因。
- [ ] 测试任何 Provider 都不能绕过预算控制器。

**验收产物：** 即使不启用 Embedding、引文和 LLM 精排，也有受预算约束的端到端编排器。

#### Task 8：baseline 集成缓冲（第 13–14 天）

**主负责人文件：** `pipeline/orchestrator.py`、`api/app.py`、端到端测试  
**协作者文件：** 最简列表、详情和错误状态

- [ ] 为成功、无结果、一个 Provider 失败、查询分析失败、预算耗尽和软截止编写端到端测试。
- [ ] 实现 `POST /v1/search`、健康检查和固定 `StructuredSearchResponse`。
- [ ] 对照 `OfficialEvaluationAdapter` 或内部评测契约生成预测文件。
- [ ] 第 14 天冻结模拟测试集 ID 清单和标签访问规则。
- [ ] 执行无 Embedding、无引文、无 LLM 精排的完整 baseline，修复集成问题。
- [ ] 运行 `uv run pytest -m "not online" tests/unit tests/integration -v`，预期全部通过。
- [ ] 运行 `uv run python -m paper_search.evaluation.runner --config configs/base.yaml --split dev --output experiments/baseline-week2`，预期生成 `predictions.jsonl`、`metrics.json`、`usage.json` 和 `snapshot_manifest.json`。
- [ ] 在独立终端运行 `uv run uvicorn paper_search.api.app:app --host 127.0.0.1 --port 8000`；随后执行 `Invoke-RestMethod http://127.0.0.1:8000/health/ready` 和一次 `POST /v1/search`，预期返回合法响应或明确的 `degraded` 状态。
- [ ] 如果第 14 天仍未通过端到端测试，第 3 周首先修复，不开始关系图或本地 Reranker。

**第 2 周阶段闸门：** 自然语言查询可在预算内返回合法结构化结果，生成可复现预测文件；基础 baseline 不要求所有高级模块已启用。

### 第 3 周：高级检索、指标和稳定性

#### Task 9：Embedding、引文扩展和 LLM 精排（第 15–17 天）

**主负责人文件：** `ranking/embedding.py`、`graph/citation_expand.py`、`ranking/rerank.py`、`control/coverage.py`  
**测试文件：** 对应 unit 和 integration 测试

- [ ] 测试 Embedding 不可用、空摘要、批处理、OOM 和 CPU 降级。
- [ ] 实现标题加摘要编码，测量 CPU/GPU 峰值内存与延迟；主配置不允许 Embedding 和本地 Reranker 同时常驻 4GB GPU。
- [ ] 实现默认 1 篇、最多 2 篇种子的一跳前向/后向引文扩展及 canonical ID 边重映射。
- [ ] 实现每批最多 15 篇、最多两批的约束精排和截断记录。
- [ ] 将高级模块接入现有 Orchestrator，每项均可独立关闭。
- [ ] 每项完成开启/关闭消融；没有达到稳定提升门槛的模块不进入主配置。

#### Task 10：实验记录、消融和参数选择（第 18–19 天）

**主负责人文件：** `evaluation/runner.py`、`configs/ablations.yaml`  
**协作者产物：** 200 个论文对标注、失败案例表和实验记录

- [ ] 保存配置、Git SHA、模型、Prompt、逐查询指标、成本和预测文件。
- [ ] 论文对标注字段固定为 `query_id, paper_id, relevance_label, matched_constraints, evidence_text, annotator`；两人共同标注 50 对，一致性达到 0.80 后由协作者完成剩余 150 对，主负责人随机复核 30 对。
- [ ] 执行查询规划、第二来源、Embedding、引文扩展、LLM 精排的必做消融。
- [ ] 只在开发集调节子查询数、候选数、融合权重、Top-K 和相关性阈值。
- [ ] 第 18 天首次运行验证集；验证集结果只用于选方案，不回到开发流程调参。
- [ ] 计算各查询类型和领域的指标，完成 bootstrap 置信区间。
- [ ] 选择主配置和低成本配置；未达门槛模块改为条件触发或关闭。

#### Task 11：稳定性、泛化和最小展示（第 20–21 天）

**主负责人：** 限流、断点续跑、缓存回放、批量并发上限、泛化测试  
**协作者：** 筛选器、详情、成本面板和演示查询；关系图仅在阶段闸门通过后实施

- [ ] 执行 429、5xx、超时、空摘要、重复记录、无结果和 Provider 切换测试。
- [ ] 执行至少 30 条无缓存批量评测，确认单样本失败不终止任务。
- [ ] 完成中英文改写、未见领域、缺失摘要和简单查询路由测试。
- [ ] UI 不得生成后端未返回的论文信息。
- [ ] 完成一次全新环境安装演练。
- [ ] 只有主配置达到第 14 节 F1 与效率门槛后，协作者才实现关系图；否则保持最简列表和成本面板。

**第 3 周阶段闸门：** 可复现预测文件、主配置、低成本配置、至少三组核心消融和失败归因齐全；高级模块按证据启用，不要求全部进入主配置。

### 第 4 周：创新、冻结和交付

#### Task 12：预算感知的自适应查询演化（第 22–24 天）

1. CoverageAnalyzer 统计每个强约束的候选覆盖情况；
2. 对未覆盖或低覆盖约束生成定向下一轮查询；
3. 估计新增查询的调用和 Token 成本；
4. 执行后计算新增高相关论文数及 F1/Recall 增量；
5. 边际收益低于阈值或预算不足时停止；
6. 与固定一轮、固定两轮检索做相同预算和相同数据对照。

交付验收：完成与固定一轮、固定两轮的同数据同预算对照并报告负结果。进入主配置必须先满足“稳定提升”和第 14.3 节全部泛化门槛，并在此基础上满足以下任一收益条件：A）宏平均 F1 绝对提升至少 `0.02`；B）InternalScore 提升至少 `0.02`、F1 下降不超过 `0.005`、失败率增加不超过 1 个百分点。未达到门槛时使用完整 baseline 参赛，不强行启用创新模块。

#### Task 13：最终验证和版本冻结（第 25–26 天）

- [ ] 冻结依赖、模型 ID、Prompt、阈值、预算和随机种子。
- [ ] 在模拟测试集运行一次，禁止根据结果继续调参。
- [ ] 重复运行关键实验，检查结论一致性。
- [ ] 验证无缓存、有缓存、备用 Provider 和低预算四种模式。
- [ ] 生成最终指标表、成本表和消融表。
- [ ] 给最终版本打 Git 标签。

#### Task 14：展示、文档和答辩（第 27–28 天）

- [ ] 协作者完成前端视觉收尾和演示路径。
- [ ] 主负责人完成架构、算法、实验、创新和局限说明。
- [ ] 双人按演示脚本完成至少两次计时演练。
- [ ] 在一台新环境完成部署和运行。
- [ ] 最后 2–3 天不增加功能，只修复阻断问题。

## 13. 团队分工

| 工作 | 主负责人 | 协作者 | 交付物 |
|---|---|---|---|
| 架构和技术决策 | 负责 | 了解 | PRD、接口、配置 |
| 查询理解和规划 | 负责 | 人工抽查 | Prompt、解析指标 |
| 检索和排序 | 负责 | 测试 | 模块、消融结果 |
| 数据集整理 | 审核 | 负责 | 开发/验证样本 |
| 失败案例归因 | 联合 | 负责整理 | 失败类型统计 |
| 实验执行 | 负责设计 | 负责记录 | 实验目录和表格 |
| 前端 | 提供接口 | 负责 | Streamlit 页面 |
| 关系图与展示（阶段闸门后） | 校验真实性 | 负责 | 图、筛选和面板 |
| 报告和答辩 | 负责技术部分 | 负责整合展示 | 报告、演示稿 |

协作者任务必须具有明确输入、输出和验收，不得让其学习进度阻塞核心检索链路。

## 14. 评测方案

### 14.0 固定评测契约

若赛事发布官方提交 Schema 或评分器，必须先实现 `OfficialEvaluationAdapter`，并用官方样例逐字段验证；官方口径优先于本文档的内部口径。官方材料尚未提供的部分采用以下固定内部契约：

1. 预测正例集合仅取 `StructuredSearchResponse.selected_paper_ids`，先去重再评分；
2. DOI 统一小写并去掉 `https://doi.org/` 前缀；没有 DOI 时依次使用官方论文 ID、OpenAlex/Semantic Scholar 映射；标题模糊匹配只用于内部诊断，不替代官方 ID；
3. 预测集合按唯一规则生成：先保留 `final_score >= threshold` 的候选，再按 `final_score` 降序截取前 K 篇；分数相同时依次按约束覆盖率降序、`canonical_id` 字典序升序打破平局。高度相关和部分相关分组不改变此集合；
4. 在开发集对 `K ∈ {10, 20, 30, 50}` 和分数阈值 `0.45–0.75`（步长 0.05）做网格选择，选择宏平均 F1 最高且成本满足预算的组合；若多个组合 F1 相同，依次选择预测更少、成本更低、K 更小的组合。随后冻结，不在验证集和模拟测试集继续调节；
5. 主指标是逐查询 F1 的宏平均；微平均 F1、Precision、Recall 和 Recall@K 作为辅助指标；
6. 金标准和预测均为空时该查询 Precision、Recall、F1 记为 1；只有一方为空时 F1 记为 0；
7. 预测文件、逐查询评分表和成本表是第一优先级交付，前端不参与评分计算。

内部工程目标为：主配置在开发集宏平均 F1 不低于 `0.30`，且相对“原始查询 + OpenAlex”基线绝对提升至少 `0.03`；验证集相对基线绝对提升至少 `0.02`。这些不是官方分数承诺，官方公开榜单出现后应改用“超过官方 baseline”作为最终竞争目标。

### 14.1 数据划分

- 首选数据源固定为 Hugging Face `CarlanLark/pasa-dataset`。第 1 天完成访问条件确认，并把下载时解析到的 dataset revision SHA、文件 SHA-256、许可/访问条件和下载日期写入 `data/manifest.json`；此后所有实验只读取该冻结副本。
- 开发集从 `AutoScholarQuery/dev.jsonl` 按种子 `20260714` 分层抽取 60 条；验证集从 `AutoScholarQuery/test.jsonl` 抽取 30 条；模拟测试集使用 `RealScholarQuery/test.jsonl` 全部 50 条。三个分区分开报告，不把分数混为一个平均值。
- 若到第 1 天结束仍无法获得 PaSa 数据访问权限，唯一备用方案为 `allenai/asta-bench` 标签 `v0.3.1` 的 PaperFindingBench validation/test 数据；在 `data/manifest.json` 写入启用备用方案的原因。两种数据源不得混合构造同一个 F1 分区。
- 赛事公开数据发布后，建立独立 `competition` 分区和官方适配器；不得用参考数据的阈值直接宣称官方成绩。
- 开发集用于 Prompt、权重、阈值和预算调节；第 7 天冻结。
- 验证集用于方案选择，不能继续调参；第 7 天冻结并在第 18 天首次使用。
- 模拟测试集在第 14 天冻结，方案冻结后只运行一次。
- 另建 24 条不参与 F1 调参的压力集，覆盖七类查询、同义改写、长查询、歧义、缺失元数据及中英文表达。
- 团队不修改公开金标准。人工工作主要标注查询约束和 200 个候选论文对的相关/不相关标签；两人独立标注，Cohen's kappa 目标不低于 0.80，分歧通过复核形成最终标签。

### 14.2 指标

质量指标：Precision、Recall、F1、Recall@5/10/20。  
效率指标：搜索 API 次数、LLM 次数、输入/输出 Token、P50/P95 延迟、失败率、缓存命中率。  
结构指标：Schema 合法率、有效链接率、理由完整率、关系边可验证率。

内部近似竞赛得分：

```text
F1_normalized = macro_f1

EfficiencyScore = clamp(
  1 - 0.35 × search_api_calls / 12
    - 0.35 × llm_calls / 5
    - 0.20 × total_tokens / token_hard_limit
    - 0.10 × latency_seconds / 80,
  0, 1
)

StructuredOutputScore = mean(
  schema_valid,
  valid_paper_link_rate,
  reason_complete_rate,
  verifiable_citation_edge_rate
)

InternalScore = 0.70 × F1_normalized
              + 0.20 × EfficiencyScore
              + 0.10 × StructuredOutputScore
```

该分数只用于内部比较，不宣称等同官方分数。

结构化验收线：响应 Schema 合法率 100%，论文 ID/链接可验证率不低于 99%，理由字段完整率不低于 95%，展示出来的引文边可验证率 100%，虚构论文或关系数量必须为 0。

计算契约如下：每条查询先计算，再对查询做宏平均。`schema_valid` 是响应通过 Pydantic Schema 校验的 0/1；`valid_paper_link_rate` 的分母是所有展示论文，分子是 DOI/Provider ID 能在本次冻结原始响应中找到且 URL 由该 ID 确定生成的论文；`reason_complete_rate` 的分母是所有展示论文，分子是同时具有匹配约束、未匹配约束和非空理由的论文；`verifiable_citation_edge_rate` 的分母是所有展示边，分子是能通过 `source_edge_hash` 回溯到冻结响应的边。论文或边集合为空时相应比率记为 N/A，并从该查询的结构化均值中排除，不能记为 1；如果四项全部 N/A，则该查询 StructuredOutputScore 记为 0。

### 14.3 泛化与鲁棒性测试

- 按领域留出一组未见领域查询，宏平均 F1 相对总体验证集下降不得超过 0.05；
- 对至少 20 条查询生成不改变语义的改写，预测集合 Jaccard 和 F1 均需报告；
- 对 20 条中英文对应查询比较结果；中文查询由 QueryAnalyzer 生成英文检索词，若 F1 下降超过 0.05，则评估多语言 Embedding 替代项；
- 分别关闭一个 Provider、切换备用 LLM、删除摘要字段，验证降级模式；
- 对简单查询验证不会默认触发第二轮和全部高级模块；
- 任一主要查询类型的宏平均 F1 相对基线不得下降超过 0.02，否则必须采用按类型路由或关闭相关模块。

### 14.4 必做消融

1. 原始查询 + OpenAlex；
2. 加查询规划；
3. 加 Semantic Scholar；
4. 加 Embedding 初排；
5. 加引文扩展；
6. 加 LLM 精排；
7. 固定两轮检索；
8. 自适应查询演化；
9. 低预算配置与平衡配置。

每组同时报告 F1、Recall、调用数、Token 和延迟。

## 15. 完整验收标准

### 15.1 功能验收

- 支持复杂组合约束查询；
- 支持查询解析、规划、多源召回、过滤、初排、引文扩展、精排和结构化输出；
- 所有候选可追溯到来源和子查询；
- 所有过滤和排序保留原因；
- 高度相关和部分相关结果明确分组；
- 前端和批量评测共用同一后端。

### 15.2 质量验收

- 主配置达到第 14.0 节的开发集和验证集 F1 门槛；
- 查询规划、引文扩展和精排均完成消融；只有达到稳定提升门槛的模块进入主配置；
- QuerySpec 结构化成功率不低于 99%；
- 强约束抽取 Recall 不低于 90%；
- 精排人工判断准确率目标不低于 85%；
- DOI 去重准确，模糊去重抽查准确率不低于 98%；
- 不生成虚构论文、DOI、链接或引文关系。
- 主要查询类型相对基线下降不超过 0.02，未见领域相对总体验证集下降不超过 0.05。

### 15.3 效率验收

- 默认搜索 API 调用目标 ≤8、硬上限 12；
- 默认 LLM 调用目标 ≤3、硬上限 5；
- 单查询最大 2 轮；
- 无缓存 P50 ≤30 秒、P95 ≤80 秒，90 秒硬终止；
- 本地模型峰值显存 ≤3.2GB；
- 每次实验同时报告质量和成本。
- 批量评测硬失败率不超过 2%，返回显式部分结果的比例不超过 5%；两者必须分别报告。

### 15.4 工程验收

- 全新环境可按 README 安装和运行；
- 单元测试、集成测试、评测测试全部通过；
- Ruff 和 mypy 通过；
- API 密钥未进入代码、日志和仓库；
- 批量评测可断点续跑；
- 配置、Prompt、模型和依赖版本固定；
- 最终结果可由 Git SHA 和配置哈希复现。

### 15.5 竞赛交付完成定义

以下条件全部满足才视为完成：

1. 端到端系统和前端可运行；
2. 主配置和低成本配置均可运行；
3. 最终开发、验证和模拟测试结果齐全；
4. 至少三组核心消融和一组创新点对照实验齐全；
5. 所有论文和关系均可验证；
6. 部署、使用、评测和故障降级有文档；
7. 在新环境复现成功；
8. 演示脚本完成两次计时演练；
9. 最后版本已冻结并打标签。

## 16. 风险与降级方案

| 风险 | 早期信号 | 降级/处理 |
|---|---|---|
| API 限流或不稳定 | 429、P95 激增 | 缓存、退避、并发上限、切换备用源 |
| API 预算过高 | F1 小幅提高但 Token 激增 | 批量精排、减少候选、条件触发 LLM |
| 4GB 显存不足 | OOM、系统卡顿 | Embedding 改 CPU，关闭本地 Reranker |
| 查询分解丢约束 | Recall 下降、错误案例集中 | 强约束继承校验，保留原始查询通道 |
| 引文扩展噪声 | Precision 明显下降 | 减少种子和扩展量，按查询类型关闭 |
| 标签不完整 | 找到合理论文却被判错 | 同时报告严格指标和人工误差分析，不修改测试标签 |
| 队友学习进度慢 | 前端里程碑延误 | 保留最简列表页，关系图降为可选展示 |
| 创新点未产生收益 | 第 24 天仍无稳定提升 | 使用完整 baseline 参赛，创新作为负结果分析 |
| 借用 GPU 不可用 | 微调无法进行 | 微调始终为可选项，不影响主线 |
| 最后阶段功能膨胀 | 第 25 天仍新增模块 | 强制冻结，只修复阻断问题 |

## 17. 后续优化路径

### P0：四周内必须完成

- 查询约束抽取；
- 多查询检索；
- 标准化、去重和过滤；
- 基础排序与结构化输出；
- 评测、缓存、日志和预算控制。

### P1：高收益优化

- 多搜索源融合；
- 同义词、缩写和实体扩展；
- 轻量 Embedding 初筛；
- 一跳引文扩展；
- 约束级 LLM 精排；
- 按查询类型路由搜索策略。

### P2：本次推荐创新点

- 预算感知的自适应查询演化；
- 基于未覆盖约束生成下一轮查询；
- 用新增高相关论文数量和质量—成本比决定停止；
- 为不同难度查询动态分配预算。

### P3：有额外 GPU 和时间后

- 微调轻量 Cross-Encoder Reranker；
- 用参考数据构造论文对和排序样本；
- 建立小规模本地元数据/摘要索引；
- 用查询分类器选择检索策略；
- 用多臂老虎机选择搜索源或查询模板。

### P4：当前周期不进入主线

- 从零训练大型检索模型；
- 完整复现 PaSa 强化学习；
- 无限制多 Agent；
- 多跳全文知识图谱；
- 自建全量学术搜索引擎。

## 18. 参考资料

- 赛题三原始说明：`E:\桌面\AI\赛题三.docx`
- PaSa 论文：https://arxiv.org/abs/2501.10120
- PaSa 官方代码：https://github.com/bytedance/pasa
- PaSa 官方数据：https://huggingface.co/datasets/CarlanLark/pasa-dataset
- SPAR 论文：https://arxiv.org/abs/2507.15245
- LitSearch 论文：https://arxiv.org/abs/2407.18940
- AstaBench 官方仓库：https://github.com/allenai/asta-bench
- OpenAlex Works API：https://developers.openalex.org/api-reference/works/list-works
- OpenAlex 引文网络示例：https://developers.openalex.org/guides/recipes
- Semantic Scholar API：https://www.semanticscholar.org/product/api
- 阿里云百炼模型列表：https://help.aliyun.com/zh/model-studio/models
- BGE Small 模型卡：https://huggingface.co/BAAI/bge-small-en-v1.5
- MiniLM Cross-Encoder 模型卡：https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2
