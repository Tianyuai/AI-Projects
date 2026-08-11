# Title-informed 多查询检索干净基线设计

**日期：** 2026-08-11

**状态：** 对话设计已批准，书面 spec 待用户复核

**范围：** 冻结 dev 上的独立检索诊断；本设计不执行实验、不联网、不修改生产检索

## 1. 决策摘要

下一阶段建立一个独立、干净的 `title-informed` 多查询检索基线，验证单一假设：仅从冻结 `EvaluationQuery.query` 确定性构造少量互补 OpenAlex 查询，能否显著降低封存 recomposition 基准的 `not_retrieved=101/143` 主体召回缺口，并在沿用现有多子查询 OpenAlex 聚合路径的情况下让 selected Top-50 达到或超过旧版 title 外部基准 30。

本设计固定以下决策：

- 查询构造的唯一内容输入是冻结 `EvaluationQuery.query`；本文把它称为 reference title；
- 不使用生产 `QuerySpec`、Gold、identifier map、历史候选、论文标题、历史命中或评分反馈构造查询；
- facet 提取只使用版本化、确定性的规则，不使用 LLM、网络、语料频率、逐查询词典或人工例外；
- 每个 reference title 始终产生四个逻辑槽位：完整标题、主体–任务、方法–任务、数据集–任务；其中有效 OpenAlex 查询最多四条；
- 缺少 facet 或 canonical 重复时只产生 `no_op`，不得回填或增加组合；
- 检索覆盖先于 Top-50 评估：先要求 `not_retrieved <= 91`，再要求 `selected_top50 >= 30`；
- 若覆盖门槛失败，停止查询工程；若覆盖通过而 Top-50 失败，另开保留/排序设计；
- 本实验不修改生产代码路径、实验注册、candidate lock、readiness 或正式 capture/replay/compare。

## 2. 权威起点与问题定义

### 2.1 固定证据

本设计只读依赖：

- `docs/evidence/identifier-map-semantic-rescore-2026-08-11.json`，文件 SHA-256 为 `sha256:d09cbb52ce444d0ddcf2a49fa11468afcb5b9d648f4de58d8193d3ef1a0ecdf9`；
- `docs/evidence/sealed-query-recomposition-offline-2026-08-11.json`，文件 SHA-256 为 `sha256:c3b0fda41484d33d3de1933c7820c355708106544dcc7cbbb13bfbaea057f9d6`；
- 冻结 dev Gold generation hash `sha256:24009cf03ad069131793b9a190024e239082277bd0e48149a1efbbbb7978e215`；
- verified identifier 语义和固定的 143 个 Gold 关联分母。

上述证据不得覆盖或重算。未来 source lock 必须重新核对这些哈希，但不得把证据中的 Gold 内容送入查询构造。

### 2.2 固定外部基准

外部聚合基准保持只读，来源不可互换：

| 基准 | selected Gold | 用途 |
| --- | ---: | --- |
| 当前正式基线 2026-08-10 | 17/143 | 当前生产质量与既有 selected 保留基准 |
| 旧正式基线 2026-08-09 / append | 19/143 | 只读外部效果参照 |
| 封存 `rrf_slots_k60` | 25/143 | 只读可用排序信号参照 |
| 旧版 title | 30/143 | Top-50 充分性门槛 |

当前正式基线的 verified identifier 聚合指标为：

- macro F1 `0.01577682933599093`；
- macro recall `0.19708333333333333`；
- micro recall `0.11888111888111888`；
- MRR `0.11377801120448179`；
- NDCG `0.11808437599703922`；
- pipeline stages `103/0/23/17`。

封存 recomposition 的固定阶段基准为 `101/0/23/19`。因此封存 retrieved 集合包含 42 个 Gold 关联。本文把 `101/0/23/19` 称为“覆盖基准”，把当前正式基线的 `103/0/23/17` 称为“质量基准”；两者职责固定，不宣称来自同一运行。新基线必须把覆盖基准的 42 个关联作为只读保留集合，但不得把该集合用于查询生成、槽位选择、排序或执行早停。

### 2.3 已封存结论

`append_v2 / round_robin_slots / rrf_slots_k60` 已一次性比较并封存为 `19/24/25`。RRF 是可用排序信号，但 25 低于 30，正式结论为 `signal_insufficient`，reason code 为 `usable_signal_below_legacy_benchmark`。

本设计不得重新打开 recomposition：不增加变体、不调整 RRF 参数、不逐查询选择方法，也不把新查询族与旧重组方案组合成同一实验。

## 3. 假设、范围与非目标

### 3.1 可证伪假设

在查询构造层不能读取 Gold、identifier map、生产 `QuerySpec`、历史结果或评分的前提下，四个固定语义槽位能够：

1. 把 `not_retrieved` 从 101 降到不高于 91；
2. 至少新增找回 10 个 Gold 关联，且分布在至少 5 个查询；
3. 保留封存 retrieved 集合中已有的全部 42 个 Gold 关联；
4. 在不改变下游过滤和排序语义的情况下，使 selected Top-50 达到至少 30，并保留当前正式基线的全部 17 个 selected Gold 关联。

### 3.2 本实验包含

- 60 个冻结 dev reference title 的无 Gold 查询投影；
- 一个固定的确定性 facet 规则版本；
- 四个固定查询槽位及严格 `no_op` 语义；
- 一个独立 diagnostic source lock 和未来执行锁；
- 有限 OpenAlex 诊断 capture、dependency snapshot v2 和零网络 replay；
- 延迟加载 Gold 的 aggregate-only 评分；
- 完整性、覆盖、Top-50、预算、隐私和停止 Gate。

### 3.3 明确不包含

- LLM 查询生成、production `QuerySpec` 辅助 facet、旧 title-candidate 生成标题；
- Gold ID、Gold 标题、identifier map、历史命中或评分反馈驱动的查询；
- 逐查询词典、人工 override、结果驱动追加、参数网格或 query-level cherry-pick；
- 新排序、权重、保留槽、RRF 参数、LLM rerank、embedding 或其他数据源；
- 生产 orchestrator、默认实验、ablation、API 或 UI 修改；
- candidate lock、readiness、正式 live capture/replay/compare 或 validation；
- 在当前设计阶段运行 preflight、读取查询正文、联网、读取 `.env` 或 ledger。

## 4. 输入边界与 Gold 隔离

### 4.1 查询投影

查询源固定为 `data/dev/gold.jsonl`，文件哈希必须是第 2.1 节的 Gold generation hash。可信投影层保留外部 `query_id` 关联，但查询构造函数只接收：

```python
class ReferenceQuery(DomainModel):
    query: NonEmptyStr


def build_slots(query: str, rules: TitleFacetRules) -> SlotPlan: ...
```

可信 orchestration 边界负责从 `EvaluationQuery` 投影 `(query_id, ReferenceQuery)`，但只把 `ReferenceQuery.query` 传入 `build_slots`。`query_id` 只能用于外部关联、冻结顺序、哈希和审计，不能进入规则选择或查询文本。构造核心的函数签名、模型和序列化 payload 中不得存在 `query_id`、`relevant_paper_ids`、identifier map、历史运行、候选或评分字段。

投影层必须通过标签扰动不变性测试：保持 `query_id/query` 不变并任意替换、清空或重排 `relevant_paper_ids` 时，ReferenceQuery 投影、SlotPlan 和 source-lock 查询投影哈希必须完全相同。

### 4.2 Downstream control 白名单

为保持现有硬过滤是控制变量，可信 source preflight 从当前正式基线的冻结 `QuerySpec` 只投影：

```python
class DownstreamControl(DomainModel):
    year_from: int | None
    year_to: int | None
    venues: tuple[NonEmptyStr, ...]
    exclusions: tuple[NonEmptyStr, ...]
```

这些是当前 `apply_hard_filters` 实际读取的全部 query 字段，也是 OpenAlex inherited hard filters 的完整来源。`DownstreamControl` 不得包含 query text、research goal、topic、method、task、dataset、must/should-have、candidate、score、Gold 或 provider-query 字段。

下游适配器可在检索完成后用原始 reference query 填充 `QuerySpec.original_query/research_goal`，并只从 `DownstreamControl` 填充年份、venue 和 exclusion；其余字段保持空。slot builder 不能接收该对象。

### 4.3 三层隔离

1. **构造层**只接收 `ReferenceQuery` 和冻结规则；
2. **capture/replay orchestration 层**接收 source lock、reference query source、隔离的 downstream control projection、OpenAlex 配置、预算和快照适配器；其中 slot builder 的调用点只能传入 `ReferenceQuery` 和冻结规则；
3. **评分层**只在 capture/replay matched 且证据完整后，延迟加载 Gold 和 verified identifier map。

未来在线命令不得接受 Gold、identifier map 或评分阈值路径。评分必须是独立子命令或封存后 deferred loader，且技术失败或 replay mismatch 时不得调用。

### 4.4 防泄漏证明

实现测试必须使用：

- 一旦被读取就抛错的 Gold/identifier-map trap；
- 含 sentinel query、Gold ID、paper title、绝对路径和 credential-shaped 文本的隐私夹具；
- 接口签名检查，证明构造模型没有标签、`QuerySpec`、downstream control 或历史结果字段，capture 命令和网络调用模型没有标签字段；
- 标签扰动不变性和 query-ID 扰动不变性测试；
- 文件访问 spy，证明网络阶段没有打开 Gold、identifier map 或历史评分证据；
- 查询槽位哈希检查，证明评分前后构造结果不变。

Gold 只决定最终评分，不能决定查询数量、文本、顺序、执行、request retry、run rerun、停止或候选合并。本文未加修饰的“禁止论文标题”专指 Gold、candidate 或 result paper title，不包括作为唯一动态内容输入的 reference title。

## 5. `title-query-facets-v1` 规则契约

### 5.1 规则载体

新增诊断规则文件：

`configs/diagnostics/title_informed_retrieval_v1.yaml`

它必须逐字编码本节预注册的 v1 值，不得引入额外 marker 或参数。文件固定并版本化：

- schema/version `title-query-facets-v1`；
- Unicode 和空白规范化规则；
- 中英文分隔符；
- 通用请求措辞的固定停用列表；
- 主体、方法、任务、数据集的角色标记；
- span 字符和长度边界；
- 四个槽位及顺序；
- 每槽 OpenAlex limit 50；
- canonical dedup 与 `no_op` 规则。

规则文件在任何评分前整体冻结，source lock 记录其 SHA-256。实施后不得根据实验结果修改同一版本；任何实质修改都是新的假设和新设计。

### 5.2 规范化

对每个 reference title：

1. Unicode NFKC；
2. 把每个 Unicode whitespace 连续段替换为一个 ASCII space，再去除首尾空格；
3. 先且仅先移除至多一个 leading scaffold 和至多一个 trailing scaffold；
4. 在移除 scaffold 后的 core text 上识别 marker、分段和提取 facet；
5. 构造四个槽位；
6. 对每个非空槽位调用纯本地、确定性、无网络和无环境读取的 `paper_search.retrieval.openalex.canonicalize_openalex_search_query`；
7. 只使用 canonical identity 做槽位去重，按槽位顺序保留第一次出现；casefold 仅用于英文 marker/scaffold 匹配，不作为第二套槽位去重规则。

anchor canonical 为空是完整性失败。已满足 facet 长度/字符规则的非 anchor 槽位若 canonical 为空，也视为规则与 canonicalizer 不兼容的完整性失败；不得转换成 `no_op` 或回填。不得进行翻译、同义词扩展、拼写修复、语料统计、分词模型推断或联网实体解析。

### 5.3 固定 grammar

`title-query-facets-v1` 使用以下且仅以下规则：

- hard separators：`. , ; : ? ! / |`、`。 ， ； ： ？ ！ 、`、em dash 和 en dash；ASCII `-` 不作为 separator，避免拆开连字符术语；
- 英文 leading scaffold：`find|search for|retrieve|recommend`，后接可选 `academic|research`、必选 `papers|articles|literature`、必选 `on|about|for`；匹配不区分大小写且只允许正常空白；
- 中文 leading scaffold：`查找|检索|搜索|推荐`，后接可选 `关于|有关|针对`；
- 中文 trailing scaffold：紧邻结尾的 `论文|文献`，以及它前面的单个 `的`；
- dataset heads：`dataset|datasets|benchmark|benchmarks|corpus|corpora|数据集|基准|语料库`；
- method markers：`using|via|with|based on|leveraging|采用|使用|利用|通过|基于`；
- task markers：`for|to|toward|towards|aimed at|用于|面向|针对|以实现`。

“正常空白”固定指一个或多个 Unicode whitespace，经第 5.2 节后已统一为单个 ASCII space。英文 marker/scaffold 只有在其前后字符不存在或不是 Unicode letter、number、mark 或 `_` 时才匹配；中文 marker 按连续字面量匹配。所有 marker 都在 scaffold 移除后的 core text 上识别，按起始位置排序；同起点时使用 dataset head、method marker、task marker 的固定优先级。

span 规则固定为：

- `dataset`：从同一 hard-separator clause 的开始或前一个 role marker 的结束处，截取到 dataset head 结束；
- `method`：从 method marker 结束处，截取到下一个 role marker 或 hard separator 之前；
- `task`：从 task marker 结束处，截取到下一个 role marker 或 hard separator 之前；
- `subject`：移除 leading/trailing scaffold 后，从 core text 开始截取到第一个 role marker 或 hard separator 之前；若没有 role marker，则使用第一个非空 clause。

截取结果只做空白和两端 separator 修剪，不删除内部词。有效 span 必须包含至少一个 Unicode 字母、数字或 CJK 字符，长度为 2–120 个 Unicode code points，且不含控制字符。每类只检查按原文顺序出现的第一个语法候选；第一个候选无效时该类保持缺失，不截短、不继续扫描后续候选。不同角色候选可以重叠；只有规范化文本完全相同的候选才按 dataset、method、task、subject 的优先级保留一个角色。

规范性合成例：

- `Find papers on Alpha systems using Beta method for Gamma task on Demo dataset`：先移除英文 leading scaffold；`subject=Alpha systems`、`method=Beta method`、`task=Gamma task on Demo`、`dataset=Gamma task on Demo dataset`；槽位按第 6 节构造并由 canonical identity 去重；
- `检索关于甲系统基于乙方法用于丙任务在示例数据集上的论文`：先移除中文 leading/trailing scaffold；`subject=甲系统`、`method=乙方法`、`task=丙任务在示例`、`dataset=丙任务在示例数据集`。

示例只说明 grammar，不允许作为逐查询词典或特殊分支。

### 5.4 facet 提取

提取器只产生四类有序 span：`subject`、`method`、`task`、`dataset`。每类只检查按 reference title 阅读顺序出现的第一个语法候选；该候选无效时角色缺失。

- marker 和分隔符只能来自冻结规则文件；
- span 必须是 reference title 的规范化连续片段，不得生成新词；
- 同一 span 命中多个角色时按 dataset、method、task、subject 的固定角色优先级归类一次；
- 没有确定性证据时保持缺失，不做推断；
- 不允许 per-query override 或人工修正表。

facet 抽取失败但 anchor 有效时不是运行失败；对应组合槽位产生 `no_op`。

## 6. 固定查询族

每个 reference title 依次构造四个且仅四个逻辑槽位：

1. `anchor_full`：规范化后的完整 reference title；
2. `subject_task`：第一个 `subject` + 第一个 `task`；
3. `method_task`：第一个 `method` + 第一个 `task`；
4. `dataset_task`：第一个 `dataset` + 第一个 `task`。

组合只以单个空格连接两个 span。每个槽位先得到不可变 `plan_status`：有效查询为 `active`，缺少任一必要 facet 为 `no_op_missing_facet`，经 OpenAlex canonicalizer 后与较早槽位重复为 `no_op_duplicate`。`no_op` 在本文中只表示后两种 plan status 的类别名，不是可序列化状态。不得回填、交换角色、加入单 facet 查询或构造额外组合。

`anchor_full` 必须始终有效；否则 source preflight 为 `integrity_failure`。每个有效槽位固定请求最多 50 个 OpenAlex Works。每条 reference title 最多 4 个逻辑搜索操作，全批次最多 240 个。

## 7. 分阶段数据流

### 7.1 离线 source preflight

source preflight 不联网、不读取 `.env` 或 ledger，负责：

1. 验证固定证据、Gold generation、query set 和规则文件哈希；
2. 按冻结顺序投影 60 个 `ReferenceQuery`；
3. 从当前正式基线的哈希绑定运行中投影每查询的 downstream `QuerySpec`，形成与 slot builder 隔离的 `DownstreamControl`；
4. 仅以 `ReferenceQuery` 和冻结规则调用 slot builder，构造所有四槽状态；
5. 生成私有 source lock；
6. 输出 aggregate-only 的槽位计数和最坏预算，不输出 query ID 或文本。

source lock 固定：

- 60-query 顺序的整体哈希；
- 输入文件和 reference query 字段哈希；
- 每条 query 的原文哈希；
- 四槽的有序状态和有效文本哈希；
- 当前正式基线运行绑定及 `DownstreamControl` 投影哈希；
- 规则、canonicalizer、adapter、代码和 policy 哈希；
- 每槽 limit、最大逻辑操作数、重试和超时上限；
- 固定外部证据哈希与聚合基准。

source lock schema 固定为 `title-informed-retrieval-source-lock-v1`，规范路径为：

`runs/_locks/title_informed_retrieval-v1-source/source.lock.json`

使用规范 JSON、同目录临时文件、落盘同步和 no-replace 原子发布。source lock 不得含 raw query、逐 query 的可公开哈希、Gold ID 或 paper ID；它是私有文件，可包含完成重放绑定所需的有序整体承诺。

### 7.2 执行锁与授权

未来在线执行前，需经单独授权创建 execution lock。run ID 固定格式为 `YYYYMMDDTHHMMSSZ-<12 lowercase hex>`；execution lock schema 为 `title-informed-retrieval-execution-lock-v1`，路径为：

`runs/_locks/title_informed_retrieval-v1-<run-id>/execution.lock.json`

execution lock 绑定：

- source lock hash；
- 最新项目 ledger checkpoint；
- `configs/budget_balanced.yaml`，固定 SHA-256 `sha256:4041783f059d58d9f7e3949b95a60ceab94e6bafb114457064618adf0de358ac`；
- `configs/quality_gates_v1.yaml`，固定 SHA-256 `sha256:0e135d194b46de30cd89bdcd4d66ebda0d27f7f01056b4f1ba5b4fed851fb058`；
- 当前项目 ledger 返回的 pricing policy identity/hash；
- `DependencySnapshotManifestV2` 所在文件，固定 SHA-256 `sha256:5dd4dddac112f1c3413965b5e9509101ef193e5356656b4b7351a6e0ebec71ad`；
- 第 11.1 节的 retry/timeout policy；
- 预留上限和授权身份；
- 唯一 diagnostic run ID。

授权分为两个单元：

1. **execution-lock 授权**：只允许读取最新 ledger checkpoint、验证预算并 no-replace 创建一个绑定 run ID 的 execution lock；不读取 `.env`、不 reservation、不联网；
2. **diagnostic-run 授权**：只对该 execution lock 有效，允许临时读取必要 OpenAlex keys、建立 reservation、执行 capture、自动零网络 replay、deferred scoring 和 no-replace 聚合发布；不允许生产变更或 validation。

任一授权不能复用于其他 run ID。创建 execution lock 和运行都不属于本设计文档提交。

### 7.3 有限 diagnostic capture

未来获批后，runner：

1. 重新构造全部槽位并核对 source lock；
2. 核对 execution lock 和最新 ledger checkpoint；
3. 在独立调用点仅以 `ReferenceQuery` 和冻结规则调用 slot builder；
4. 为全部 `active` 槽位建立一一对应的 reservation；`no_op_*` 槽位不建立 reservation；任何部分预留失败都在零请求状态终结或释放已建立 reservation，并把全部 `active` 槽标记为 `not_scheduled`；
5. 按 60-query 顺序和四槽顺序串行执行；
6. `no_op_*` 不调用 OpenAlex，`execution_terminal=not_applicable`，actual usage 为零且没有 ledger receipt；
7. 每个有效槽位调用一次逻辑 OpenAlex search，最多 50 个结果；
8. 检索完成后才把结果与隔离的 `DownstreamControl` 交给下游投影；
9. 封存 dependency snapshot v2、终态、usage 和安全 provenance；
10. 首个不可恢复失败后停止调度；后续 `active` 槽写 `not_scheduled`，后续 `no_op_*` 槽仍保持 `not_applicable`。

这次运行是独立诊断 capture，不是正式生产 capture，不使用 candidate lock 或 readiness。

### 7.4 零网络 replay

capture 结束后立即在禁网适配器下：

1. 重建 reference title 槽位；
2. 从 snapshot 重新解析 OpenAlex 结果；
3. 重建逐槽、合并、过滤和 Top-50 业务投影；
4. 比较 capture/replay 规范业务哈希。

业务哈希不包含时间、usage、路径、headers、request ID 或 snapshot ref。只有 `matched` 才允许 deferred scoring。

### 7.5 延迟评分

评分层在技术证据通过后加载 Gold、verified identifier map，以及由公共 evidence 哈希绑定的以下私有只读材料：

- 质量基准：`runs/dev-20260810T104256Z-d9e89476d484`，使用 `scripts.rescore_identifier_semantics.load_formal_source` 和公共 rescore evidence 中 `formal_baseline_2026_08_10` 的 binding hashes；
- 覆盖基准：`runs/_diag_query_evolution_query-evolution-prompt-v2-full-20260810`，使用 `scripts.rescore_identifier_semantics.load_verified_probe_materials`、固定 `append_v2` 投影和 sealed recomposition evidence 的 binding hashes。

Gold 关联 identity 固定为 `(query_id, identifier_map.resolve(gold_identifier))`。17-selected 是质量基准的 selected resolved association set；42-retrieved 是覆盖基准的 retrieved resolved association set。新 retrieved 增量固定为 `new_retrieved_set - coverage_baseline_retrieved_set`。评分层核对全部 generation/binding hash 后计算这些集合；私有材料不得进入构造、capture 或 replay，也不得修改槽位、候选或 Top-50。

## 8. 下游处理的单变量边界

本实验的 treatment 是“固定四槽查询计划”；它不声称只改变字符串，而是把现有 3–5 子查询计划替换为本设计的四个固定逻辑槽，同时保持同一 OpenAlex provider 内的聚合、过滤和排序实现不变。查询构造不得使用 production `QuerySpec`；capture/replay 的下游投影只复用第 4.2 节 `DownstreamControl` 白名单。

处理路径精确绑定当前提交 `f04f68d` 的以下实现与文件 SHA-256：

- `paper_search.retrieval.openalex.canonicalize_openalex_search_query`：`sha256:c79eed6b0f89eced27dd44ef54768ddd68a6ebf1d64a43d5ac67fb50e8b081a2`；
- `MockSearchOrchestrator._combine_provider_results` 所在文件：`sha256:2c745698d148cd69e22abbaf464ebfc87e804d64c574282b9ec1970739255a2e`；
- `paper_search.processing.deduplicate.deduplicate_papers`：`sha256:b8fc17b4fe6a99c42553e112a2e7085a1edd78b96c238ac1252bbbb42a2d5584`；
- `paper_search.processing.filter.apply_hard_filters`：`sha256:23121a659f2080cf43474081887abc60e78451967df27408ecfd8ff590cc9e2d`；
- `paper_search.ranking.fusion.fuse_provider_results`：`sha256:8ded9c86c60637e58a3155026c1dc5b5661f85bed031cbb3220a478ff7058950`。

future implementation 不得修改这些生产文件。source preflight 必须核对哈希；漂移时停止并要求新设计复核，而不是接受“行为看似相同”。固定处理顺序为：

1. 使用 `DownstreamControl.year_from/year_to/venues` 形成 OpenAlex inherited hard filters；
2. 对四槽 ProviderResult 直接调用 `_combine_provider_results`，按槽位顺序串接并以 canonical ID 首次出现优先，形成唯一的 `openalex` source；
3. 对该 source 调用 `deduplicate_papers`；
4. 使用第 4.2 节构造的 filter-only QuerySpec 调用 `apply_hard_filters`；
5. 以 `{"openalex": combined_result}` 调用 `fuse_provider_results(method="rrf")`；
6. 只保留 hard-filter accepted IDs，按 fusion 输出顺序截断 Top-50。

四槽不得作为四个 RRF source。步骤 2 的固定追加是现有生产多子查询同 provider 的聚合语义，不是新的选择器；任何将槽位分别送入 RRF、改变槽位顺序、增加权重或保留槽的实现都违反本设计。

production `QuerySpec` 只能进入 downstream filter/ranking 控制路径，不能进入 facet extractor、slot builder 或 OpenAlex query text。

## 9. 指标与守恒

每个结果报告：

- `true_positive_count`；
- macro F1、macro recall、micro recall、MRR、NDCG；
- `not_retrieved`、`filtered_out`、`ranked_outside_top50`、`selected_top50`；
- 新增 retrieved Gold 关联数和覆盖的查询数；
- 是否保留全部 42 个封存 retrieved Gold；
- 是否保留全部 17 个当前正式 selected Gold；
- 每个槽位的 aggregate `active/no_op_missing_facet/no_op_duplicate` 数量；
- logical/actual OpenAlex 操作、重试、错误、usage 和 replay 状态。

阶段计数必须满足：

```text
not_retrieved
+ filtered_out
+ ranked_outside_top50
+ selected_top50
= 143
```

公开证据只保留聚合数量和布尔值，不发布逐查询记录。

阶段和指标不得重新定义。实现必须调用并绑定 `f04f68d` 的：

- `paper_search.evaluation.semantic_rescore.score_source`，文件 SHA-256 `sha256:99347925a848fa36afe5743e11ca5a41a283ac8a5477eee5e02536efaf816384`；
- `paper_search.evaluation.metrics.evaluate`，文件 SHA-256 `sha256:9912e0997368d00e3ed14f56599e58ba3e14eb79b885cbc4a3712a69e0bc4c59`；
- `paper_search.evaluation.ranking_metrics.evaluate_ranking`，文件 SHA-256 `sha256:99ad44e9fb2bf8362b3db915bd4915fd5d4e7e77de704278823a8dafddd1f3b1`。

这固定了空命中、去重、排序相关性、MRR/NDCG 和 macro/micro 聚合口径。Gate 比较使用上述函数的未舍入有限 float，直接执行 `>=`/`<=`，不使用 epsilon 或显示值四舍五入。

## 10. Gate 与固定结论

### 10.1 Gate A：完整性

Gate A 分为严格有序的两个子 Gate。

**Gate A1：技术完整性**，在不加载 Gold/map/历史评分材料时必须全部满足：

- 固定 evidence、query set、规则、代码、source/execution lock 和 policy 哈希一致；
- 60 个查询和四槽顺序完整；
- 所有槽位具有唯一 plan status 和允许的 execution terminal；
- 每个 active 槽的 reservation/receipt 终态与 usage 一致，no-op 槽没有 receipt；
- 无 integrity、provenance、snapshot、accounting 或 privacy failure；
- capture/replay 业务哈希 matched。

Gate A1 失败时不得调用 deferred loader。

**Gate A2：评分完整性**，仅在 Gate A1 通过后加载第 7.5 节材料并必须全部满足：

- verified identifier generation 未漂移；
- 质量/覆盖基准的 binding hashes 与 17/42 集合重建通过；
- 指标有限且 143 阶段计数守恒。

任一子 Gate 失败的结论都是 `integrity_failure`。此时不解释效果；A2 失败可以保留私有评分诊断，但不发布逐查询评分。

### 10.2 Gate B：检索覆盖

Gate A 通过后，必须全部满足：

- `not_retrieved <= 91`；
- `new_retrieved_set - coverage_baseline_retrieved_set` 至少包含 10 个关联；
- 新增命中分布在至少 5 个查询；
- 原有 42 个 retrieved Gold 关联全部保留。

失败结论：`coverage_insufficient`。停止查询工程，转向 OpenAlex/其他数据源覆盖、identifier mapping 或 Gold/reference 输入诊断。

### 10.3 Gate C：Top-50 可用性

Gate B 通过后，必须全部满足：

- `selected_top50 >= 30`；
- 当前正式基线的 17 个 selected Gold 关联全部保留；
- macro F1 不低于 `0.01577682933599093`；
- macro recall 不低于 `0.19708333333333333`；
- micro recall 不低于 `0.11888111888111888`；
- MRR 不低于 `0.11377801120448179`；
- NDCG 不低于 `0.11808437599703922`；
- `filtered_out=0`。

失败结论：`coverage_only`。只证明检索覆盖提升，下一步另写保留/排序设计；不得在本实验中调排序。

### 10.4 全部通过

Gate A/B/C 全部通过时结论为 `retrieval_baseline_candidate`。它只允许申请独立的 production-equivalent 集成或有限 live canary 设计，不代表生产晋级，也不自动创建任何锁或运行。

### 10.5 Reason codes

公开报告只允许固定 reason codes，顺序如下：

- `experiment_integrity_failed`；
- `not_retrieved_not_materially_reduced`；
- `retrieved_gold_not_preserved`；
- `coverage_gain_too_concentrated`；
- `legacy_top50_benchmark_not_met`；
- `formal_selected_gold_not_preserved`；
- `ranking_metric_regressed`；
- `hard_filter_regressed`；
- `preregistered_retrieval_gate_passed`。

Gate 按 A、B、C 顺序判定，报告与首个失败 Gate 一致，不从多个失败中选择更有利的解释。
`reason_codes` 包含首个失败 Gate 中所有失败条件对应的 code，严格按上表顺序排列；不包含后续未评估 Gate 的 code。`preregistered_retrieval_gate_passed` 只表示 Gate A/B/C 全部通过。

## 11. 预算、账本和错误处理

### 11.1 固定上限

- 60 个查询；
- 每查询 4 个逻辑槽；
- 最多 240 个逻辑 OpenAlex 操作；
- 每个有效操作最多 3 次 HTTP 尝试；
- 最多 720 次 HTTP 尝试；
- 每槽最多 50 个 Works；
- 从第一笔 reservation 创建前开始计时的全局运行超时 3600 秒，覆盖 reservation、网络、settlement 和 capture sealing；
- ledger reservation TTL 3900 秒。

source preflight 只计算最坏预算。execution lock 创建时才绑定最新 ledger checkpoint；任一预算、pricing 或 reservation 条件不满足即零请求停止。到达 3600 秒后不再调度请求，并保留最多 300 秒只用于 fail-close/settlement；超出 TTL 或未闭合 receipt 固定为 accounting failure。

request retry 复用并哈希绑定当前 OpenAlex adapter：只对 timeout、network error、HTTP 429 和 5xx 重试；单逻辑操作最多 3 次 HTTP 尝试；退避为 `min(8, 2^retry_index) + jitter[0,1)`。其他 4xx 不重试。本文“不得自动重跑”专指完整 diagnostic run rerun，不禁止这里预注册的 request retry。

### 11.2 终态

每个槽位具有两个正交字段。`plan_status` 只允许：

- `active`；
- `no_op_missing_facet`；
- `no_op_duplicate`。

最终 `execution_terminal` 只允许：

- `retrieved`；
- `not_applicable`；
- `integrity_failure`；
- `dependency_failure`；
- `accounting_failure`；
- `snapshot_failure`；
- `cancelled`；
- `not_scheduled`。

`plan_status=no_op_*` 必须对应 `execution_terminal=not_applicable`、零 usage、无 reservation/receipt。`plan_status=active` 不得使用 `not_applicable`；它必须有一个 receipt，并最终对应 `retrieved`、失败、`cancelled` 或 `not_scheduled`。未调度槽位不能省略。首个不可恢复失败后停止调度并安全终结剩余 active reservation。

状态转换固定为：

| 条件 | plan status | execution terminal | receipt |
| --- | --- | --- | --- |
| 缺 facet / canonical duplicate | `no_op_*` | `not_applicable` | 不创建 |
| 预留阶段整体成功、检索成功 | `active` | `retrieved` | settled actual |
| 预留阶段部分失败 | `active` | `not_scheduled` | 已创建者 release/零 usage 终结；未创建者无 receipt，运行整体 A1 失败 |
| 首个执行失败槽 | `active` | 对应 failure/cancelled | settled 或 fail-closed |
| 首个失败后的 active 槽 | `active` | `not_scheduled` | 零 usage 终结 |

每个 active 槽拥有 receipt 是完整预留成功后的 Gate A1 要求；预留阶段部分失败属于零请求技术失败，不进入效果评分。

### 11.3 错误分类

- anchor 为空、规则/槽位/锁漂移：`integrity_failure`；
- 请求级失败或重试耗尽：`dependency_failure`；
- controller、usage 或 ledger 不一致：`accounting_failure`；
- snapshot 写入、密封或验证失败：`snapshot_failure`；
- 全局超时或操作者取消：`cancelled`。

OpenAlex 部分成功页继续保留有效论文并记录 `invalid_work` 警告；请求级、provenance、结算或 snapshot 不完整仍阻断 Gate A。

技术失败不自动 rerun 整个 diagnostic run。任何 run rerun 必须先人工确认是证据完整性问题而非结果不佳，并以新 run ID、execution lock 和 diagnostic-run 授权执行；不得借 rerun 修改规则、槽位或门槛。

## 12. 证据与隐私契约

### 12.1 私有产物

写入 Git 忽略目录：

`runs/_diag_title_informed_retrieval_<run-id>/`

可包含 source/execution lock、槽位私有投影、逐操作终态、snapshot、replay trace、usage 和私有评分明细。它们不得提交。

私有目录完成后生成规范 `evidence-manifest.json`，逐文件记录相对路径、byte length 和 SHA-256，并以 manifest 自身 SHA-256 作为公开的单一私有证据包承诺。该目录至少保留到 paper-search 项目正式归档；不得因 Gate 失败或后续实验而删除、覆盖或复用。

### 12.2 公开产物

未来正式结果使用：

- `docs/evidence/title-informed-retrieval-baseline-<run-id>.json`；
- `docs/title-informed-retrieval-baseline-<run-id>.md`。

`run-id` 使用第 7.2 节的固定格式。JSON schema 固定为 `title-informed-retrieval-baseline-v1`，字段白名单只包含：schema/run ID、整体输入与 policy/code hashes、私有 evidence-manifest hash、aggregate plan-status/terminal/usage/stage/metric、Gate、结论和固定 reason codes。

公开文件禁止逐 query、逐 slot、逐 Gold、逐 paper 的哈希或 commitment，避免低熵值被枚举反推。query set、slot plan、17/42 集合只通过私有 evidence manifest 的整体承诺绑定。

规范 JSON 使用 UTF-8、排序 key、紧凑分隔符、`allow_nan=False` 和单个末尾换行。Markdown 只从已验证 JSON 渲染，不重新评分。两个目标都使用 no-replace 发布；若 JSON 已发布而 Markdown 失败，只允许从 JSON 恢复 Markdown。

发布前必须扫描并拒绝：

- query ID 和 query 文本；
- 槽位文本；
- Gold、paper ID 和论文标题；
- raw response、request ID 和 snapshot 路径；
- 绝对本机路径、`.env` 内容和 credential-shaped 文本。

## 13. 模块边界

未来实施预计新增：

- `configs/diagnostics/title_informed_retrieval_v1.yaml`：冻结规则；
- `src/paper_search/evaluation/title_informed_retrieval.py`：纯 facet、slot、stage、metric 和 Gate；
- `scripts/probe_title_informed_retrieval.py`：source preflight、execution、replay、deferred scoring 和发布；
- `tests/evaluation/test_title_informed_retrieval.py`；
- `tests/scripts/test_probe_title_informed_retrieval.py`；
- 必要的合成 snapshot/Gold 夹具。

模块不得注册生产实验或修改 orchestrator 默认行为。若复用生产 canonicalizer、去重、过滤或融合函数，只允许行为不变的公开导出或薄适配，不得顺带重构无关代码。

## 14. 测试策略

实施必须遵循 TDD，至少覆盖：

1. 合成中英文 reference title 的 NFKC、scaffold-first 顺序、marker 边界、重叠角色、空白、角色提取和连续片段约束；
2. 四个固定逻辑槽、缺 facet、canonical duplicate、非 anchor canonical-empty、anchor failure 和禁止回填；
3. 逐查询 override 和额外组合被拒绝，slot builder 只接收 query string/rules，拒绝 query ID、QuerySpec、Gold 和历史结果字段；
4. Gold trap、文件访问 spy、标签/query-ID 扰动不变性、DownstreamControl 白名单和 sentinel 隐私夹具；
5. 60-query 顺序、query/rule/code/policy/source lock 哈希；
6. 最大 240 logical / 720 attempts / 3600 秒 / 3900 秒 TTL；
7. `httpx.MockTransport` 正常响应、部分成功、429、5xx、timeout、重试耗尽；
8. reservation 部分失败回滚、plan status/execution terminal 状态表、无 receipt `no_op_*`、失败后 active `not_scheduled` 和全部账本终态；
9. dependency snapshot v2 capture、禁网 replay 和业务哈希 matched/mismatched；
10. 同一 OpenAlex 来源按槽位顺序追加、canonical ID 首次出现、现有去重/过滤/RRF/Top-50 语义；
11. 从两个固定私有来源重建 `(query_id, resolved Gold ID)` 的 42/17 集合，验证 143 守恒、91/30 边界、指标非回退和四种固定结论；
12. Gate A1 → deferred loader → Gate A2 时序、同 Gate 多 reason-code 固定顺序；
13. aggregate-only 字段白名单、禁止逐记录哈希、evidence manifest、双隐私扫描、原子写入、no-overwrite 和 Markdown 恢复。

自动测试不得读取 `.env`、真实 ledger 或网络。最终验证包括聚焦测试、全量离线 pytest、Ruff、mypy 和 `git diff --check`。

## 15. 验收与停止边界

### 15.1 设计实施可以判定完成

- 四槽规则和 source/execution lock 契约已实现并测试；
- 构造层无法访问 Gold、QuerySpec、downstream control 或历史评分；capture 网络阶段无法访问 Gold、identifier map 或历史评分，且 downstream control 不可传入 slot builder；
- 下游只复用冻结控制语义，没有新排序变量；
- 预算、账本、snapshot、replay、隐私和 no-overwrite 契约通过；
- Gate/结论边界由合成测试覆盖；
- 全量离线质量检查通过；
- 只完成 offline source preflight，未执行在线诊断。

### 15.2 只可经第 7.2 节对应授权执行

- execution-lock 授权：读取 ledger checkpoint 并创建一个 execution lock；
- diagnostic-run 授权：为该 lock 临时读取必要 OpenAlex keys，执行 diagnostic capture、自动 replay/deferred score 和聚合发布。

### 15.3 本实验始终不允许

- 修改生产检索、实验注册、candidate lock 或 readiness；
- 启动正式 live capture/replay/compare；
- 读取或运行 validation；
- 将一次运行授权解释为生产变更授权。

上述事项即使实验通过，也必须先另写设计并再次评审，不能在本实验内通过单次授权解锁。

### 15.4 实验后的强制停止

- `integrity_failure`：停止并人工诊断证据链；
- `coverage_insufficient`：停止查询工程，转向数据源、identifier mapping 或 Gold/reference 诊断；
- `coverage_only`：停止本实验，另写保留/排序设计；
- `retrieval_baseline_candidate`：停止在设计候选状态，申请独立 production-equivalent 集成或有限 canary 设计。

任何分支都不自动追加查询、调规则、重跑、重建 candidate lock、刷新 readiness 或启动正式闭环。

## 16. 备选方案与取舍

### 16.1 采用：固定语义槽位

优点是查询族少、互补、可解释，调用数和缺失行为可锁定。缺点是确定性多语言 facet 提取较保守，部分 query 可能只有 anchor；该缺点正是可证伪设计的一部分，不允许结果后回填。

### 16.2 未采用：词法压缩阶梯

完整 query、去通用措辞、Top-N 关键词和固定窗口更容易实现，但语义角色弱，`N` 和窗口长度容易成为参数网格，难以解释哪类线索真正改善召回。

### 16.3 未采用：有界 facet 组合

枚举主体/方法/任务/数据集二元组合可能提高召回，但查询数更多，接近组合搜索，也增加逐查询选择和结果驱动截断风险。

### 16.4 未采用：复用 production QuerySpec 或 LLM

production QuerySpec 会把旧解析偏差带入新基线；LLM 会引入网络、提示词、模型、随机性和预算变量。两者都削弱“reference title 单一输入”的独立性。

## 17. 后续流程

本设计批准并提交后，下一步只能先进行书面 spec 复核。用户再次批准书面 spec 后，才使用 `writing-plans` 编写精简实施计划。

实施计划必须把工作分为：纯规则与锁、无网 source preflight、MockTransport capture/replay、deferred scoring 与发布、全量离线验证。任何在线执行仍保持单独授权。
