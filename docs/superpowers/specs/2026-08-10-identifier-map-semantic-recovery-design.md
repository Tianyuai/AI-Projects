# Identifier Map Semantic Recovery Design

## 背景与结论

当前正式 dev capture `dev-20260810T104256Z-d9e89476d484` 与 replay
`dev-20260810T123542Z-e097e1aa48b2` 的运行、快照、账本和重放证据仍然有效，
但其相关性指标不再作为可信质量基线。正式运行绑定的 identifier map 哈希为
`sha256:6ea6dbcd20a3f572d9f0dd0a0eef938ff01773db792401adf7fde1e489396e82`；
现有实现只验证格式、冲突、循环、字节哈希和 gold 覆盖率，不验证映射两端是否为
同一论文。

本地、无需网络的聚合复核已经给出确定性反例：141 个 dev gold arXiv 标识中，
43 个被映射到带 arXiv 编号的 DataCite DOI，其中 41 个 DOI 内嵌编号与源 arXiv
编号不同。当前密封输出中另有 12 个可通过完全相同 arXiv 编号直接确认的命中，
而现有正式指标只计入 6 个。因此 `0.0038000670` 只保留为历史运行结果；旧
`134/134 available`、漏斗归因和方法晋级结论均须在语义恢复后离线重算。

## 目标与非目标

本阶段目标是建立一个 fail-closed 的论文标识语义契约，重建受该契约约束的私有
identifier map，并从既有密封产物恢复可信 dev 基线。完成前停止新的检索、排序和
live capture 实验。

本阶段不：

- 改写任何历史 run、snapshot、ledger 或 evidence；
- 修改现有 `data/manifest.json` 或替换其已绑定的旧 map；
- 读取 validation 查询、运行 validation 或用 validation 指标作决策；
- 把 gold、映射条目、查询文本或私有元数据写入公开报告；
- 修改生产检索、排序、过滤或可选模块行为；
- 重跑已经否决的 Query Evolution、标题排序或其他方法；
- 在离线可信基线恢复前重建 candidate lock、刷新 readiness 或发起 live capture。

## 方案选择

采用“语义校验契约 + split-scoped 私有等价映射 + 密封产物离线重算”。该方案
保留现有 `IdentifierMap` 在评分和去重中的接口，只为新 dev freeze 增加可验证的
语义前置条件和证据绑定，改动范围小于立即把整个领域模型改造成多标识集合。

未采用的方案：

- 只人工修正 JSON：速度快，但不能阻止同类错误再次进入正式评分，也缺少可复现
  证据；
- 立即在 `Paper` 和所有产物中引入多标识等价类：长期更完整，但会同时改变检索
  规范化、去重、artifact schema 和 replay 契约，不适合作为当前可信度恢复的第一步。

多标识领域模型仅在本方案证明现有 map 接口无法无损表示权威等价关系时重新评估。

## 语义判定契约

语义审计以规范化后的 `alias -> terminal` 关系为单位。每个会影响评分的显式关系
必须得到至少一种精确正向证据；覆盖率、标题相似度、搜索排名或“目标存在”均不构成
同一论文证据。

允许的正向证据只有：

1. `arxiv_datacite_exact`：`10.48550/arxiv.<id>` 中的规范化编号与 arXiv alias
   完全一致；
2. `semantic_scholar_exact`：以 arXiv 和目标 DOI 精确反查得到同一 Semantic
   Scholar paper ID，且返回的 external IDs 不与源标识冲突；
3. `openalex_location_exact`：目标 DOI/OpenAlex work 的位置或外部标识中包含完全
   相同的规范化 arXiv ID。

明确的 DataCite arXiv 编号不同属于 `semantic_mismatch`。OpenAlex 目标显式关联
其他 arXiv ID 且不含源 ID 也属于 `semantic_mismatch`。仅有不同 Semantic Scholar
paper ID 不单独判错，因为供应商可能存在重复记录；它记为冲突信号，并在没有其他
精确正向证据时落入 `unresolved`。

审计终态只有 `verified`、`semantic_mismatch` 和 `unresolved`。正式 dev 评分要求：

- 所有 dev gold alias 均显式覆盖；
- 所有实际参与解析的 map 关系均为 `verified`；
- `semantic_mismatch = 0`、`unresolved = 0`；
- 不允许通过强制映射、模糊标题或忽略失败来凑满覆盖率。

## 组件边界与数据流

### 私有身份采集

一个有界、单用途采集器只接收规范化标识，不接收查询文本、相关性标签、排名或
候选列表。它优先复用已有不可变 provider snapshot；缺失的精确身份元数据需要单独
在线授权，并使用项目账本、固定请求上限、重试上限和不可变私有 snapshot。网络响应
不得直接决定通过，必须先封存再由离线审计器消费。

私有证据保存在已忽略的 `data/annotation_work/` 下。它可以包含身份对应关系，但不
提交 Git，也不进入聊天、公开 Markdown 或聚合 JSON。

### 离线语义审计

纯离线审计器一次读取 identifier map、dev gold 和私有身份 snapshot，先验证各自
字节哈希，再执行规范化和三态判定。它不得访问网络，也不得根据 gold 增删或排序
候选。审计结果分为：

- 私有逐关系结果：用于修图和复核，保持在忽略目录；
- 公开聚合报告 `identifier-map-semantic-audit-v1`：只包含输入哈希、总数、三态
  计数、证据类型计数、直接 arXiv 命中 sanity check、原因码和通过状态。

公开报告禁止包含任何论文标识、标题、作者、查询 ID、查询文本或逐项位置索引。

### 私有 map 重建

重建器只处理 dev gold 并只把 `verified` 关系写入新的 dev map。每个 gold arXiv
alias 的 terminal 固定为由自身编号确定的 `doi:10.48550/arxiv.<same-id>`；这只是
稳定的内部等价组锚点，不要求它成为生产检索请求。由此 terminal 选择不再依赖供应商
返回顺序、某个“首选 DOI”或人工排序。

从现有密封 baseline、标题候选和 Query Evolution 产物收集到的 DOI/OpenAlex alias，
以及精确身份元数据返回的其他 provider alias，只有在它们与该 arXiv 锚点的关系得到
精确证明后才加入同一组。每个 dev gold 组还必须完成规定的 provider 身份检查；
provider 元数据缺失或互相冲突时，该组仍为 `unresolved`，不能仅凭锚点自映射宣称
正式语义审计通过。映射不得依赖输入顺序、列表位置、标题相似度或搜索排名。

`semantic_mismatch` 被删除，`unresolved` 保留为私有待复核项，不写入正式 map。
若仍有 unresolved gold，流程停止，不生成“通过”的正式 map。旧 map 和现有
`data/manifest.json` 保持原字节不变；新 map 使用新文件名和新哈希，且不得包含为
validation 构建或推断的映射。

### 正式绑定

新建 split-scoped freeze identity，不迁移现有 V2 manifest。新 schema 为 dev 分区
绑定独立的 map、语义审计报告和私有证据集合哈希；本阶段不创建 validation 绑定。
新文件写入独立路径，现有 `data/manifest.json` 不修改。V2 loader 继续只用于验证
历史 artifact；新的 candidate lock 必须使用 split-scoped schema，validation lock 在
缺少独立 validation 语义绑定时必须拒绝创建。

候选锁和正式运行必须同时绑定 dev map 哈希、语义审计报告哈希及其私有证据集合
哈希。runner 和 Gate 0 在 provider 构造前只验证 lock 选定的 dev 分区，不读取
validation partition，并验证：

- 审计报告 schema、状态和哈希有效；
- 报告绑定的 map、gold split 和证据集合与本次输入完全一致；
- map、报告和证据均为 dev scope，且不宣称 validation 结论；
- map 的覆盖、格式和语义状态全部通过。

缺失、过期、错 scope、哈希不一致或非 `passed` 的报告均以固定、无标识值的错误
停止。历史 run、V2 manifest 和旧 map 不迁移、不重写，只在 HANDOFF 和路线图中
标注其质量结论已被取代。

### 离线重算与算法决策

新 map 通过后，按相同代码和密封输入离线重算：当前正式 baseline、标题候选历史
对照和 Prompt-v2 Query Evolution。每组结果单独绑定原始 run 哈希、新 map 哈希和
语义审计哈希，不跨历史运行混用候选口径。

直接可确认的 12 个同 arXiv 编号命中必须全部进入评分；否则语义恢复仍失败。重算
后重新统计 retrieved、post-filter、Top-50 和 macro/micro 指标：

- 若可信候选召回已经充足而 Top-50 流失主导，下一实验只设计 selector/reranker；
- 若未检索到仍主导，下一实验只从 Query Evolution 或多源检索中选择一个单变量；
- 若历史方法在可信评分下已产生正向结果，优先复用该密封 evidence，不重复网络运行。

## 错误处理、隐私与授权

- 所有语义失败均使用固定原因码，异常消息不包含标识值；
- 私有 map、逐关系证据和 unresolved 清单必须继续被 Git 忽略；
- 日志和公开报告在写盘后重新读取并执行禁止字段/标识模式扫描；
- 在线身份采集与正式 live capture 是两种独立授权，前者获批不自动授权后者；
- 不读取 `.env`，除非用户明确授权某次有界身份采集；只把必要 key 临时注入该进程；
- 不读取或评分 validation 查询。未来 validation 需要单独设计、单独 map audit 和单独
  不可撤销授权。

## 测试策略

采用 TDD，最小覆盖以下行为：

1. 同 arXiv DataCite DOI 通过，编号不同必然失败；
2. 同一 Semantic Scholar paper ID 或 OpenAlex 精确位置关联通过；
3. 单独的 S2 paper-ID 分歧只产生 unresolved，不误判为确定错配；
4. 模糊标题、搜索排名、目标存在和缺失证据均不能通过；
5. 错误链、冲突链及未经验证的预测 alias 不能诱发假命中；
6. 聚合报告不泄露任何私有标识或查询信息；
7. map、gold、证据或报告任一字节变化都会使绑定失败；
8. 审计失败发生在 provider 构造和任何网络调用之前；
9. split-scoped dev Gate 不读取 validation partition，缺少独立语义绑定时不能创建
   validation lock；
10. 旧 V2 artifact 保持可验证但被标记为历史语义未认证；
11. 密封输出中的 12 个直接 arXiv 命中全部计入新评分。

专项测试通过后运行受影响测试、Ruff、mypy、完整离线 pytest 和 `git diff --check`。
在线接口只用确定性 mock 测试；真实身份采集不属于单元测试。

## 验收门槛与停止条件

语义恢复只有同时满足以下条件才完成：

- dev map 的所有评分关系均为 `verified`，无 mismatch、无 unresolved；
- 语义报告、map、dev gold 和私有证据哈希闭合；
- 公开产物通过隐私扫描；
- 12 个直接同 arXiv 命中全部被计入；
- 当前 baseline 和历史候选均完成离线重算，漏斗可解释；
- 专项、静态检查和完整离线测试通过。

任一 gold 关系无法精确验证时停止在 map 重建阶段，不发布正式 F1。任一离线算法候选
不能同时提高可信 Top-50 gold 和 macro F1、且保留既有可信命中时，不重建 candidate
lock、不刷新 readiness、不申请 live capture。validation 始终保持独立授权。
