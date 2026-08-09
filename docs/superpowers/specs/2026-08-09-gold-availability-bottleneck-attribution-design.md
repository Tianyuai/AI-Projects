# Gold 可用性与检索瓶颈归因设计

日期：2026-08-09  
状态：已确认，待实施计划

## 目标

在不运行 live capture、不改变生产检索和排序的前提下，回答两个问题：

1. 冻结 dev gold 中有多少唯一论文可由 OpenAlex 精确解析；
2. 对每个查询–论文关联，损失发生在未召回、硬过滤还是 Top-50 截断。

最终产物必须给出可审计的瓶颈分布；只有诊断完整且最大可恢复损失桶唯一时，才给出一个主要方向。诊断不完整或最大桶并列时必须明确停止，不为得到结论而强行决策。

## 非目标

- 不把 gold 标识符转换成检索关键词；
- 不使用标题、作者或摘要做模糊匹配；
- 不同时探测 Semantic Scholar、Crossref 或其他新数据源；
- 不修改生产 provider、过滤、融合或排序；
- 不重建候选锁，不运行 readiness、capture、replay、compare 或 validation；
- 不在输出中保存 gold ID、查询文本、论文标题、响应正文、request ID 或密钥。

## 固定输入与计数口径

主要流水线证据使用当前通过 Gate 的正式 capture：

`runs/dev-20260809T061903Z-9bd861e90299`

固定输入包括：

- `data/dev/gold.jsonl`；
- `data/identifier-map.json`；
- capture 的 `executions.jsonl` 与 `business-results.jsonl`；
- capture 的 `run.json`、输入哈希和源提交信息。

CLI 必须先确认该 capture 的 Gate、formal validity 和 provenance 均通过，再读取阶段集合。运行开始和写报告前各计算一次输入哈希；两次不一致即终止，避免在线探针期间输入漂移。

当前输入有三套不同但都必要的计数：

| 口径 | 数量 | 用途 |
|---|---:|---|
| 原始 gold 标识符记录 | 143 | 输入完整性审计 |
| 归一化查询–论文关联 | 139 | 与评测和流水线阶段对齐 |
| 归一化唯一论文 | 134 | 在线精确反查请求上限 |

143 条原始记录均为 arXiv ID。经冻结 identifier map 归一化后，134 篇唯一论文全部具有可直接反查的终端标识符：128 个 DOI、6 个 OpenAlex ID。因此当前数据不需要 arXiv URL、标题或全文搜索兜底。

实现不得只硬编码这些数字。CLI 必须从输入重新计算，并在数字不符时 fail closed；报告同时保留三套口径，避免再次混淆 143、139 和 134。

归一化统一调用现有 `normalize_paper_id` 和 `IdentifierMap.resolve`。先在每个查询内按 resolved ID 去重，得到 139 个查询–论文关联；再跨查询按 resolved ID 去重，得到 134 篇唯一论文。不得自行实现第二套 DOI、arXiv 或 OpenAlex 规范化规则。

## 两维归因模型

### 维度 A：OpenAlex 精确可用性

以 134 篇归一化唯一论文为单位，每篇只能进入一个状态：

- `available`：OpenAlex 单实体接口返回 200，且响应标识符经 `normalize_paper_id` 后与请求的 resolved ID 精确一致。DOI 请求比较响应顶层 DOI；OpenAlex ID 请求比较响应顶层 Work ID；
- `exact_not_found`：单实体接口明确返回 404；含义仅为“OpenAlex 精确接口未解析该标识符”，不外推为论文不存在；
- `unknown_transient`：超时、429 或 5xx 经有限重试后仍未得到确定结果；
- `invalid_identifier`：归一化后不是 DOI 或 OpenAlex ID；当前固定输入中预期为 0；
- `integrity_failure`：200 响应不可解析或标识符不匹配。该状态阻止形成方向性结论。

请求方式：

- OpenAlex ID：`GET /works/{W-id}`；
- DOI：`GET /works/{URL-encoded DOI URL}`；
- 只选择验证身份所需的最小字段；
- 每个唯一论文最多一个确定结果，重复查询关联复用该结果；
- 请求计划按规范化 resolved ID 词典序固定，保证相同输入产生相同顺序。

OpenAlex 官方单实体接口支持用 OpenAlex ID 或 DOI 获取 Work；Work 的官方 ID/filter 字段不提供 arXiv ID，因此禁止用 arXiv 文本搜索冒充精确 availability：

- <https://developers.openalex.org/api-reference/works/get-a-single-work>
- <https://developers.openalex.org/api-reference/works>

认证失败、配置错误、输入哈希变化或预算无法覆盖完整探针时，整个探针 fail closed，不生成“不可用”结论。在线调用必须复用项目现有 OpenAlex 认证、限流和预算账本机制，不新增旁路 key loader，也不直接修改 SQLite。启动前读取 `project_checkpoint()`，按唯一论文及重试上限做预算预检，每次实际 HTTP 尝试按现有账本接口结算，并在报告中写入前后 checkpoint 的聚合哈希和实际尝试数。

每个唯一论文最多 3 次总尝试：首次请求加最多 2 次重试。只对超时、429 和 5xx 重试；遵循合法 `Retry-After`，否则按 1 秒、2 秒退避，单次等待上限 10 秒。理论 HTTP 尝试硬上限为 `134 × 3 = 402`；达到上限立即停止。401/403、其他非 404 的 4xx、预算错误和完整性错误不重试。

### 维度 B：正式流水线位置

以 139 个归一化查询–论文关联为单位，从正式 capture 的三个有序集合做互斥分类：

- `selected_top50`：存在于 `selected_paper_ids`；
- `ranked_outside_top50`：存在于 `post_filter_paper_ids`，但不在最终选择中；
- `filtered_out`：存在于 `retrieved_paper_ids`，但不在硬过滤后集合中；
- `not_retrieved`：不在 `retrieved_paper_ids`；

在分类前必须验证：

```text
selected_paper_ids ⊆ post_filter_paper_ids ⊆ retrieved_paper_ids
```

并验证 gold、execution 和 business result 的查询集合完全一致。查询集合、字段、Gate、provenance 或产物契约不完整时直接终止，不生成报告；不把产物错误伪装成论文级分类。任一不变量失败即终止，不输出方向性结论。

当前只读预检得到 14 个 retrieved、14 个 post-filter、8 个 selected exact-gold 关联；这只是设计校验，正式报告必须由实现重新计算。

## 交叉统计与决策规则

报告包含：

1. 唯一论文级 availability 状态计数；
2. 查询–论文关联级流水线阶段计数；
3. availability × 流水线阶段交叉表；
4. 每个桶覆盖的查询数；
5. 输入哈希、源运行、源提交、实际请求/重试/错误聚合计数。

只有 `unknown_transient = 0`、`invalid_identifier = 0`、`integrity_failure = 0` 时，报告才标记 `diagnostic_complete: true` 并选择主要方向。否则可以写出部分在线证据，但 `recommended_direction` 必须为 `null`。离线产物不完整属于全局失败，不写部分报告。

完整诊断把唯一论文 availability 映射回 139 个查询–论文关联，再按以下可恢复损失桶比较关联级绝对数量，不混用唯一论文数与关联数，也不设置人为百分比阈值：

| 最大桶 | 推荐方向 |
|---|---|
| `exact_not_found` 且未进入 Top-50 | 对新数据源做有限 exact-ID probe |
| `available × not_retrieved` | 生产一致的检索/Query Evolution 小探针 |
| `available × filtered_out` | 只诊断对应硬过滤规则 |
| `available × ranked_outside_top50` | 实质不同的 selector/rerank 离线实验 |

若最大桶并列，`recommended_direction` 为 `null`，报告并列项，不擅自选择。已否决的标题权重、保留槽、Embedding、Topic、Citation、普通 Query Rewrite 和旧 LLM variants 不因本报告自动重开。

## 组件边界

建议新增一个独立诊断 CLI，不扩张生产检索接口：

- 纯函数层：输入计数、标识符规划、响应分类、流水线阶段分类、交叉聚合和隐私校验；
- 在线适配层：最小 OpenAlex 单实体 GET、有限重试、预算结算；
- CLI 层：校验固定输入、执行 probe、原子写入聚合 JSON 和简洁 Markdown。

建议产物：

- `scripts/analyze_gold_bottlenecks.py`；
- `tests/scripts/test_analyze_gold_bottlenecks.py`；
- `docs/evidence/gold-bottleneck-attribution-2026-08-09.json`；
- `docs/gold-bottleneck-attribution-2026-08-09.md`。

不持久化逐论文中间状态。探针中断时不根据未完成样本作结论；再次执行时重新读取当前项目预算 checkpoint。

聚合 JSON 使用固定 schema `gold-bottleneck-attribution-v1`，完整结构如下；所有字段必填，不允许额外键：

```text
schema_version: "gold-bottleneck-attribution-v1"
source_run_id: <固定格式 run ID>
source_git_sha: <40 位 Git SHA>
input_hashes:
  gold_sha256: <64 位 SHA-256>
  identifier_map_sha256: <64 位 SHA-256>
  executions_sha256: <64 位 SHA-256>
  business_results_sha256: <64 位 SHA-256>
  gates_sha256: <64 位 SHA-256>
  run_sha256: <64 位 SHA-256>
counts:
  query_count: <非负整数>
  raw_gold_identifier_count: <非负整数>
  normalized_query_work_count: <非负整数>
  unique_work_count: <非负整数>
  doi_work_count: <非负整数>
  openalex_work_count: <非负整数>
availability:
  available: <非负整数>
  exact_not_found: <非负整数>
  unknown_transient: <非负整数>
  invalid_identifier: <非负整数>
  integrity_failure: <非负整数>
pipeline_stages:
  selected_top50: <非负整数>
  ranked_outside_top50: <非负整数>
  filtered_out: <非负整数>
  not_retrieved: <非负整数>
cross_tab:
  <每个 availability 状态>:
    <每个 pipeline_stages 状态>: <关联数，非负整数>
query_coverage:
  <每个 availability 状态>:
    <每个 pipeline_stages 状态>: <不同查询数，0..query_count>
usage:
  unique_requests_planned: <非负整数>
  http_attempts: <非负整数>
  retries: <非负整数>
  http_200: <非负整数>
  http_404: <非负整数>
  http_429: <非负整数>
  http_5xx: <非负整数>
  timeouts: <非负整数>
  ledger_checkpoint_before_sha256: <64 位 SHA-256>
  ledger_checkpoint_after_sha256: <64 位 SHA-256>
diagnostic_complete: <布尔值>
recommended_direction: <允许的方向枚举或 null>
reason_codes: <允许的原因码数组，去重并按词典序排列>
```

`cross_tab` 和 `query_coverage` 的第一层必须恰好包含五个 availability 状态，第二层必须恰好包含四个 pipeline 状态。允许的 `recommended_direction` 只有：

- `new_data_source_probe`；
- `retrieval_query_evolution_probe`；
- `hard_filter_diagnosis`；
- `selector_rerank_offline`。

允许的 `reason_codes` 只有：

- `exact_not_found_dominant`；
- `available_not_retrieved_dominant`；
- `available_filtered_out_dominant`；
- `available_ranked_out_dominant`；
- `unknown_transient_present`；
- `invalid_identifier_present`；
- `integrity_failure_present`；
- `largest_bucket_tie`；
- `no_recoverable_loss`。

计数必须满足：availability 合计等于 `unique_work_count`；pipeline 与 `cross_tab` 合计均等于 `normalized_query_work_count`；DOI、OpenAlex 与 `invalid_identifier` 数之和等于 `unique_work_count`；HTTP 状态与超时计数之和等于 `http_attempts`。不得接受或透传任意 provider 字段。

## 错误处理与隐私

- 404 与瞬时失败严格分开；
- 401/403、预算不足、输入变化和响应完整性错误为全局失败；
- 429、超时和 5xx 只做有限退避重试，耗尽后为 `unknown_transient`；
- JSON 使用固定 schema、原子写入、`allow_nan=False`；
- 输出先由白名单 schema 构造，再递归拒绝禁用键名 `query_id`、`paper_id`、`title`、`request_id`、`response`；
- 值级隐私检查只允许数字、布尔值、`null`、固定枚举/原因码、schema 名、run ID、Git SHA 和 SHA-256；任何 DOI、arXiv/OpenAlex ID、URL、自由文本或非白名单字符串都阻止写文件；
- 标准输出只打印 schema、完成状态、三套总数和推荐方向；
- 不读取、打印或提交 `.env`；不得把密钥写入 URL、日志或产物。

## TDD 与验收

先用合成标识符和伪 HTTP transport 写失败测试，再实现：

- 143/139/134 三口径去重规则；
- DOI 与 OpenAlex ID 请求规划；
- 200 精确匹配、404、429/5xx/超时重试、认证失败和响应不匹配；
- 五个 availability 状态与四个流水线状态的互斥性；
- 集合不变量和输入哈希变化 fail closed；
- availability 结果在重复查询关联间复用；
- 交叉表总数守恒；
- 完整/不完整诊断的推荐规则；
- 聚合 JSON 的键白名单、禁用字段扫描和值级隐私校验。

完成标准：

1. 聚焦测试、全量 pytest 与 Ruff 通过；
2. mypy 不新增任务相关错误；
3. 正式流水线的 60/60 查询、139 个归一化关联和 134 篇唯一论文精确重建；
4. 在线请求不超过 134 个唯一论文的必要请求加有限重试；
5. 仅在诊断完整时给出一个非并列推荐方向；
6. 不运行 live capture，不修改生产排序或候选锁。
