# Task 3 OpenAlex 检索、缓存和标准化设计

**日期：** 2026-07-16
**分支：** `codex/task3-openalex`
**状态：** 已批准，待实施计划
**范围：** PRD 第一周 Task 3 的 OpenAlex 搜索链路

## 1. 目标

实现一条可离线回放、受预算约束、可生成不可变实验快照的 OpenAlex 搜索链路：输入自然语言查询和年份过滤后，系统返回统一的 `Paper` 列表以及完整的 `ProviderResult` 用量、来源和错误信息。

本阶段优先支撑 Task 4 和第一周阶段闸门，不提前实现引文关系图。

## 2. 范围

### 2.1 本阶段实现

- OpenAlex `/works` 搜索；
- 字段裁剪、年份过滤、每页和总结果上限；
- cursor 分页；
- SQLite 原始响应缓存、TTL 和 429 冷却；
- 有限重试、指数退避、抖动和错误分类；
- OpenAlex Work 到统一 `Paper` 的标准化；
- 从本次运行实际使用的缓存条目导出不可变快照和 SHA-256 manifest；
- 固定 fixture 离线测试和可选在线烟测。

### 2.2 明确不实现

- `references` 和 `citations` 引文接口；
- Semantic Scholar；
- Task 4 的跨来源去重、硬过滤和 BM25 排序；
- UI；
- 通用多 Provider 框架；
- 关系图。

引文相关功能必须等第一周阶段闸门通过后再设计和实现。

## 3. 设计选择

采用分层实现：

1. `OpenAlexProvider` 编排 HTTP、缓存、分页、重试和预算；
2. `SQLiteResponseCache` 管理原始响应、TTL、冷却和快照；
3. `normalize_openalex_work()` 作为纯函数完成数据转换。

没有采用单体 Provider，因为 HTTP、SQLite 和标准化耦合后难以独立测试。没有提前抽象通用多 Provider 平台，因为当前只有 OpenAlex 搜索需求，提前抽象会扩大第一周范围。

## 4. 文件与职责

### 4.1 `src/paper_search/retrieval/openalex.py`

提供 `OpenAlexProvider`：

```python
class OpenAlexProvider:
    async def search(
        self,
        query: str,
        filters: dict[str, object],
        limit: int,
        reservation: BudgetReservation,
    ) -> ProviderResult[list[Paper]]: ...
```

构造函数注入：

- `httpx.AsyncClient`；
- `SQLiteResponseCache`；
- API Key；
- 时钟；
- 异步 sleep；
- 抖动生成器；
- 缓存版本。

注入时钟、sleep 和抖动是为了让 TTL、冷却和退避测试不依赖真实等待。

### 4.2 `src/paper_search/storage/cache.py`

提供 `SQLiteResponseCache`，职责包括：

- 规范化请求参数并生成稳定缓存键；
- 保存成功的原始响应和安全响应元数据；
- 判断 7 天 TTL；
- 保存并判断 429 的 60 秒冷却截止时间；
- 返回本次命中或刚写入的缓存条目标识；
- 按有序缓存键导出正式快照和 manifest；
- 对不同内容的已有正式快照拒绝覆盖。

缓存键输入固定为：

```text
provider + endpoint + canonical_non_secret_params + cache_version
```

`api_key` 必须在规范化参数、缓存键、数据库、日志、provenance 和快照中删除。

### 4.3 `src/paper_search/processing/normalize.py`

提供：

```python
def normalize_openalex_work(raw_work: Mapping[str, object]) -> Paper: ...
```

它不读取缓存、不访问网络、不依赖系统时间。输入非法时抛出明确的 `ValueError`，由 Provider 将单条坏记录转换为结构化错误并继续处理其他记录。

## 5. 请求契约

请求使用当前 OpenAlex 官方接口：

```text
GET https://api.openalex.org/works
```

官方文档：

- Works API：https://developers.openalex.org/api-reference/works/list-works
- 过滤：https://developers.openalex.org/guides/filtering
- 错误处理：https://developers.openalex.org/api-reference/errors

请求参数：

- `api_key`：只发送给 OpenAlex，不进入任何持久化数据；
- `search`：非空查询；
- `filter`：由允许的项目过滤器生成；
- `select`：只请求标准化需要的字段；
- `per_page`：`min(50, remaining_limit)`；
- `cursor`：第一页为 `*`，后续使用响应中的 `next_cursor`。

固定 `select` 字段为：

```text
id,doi,title,display_name,abstract_inverted_index,authorships,
publication_year,primary_location,cited_by_count,is_retracted
```

年份转换规则：

- `year_from` → `from_publication_date:YYYY-01-01`；
- `year_to` → `to_publication_date:YYYY-12-31`；
- 两者都有时用逗号连接；
- 未知过滤字段立即抛出 `ValueError`，不向外部 API 透传。

本阶段允许的 `filters` 键只有 `year_from` 和 `year_to`。年份复用项目现有的 `1900..current year + 1` 约束。

查询必须非空，`limit` 必须为 `1..300`。单个子查询默认由调用方传入不超过 50；300 是 PRD 原始候选硬上限。

## 6. 分页、预算和用量

每次实际 HTTP 尝试都消耗一个 `search_api_calls` 额度，包括：

- 首次请求；
- 后续分页；
- 429、5xx 或超时后的重试。

Provider 只可使用 `reservation.reserved.search_api_calls`。当前 `BudgetController` 没有子预留接口，因此 Task 3 在 Provider 内部维护已消费计数；达到预留上限后不得继续请求。

缓存命中和 429 冷却命中不产生外部调用，`UsageActual.search_api_calls` 为 0。

多页搜索在以下任一条件满足时停止：

- 已获得 `limit` 条原始记录；
- `next_cursor` 为空；
- 预算耗尽；
- 当前页最终失败；
- 返回空页。

后续页失败时保留之前成功页面的数据，并在 `errors` 中说明停止原因。

## 7. 重试与错误分类

每页最多 3 次实际尝试：首次请求加最多 2 次重试。

运行时退避为带抖动的指数退避，测试中注入确定性抖动：

```text
delay = min(8.0, 2**retry_index) + jitter
```

错误分类：

| 情况 | code | retryable | 缓存 |
|---|---|---:|---:|
| 429 | `rate_limited` | true | 仅保存 60 秒冷却截止时间 |
| 500–599 | `server_error` | true | 否 |
| `httpx.TimeoutException` | `timeout` | true | 否 |
| 400 | `invalid_request` | false | 否 |
| 401/403 | `authentication_error` | false | 否 |
| 其他 4xx | `client_error` | false | 否 |
| 非法 JSON 或顶层结构 | `invalid_response` | false | 否 |
| 单条 Work 非法 | `invalid_work` | false | 成功页面仍可缓存 |
| 预留额度耗尽 | `budget_exhausted` | false | 不适用 |

错误不得包含 API Key、完整请求 URL 或原始受限凭据。若响应提供安全的请求 ID，则写入 `ErrorDetail.request_id`。

Provider 对正常的外部失败返回 `ProviderResult`，不让单个查询异常终止批量评测。调用方输入非法仍使用 `ValueError` 尽早失败。

## 8. 缓存模型

SQLite 至少保存：

- `cache_key`；
- provider；
- endpoint；
- canonical non-secret params JSON；
- cache version；
- HTTP status；
- raw response bytes；
- raw response SHA-256；
- safe response metadata JSON；
- requested_at；
- expires_at；
- cooldown_until。

成功搜索响应 TTL 固定为 7 天。过期记录保留到显式清理，但普通读取视为 miss。429 只保存冷却状态，不保存错误响应正文。

安全响应元数据只允许保存：

- `content-type`；
- `x-request-id`；
- `x-ratelimit-limit`；
- `x-ratelimit-remaining`；
- `x-ratelimit-credits-used`；
- `x-ratelimit-reset`。

缓存写入使用 SQLite 事务。缓存命中返回原始 bytes，由与在线响应相同的解析路径处理，保证离线回放行为一致。

## 9. 标准化规则

`normalize_openalex_work()` 执行以下映射：

| OpenAlex 字段 | `Paper` 字段 |
|---|---|
| `doi` | `doi`，去掉 URL 前缀并小写 |
| `id` | `openalex_id`，规范为 `W...` |
| `title`，回退 `display_name` | `title` |
| `abstract_inverted_index` | `abstract` |
| `authorships[*].author.display_name` | `authors` |
| `publication_year` | `publication_year` |
| `primary_location.source.display_name` | `venue` |
| `primary_location.landing_page_url`，回退 OpenAlex URL | `url` |
| `cited_by_count` | `citation_count` |
| `is_retracted` | `is_retracted` |
| 固定值 | `sources=["openalex"]` |

canonical ID 优先级：

1. 合法 DOI → `doi:<normalized-doi>`；
2. 否则合法 OpenAlex ID → `openalex:<W...>`。

标题为空或 DOI/OpenAlex ID 都无法形成稳定 ID 时，该记录无效。

倒排摘要按每个 token 的全部整数位置展开；位置可以不连续，但不得为负数、布尔值或重复位置。最终按位置排序并用单个空格连接。缺失或 `null` 摘要是合法情况，映射为 `None`。

## 10. ProviderResult 契约

`data` 是成功标准化的论文，按 OpenAlex 页面与页内顺序保留。Task 3 不做跨记录去重；去重属于 Task 4。

`usage`：

- `search_api_calls` 为实际 HTTP 尝试次数；
- 其他调用和 Token 字段为 0；
- `cost_cny=None`；
- `elapsed_ms` 为整个搜索耗时。

`provenance` 至少包含现有模型要求的五个字段：

- `provider=openalex`；
- `endpoint=/works`；
- `model_id=openalex-api`；
- `requested_at`；
- `response_hash`。

另包含 `cache_keys`：按页面顺序排列的缓存键 JSON 数组字符串。runner 使用该字段导出本次搜索实际引用的快照，避免 Provider 保存并发不安全的“最后一次搜索”状态。

单页 `response_hash` 为原始响应 SHA-256。多页时，按页序列化各页原始哈希后再计算聚合 SHA-256。没有成功页面时，使用空哈希列表的确定性 SHA-256。

`cache_hit` 只有在本次搜索的所有成功页面都来自缓存且没有外部请求时为 true。混合缓存/在线分页为 false。

`latency_ms` 与 `usage.elapsed_ms` 相同。`errors` 按发生顺序记录。

## 11. 不可变快照

快照导出输入为本次运行实际使用的有序缓存键集合和目标运行目录。输出：

```text
experiments/<run_id>/snapshots/openalex-0001.json
experiments/<run_id>/snapshots/openalex-0002.json
experiments/<run_id>/snapshot_manifest.json
```

每个快照文件保存原始响应 bytes，不重新格式化 JSON。manifest 使用稳定排序、UTF-8 和原子写入，记录：

- manifest contract version；
- provider；
- endpoint；
- canonical non-secret params；
- requested_at；
- response SHA-256；
- snapshot relative path；
- snapshot file SHA-256。

相同内容的重复导出是幂等操作。目标文件或 manifest 已存在但内容不同则拒绝覆盖。正式实验后续只能引用该 manifest 中的响应；Task 3 提供不可变产物和校验函数，Task 4 的 runner 负责强制指标引用关系。

## 12. 测试策略

### 12.1 固定 fixture

创建不含密钥的 OpenAlex JSON fixture：

- 完整成功页；
- 空结果页；
- 缺失摘要页；
- 第二页；
- 非法顶层结构；
- 包含单条非法 Work 的成功页；
- 429 与 5xx 错误正文。

### 12.2 `tests/unit/test_normalize.py`

覆盖完整映射、缺失摘要、倒排索引、canonical ID 优先级、标题缺失、稳定 ID 缺失和非法倒排位置。

### 12.3 `tests/unit/test_cache.py`

覆盖稳定键、参数顺序、密钥脱敏、7 天 TTL、429 冷却、过期 miss、事务写入、离线回放、快照稳定性、manifest 哈希和覆盖保护。

### 12.4 `tests/unit/test_openalex.py`

使用 `httpx.MockTransport` 或等价注入 transport 覆盖：

- search/select/filter/per_page/cursor；
- 空结果；
- 分页和 limit；
- 429、5xx、超时有限重试；
- reservation 消耗；
- 预算耗尽；
- 后续页失败的部分结果；
- 非法 JSON、非法顶层结构、单条坏记录；
- 缓存命中零调用；
- 冷却期零调用；
- API Key 不出现在错误、缓存和 provenance。

### 12.5 `tests/integration/test_openalex_live.py`

使用 `@pytest.mark.online`。只有进程环境提供 `OPENALEX_API_KEY` 时执行，不从本地敏感配置文件读取。固定运行 3 条无敏感内容查询，输出只包含状态、数量、延迟、哈希和脱敏快照信息。

没有 Key 时明确 skip。在线失败不得被离线 fixture 测试掩盖，但也不得阻止无凭据开发者运行默认测试。

## 13. 验收命令

```powershell
uv run pytest -m "not online" tests/unit/test_openalex.py tests/unit/test_cache.py tests/unit/test_normalize.py -v
uv run pytest -q
uv run ruff check .
uv run mypy src
```

有进程环境 Key 时额外运行：

```powershell
uv run pytest -m online tests/integration/test_openalex_live.py -v
```

验收成功标准：

- 离线 fixture 全部通过；
- 输入查询可返回统一 `Paper`；
- 有效缓存可在断网状态回放；
- HTTP 尝试不超过预留额度和每页 3 次上限；
- 快照和 manifest 可验证且不可变；
- 全量 pytest、Ruff 和 mypy 通过；
- 仓库、缓存、错误和快照中不存在 API Key。

## 14. PRD 状态更新规则

只有离线实现和相应测试实际通过后，才勾选 Task 3 的代码、fixture、缓存、快照、标准化、有限重试与离线测试条目。

真实 3 查询烟测和 online 测试只有实际使用进程环境 Key 成功运行并生成安全快照后才可勾选。没有 Key 或在线服务异常时保持未完成，并记录真实原因，不以离线测试替代。
