# VivaAI 评委操作手册

本文从一次全新 GitHub 拉取开始，覆盖安装、零网络验证、浏览器 UI、HTTP API、单条在线查询、JSONL 测试集调度、live/replay 以及未来生产排序器升级。所有命令均在仓库根目录运行，除非步骤明确要求切换目录。

## 1. 拉取与安装

运行环境固定为 Python 3.11（不支持 3.12）。推荐使用 `uv` 按锁文件安装：

```powershell
git clone https://github.com/Tianyuai/AI-Projects.git
Set-Location AI-Projects
uv python install 3.11
uv sync --locked
uv run --no-sync --no-env-file python -m paper_search.health
```

`uv sync --locked` 不允许安装器擅自改写依赖锁；后续命令使用 `--no-sync --no-env-file`，既不会临时改变环境，也不会自动读取本地 `.env`。

若收到的是 ZIP，解压后进入唯一的 `vivaai-paper-search-evaluator-runtime` 目录，执行相同的 `uv sync --locked`。随后先验证文件没有缺失或改动：

```powershell
uv run --no-sync --no-env-file python scripts/verify_evaluator_release.py --release-root .
```

输出中的 `valid` 必须为 `true`。校验器同时检查 `MANIFEST.sha256`、生产锁、模型选择以及所有被绑定制品的 SHA-256。

## 2. 零网络完整回放

仓库附带一条合成查询及确定性供应商响应，不含密钥、私有训练数据、Gold 标签、隐藏测试查询或外部生产回执：

```powershell
uv run --no-sync --no-env-file python scripts/run_evaluator_package.py `
  --queries examples/safe-replay/queries.jsonl `
  --output runs/safe-replay/predictions.jsonl `
  --lock examples/safe-replay/replay.lock.yaml `
  --snapshot-manifest examples/safe-replay/snapshots/smoke/snapshot-manifest.json `
  --artifact-root examples/safe-replay `
  --capture-output-root runs/safe-replay/captures `
  --mode replay
```

成功时摘要包含 `mode: replay`、查询数和论文 ID 数，预测写入 `runs/safe-replay/predictions.jsonl`。Replay 与请求身份严格绑定；修改样例查询文字会安全失败，不会错误返回其他查询的缓存。

## 3. API 配置

Live 只从当前进程环境读取凭据，绝不要求把密钥提交到仓库。PowerShell 示例：

```powershell
$env:LLM_API_KEY = '<评测方提供的 LLM 密钥>'
$env:OPENALEX_API_KEY = '<可选的 OpenAlex 密钥>'
$env:OPENALEX_API_KEY_2 = '<可选的第二个 OpenAlex 密钥>'
$env:OPENALEX_MAILTO = '<可选的联系邮箱>'
$env:SEMANTIC_SCHOLAR_API_KEY = '<可选的 Semantic Scholar 密钥>'
```

变量说明：

| 变量 | Live 是否必需 | 用途 |
| --- | --- | --- |
| `LLM_API_KEY` | 是 | 生产锁指定的查询分析模型 |
| `OPENALEX_API_KEY` | 否但推荐 | OpenAlex 检索；无密钥时使用公开接口 |
| `OPENALEX_API_KEY_2`、`_3`… | 否 | 连续编号的备用 OpenAlex 密钥，空缺后停止读取 |
| `OPENALEX_MAILTO` | 否 | OpenAlex polite-pool 联系信息 |
| `SEMANTIC_SCHOLAR_API_KEY` | 否 | Semantic Scholar 检索配额 |

`.env.example` 仅列出变量名称。评委入口使用 `--no-env-file`，所以应像上面一样设置当前进程环境；不要把真实值写入仓库。模型名称、提示词、预算、价格和 API 端点由生产锁固定，不能通过环境变量绕过锁。

## 4. 单条 query 在线测试

`examples/evaluator/queries.jsonl` 是格式示例。若只运行一条查询，新建 UTF-8 文件 `single-query.jsonl`：

```json
{"query_id":"online-001","query":"Find recent papers on retrieval-augmented generation for scientific question answering."}
```

执行 live，并立刻对刚捕获的响应做零网络精确回放：

```powershell
uv run --no-sync --no-env-file python scripts/run_evaluator_package.py `
  --queries single-query.jsonl `
  --output runs/online-single/predictions.jsonl `
  --lock deliverables/evaluator/live-evaluator.lock.yaml `
  --artifact-root . `
  --capture-output-root runs/online-single/captures `
  --mode live `
  --verify-replay
```

启动前会校验生产锁及 F5/F4/B0、监督词法桥、PASA 别名和配置哈希。成功查询被原子封存；`--verify-replay` 要求 live 与 replay 的最终论文 ID 及顺序完全一致。任一查询失败时整批失败关闭，不伪造或错位输出。

## 5. 直接运行公开集或隐藏集 JSONL

输入是 UTF-8 JSONL，每行只能包含一个唯一 `query_id` 和 `query`：

```json
{"query_id":"judge-001","query":"Find papers about scientific document retrieval."}
```

将第 4 节命令中的 `--queries` 换成评测文件即可。输出保持输入顺序，每行形如：

```json
{"query_id":"judge-001","selected_paper_ids":["arxiv:...","doi:...","openalex:..."]}
```

可单独复核提交合同：

```powershell
uv run --no-sync --no-env-file python scripts/validate_evaluator_submission.py `
  --queries <queries.jsonl> `
  --predictions <predictions.jsonl>
```

该检查会拒绝非法 UTF-8、空行、未知字段、重复或错序 query、缺失预测和重复论文 ID。测试集不应包含 Gold 标签，系统也不会读取隐藏答案。

## 6. UI 与 HTTP API

### 6.1 零网络 UI

安全样例的制品根目录是 `examples/safe-replay`，因此在该目录启动服务：

```powershell
Push-Location examples/safe-replay
uv run --project ../.. --no-sync --no-env-file paper-search serve `
  --lock replay.lock.yaml `
  --mode replay `
  --snapshot-manifest snapshots/smoke/snapshot-manifest.json `
  --capture-output-root ../../runs/safe-replay-ui `
  --host 127.0.0.1 `
  --port 8000
```

浏览器打开 `http://127.0.0.1:8000/`，模式选择 `replay`，输入精确查询 `resource-aware scholarly paper search`。页面会显示排序结果、查询理解、检索与去重过程、预算/费用、来源与诊断信息。按 `Ctrl+C` 停止后执行 `Pop-Location`。

### 6.2 在线 UI

先按第 4 节完成一次 live 查询，以获得可验证的 replay 基座。另开 PowerShell，在仓库根目录选择刚生成的封存并启动统一服务：

```powershell
$capture = Get-ChildItem runs/online-single/captures -Directory -Filter 'evaluator-*' |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1
uv run --no-sync --no-env-file paper-search serve `
  --lock "$($capture.FullName)\replay.lock.yaml" `
  --mode replay `
  --snapshot-manifest "$($capture.FullName)\snapshot-manifest.json" `
  --capture-output-root runs/online-single/captures `
  --allow-live `
  --host 127.0.0.1 `
  --port 8000
```

`--allow-live` 只在所给 replay 锁能追溯到 `capture-output-root` 内真实封存时生效。UI 中可在 `live` 与 `replay` 间切换；两种模式仍走同一个 `/v1/search` 和应用服务。

### 6.3 HTTP 调用

就绪检查：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health/ready
```

单条 API 请求：

```powershell
$body = @{
  query_id = 'api-001'
  query = 'resource-aware scholarly paper search'
  budget_profile = 'balanced'
  include_trace = $true
  mode = 'replay'
} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/v1/search `
  -ContentType 'application/json' -Body $body
```

对已启动的服务调度整份 JSONL：

```powershell
uv run --no-sync --no-env-file python scripts/run_evaluator_batch.py `
  --queries <queries.jsonl> `
  --output <predictions.jsonl> `
  --base-url http://127.0.0.1:8000 `
  --mode live
```

批处理按输入顺序串行调用统一 API，以控制供应商限流、成本和结果对应关系。安全样例或已封存查询可改为 `--mode replay`；未知查询不能由旧快照回答。

## 7. 代码包生成与验证

从 GitHub 克隆后可重建相同结构的提交包：

```powershell
uv run --no-sync --no-env-file python scripts/build_evaluator_release.py
uv run --no-sync --no-env-file python scripts/verify_evaluator_release.py `
  --release-root deliverables/submission/vivaai-paper-search-evaluator-runtime
```

发布器使用显式白名单，只收录运行代码、UI、必要配置、文档、安全 replay，以及生产锁/模型选择实际引用且哈希匹配的制品。`.env`、缓存、临时运行目录、数据库、私有训练数据、隐藏/最终测试数据和外部回执均不会进入包。ZIP 可重复生成；目录内清单覆盖每个文件。

## 8. 生产 F5 后续替换

扩充训练完成后不要覆盖现有模型目录，也不要修改 UI/API/CLI/批处理代码。采用以下稳定发布方式：

1. 将新 F5 与需要的 F4 回退写入新的版本化目录，并生成各自 `manifest.json` 与权重哈希。
2. 通过既有晋升流程更新 `artifacts/models/production-document-ranker-selection.json`，仍保持 `F5 → F4 → B0` 的失败回退语义；禁止按 query 动态选模。
3. 使用 `scripts/bind_production_document_ranker.py` 生成新的候选生产锁，完成既有门禁后再替换 `deliverables/evaluator/live-evaluator.lock.yaml`。不要手工改写旧锁或评测结论。
4. 运行完整测试、live 模拟、replay、自检与敏感信息扫描，再重新生成提交 ZIP 和校验值。

发布器会从“生产锁 + 模型选择”动态发现所需制品，因此模型版本改变时不需要重建项目架构。旧版本化目录可以保留用于审计与回滚；Git 发布只纳入新生产锁实际绑定的版本。

## 9. 常见失败

- `LLM_API_KEY is required`：仅 live 需要；在当前 PowerShell 设置变量后重试。
- `hash mismatch`：文件字节与锁不一致。重新拉取，不要编辑模型、锁或被绑定配置。
- `snapshot unavailable/request mismatch`：replay 的 lock、manifest 与 query 不属于同一封存；未知 query 必须 live。
- `/health/ready` 非 200：不要开始批测；检查终端中的锁、快照和制品路径。
- Python 版本不符：运行 `uv python install 3.11` 后再次 `uv sync --locked`。
