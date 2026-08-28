# VivaAI 公开集/隐藏集查询与提交说明

本说明定义一套不依赖 Gold 标签的最小交换格式。公开测试集与隐藏测试集使用完全相同的输入、运行和输出流程；评委无需操作 UI，也无需了解项目内部训练数据。

## 1. 文件约定

评委提供 UTF-8 JSONL 查询文件，每行一个对象，且只包含：

```json
{"query_id":"public-001","query":"Find papers on retrieval-augmented generation for scientific question answering."}
```

系统生成 UTF-8 JSONL 预测文件，每行一个对象：

```json
{"query_id":"public-001","selected_paper_ids":["arxiv:2501.10120","openalex:W1234567890"]}
```

约束如下：

1. `query_id` 必须非空且在输入文件中唯一。
2. 输入不包含答案、Gold 论文或数据集分区名称。
3. 输出与输入必须一一对应并保持相同顺序。
4. `selected_paper_ids` 按相关性从高到低排列，不允许重复。
5. 论文标识优先使用 `arxiv:`，没有 arXiv ID 时使用 `doi:` 或 `openalex:`。
6. 单条查询失败时批处理失败关闭，不生成伪造或错位预测。

可直接查看 [查询样例](../examples/evaluator/queries.jsonl) 和 [预测格式样例](../examples/evaluator/predictions.jsonl)。预测样例仅用于演示文件格式，不代表样例查询的正式检索答案。

## 2. 推荐：单命令运行

评委不需要先启动 HTTP 服务。公开集和隐藏集都可以直接执行：

```powershell
python scripts/run_evaluator_package.py `
  --queries <queries.jsonl> `
  --output <predictions.jsonl> `
  --lock deliverables/evaluator/live-evaluator.lock.yaml `
  --artifact-root . `
  --capture-output-root runs/evaluator `
  --mode live `
  --verify-replay
```

未知公开/隐藏查询使用 `live`。该命令在同一进程内完成锁与模型哈希校验、查询执行、预测写入和提交合同校验。每条成功查询还会在 `runs/evaluator/evaluator-*` 原子封存完整回执；`--verify-replay` 立即从该回执离线重放，并要求最终论文 ID 及顺序精确一致。

对已有封存执行离线 replay 时，改用该封存目录中的 `replay.lock.yaml` 与 `snapshot-manifest.json`：

```powershell
python scripts/run_evaluator_package.py `
  --queries <该封存对应的查询.jsonl> `
  --output <replay-predictions.jsonl> `
  --lock <capture-dir/replay.lock.yaml> `
  --snapshot-manifest <capture-dir/snapshot-manifest.json> `
  --artifact-root . `
  --capture-output-root runs/evaluator-replay `
  --mode replay
```

Replay 是严格请求级重放，只能运行该快照实际捕获过的 query；未知隐藏 query 必须走 live。历史快照如与当前检索请求协议不同，也不会被伪装成当前版本通过。

## 3. 可选：启动 UI 与 HTTP 服务

服务始终绑定一个可复现 replay 快照；需要在线查询时额外开放 live。API 密钥通过本地环境变量提供，不写入提交文件。

```powershell
paper-search serve `
  --lock <replay.lock.yaml> `
  --mode replay `
  --snapshot-manifest <snapshot-manifest.json> `
  --capture-output-root <capture-output-directory> `
  --allow-live `
  --host 127.0.0.1 `
  --port 8000
```

就绪检查：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health/ready
```

只有返回 HTTP 200 和 `status=ready` 后才开始评测。浏览器访问 `http://127.0.0.1:8000/` 可使用 UI 做人工演示；自动评分不依赖 UI。

## 4. 对已启动服务运行批量查询

```powershell
python scripts/run_evaluator_batch.py `
  --queries examples/evaluator/queries.jsonl `
  --output predictions.jsonl `
  --base-url http://127.0.0.1:8000 `
  --mode live
```

批处理客户端按输入顺序逐条调用 `/v1/search`，默认单条超时 180 秒。顺序执行便于控制供应商限流、API 次数和成本，也让公开集与隐藏集运行行为一致。

如评委提供的是已经捕获并随包发布的查询快照，可将 `--mode live` 改为 `--mode replay`；未知隐藏查询通常使用 live。

## 5. 提交前校验

```powershell
python scripts/validate_evaluator_submission.py `
  --queries examples/evaluator/queries.jsonl `
  --predictions examples/evaluator/predictions.jsonl
```

成功时输出查询数、预测数和返回论文 ID 总数。以下情况会直接失败：非法 UTF-8、空行、未知字段、重复 `query_id`、预测缺失、预测顺序错误或单条结果中的重复论文 ID。

## 6. Live/Replay 交付演练

推荐在单命令 live 批测中直接加入 `--verify-replay`，它会对每条新回执执行最严格的逐条一致性检查，无需人工管理两个服务。

如需验证已经分别启动的两个服务，可运行：

```powershell
python scripts/run_delivery_rehearsal.py `
  --queries <frozen-queries.jsonl> `
  --output-dir <rehearsal-output> `
  --live-base-url http://127.0.0.1:8000 `
  --replay-base-url http://127.0.0.1:8001
```

只有所有 query 的最终论文 ID 及顺序完全一致时才生成 `passed=true` 报告；任何错位或排序差异都会失败关闭。两个服务的 replay 端必须包含相同查询对应的当前协议快照。

## 7. 评委最短操作流程

1. 解压项目与运行环境。
2. 将公开或隐藏查询保存为 `queries.jsonl`。
3. 配置评测方持有的 API 密钥。
4. 运行带 `--verify-replay` 的 `run_evaluator_package.py`，生成并校验 `predictions.jsonl`，同时完成每条查询的 live/replay 演练。
5. 将 `predictions.jsonl` 交给官方 scorer；需要人工查看时再启动 UI。

## 8. 当前生产方法

生产锁固定采用 `统一上下文多特征融合 F5 → 可靠性融合 F4 → 基础排序 B0`：F5 是默认排序器，F4 是制品或初始化失败时的第一性能回退，B0 是紧急回退。禁止按单条 query 动态选择模型。当前 F5 使用 18,314 条冻结训练查询完成三折 OOF，在全量汇总上优于 F4/B0；538 条独立冻结 auto_dev 的各召回截断点均不低于 B0，live/replay 538/538 完全一致，test 分区未触碰。

评委从全新 GitHub 拉取后的安装、API 环境变量、UI 操作、零网络安全样例和未来 F5 版本化替换，见 `docs/judge-guide.md`。生产版本变化时，批量输入输出合同和统一 `/v1/search` 服务入口保持不变。
