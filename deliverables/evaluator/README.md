# VivaAI 评委运行入口

评委只需准备 UTF-8 JSONL 查询文件：

```json
{"query_id":"judge-001","query":"Find papers about scientific document retrieval."}
```

未知公开集或隐藏集执行：

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

`live` 使用评测方配置的 API 密钥；每条成功查询都会原子封存自己的 `replay.lock.yaml` 与 `snapshot-manifest.json`。`--verify-replay` 会立即离线重放该新回执，并要求最终论文 ID 及顺序精确一致。程序还会自动验证输入顺序、输出覆盖、重复论文 ID、生产模型哈希和 test 隔离；任何查询失败都会终止整批，不生成伪造结果。

已有封存查询可单独使用 `--mode replay`，但 lock、manifest 和 query 必须来自同一次封存；replay 不能回答快照中从未执行过的未知查询。

生产排序顺序固定为 F5、F4 reliability、B0。回退只允许发生在模型制品或初始化失败时，禁止依据单条查询结果切换模型。

完整格式、HTTP/UI 模式和 live/replay 演练见 `docs/evaluator-submission-quickstart.md`。
