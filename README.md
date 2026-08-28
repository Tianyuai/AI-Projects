# VivaAI 智能论文搜索与推荐

面向复杂学术查询的端到端论文检索系统。发布面包含浏览器 UI、HTTP API、命令行、JSONL 批量评测、在线 live 捕获与零网络 replay；这些入口统一调用同一个应用服务，避免演示与评测逻辑分叉。

当前生产锁固定默认排序器 F5，并绑定 F4、B0 回退制品、监督词法桥、PASA 身份别名、预算/价格/提示词配置及其 SHA-256。运行时会先校验制品，禁止按单条查询动态换模型，且不会读取或打包私有训练数据、Gold 标签或隐藏测试集。

## 评委最短验证

要求 Python 3.11 与 [uv](https://docs.astral.sh/uv/)。在 PowerShell 中：

```powershell
git clone https://github.com/Tianyuai/AI-Projects.git
Set-Location AI-Projects
uv python install 3.11
uv sync --locked
uv run --no-sync --no-env-file python scripts/run_evaluator_package.py `
  --queries examples/safe-replay/queries.jsonl `
  --output runs/safe-replay/predictions.jsonl `
  --lock examples/safe-replay/replay.lock.yaml `
  --snapshot-manifest examples/safe-replay/snapshots/smoke/snapshot-manifest.json `
  --artifact-root examples/safe-replay `
  --capture-output-root runs/safe-replay/captures `
  --mode replay
```

该样例完全离线，不需要任何 API 密钥。完整的安装、UI、API、单条在线查询、测试集 JSONL 调度、live/replay 与 F5 替换说明见 [评委操作手册](docs/judge-guide.md)。输入输出合同见 [批量提交说明](docs/evaluator-submission-quickstart.md)。

## 发布包自检

```powershell
uv run --no-sync --no-env-file python scripts/build_evaluator_release.py
uv run --no-sync --no-env-file python scripts/verify_evaluator_release.py `
  --release-root deliverables/submission/vivaai-paper-search-evaluator-runtime
```

生成的 ZIP、目录内 `MANIFEST.sha256` 和 `RELEASE.json` 共同给出精确文件清单、生产锁/模型选择绑定与校验值。仓库中的训练扩充脚本不在评委运行包内；未来模型晋升只需发布新的版本化制品、更新稳定选择文件并生成新的生产锁，无需改动 UI/API/CLI/JSONL 调度层。
