# VivaAI 可提交评测代码包

统一包名：`vivaai-paper-search-evaluator-runtime`。

运行 `scripts/build_evaluator_release.py` 后，本目录生成：

- `vivaai-paper-search-evaluator-runtime.zip`：可独立安装、运行与验证的评委代码包；
- `vivaai-paper-search-evaluator-runtime.zip.sha256`：ZIP 的外部 SHA-256；
- `vivaai-paper-search-evaluator-runtime/`：本地自检用解压等价目录（不提交 Git）；
- 目录内 `MANIFEST.sha256`：逐文件清单与 SHA-256；
- 目录内 `RELEASE.json`：生产锁、模型选择、默认/回退模型与文件数量元数据。

代码包只收录统一应用服务、UI/API/CLI/JSONL 入口、必要依赖与配置、生产锁实际绑定的模型/PASA 制品，以及合成的零网络 replay 样例。不收录 `.env`、密钥、缓存、运行目录、私有训练数据、Gold/隐藏/最终测试数据或未授权外部回执。

完整使用说明见包内 `docs/judge-guide.md`；从本仓库重新构建并验证：

```powershell
uv run --no-sync --no-env-file python scripts/build_evaluator_release.py
uv run --no-sync --no-env-file python scripts/verify_evaluator_release.py `
  --release-root deliverables/submission/vivaai-paper-search-evaluator-runtime
```
