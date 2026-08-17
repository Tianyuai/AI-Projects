# 协作交接

稳定分支只承载已验证的端到端主基线和通用模块优化框架。新优化应从最新 `main` 建分支，避免把运行产物、私有数据、密钥或一次性分析脚本提交到仓库。

提交前至少运行：

```bash
uv run --no-sync --no-env-file pytest -q
uv run --no-sync --no-env-file ruff check .
```

影响运行配置、接口或评估契约时，同时更新对应测试与当前说明文档。
