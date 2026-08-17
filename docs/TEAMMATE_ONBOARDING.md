# 队友上手

1. 拉取最新 `main`，执行 `uv sync --locked`。
2. 运行 `uv run --no-sync --no-env-file pytest -q` 验证环境。
3. 从 `main` 创建个人功能分支，只修改一个明确模块。
4. 默认使用 replay 开发；live 模式需本地 `.env` 和明确网络授权。
5. 私有数据、运行产物和临时分析代码不得提交。

涉及哈希绑定的冻结数据时，不要在旧 checkout 中只执行 `git pull --ff-only`；请创建全新 clone 或全新 worktree 后再验证。

当前主链路与模块边界见 `docs/architecture/current-system.md`。
