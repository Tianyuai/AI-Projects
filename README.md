# Paper Search

面向学术论文检索的端到端系统。当前稳定面只包含主基线：查询分析、OpenAlex/Semantic Scholar 检索、去重与词法融合、预算控制、可回放快照、正式评估、API 与浏览器界面。

## 快速开始

```bash
uv sync --locked
uv run --no-sync --no-env-file pytest -q
uv run --no-sync --no-env-file paper-search serve --mode replay
```

打开 `http://127.0.0.1:8000/` 使用界面。实时模式需要在本地 `.env` 配置密钥，并显式授权网络访问；密钥与运行产物不会进入仓库。

```bash
uv run --no-sync --no-env-file python -m paper_search.health
```

模块优化入口位于 `paper-search recall`，仓库仅保留一个通用盲测方案及 replay/live 两种后端配置。私有数据集和实验产物由协作者在本地提供。

协作说明见 [docs/TEAMMATE_ONBOARDING.md](docs/TEAMMATE_ONBOARDING.md)，当前结构见 [docs/architecture/current-system.md](docs/architecture/current-system.md)。
