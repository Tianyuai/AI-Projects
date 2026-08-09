# Integrity and Typing Stabilization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不改动生产检索与排序的前提下，把 OpenAlex 完整性失败细分为可行动的隐私安全证据，清零既有 15 个 mypy 错误，并完成一次有限在线复验。

**Architecture:** 保留现有五类 availability 状态，只在聚合报告中增加固定的“失败原因 × 标识符类型”计数，并将报告升级为 v2。诊断仍只在内存保存逐论文结果；在线重跑使用唯一账本运行 ID，随后原子写出并重新读取验证产物。四个旧文件的 mypy 修复独立完成，只做类型收窄和变量命名修正，不改变业务行为。

**Tech Stack:** Python 3.11、httpx、Pydantic v2、SQLiteBudgetLedger、pytest、Ruff、mypy strict。

## Global Constraints

- 工作区为 `D:\AI Projects\.worktrees\week3`，保留现有未跟踪的 `data/budget_ledger.sqlite3` 和 `deliverables/`。
- 在线阶段只运行一次 134 篇 gold 的 OpenAlex exact-ID 诊断；不运行 readiness、capture、replay、compare、validation，不重建候选锁。
- 仅从 `D:\AI Projects\Projects\.env` 临时加载连续的 `OPENALEX_API_KEY` 至 `OPENALEX_API_KEY_7`，不得打印、写盘或提交密钥。
- 不持久化论文 ID、查询文本、标题、URL、原始响应或 request ID；报告只允许固定枚举、计数和哈希。
- 不修改生产 provider、过滤、融合、排序、历史 run、`data/manifest.json` 或 `runs/candidate.lock.yaml`。
- 严格执行 RED → GREEN；在线调用必须等离线测试、Ruff 和目标 mypy 全部通过后才能开始。

---

### Task 1: 完整性失败可观测性与报告闭环

**Files:**
- Modify: `scripts/analyze_gold_bottlenecks.py`
- Modify: `tests/scripts/test_analyze_gold_bottlenecks.py`

**Interfaces:**
- `IntegrityFailureReason`: `missing_expected_field | unparseable_identifier | canonical_mismatch`。
- `ProbeBatch.integrity_reason_by_work`: 仅存活于进程内；非失败记录不得带原因。
- 报告新增 `integrity_failure_breakdown[reason][doi|openalex]`，合计必须等于 `availability.integrity_failure`。
- schema 升级为 `gold-bottleneck-attribution-v2`；`assert_safe_report` 负责固定键、计数守恒和序列化重读校验。

- [ ] **Step 1: 写 RED 测试**

  增加三个 200 响应分类测试、非法 reason/status 组合测试、breakdown 守恒测试、JSON 写盘重读测试，以及唯一账本运行 ID 测试。

- [ ] **Step 2: 运行 RED**

  Run: `D:\AI Projects\Projects\.venv\Scripts\python.exe -m pytest tests/scripts/test_analyze_gold_bottlenecks.py -q`

  Expected: 新增测试因缺少原因分类、v2 字段或唯一运行 ID 而失败。

- [ ] **Step 3: 最小实现**

  将布尔 `_response_matches` 改为返回状态和失败原因的纯分类函数；聚合固定矩阵并补齐 schema/隐私/守恒校验；诊断运行 ID 加入 UTC 微秒时间戳。不得增加原始响应日志或第二数据源。

- [ ] **Step 4: 运行 GREEN 与静态检查**

  Run: `D:\AI Projects\Projects\.venv\Scripts\python.exe -m pytest tests/scripts/test_analyze_gold_bottlenecks.py -q`

  Run: `D:\AI Projects\Projects\.venv\Scripts\python.exe -m ruff check scripts/analyze_gold_bottlenecks.py tests/scripts/test_analyze_gold_bottlenecks.py`

  Run: `D:\AI Projects\Projects\.venv\Scripts\python.exe -m mypy src scripts/analyze_gold_bottlenecks.py`

  Expected: 聚焦测试与 Ruff 通过；mypy 仅报告 Task 2 已冻结的 15 个旧错误，诊断脚本无新增错误。

### Task 2: 清零 15 个既有 mypy 错误

**Files:**
- Modify: `src/paper_search/query/parser.py`
- Modify: `src/paper_search/application/readiness.py`
- Modify: `src/paper_search/retrieval/snapshot_adapters.py`
- Modify: `src/paper_search/llm/snapshot_adapters.py`

**Interfaces:** 不新增公共行为；只明确已有 Literal、HTTP 参数、可空 snapshot ref 和局部变量作用域。

- [ ] **Step 1: 保留当前 RED 证据**

  Run: `D:\AI Projects\Projects\.venv\Scripts\python.exe -m mypy src scripts/analyze_gold_bottlenecks.py`

  Expected: 恰好 15 errors / 4 files。

- [ ] **Step 2: 分文件做最小修复**

  - parser：避免同作用域变量重定义，并把模型返回值先保持为 `object` 后再收窄；
  - readiness：从定义模块导入 `DependencyStatus`，为 capability/state 和 HTTP params 使用准确类型；
  - retrieval replay：只有 `SnapshotRef` 非空时才加入 provenance；
  - LLM capture：三个分支使用独立的安全 header 局部变量名。

- [ ] **Step 3: 运行针对性回归和 GREEN**

  Run: `D:\AI Projects\Projects\.venv\Scripts\python.exe -m pytest tests/unit/test_query_parser.py tests/application/test_readiness.py tests/unit/test_retrieval_snapshot_adapters.py tests/unit/test_llm_snapshot_adapters.py -q`

  Run: `D:\AI Projects\Projects\.venv\Scripts\python.exe -m mypy src scripts/analyze_gold_bottlenecks.py`

  Expected: 测试通过，mypy 为 `Success: no issues found`。

### Task 3: 一次在线复验、证据更新与最终验证

**Files:**
- Modify: `docs/evidence/gold-bottleneck-attribution-2026-08-09.json`
- Modify: `docs/gold-bottleneck-attribution-2026-08-09.md`
- Modify: `docs/experiment-decisions.md`
- Modify: `docs/retrieval-roadmap.md`
- Modify: `HANDOFF.md`

- [ ] **Step 1: 在线前离线门禁**

  重新运行 Task 1 聚焦测试、目标 mypy 和 Ruff；任一失败即停止，不发起网络请求。

- [ ] **Step 2: 运行唯一一次 exact-ID 在线诊断**

  在单个子进程中从获准 `.env` 只注入 7 个 OpenAlex key，使用正式 ledger 和冻结 source run，覆盖写出 v2 JSON/Markdown；不回显环境变量。

- [ ] **Step 3: 验证并解释产物**

  重新读取 JSON，通过 `assert_safe_report`，确认 60/143/139/134、HTTP 尝试上限、账本前后 checkpoint、breakdown 守恒，并根据失败原因更新 Markdown 和交接文档。若仍有完整性失败，保持 `diagnostic_complete=false`，不擅自选择提升方向。

- [ ] **Step 4: 最终全量验证**

  Run: `D:\AI Projects\Projects\.venv\Scripts\python.exe -m pytest -q`

  Run: `D:\AI Projects\Projects\.venv\Scripts\python.exe -m ruff check .`

  Run: `D:\AI Projects\Projects\.venv\Scripts\python.exe -m mypy src scripts/analyze_gold_bottlenecks.py`

  Expected: pytest、Ruff、目标 mypy 全部通过；Git 差异只包含本计划列出的文件和既有未跟踪产物。
