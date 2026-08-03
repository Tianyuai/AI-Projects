# 新对话恢复提示词

复制下面的整个代码块，粘贴到新开的 Codex 对话窗口并发送：

```text
你是本项目的接手 Codex 实现控制器。请把以下内容视为可信交接上下文，并从当前工作树继续，不要重新规划已完成工作。

【项目与分支】
- 工作树：D:\AI Projects\.worktrees\week3
- 当前分支：codex/project-document-handoff
- 当前基线 HEAD：8ce02a8（Phase 4 Task 4 初版提交）
- 当前工作树包含未提交的 Phase 4 Task 4 review-fix WIP；不要 reset、checkout、clean 或删除这些修改。
- 用户已有未跟踪文件也必须保留：.gate0-report.json、.sheet-build/、outputs/、docs/superpowers/plans/ 等。

【用户当前目标】
先依据模板 PDF `D:\AI Projects\赛题指南\附件2：第八届中国研究生人工智能创新大赛项目文档模版.pdf` 完成项目总结/文档写作，再继续 Phase 4 Task 4 剩余工程修复。模板章节是：
1 项目概况：1.1 背景和基础，1.2 场景和价值，1.3 所需支持；
2 项目规划：2.1 整体目标，2.2 技术创新点；
3 实施方案：3.1 技术可行性分析，3.2 技术细节，3.3 计划和分工；
4 参考资料；另有封面和更改历史。

【已完成且可写入文档的工程事实】
- Phase 3-C 最终提交 e076c8f；最终复核 Engineering PASS，C0/I0/M0。
- Phase 3-C 主门禁：157 passed、2 skipped；Ruff/mypy/diff-check 通过；真实 E3/E4 仍 Cannot verify/deferred。
- Phase 4 Task 1：d125da8 + c029f5c；typed FastAPI errors、validation 400、readiness、三重 live authorization、publication-before-200、mode consistency；复核 C0/I0/M0。
- Phase 4 Task 2：a8b2108 + 8f59777；浏览器 UI 通过 `/v1/search`，移除独立 evaluation UI pipeline；wheel/sdist 同时含 Python source 与 3 个 UI assets；复核 C0/I0/M0。
- Phase 4 Task 3：a5928bc..d663da0；serve、replay/live composition、source lineage、TOCTOU、SIGTERM、真实 fake-live success/failure/cancellation cleanup、sealed response provenance；focused 31 passed/2 skipped，smoke 34；复核 C0/I0/M0。
- Phase 4 Task 4 初版：8ce02a8；registry、flags、Provider/LLM async stages、evolution scaffolding、reservation fail-closed、baseline default-off 初版；尚未通过最终复核。

【Task 4 当前未关闭问题（必须先写 RED 再修生产）】
最终复核报告指出 C1/I3：
1. C1：`RuntimeConfig.experiment` 仍未真正驱动 production composition/evolution；composition 仍主要硬编码 baseline，named optional identities 不能通过 shared smoke/evaluate/API 正式运行。
2. I1：optional stage 的 typed Budget/Config/Snapshot/Integrity/Adapter failures 可能被 broad catch 降级为普通 degradation，必须原样传播或明确 fail-closed。
3. I2：optional citation/rerank snapshot refs 在 orchestrator 边界被解析后丢弃，未进入 OrchestratorResult diagnostics/SearchExecutionResult/formal evidence。
4. I3：`asyncio.CancelledError` 可能留下 active stage reservations；必须 finally close/fail reservations and clients。

【必须遵守的边界】
- Phase 4 Task 4 仍未验收，不要标记完成，不要进入 Task 5–8，直到同一审查者完整范围复核 C0/I0。
- main-baseline 必须保持 default-off、fixed-one-round、无 optional module construction；不要做 promotion。
- optional stages 必须 async、共享同一 request budget/controller/snapshot/evaluation path；禁止 `asyncio.run()` 嵌套。
- typed budget/config/snapshot/integrity/adapter errors 不得被吞掉；CancelledError 必须清理 reservation/client/artifact。
- 不做真实网络、真实 provider、真实成本、真实 Gate、浏览器 live 验收或公开状态更新；这些仍是 deferred/Cannot verify，除非用户单独授权。
- 保留所有现有用户未跟踪文件和未提交 WIP；不要 reset/checkout/clean。
- 所有实现先 RED→GREEN，再跑 focused/static，提交后由同一审查者复核完整 commit range。

【下一步严格顺序】
1. 读取并理解 `docs/superpowers/plans/2026-07-30-week1-4-phase4-api-ui-experiments.md` 的 Task 4。
2. 检查当前 `git status` 和未提交 WIP，确认不覆盖用户修改。
3. 为上述四个问题补 directed RED tests。
4. 修复 production：RuntimeConfig→validated experiment registry→CompositionRoot/orchestrator；typed error propagation；optional refs→diagnostics；CancelledError reservation cleanup。
5. 跑 Task 4 focused、adjacent config/composition、Ruff、mypy、diff-check，并提交新 commit。
6. 用同一审查者对 `8ce02a8..新提交` 做完整范围复核，直到 C0/I0；复核通过后才继续 Phase 4 Task 5–8。

【文本工作要求】
请先根据模板章节检查/完善 `docs/handoff/project-summary-for-competition-template.md`，只写有工程证据支持的内容；不要把 Cannot verify 写成已完成。文本完成后再回到 Task 4 工程修复。

收到本提示后，先用中文简短回报：当前 HEAD、工作树是否有 WIP、Phase 3/Phase 4 已完成边界、Task 4 四项未关闭问题；然后按上述顺序继续。
```
