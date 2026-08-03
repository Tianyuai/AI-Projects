# 新对话恢复提示词

复制下面的整个代码块，粘贴到新开的 Codex 对话窗口并发送：

```text
你是本项目的接手 Codex 实现控制器。请把以下内容视为可信交接上下文，并从当前工作树继续，不要重新规划已完成工作。

【项目与分支】
- 工作树：D:\AI Projects\.worktrees\week3
- 当前分支：codex/project-document-handoff
- 当前 HEAD：a069165（最新文档交接更新提交）
- Task 4 review-fix 工程提交：ddf9972；文档交接提交：48d3aa8；Task 4 初版基线：8ce02a8。
- 当前没有应被覆盖的 Task 4 源码 WIP；但用户已有未跟踪文件必须保留，不要 reset、checkout、clean 或删除任何现有修改。
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
- Phase 4 Task 4 初版：8ce02a8；registry、flags、Provider/LLM async stages、evolution scaffolding、reservation fail-closed、baseline default-off 初版。
- Phase 4 Task 4 review-fix：ddf9972；已接通 RuntimeConfig→validated registry→production CompositionRoot/service/orchestrator/evolution，保护 typed failures，保留 optional snapshot refs，并处理 CancelledError reservation cleanup；本地 249 个 focused/adjacent 测试、10 个 smoke/serve/formal-path 测试、Ruff、mypy、diff-check 通过，但尚未完成同一审查者的完整范围最终复核。

【Task 4 当前边界（实现已提交，验收未闭合）】
`ddf9972` 已针对先前 C1/I3 复核意见完成实现修复；当前唯一剩余门槛是由同一审查者对 `8ce02a8..ddf9972` 做完整范围复核并确认 C0/I0。不要仅凭本地测试把 Task 4 标记为最终通过，也不要进入 Task 5–8。

【必须遵守的边界】
- Phase 4 Task 4 尚未最终验收，不要标记完成，不要进入 Task 5–8，直到同一审查者完整范围复核 C0/I0。
- main-baseline 必须保持 default-off、fixed-one-round、无 optional module construction；不要做 promotion。
- optional stages 必须 async、共享同一 request budget/controller/snapshot/evaluation path；禁止 `asyncio.run()` 嵌套。
- typed budget/config/snapshot/integrity/adapter errors 不得被吞掉；CancelledError 必须清理 reservation/client/artifact。
- 不做真实网络、真实 provider、真实成本、真实 Gate、浏览器 live 验收或公开状态更新；这些仍是 deferred/Cannot verify，除非用户单独授权。
- 保留所有现有用户未跟踪文件和未提交 WIP；不要 reset/checkout/clean。
- 所有后续实现先 RED→GREEN，再跑 focused/static；当前优先做完整范围复核，不要重复实现已经提交的四项修复。

【下一步严格顺序】
1. 先根据模板章节检查/完善 `docs/handoff/project-summary-for-competition-template.md`，这是下一步文本工作；只写有证据支持的内容。
2. 检查当前 `git status`，保留 `.gate0-report.json`、`.sheet-build/`、`outputs/`、`docs/superpowers/plans/` 等未跟踪文件。
3. 阅读 Task 4 计划和 `.superpowers/sdd/phase4-task-4-report.md`，确认 `ddf9972` 的变更范围与本地证据。
4. 由同一审查者对 `8ce02a8..ddf9972` 做完整范围复核；若发现问题，再为问题补 directed RED→GREEN，并更新提交。
5. 只有 C0/I0 复核通过后，才继续 Phase 4 Task 5–8。

【文本工作要求】
请先根据模板章节检查/完善 `docs/handoff/project-summary-for-competition-template.md`，只写有工程证据支持的内容；不要把 Cannot verify 写成已完成。文本完成后再回到 Task 4 的完整范围复核，不要重复已完成的修复。

收到本提示后，先用中文简短回报：当前 HEAD、是否存在用户未跟踪文件、Phase 3/Phase 4 已完成边界、Task 4 当前“实现已提交但待最终复核”的状态；然后按上述顺序继续。
```
