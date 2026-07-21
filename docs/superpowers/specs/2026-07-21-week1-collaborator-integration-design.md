# Week 1 协作者交付整合与标注启动设计

**日期：** 2026-07-21
**分支：** `codex/week1-collaboration`
**协作者提交：** `d6adb6e1f1ab12c40cf87315951de1cfe9742121`
**状态：** 待用户复核

## 1. 背景与已验证事实

协作者提交直接基于主负责人已验证的 `5b8800f`，新增 `data/manifest.json`、六份安全 ID 清单，并修复 PaSa canonical answer ID 的稳定去重。独立离线验证结果为聚焦测试 `97 passed`、全量测试 `423 passed, 1 skipped`，Ruff 与 mypy 均通过。唯一 skip 是未向进程提供 `OPENALEX_API_KEY` 的在线 OpenAlex 测试。

数据内容满足当前准备阶段契约：

- dataset revision 固定为 `232428b0c867268c3b8ded90db4d98c1b30501d6`；
- dev、validation、simulated_test 分别为 60、30、50 条；
- type/domain、constraints、overlap 工作包分别为 90、40、20 条；
- 每份 ID 清单非空、唯一，dev 与 validation 不重叠；
- type/domain 恰好覆盖 dev 与 validation 的并集；
- constraints 是 dev 子集，overlap 是 constraints 子集；
- manifest 真实保持 `waiting_for_human_label_freeze`，不包含 `gold_sha256`、`labels_complete` 或 `zero_answer_policy`，没有伪造正式冻结。

审查发现仓库没有 `.gitattributes`，而 Windows Git 全局配置为 `core.autocrlf=true`。六份 ID 清单的 manifest 哈希与 Git blob 的 LF 精确字节一致，但 fresh Windows checkout 或 `git archive` 会转换成 CRLF，导致正式精确字节哈希校验失败。这是整合前必须修复的可复现性缺陷。

## 2. 目标

1. 保留协作者提交和作者身份，不重写其提交；
2. 固定正式 manifest、split ID 和未来安全 freeze report 的 checkout 字节为 LF；
3. 用自动化测试证明 Git 属性覆盖所有受精确字节哈希约束的可提交数据文件；
4. 真实更新 PRD：数据准备与安全元数据已完成，真人标注、gold、正式冻结和 baseline 仍未完成；
5. 在权威 onboarding 中发布“Task 2 标注工作包 v1 已冻结”通知，允许两名成员开始独立真人标注；
6. 明确工作包冻结仅固定 revision、manifest、ID 清单和私密 source 文件哈希，不把 manifest 改为 `frozen`；
7. 为后续真人标签交付、审核、正式冻结、真实 baseline 和 Week 1 gate 保留严格证据边界。

## 3. 非目标

- 不读取、打印、搜索或复制 `.env`；
- 不提交 PaSa 原始数据、真实查询、gold 或人工标签；
- 不使用 LLM 生成 90/40/20 条人工标签；
- 不运行正式冻结命令，不把 manifest 状态改成 `frozen`；
- 不运行真实在线 baseline；
- 不宣称 Week 1 gate 已通过；
- 不修改受保护的 `docs/superpowers/specs/2026-07-15-task2-evaluation-design.md`；
- 不创建 PR、不合并到 `main`、不推送，除非用户另行明确授权。

## 4. 选定方案

### 4.1 保留协作者提交

`codex/week1-collaboration` 已通过 fast-forward 指向 `d6adb6e`。后续修复使用独立主负责人提交，不 amend、不 rebase、不 squash 协作者提交，以保留作者和审计链。

### 4.2 固定精确字节 checkout

新增仓库根目录 `.gitattributes`，至少包含：

```gitattributes
data/manifest.json text eol=lf
data/splits/*.json text eol=lf
data/freeze_reports/*.json text eol=lf
```

manifest 和 split ID 已进入 Git，必须在所有平台保持与 Git blob 相同的 LF 字节。安全 freeze report 未来可能作为证据提交，因此同步固定 LF。受限 gold、原始数据和私密标注继续由 `.gitignore` 隔离，不因本设计进入 Git。

### 4.3 真实状态更新

PRD 只勾选“准备 60/30/50 样本并记录 revision、哈希、访问条件、抽样脚本和种子”。以下条目继续未勾选：90 条类型/领域真人标注、40 条约束标注、20 条双人独立标注、正式 gold 冻结、真实 baseline、模糊去重人工审计和 Week 1 gate。

`docs/TEAMMATE_ONBOARDING.md` 增加明确通知：

> Task 2 标注工作包 v1 已冻结。

通知必须同时说明它只授权开始独立真人标注，当前 manifest 仍为 `waiting_for_human_label_freeze`；两名成员必须使用提交 `d6adb6e` 中的 revision、manifest、ID 清单以及各自本地与 manifest 哈希一致的私密 source 文件。完成前不得交换答案。

## 5. TDD 与验证设计

### 5.1 RED

先在 `tests/evaluation/test_prepare_data.py` 增加仓库数据 checkout 契约测试，要求 `.gitattributes` 对 manifest、split ID 和 freeze report 声明 `text eol=lf`。在 `.gitattributes` 尚不存在时运行并确认因缺失契约而失败。

### 5.2 GREEN

新增最小 `.gitattributes`，机械地把当前七个已提交数据文件恢复为 LF 精确字节，再运行测试确认通过。不得通过修改 manifest 哈希或在审核器中规范化换行绕过精确字节契约。

### 5.3 完整验证

1. 逐份计算当前工作树 ID 文件 SHA-256，并与 manifest 声明逐字节比较；
2. 从提交快照建立 fresh Windows checkout/归档，重新验证相同哈希；
3. 运行 official adapter、prepare data、freeze 和 Week 1 pipeline 聚焦测试；
4. 运行全量离线 pytest、Ruff 和 mypy；
5. 运行 `git diff --check`、范围检查、受保护文件检查和新增行秘密扫描；
6. 请求独立代码与数据契约审查，解决所有 Critical/Important 问题。

## 6. 人工工作与后续闸门

本轮工程交付完成后，协作者可以开始 90 条 type/domain 与 40 条 constraint 标注，主负责人必须独立完成固定 20 条 overlap 标注。三份私密标签通过受控渠道交接后，主负责人依次执行：

1. audit-only 冻结审核；
2. 核对两个关键字段 kappa 均不低于 0.80；
3. 明确选择每个 partition 的 `zero_answer_policy`；
4. 显式 `--approve` 生成安全 report 并把 manifest 原子转换为 `frozen`；
5. 在冻结数据上运行真实在线 Week 1 baseline；
6. 完成模糊去重准确率、去重/过滤 Recall 损失、失败率、成本和正式产物审核；
7. 最后判定 Week 1 gate。

缺少真人标签、gold 或在线运行证据时，任何一个后续步骤都不得提前宣称完成。

## 7. 预计文件范围

```text
.gitattributes
PRD.md
docs/TEAMMATE_ONBOARDING.md
docs/superpowers/specs/2026-07-21-week1-collaborator-integration-design.md
tests/evaluation/test_prepare_data.py
```

除机械恢复 LF 字节外，协作者提交的数据文件内容不变。实施提交只包含上述明确范围，不包含 `.env`、受限数据、人工标签、gold 或受保护文档。
