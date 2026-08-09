# Dev 质量 Gate 离线根因诊断

日期：2026-08-09

状态：已解决并完成 live capture/replay 闭环。下文保留失败轮次的历史根因证据。

## 闭环结果

`a720bbe` 修正 repair 在精确成本上限下的预算分配，`45ef874`
统一了灵活 repair 响应的归一化。修复后的正式闭环为：

| 项目 | 结果 |
|---|---|
| live capture | `runs/dev-20260809T061903Z-9bd861e90299` |
| replay | `runs/dev-20260809T063333Z-6897d295a3c8` |
| Gate | 两轮均 `passed`，`formal_valid: true`，`quality_passed: true` |
| 查询分析 | model-produced analysis 60/60，primary planner 60/60 |
| 检索响应 | parseable retrieval response 60/60 |
| 复现性 | capture/replay 业务结果 `equivalent: true` |
| 基线指标 | macro F1 `0.0050946874`，micro recall `0.0575539568` |

因此，原 59/60 失败不再是后续任务的阻塞项。OpenAlex
`invalid_work` 仍是 provider 数据质量边界，但不影响基线 Gate 通过。

## 历史失败轮次结论

本轮 dev capture 与 replay 均通过结构校验，`compare-replay` 结果为
`equivalent: true`。正式 Gate 仍为 `failed`；在适用的 baseline quality checks 中，唯一未通过的质量项是
`model-produced-analysis-rate`：`59/60 = 0.9833333333`，阈值为 `0.99`。

三项重点 Gate 中，`retrieval-response-rate` 已通过；`strong-constraint-recall`
和 `fuzzy-merge-accuracy` 在 dev split 中不适用，不能用 `0/0` 判断算法失败。

已确认的离线根因有两个层次：

1. 一次 LLM 返回的 assistant 内容不是合法 JSON，导致该 query 进入 rules fallback；
   生产编排调用 `QueryParser.parse` 时没有传入 lock 所声明的单次 repair callback，
   因而没有发生第二次修复调用。当前代码已接入这一次独立 repair，并纳入预算结算。
2. OpenAlex 有 10 个成功封存的响应各包含一个缺少标题的 Work。decoder 将其记为
   `invalid_work`；评估适配层为安全起见将该未列入白名单的错误码归一化为
   `provider_error`。这些响应仍是可解析响应，因此没有造成
   `retrieval-response-rate` 失败。

## 证据范围

本报告只使用本地正式 capture/replay 产物和源码，不发起网络请求、不新增 capture，
也不读取或展示原始 query、响应正文、密钥或 request ID。

| 项目 | 证据 |
|---|---|
| capture | `runs/dev-20260809T035350Z-7e0f28699548` |
| replay | `runs/dev-20260809T040808Z-af862dc2d4ab` |
| capture/replay 校验 | 两者 `valid: true`；比较 `equivalent: true` |
| 当前源提交 | `60faaad78203c63bac1d8a6723ecd9e85304924a` |
| formal validity | `true` |
| quality gate | `failed` |

## 三项重点 Gate

### 1. strong-constraint-recall

```text
applies: false
measure: audited_strong_constraint_recall = 0/0
threshold: 0.90
```

该 Gate 只适用于 `frozen_audit`。本轮 dev run 没有约束标注审计输入，候选锁也
没有启用 constraint reranking，因此不存在可用于本轮的分母。

结论：这是 frozen audit 输入尚未准备，不是本轮已证实的约束提取或排序算法失败。

### 2. retrieval-response-rate

```text
applies: true
measure: parseable_configured_retrieval_response_rate = 60/60 = 1.0
threshold: 0.95
```

本轮 273 个 OpenAlex snapshot response 文件均存在且非空，离线 decoder 能解析响应
顶层结构；没有空响应、HTTP 错误封存或网络错误证据。

进一步检查 33 个与错误诊断关联的 OpenAlex 成功快照，发现其中 10 个 Work 缺少
`title` 和 `display_name`，decoder 返回 `invalid_work`。这只丢弃了单个无效 Work，
没有使整个检索响应不可解析，因此本 Gate 通过。

结论：本轮没有证据支持修改 retrieval decoder、候选排序或合并逻辑。

### 3. fuzzy-merge-accuracy

```text
applies: false
measure: audited_fuzzy_merge_accuracy = 0/0
threshold: 0.98
```

该 Gate 同样只适用于 `frozen_audit`。本轮没有 fuzzy-merge decision 标注集，不能
从 dev prediction 或 replay 结果计算准确率。

结论：这是 frozen dedup audit 输入缺失，不是已证实的 fuzzy merge 代码失败。

## 本轮实际 Gate 失败：`model-produced-analysis-rate`

```text
applies: true
measure: model_produced_analysis_rate = 59/60 = 0.9833333333
threshold: 0.99
```

离线检查显示：

- 60 个 LLM 快照中 59 个 assistant 内容可解析为 JSON object，1 个不可解析；
- 异常快照的 JSON envelope 完整，包含一个 choice 和字符串 content，失败发生在
  assistant content 的 JSON 解析阶段，不是封存文件损坏；
- 该记录的 `llm_calls` 为 1，`planner_status` 为 `rules_fallback`；
- `QueryParser` 支持在传入 callback 时执行一次 repair，但生产编排调用
  `parse(query, analysis_result)` 时没有传入 repair 参数；
- 候选锁仍声明 `repair_attempts: 1`，所以当前实现与锁定预算能力不一致。

结论：本 Gate 的直接根因是一次 malformed assistant JSON；代码层放大因素是 repair
callback 没有接入生产编排，导致本可用的一次修复预算未被使用。该问题已修复，并由新 live
capture 验证为 60/60。

## OpenAlex `provider_error` 的解释

execution 记录中的 OpenAlex `provider_error` 是评估适配层的安全归一化结果，不足以
直接说明网络失败。源码允许 `invalid_work` 等 decoder 错误进入 provider result，
而评估适配层只保留白名单错误码；未列入白名单的错误码会被改写为 `provider_error`。

本轮对封存响应重新运行纯离线 decoder 后，10 个错误全部归因到同一条消息：
`OpenAlex work must have a title`。因此这部分应归类为 provider 数据质量边界，
而不是网络可用性或 capture/replay 完整性问题。

## 已完成的最小后续动作

1. 已为 malformed LLM JSON 补失败测试，接入既有的单次 repair callback，并修正预算和响应归一化。
2. 已重建候选锁、运行 readiness，并完成 capture → verify → replay → compare。
3. `invalid_work` 的部分成功处理转入检索质量提升；不再将其与 Gate 可用性问题混同。
4. frozen constraint audit 与 frozen dedup audit 仍属独立审计输入，不用 dev 结果替代。

## 最终判断

正式 capture/replay、账本一致性、model-produced-analysis-rate 和
retrieval-response-rate 已闭环。OpenAlex 的 `invalid_work` 属于已识别但未阻塞
Gate 的 provider 数据质量问题。后续工作已转向召回和最终候选保留，不再重复诊断 59/60。
