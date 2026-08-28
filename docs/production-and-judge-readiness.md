# 生产接入与评委使用状态台账

更新日期：2026-08-29。

## 当前方法与生产规则

- 生产默认：`F5 gated fusion`。
- 第一性能回退：`F4 reliability`。
- 紧急安全回退：`B0`。
- 运行顺序固定为 `F5 → F4 → B0`；仅允许因制品缺失、哈希错误或初始化失败触发回退，禁止按单条 query 的结果动态选模型。
- 当前生产选择基于 18,314 条冻结训练查询的三折 OOF 证据以及 538 条独立冻结 `auto_dev` 非退化门禁；test 分区未触碰。

## 已完成

- [x] F5、F4、B0 三套制品及内容哈希进入统一生产锁。
- [x] 应用组成层按固定三级顺序加载，F5 不可用时先回退 F4，再回退 B0。
- [x] CLI、API、UI 与评委批量入口共用同一应用服务和生产排序路径。
- [x] 提供只含 `query_id`、`query` 的批量输入契约与固定 JSONL 输出契约。
- [x] 提供公开集/隐藏集共用的一键 live 命令；live 不依赖历史 replay 快照。
- [x] 每条 live 成功查询原子封存 lock、manifest、依赖响应与业务结果。
- [x] `--verify-replay` 会立即重放新封存回执，并要求最终论文 ID 及顺序精确一致。
- [x] 自动检查 query 唯一性、输入输出顺序、覆盖、结果 ID 去重、制品哈希及 test 隔离。
- [x] 新封存 live → replay 的端到端工程测试已通过；replay 结构性离线测试已通过。
- [x] 可重复构建的评委代码包包含逐文件清单、外部 ZIP 校验值和零网络 replay 样例。

## 评委入口

生产 live 锁：`deliverables/evaluator/live-evaluator.lock.yaml`。

```powershell
python scripts/run_evaluator_package.py `
  --queries <queries.jsonl> `
  --output <predictions.jsonl> `
  --lock deliverables/evaluator/live-evaluator.lock.yaml `
  --artifact-root . `
  --capture-output-root runs/evaluator `
  --mode live `
  --verify-replay
```

详细说明见 `deliverables/evaluator/README.md` 和 `docs/evaluator-submission-quickstart.md`。

## 仍需在最终交付环境执行

- [ ] 用最终提交机器、最终密钥和少量真实 OpenAlex/LLM 请求跑一次 release rehearsal；这会消耗在线配额，不能用模拟依赖替代声明。
- [ ] 记录最终环境安装耗时、单 query/整批延迟、供应商限流和失败恢复结果。
- [ ] 用比赛方最终提供的输入样例复核字段映射；若赛题仍无样例，则沿用当前最小 JSONL 契约。
- [ ] 由提交负责人另行完成演示视频、平台上传和交付签字；这些外部动作不进入代码仓库。

## 已识别兼容性边界

- Replay 是严格请求级重放，只能回答对应快照实际捕获过的 query。
- 2026-08-06 的历史演示快照使用旧 OpenAlex 请求字段；当前版本会因请求身份不同而失败关闭，不把旧快照伪装成当前 live/replay 一致性证据。
- 正式演练应使用当前版本新生成的回执；一键入口已自动完成该动作。
