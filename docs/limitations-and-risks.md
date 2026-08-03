# 限制与风险

## 当前证据边界

当前修订已验证：

- 集成应用、replay/live 组合、API、UI、正式运行器、验证器与实验注册表通过离线工程套件；
- fake-provider live 捕获与 replay 生命周期测试覆盖发布、失败、取消、血缘与隔离；
- 合成正式 capture/replay 夹具通过 `verify-run` 与 `compare-replay`；
- 真实浏览器已对 loopback replay UI 执行两轮演示，可见业务内容与来源稳定；
- 一次在显式网络/成本授权下的真实 live 浏览器捕获完成（2026-08-03，成本 0.001181 CNY），并暴露、修复了两个 DashScope LLM 兼容性缺陷（`json_object` 提示词与 thinking 读超时）；
- `main-baseline` 不构造任何可选阶段，已命名可选身份只构造其声明行为。

上述证据尚未确立：

- 正式 dev 或 validation 的实时 capture/replay 与真实 macro-F1（Gate 0 r5 已通过，公开 `data/manifest.json`/`data/gate0_evidence.json` 已核准 V2 frozen）；
- 真实 provider 可用性、检索质量、可测成本或生产就绪性（Gate 0 readiness 为就绪证据，单次有界 live 请求不足以形成正式质量证据）；
- 正式 dev 或 validation capture/replay 结果；
- 任一可选模块的比较收益或晋升资格；
- 全部 provider 健康状态下的 live 浏览器验收。

本地安全 Gate 0 报告（2026-08-03 r5）已通过：identifier-map 223 条覆盖 dev/validation 全部 gold 标识，生产定价、质量门与就绪证据哈希一致。公开 `data/manifest.json` 与 `data/gate0_evidence.json` 已发布核准后的安全投影；正式 dev/validation 实时证据仍待授权运行。

## 主要风险

- **证据替代：** 合成夹具可证明契约与生命周期行为，但不能支持真实检索质量、provider 可靠性或成本声明。
- **授权漂移：** 若把 lock 权限、服务授权与请求模式视为可互换，live 执行不安全；三项必须全部强制。
- **敏感制品披露：** 快照、预测、失败、业务结果、gold 标签、查询与验证声明即使在无凭据时也可能暴露受保护数据。
- **验证重试偏差：** 在中断或失败后允许新验证尝试会破坏一次性策略；尝试身份绑定到归档 lock 字节。
- **Replay 完整性漂移：** 缺少精确响应字节、请求身份、策略/配置绑定或规范化业务比较的 manifest 会制造假可复现性。
- **可选阶段过度声明：** 实现与离线测试不证明正向质量增量；未经 Gate 6 证据与单独批准，baseline 默认不得改变。
- **平台测试抖动：** 子进程测试助手先预留 OS 分配的端口再让服务绑定；其他本地进程可能赢得低概率竞争。这影响测试稳定性，不影响已验证运行时契约。
- **环境歧义：** `--no-env-file` 阻止 dotenv 加载，但既不删除继承的密钥，也不阻止网络。

## 缓解与门禁

- V2 冻结、标识符映射、生产定价、质量策略与安全就绪证据已通过 Gate 0 r5；正式 dev/validation 与晋升证据仍待各自显式授权。
- 每次真实网络、成本、dev、validation、live 浏览器或晋升动作前，要求显式且受限的授权。
- 在硬预算与请求级捕获下运行 live 工作；仅当封存证据验证后发布成功。
- 以 `verify-run` 作为正式运行目录的有效性谓词，以 `compare-replay` 作为规范化 capture/replay 比较；服务端 smoke 型 live 捕获按 `run.json` complete、快照清单、replay 锁及可回放性校验。
- 保留不可撤销的 validation 尝试声明，并拒绝跨 lock 恢复。
- 将受保护制品保留在 Git 与普通对话/日志通道之外；只共享批准的聚合、哈希、安全错误码与 run ID。
- 保持 `configs/base.yaml` 为 `main-baseline`；将所有未晋升实验制品视为证据，而非启用模块的授权。
- 子进程测试未来优先使用 port-0 握手，以消除剩余端口分配竞争。

## 报告规则

每份报告必须将各门禁标记为 passed、blocked、failed 或 not run。不得把延迟证据描述为已接受，也不得从工程测试计数推断性能或部署声明。
