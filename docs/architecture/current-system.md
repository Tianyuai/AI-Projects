# 当前系统

请求经 API/CLI 进入应用服务，完成查询分析后调用检索提供方；结果经过规范化、去重、词法融合和预算截断，再以统一响应契约返回。依赖调用可在 live 模式捕获为快照，并在 replay 模式无网络复现。评估层绑定冻结输入、配置哈希和依赖快照。

稳定模块：`api`、`application`、`pipeline`、`retrieval`、`ranking`、`control`、`storage`、`evaluation`、`recall_experiments` 与 `ui`。

当前只支持 `main-baseline`。召回优化通过 `recall_experiments` 的方案、输入、运行时和报告契约进行，不在生产编排器内保留被否决实验分支。
