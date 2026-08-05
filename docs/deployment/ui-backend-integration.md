# UI 与正式后端整合记录

## 版本边界

- 正式后端语义基线：`0c88a9afd1ae530fb05635576f9ffdcafda89a3a`
- 新版 UI：`ae77493aa453a003aadaf23552c6f01c2ccffc0e`
- 发布分支：`codex/ui-backend-integration`
- 整合方式：静态 UI overlay

`ae77493` 以 `0c88a9a` 为祖先。两者之间的变更仅涉及以下三个静态资源：

```text
src/paper_search/ui/static/app.js
src/paper_search/ui/static/index.html
src/paper_search/ui/static/styles.css
```

没有修改 Python 后端、API 合约、检索逻辑、候选生成、融合排序、预算控制、锁验证或冻结实验输入。因此，运行证据中的后端算法身份仍由获批输入锁绑定的 `source_git_sha` 决定；UI 版本单独记录为 `ae77493`，不得将两者混写为同一实验身份。

机器可读文件见 `docs/deployment/ui-overlay.provenance.json`。其中三个文件的哈希均针对 Git 中的规范 blob 字节计算，不受 Windows checkout 的 CRLF 转换影响。

## 验证结果

2026-08-05 在获批的 `title-candidates` Live 输入包上完成以下验证：

- CandidateLock 与冻结文件校验：`verified OK`
- ledger checkpoint：匹配
- UI 与双模式回归测试：`9 passed`
- Live capture：`complete`
- 离线 Replay：`complete`
- Live 与 Replay 规范业务结果 SHA-256：一致
- Replay 返回：50 条 selected 结果
- Provenance 与 Diagnostics 面板：正常
- 窄屏布局：无横向溢出
- 浏览器控制台：无 warning 或 error

以上验证没有改变检索精度、相关性阈值或排序策略。结果精度与排序改进属于后续独立任务，必须使用新的实验身份和证据链。

## 不进入 Git 的运行材料

本分支不包含以下材料：

- `.env` 或任何 API 密钥；
- `candidate.lock.yaml`；
- 私有 identifier map、标注数据或 ledger；
- Live provider 原始响应；
- `captures/`、Replay 运行目录或临时测试产物；
- 真实查询文本。

部署人员必须从获批的受控渠道取得 Live 输入包，并在进程环境中只加载 `LLM_API_KEY`、`OPENALEX_API_KEY` 与 `SEMANTIC_SCHOLAR_API_KEY`。冻结的模型、端点与实验身份仍由锁和 YAML 配置决定。
