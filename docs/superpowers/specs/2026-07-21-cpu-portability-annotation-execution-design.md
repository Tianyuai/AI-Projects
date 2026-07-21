# CPU-First 可移植验收与 Task 2 人工标注执行设计

## 1. 背景

协作者在 Windows 11、Python 3.11 和 AMD RX 6750 GRE 环境中完成了大部分离线验证，但当前仓库把 `torch==2.5.1` 固定到 `pytorch-cu121`，并在健康测试中直接要求 CUDA 可用、设备名称包含主负责人本机的 NVIDIA 型号。这使 AMD、纯 CPU 和多数第三方验收环境出现与业务能力无关的失败。

与此同时，Task 2 已具备 prepared manifest、60/30/50 分区 ID、90/40/20 工作包 ID、人工标注 Schema 和冻结审计器，但协作者仍缺少一个不依赖 GPU、不会打印私密正文、可以在正式冻结前独立检查标注文件的入口。领域标签虽然要求 kebab-case，却没有冻结的受控词表，容易因为同义词而人为降低 kappa。

本设计将环境兼容和人工标注拆成两个顺序明确、接口隔离的子阶段：先建立 CPU 强制基线和可选加速器契约，再发布可验证的真人标注流程。两者都不改变 prepared 数据身份，也不把人工标签或查询正文提交到 Git。

## 2. 目标与非目标

### 2.1 目标

1. Windows/Linux 的 Python 3.11 CPU 环境可以安装项目并通过默认离线验收。
2. 没有 CUDA 不再使默认健康检查或默认 pytest 失败。
3. NVIDIA CUDA 作为显式可选加速配置保留；未选择该配置时不安装 CUDA 专用 torch。
4. 健康检查区分“核心能力不可用”和“可选加速器不可用”。
5. 所有硬件分支由确定性测试覆盖，不依赖执行测试的真实显卡或设备名称。
6. 协作者能够在仓库外完成 90 条类型/领域、40 条约束标注；主负责人独立完成固定 20 条 overlap。
7. 在正式 freeze audit 前，双方可以分别验证 JSONL Schema、ID 集合、数量和精确字节 SHA-256，且输出不包含查询、paper ID、标注正文、私密绝对路径或凭据。
8. 冻结一个明确的领域标签词表和问题升级流程，避免自由同义词破坏 kappa。

### 2.2 非目标

- 不宣称 Windows AMD 原生 GPU/ROCm 支持。
- 不要求第三方验证者拥有 NVIDIA GPU。
- 不改变 Week 1 BM25/OpenAlex baseline 的评分、过滤或去重算法。
- 不修改 prepared manifest、六个 ID 文件、gold 或抽样算法。
- 不生成、补写或改写任何真人标签。
- 不运行正式在线 baseline，不把 manifest 改为 `frozen`。

## 3. 支持矩阵与验收语义

| 环境 | 支持等级 | 默认验收 |
|---|---|---|
| Windows 11 + Python 3.11 + CPU | 必须支持 | 核心健康检查、离线 pytest、Ruff、mypy 全部通过 |
| Linux x86_64 + Python 3.11 + CPU | 必须支持 | 同上 |
| Windows/Linux + NVIDIA CUDA | 可选加速 | 只有显式选择 CUDA profile 时验证 CUDA smoke |
| Windows + AMD GPU | CPU 路径支持 | 不探测或要求 CUDA；不得因 AMD 型号失败 |
| ROCm | 未承诺 | 只有获得真实兼容机器和独立证据后再加入矩阵 |

默认 `ready` 表示：Python 版本正确、核心依赖可导入、CPU 矩阵 smoke 有限且结果正确。可选 CUDA 不存在时，报告 `accelerator.status=unavailable` 或等价的非阻塞状态；只有显式 `require_accelerator=cuda` 时才返回非零。

## 4. 依赖配置设计

### 4.1 CPU 必选验收配置

- 第三方默认验收命令必须显式选择 `--extra cpu`，从 PyTorch CPU 索引解析固定版本 torch，不再从 `pytorch-cu121` 解析。
- 裸 `uv sync` 只安装不含 embedding/torch 的核心依赖；需要运行完整检索、健康检查或测试时必须选择 `cpu` 或 `cuda` profile。
- 默认开发命令安装 `cpu` profile 和 `dev` 依赖，不再使用会无差别安装所有硬件组的 `uv sync --all-groups`。
- `uv.lock` 必须同时记录 Windows/Linux 可用的 CPU wheel；锁文件变更通过 fresh environment dry-run、导入 smoke 和全量离线测试验证。

### 4.2 CUDA 可选配置

- 仓库只维护一个 `uv.lock`：`--extra cpu` 显式选择固定 CPU 索引和版本；`--extra cuda` 显式选择固定 CUDA 索引和版本。
- `cpu` 与 `cuda` extras 必须通过 uv `conflicts` 声明互斥；同一次解析中只能有一个 torch 来源生效，不能依赖安装顺序、已有虚拟环境或额外的手工 pip 命令。
- 第三方验收说明固定使用 CPU profile。CUDA 命令放在单独章节，明确为可选。
- `uv lock`、CPU-extra dry-run 和 CUDA-extra dry-run 任一不能证明单锁互斥解析时，依赖改动视为阻塞，不提交折中配置，也不恢复全局 CUDA 默认值。

实施前先用当前固定的 uv 版本验证配置语法和 dry-run；不得凭猜测修改锁文件。

## 5. 健康检查设计

`collect_local_health` 改为设备无关的两层报告：

1. `core`：Python、核心检索依赖、CPU torch smoke；决定默认 `status`。
2. `accelerator`：后端类型、是否可用、设备摘要和可选 smoke；默认不影响 `ready`。

CLI 增加显式模式：

- 默认：CPU 必须通过；CUDA 缺失仅报告，不阻断。
- `--require-accelerator cuda`：CUDA 不可用或 smoke 失败时返回非零。

报告继续禁止读取应用密钥，不输出环境变量值。设备名可以出现在本地健康 JSON 中，但自动测试不能断言具体型号。

测试通过依赖注入或 monkeypatch 覆盖以下场景：CPU-only ready、CUDA 可用、CUDA 缺失但默认 ready、显式要求 CUDA 时失败、CPU smoke 失败、核心依赖缺失、输出不包含环境密钥。真实硬件 smoke 使用独立 `hardware` marker，默认离线 pytest 不执行。

## 6. 第三方验证流程

第三方必须从包含精确换行契约的全新 clone/worktree 开始，不能在旧 checkout 上只执行 fast-forward pull。CPU 验收顺序为：

1. 检查 Python 3.11 和 uv 固定版本范围。
2. 安装 CPU 默认 profile 与 dev 组。
3. 运行默认健康检查并保存不含秘密的 JSON。
4. 使用 `--no-env-file` 运行聚焦和全量离线测试、Ruff、mypy。
5. 验证 prepared manifest 固定 SHA-256 和六个 ID 文件哈希。
6. 报告可选加速器状态，但不把缺失 CUDA 计为失败。

在线 OpenAlex/LLM 烟测与该流程分开；只有明确授权的测试子进程可以加载 `.env`。

## 7. 人工标注职责与数据流

### 7.1 固定身份

开始前双方核对 source commit、dataset revision、prepared manifest SHA-256、60/30/50 与 90/40/20 ID 清单。真实输入位于被忽略的 `data/annotation_work/`，只能保存在访问受控目录。

### 7.2 协作者交付

协作者使用稳定代号（例如 `member-b`）产生两份私密 UTF-8 JSONL：

1. `type_domain_labels.jsonl`：90 条，每条只有 `query_id, query_type, domain, annotator`。
2. `constraint_labels.jsonl`：固定 40 条，使用完整十一字段 `AnnotationRecord`。

40 条中的 `query_type` 和 `domain` 必须复用该协作者在 90 条文件中的值，不能二次独立判断后产生自相矛盾。

### 7.3 主负责人交付

主负责人使用不同稳定代号（例如 `member-a`），针对 `overlap_annotation.ids.json` 的固定 20 条独立产生完整十一字段 `overlap_labels.jsonl`。这 20 条是协作者 40 条的子集；协作者的对应记录是第一评分者，主负责人的文件是第二评分者。

双方在都完成并先交换精确文件 SHA-256 前，不得查看对方答案或逐条讨论。

## 8. 标签规则与受控领域词表

`query_type` 继续固定为八类：`topic, method, dataset, time_venue, combined, relationship, exclusion, ambiguous`。

新增版本化的安全领域词表文件，只包含允许的通用 kebab-case 标签、定义和版本，不包含 query ID、查询文本或逐条答案。首版固定为：`artificial-intelligence, machine-learning, natural-language-processing, information-retrieval, computer-vision, speech-audio, robotics, data-mining, knowledge-graphs, recommender-systems, human-computer-interaction, software-engineering, computer-systems, networks-security, databases, theory-algorithms, computational-biology, computational-social-science, scientific-computing, multidisciplinary, other`。每个标签必须在文件中有一句边界定义；首版在查看 overlap 答案前冻结。遇到词表外领域时：

1. 标注者在私密问题清单中记录，不直接创造近义标签；
2. 双方只讨论词表定义，不交换该 query 的拟选答案；
3. 由主负责人发布新词表版本；
4. 双方独立回到受影响记录完成选择；词表尚未升级时统一使用 `other`，并保留私密问题记录。

`ambiguous` 只表示查询主要类型无法稳定判断，不能用来代替未知 domain。若无法在不泄露逐条答案的前提下升级领域词表，domain 保持 `other`，不得复制对方标签。

## 9. 安全标注校验入口

为 `paper_search.evaluation.annotation` 增加只读 CLI，至少支持：

- 验证 90 条 `TypeDomainAnnotationRecord`；
- 验证 40/20 条 `AnnotationRecord`；
- 对照指定安全 ID 清单验证数量、唯一性和集合精确一致；
- 输出记录数、输入精确字节 SHA-256、Schema/ID 是否通过和错误类别；
- 永不输出 query ID、字段值、标注正文、标注人答案、私密绝对路径或原始异常内容。

该 CLI 不计算正式 kappa，也不修改 manifest。正式 kappa 和三文件交叉约束仍由 freeze audit 一次性执行。

## 10. 一致性、gold 与正式冻结

- freeze audit 对协作者 40 条中固定 20 条与主负责人 overlap 文件计算 `query_type` 和 `domain` kappa；两个字段均须不低于 `0.80`。
- 若初始 kappa 不达标，只先披露聚合结果；修订指南后必须用新冻结且未查看分歧答案的重叠样本重新独立验证，不能用已知分歧样本抬高 kappa。
- 主负责人额外复核固定 10 条，但复核不能替代双人独立 kappa。
- 官方 gold 不因人工标注而改写。`labels_complete` 表示固定分区的官方 gold 字节、ID、数量和完整性检查通过，而不是人为增加相关论文。
- 主负责人在正式冻结前逐分区明确 `zero_answer_policy`，不存在默认策略。
- 先运行无 `--approve` 的 audit-only；所有证据通过后才允许显式批准。任何 synthetic 测试、CPU 健康通过或工作包通知都不能把 manifest 自动改为 `frozen`。

## 11. 错误处理与安全边界

- CPU 依赖无法解析：停止，不回退到未锁定 CUDA/ROCm 包。
- 可选 CUDA 不可用：默认报告为非阻塞；显式 CUDA 模式失败。
- 标注 JSONL 无法解码、Schema 错误、ID 缺失/重复/新增：返回通用错误类别和非零码，不打印敏感行。
- 私密文件路径逃逸或进入 Git：停止交接和冻结。
- 任何命令都不得读取、打印、搜索或复制 `.env`；只有用户明确授权的在线/数据准备子进程可以通过 `--env-file` 加载。

## 12. 实施与提交边界

实施采用严格 TDD：每项行为先写失败测试并确认 RED，再做最小 GREEN。建议拆成以下提交：

1. `test/feat`: CPU 默认依赖与互斥加速 profile。
2. `test/feat`: 设备无关健康检查和 CLI。
3. `test/feat`: 私密标注安全校验 CLI。
4. `docs`: 支持矩阵、领域词表和协作者操作说明。

完成声明前必须运行 CPU profile fresh sync/dry-run、聚焦测试、全量离线 pytest、Ruff、mypy、`git diff --check`、变更范围检查、排除 `.env` 与受保护设计文档的秘密扫描，以及独立代码审查。未经用户明确授权，不推送、不创建 PR、不合并、不清理 worktree。
