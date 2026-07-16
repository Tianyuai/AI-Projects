# Task 2 人工标注指南

> 状态：工程契约已定义，正式标签尚未冻结
> 当前状态值：`waiting_for_human_label_freeze`

本指南用于开发集、验证集和固定重叠样本的真人独立标注。不得使用 LLM 生成、补写或改写标签；不得把真实查询、金标准或标注文件提交到 Git。

## 1. 开始条件

只有以下条件全部满足后才开始：

1. `data/manifest.json`、`data/splits/dev.ids.json` 和 `data/splits/validation.ids.json` 已冻结；
2. 双方使用相同的 dataset revision 和本指南版本；
3. `data/annotation_work/` 中的工作包通过 Schema 校验；
4. 双方确认固定 20 条重叠 query ID，但未查看对方答案；
5. 主负责人发出“Task 2 标注工作包 v1 已冻结”通知。

## 2. Query Type

每条查询只选择一个主要类型：

| 标签 | 定义 |
|---|---|
| `topic` | 主要限定研究主题或现象 |
| `method` | 主要限定模型、算法、训练或推理方法 |
| `dataset` | 主要限定数据集、基准或评测任务 |
| `time_venue` | 主要限定年份、会议或期刊 |
| `combined` | 同时包含多个同等重要的约束维度 |
| `relationship` | 查找前置、后续、引用或同路线工作 |
| `exclusion` | 核心意图由明确排除条件驱动 |
| `ambiguous` | 原查询不足以稳定确定上述类型 |

选择最能决定入选集合的主要类型。不要为了覆盖所有细节发明新标签；无法稳定判断时使用 `ambiguous` 并写入私密问题清单。

## 3. Domain

`domain` 使用小写 kebab-case，例如 `information-retrieval`、`natural-language-processing`、`computer-vision`。优先使用团队已有标签；新领域必须先进入问题清单并由双方统一口径，不能由单个标注者临时创造近义标签。

## 4. 十一个固定字段

| 字段 | 填写规则 |
|---|---|
| `query_id` | 原样保留，不得新增、删除或修改 |
| `research_goal` | 用一句话概括用户真正要解决的问题，不堆积搜索关键词 |
| `must_have` | 缺失任一项就不应入选的明确硬约束 |
| `should_have` | 提高相关性但不构成一票否决的软约束 |
| `exclusions` | 原查询明确排除的方法、主题、领域或文献类型 |
| `year_from` | 原查询明确给出起始年时填写，边界包含；否则为 `null` |
| `year_to` | 原查询明确给出结束年时填写，边界包含；否则为 `null` |
| `venues` | 只记录原查询明确出现的会议或期刊 |
| `query_type` | 使用第 2 节固定标签 |
| `domain` | 使用第 3 节格式和团队词表 |
| `annotator` | 使用团队约定的稳定代号 |

年份必须在 1900 到当前年份加 1 之间，且 `year_from <= year_to`。不得依据常识补造原查询没有表达的年份、venue 或排除条件。列表字段没有内容时使用空列表，不得包含空字符串。

## 5. 独立标注与分歧处理

固定 20 条重叠样本由两人独立完成。完成前不得交换答案、查看对方文件或使用自动生成标签。双方提交后，主负责人按 query ID 对齐记录，对 `query_type` 和 `domain` 计算 Cohen's kappa。

- 每个关键字段的通过阈值为 `0.80`；
- query ID 缺失、新增或重复时停止比较并修复工作包；
- 任一字段低于阈值时，先分析指南歧义并更新指南版本，再重标分歧样本；
- 不允许为了提高 kappa 直接复制另一方答案；
- 一致性通过后，主负责人再独立复核固定 10 条。

## 6. 示例

以下为项目原创格式示例，不来自 PaSa：

```json
{"query_id":"example-q1","research_goal":"查找用于长文档检索的高效注意力方法","must_have":["长文档检索"],"should_have":["稀疏注意力"],"exclusions":["仅图像任务"],"year_from":2020,"year_to":null,"venues":[],"query_type":"method","domain":"information-retrieval","annotator":"member-b"}
```

真实标注仅通过私密渠道交接。Git 中只允许保存不含受限文本的数量、哈希、Schema 校验结果、kappa 和匿名化问题汇总。
