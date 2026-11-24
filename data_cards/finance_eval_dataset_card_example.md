# 数据卡：finance_eval_sample_set

## 1. 数据集概览

| 属性 | 值 |
|---|---|
| 名称 | finance_eval_sample_set |
| 版本 | 0.1.0 |
| 规模 | 30 条（easy 6 / medium 9 / hard 15） |
| 用途 | 金融 QA/SQL/幻觉评测算子回归评测 |
| 许可 | Apache-2.0（随仓库分发） |

## 2. 数据构成

- 每条样本：`text`（模拟原文）、`question`、`answer`（待评测答案）、`gold_answer`、
  `sql`（待评测 SQL）、`gold_sql`，以及三个金标字段：
  `gold_grounded`、`expected_qa_correct`、`expected_sql_correct`。
- 负样本设计覆盖：数值失真、数字置换、虚构引用、超范围回答、聚合函数错误、
  语义等价但写法不同（在文本归一化口径下判错——已注明的已知盲区，
  用 `use_ast=True` 的 AST 口径可正确识别）。

## 3. 来源与合成说明

- 全部文本为**模拟撰写**，不含真实机构与真实市场数据；
- 金标由人工标注；确定性评测部分（qa/sql）已与评测逻辑**逐条对拍一致**，
  由 `tests/test_sample_set_matches_evaluator_expectations` 持续守护。

## 4. 已知局限

- `gold_grounded` 标注依赖 LLM 判定口径，报告幻觉率时需注明；
- SQL 金标以 sqlite 方言为主，其他方言需改写；
- 样本量适合回归评测，不适合作为训练集。

## 5. 扩充与抽检要求

- `eval/expand_evalset.py` 可扩充至 200 条（离线模式生成占位骨架）；
- **人工抽检 ≥20%**：重点核对三个金标字段的标注一致性；
- 扩充样本发布前需更新本数据卡版本号与规模字段。
