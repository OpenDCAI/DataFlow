# 数据卡：finance_compliance_eval_set

## 1. 数据集概览

| 属性 | 值 |
|---|---|
| 名称 | finance_compliance_eval_set |
| 版本 | 0.1.0 |
| 规模 | 30 条（research 15 / announcement 6 / marketing 9；合规 19 / 违规 11） |
| 用途 | 金融合规算子与流水线回归评测 |
| 许可 | Apache-2.0（随仓库分发） |

## 2. 数据构成

- 每条样本：`id`、`doc_type`、`text`、`expected_compliant`（金标）、
  `expected_risk`（low/medium/high 金标）、`contains_sensitive`（金标）。
- 覆盖场景：收益承诺、荐股诱导、夸大宣传、敏感信息（手机号/邮箱/身份证/银行卡）、
  常规研报/公告/营销表述。

## 3. 来源与合成说明

- 全部文本为**模拟撰写**，不含真实机构名称、真实个人与真实账号；
- 金标字段由人工标注并逐条核对；
- 敏感信息样例使用虚假号码/证件号（符合号段格式但非真实分配）。

## 4. 已知局限

- 违规表述较"直白"，未覆盖隐蔽话术（如暗示性收益承诺）；
- doc_type 仅三类，其他类型走默认免责声明模板；
- 金标 risk 标注存在一定主观性，适合回归评测而非绝对真值。

## 5. 扩充与抽检要求

- `eval/expand_evalset.py` 可扩充至 200 条（离线模式生成占位骨架）；
- **人工抽检 ≥20%**：重点核对 `expected_compliant` / `expected_risk` 标注准确性；
- 扩充样本发布前需更新本数据卡版本号与规模字段。
