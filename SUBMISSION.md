# 提交物料包（人工执行用）

> 向 OpenDCAI/DataFlow 主仓库提交 PR 与申请 Awesome Works 收录所需物料（dataflow-finance-compliance）。
> 所有代码/文档已就绪，需要人工执行的只有标 ⚠️ 的步骤。

## 1. 提交前检查清单

- [ ] 如需在 PyPI/PR 中展示邮箱，给 `pyproject.toml` 的 authors 补 email（可选，用 GitHub noreply 邮箱）
- [ ] 本地验证：`pip install -e ".[dev]" && pytest`（离线测试，不需要 open-dataflow）
- [ ] 如已装 open-dataflow，用真实 LLM serving 跑一次小样本冒烟（注意先脱敏）

## 2. ⚠️ Fork → Push → PR 三步（沿用既定手动流程）

1. **Fork**（共用 fork，已完成）：fork 统一使用 `DingNova/dataflow-finance`（OpenDCAI/DataFlow 的唯一 fork，同一账号只能 fork 一次，四个金融子包共用此仓库，各自使用独立分支）。注意：pyproject 的 Homepage 已指向本包在 fork 中的分支路径，与旧独立仓库名无关。
2. **Push**：
   ```bash
   cd dataflow-finance-compliance
   git remote add fork https://github.com/DingNova/dataflow-finance.git
   # 若 fork remote 已存在（指向旧地址），改用：git remote set-url fork https://github.com/DingNova/dataflow-finance.git
   # ⚠️ 特别说明：fork 上此前遗留的 feat/finance-compliance-extension 分支与上游 main 完全一致
   #    （无任何扩展提交，compare 曾因此报 "identical to the base"），本地 3 个扩展提交从未推上去，
   #    历史与远端分支不相连，必须强推覆盖：
   git push -f fork feat/finance-compliance-extension
   ```
   （本地已完成 `git init -b feat/finance-compliance-extension` 与 3 个提交：初始扩展提交、补充社区扩展、移除嵌入的 DataFlow gitlink）
3. **Create PR**：compare URL 生成命令见第 4 节。

## 3. ⚠️ PR 标题与描述（预填版）

**Title**: `feat: add dataflow-finance-compliance community extension (PII mask → rule filter → risk eval → disclaimer)`

**Body**（可直接粘贴）: 

```markdown
## Summary
- New community extension `dataflow-finance-compliance`: finance compliance pipeline
- 4 operators: FinancialSensitiveInfoMasker / ComplianceRuleFilter / ComplianceRiskSampleEvaluator / DisclaimerInjector
- End-to-end FinanceCompliancePipeline (mask → filter → evaluate → disclaimer) with YAML config
- Deterministic PII masking (ID card / mobile / bank card / email) ordered to avoid cross-matching, with per-row mask summaries for audit
- LLM compliance rule filtering (return promises / touting / exaggeration) with fail-open default-keep behavior
- 30-sample labeled eval set (doc_type × compliant/violation × risk levels) + expansion script
- Data-card template + dataset card, COMPLIANCE.md (PIPL/DSL/JR-T 0197 mapping, mask ≠ anonymization boundaries)
- Offline tests (framework stubs, no LLM needed), CI on ubuntu/windows × py3.10–3.12

## Request
- [ ] List in "Awesome Works" → Community Extensions as:
  `dataflow-finance-compliance — Finance Compliance · PII Masking · Rule Filter · Risk Eval · Disclaimer. Maintainer: DingNova`

## Compliance
- Apache-2.0; simulated data only; see COMPLIANCE.md
```

## 4. ⚠️ compare URL 生成命令（Windows bash）

```bash
python - <<'PY'
from urllib.parse import quote
base = "https://github.com/OpenDCAI/DataFlow/compare/main...DingNova:dataflow-finance:feat/finance-compliance-extension?expand=1"
title = quote("feat: add dataflow-finance-compliance community extension (PII mask -> rule filter -> risk eval -> disclaimer)")
body = quote(open("SUBMISSION.md", encoding="utf-8").read().split("```markdown")[1].split("```")[0])
print(f"{base}&title={title}&body={body}")
PY
```

## 5. 合并后动作

1. ⚠️ 在 PR 评论或社区群申请 Awesome Works 收录；
2. ⚠️ PyPI 发布（需个人 token）：`python -m build && twine upload dist/*`；
3. 更新 CHANGELOG 与数据卡版本号。
