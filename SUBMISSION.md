# 提交物料包（人工执行用）

> 本文件列出向 OpenDCAI/DataFlow 主仓库提交 PR 与申请 Awesome Works 收录所需的全部物料。
> 所有代码/文档已就绪，需要人工执行的只有标 ⚠️ 的步骤。

## 1. 提交前检查清单

- [ ] 替换 `pyproject.toml` 中两处 TODO：author（DingNova → 实际 ID+邮箱）、Homepage URL
- [ ] 确认 LICENSE Copyright 行按需补充署名（可保留默认）
- [ ] 本地验证：`pip install -e ".[dev]" && pytest`（离线测试，不需要 open-dataflow）
- [ ] 如已装 open-dataflow + PDF 依赖，跑一次真实 PDF 冒烟

## 2. ⚠️ Fork → Push → PR 三步（沿用既定手动流程）

1. **Fork**：浏览器打开 `https://github.com/OpenDCAI/DataFlow/fork`（或先建个人仓库 `dataflow-finance-qa`，两种路线二选一，推荐 A）：
   - 路线 A（推荐）：独立扩展仓库，push 到 `DingNova/dataflow-finance-qa`，再向主仓库提 PR 申请收录；
   - 路线 B：直接 fork 主仓库，在 `dataflow/extensions/` 或文档 Awesome 列表处加条目。
2. **Push**：
   ```bash
   cd dataflow-finance-qa
   git init && git checkout -b feat/finance-qa-extension
   git add . && git commit -m "feat: initial dataflow-finance-qa extension (operators + pipeline + eval + data cards + compliance)"
   git remote add fork https://github.com/DingNova/dataflow-finance-qa.git
   git push fork feat/finance-qa-extension
   ```
3. **Create PR**（compare URL 由 AI 程序化生成，见下节）。

## 3. ⚠️ PR 标题与描述（预填版）

**Title**: `feat: add dataflow-finance-qa community extension (financial report → QA)`

**Body**（可直接粘贴）:

```markdown
## Summary
- New community extension `dataflow-finance-qa`: financial report/research-note PDF → source-grounded QA dataset
- 4 operators: FinancialPDFParser / FinancialQAGenerator / FinancialHallucinationFilter / NumericConsistencyRefiner
- End-to-end FinanceQAPipeline (parse → generate → filter → refine) with YAML config
- 30-sample golden eval set (factual/numeric/causal × 3 difficulty levels) + LLM expansion script
- Data-card template + dataset card, COMPLIANCE.md (JR/T 0197 mapping, source red-lines, disclaimer)
- Offline tests (framework stubs, no LLM needed), CI on ubuntu/windows × py3.10–3.12

## Request
- [ ] List in "Awesome Works" → Community Extensions as:
  `dataflow-finance-qa — Financial QA · PDF Parsing · Hallucination Filter · Numeric Consistency. Maintainer: DingNova`

## Compliance
- Apache-2.0; simulated/public data only; see COMPLIANCE.md
```

## 4. ⚠️ compare URL 生成命令（Windows bash）

```bash
python - <<'PY'
from urllib.parse import quote
base = "https://github.com/OpenDCAI/DataFlow/compare/main...DingNova:dataflow-finance-qa:feat/finance-qa-extension?expand=1"
title = quote("feat: add dataflow-finance-qa community extension (financial report -> QA)")
body = quote(open("SUBMISSION.md", encoding="utf-8").read().split("```markdown")[1].split("```")[0])
print(f"{base}&title={title}&body={body}")
PY
```

## 5. 合并后动作

1. ⚠️ 在 PR 评论或社区群申请 Awesome Works 收录；
2. ⚠️ PyPI 发布（需个人 token）：
   ```bash
   python -m build && twine upload dist/*
   ```
3. 更新 CHANGELOG 与数据卡版本号。
