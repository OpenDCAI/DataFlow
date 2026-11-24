# 提交物料包（人工执行用）

> 向 OpenDCAI/DataFlow 主仓库提交 PR 与申请 Awesome Works 收录所需物料（dataflow-finance-text2sql）。
> 所有代码/文档已就绪，需要人工执行的只有标 ⚠️ 的步骤。

## 1. 提交前检查清单

- [ ] 如需在 PyPI/PR 中展示邮箱，给 `pyproject.toml` 的 authors 补 email（可选，用 GitHub noreply 邮箱）
- [ ] 本地验证：`pip install -e ".[dev]" && pytest`（离线测试，不需要 open-dataflow）
- [ ] 如已装 open-dataflow + sqlglot，用 sqlite 内存库跑一次真实执行冒烟

## 2. ⚠️ Fork → Push → PR 三步（沿用既定手动流程）

1. **Fork**（共用 fork，已完成）：fork 统一使用 `DingNova/dataflow-finance`（OpenDCAI/DataFlow 的唯一 fork，同一账号只能 fork 一次，四个金融子包共用此仓库，各自使用独立分支）。
2. **Push**：
   ```bash
   cd dataflow-finance-text2sql
   git remote add fork https://github.com/DingNova/dataflow-finance.git
   # 若 fork remote 已存在（指向旧地址），改用：git remote set-url fork https://github.com/DingNova/dataflow-finance.git
   git push -f fork feat/finance-text2sql-extension
   # ⚠️ 必须 -f：分支历史需嫁接到上游 main（与 PR #594 同构），远端旧分支为独立历史，普通 push 会被拒绝
   ```
   （本地已完成 `git init -b feat/finance-text2sql-extension` 与初始提交）
3. **Create PR**：compare URL 生成命令见第 4 节。

## 3. ⚠️ PR 标题与描述（预填版）

**Title**: `feat: add dataflow-finance-text2sql community extension (finance question → SQL)`

**Body**（可直接粘贴）:

```markdown
## Summary
- New community extension `dataflow-finance-text2sql`: finance-domain text-to-SQL pipeline
- 4 operators: FinanceSchemaLinker / FinanceSQLGenerator / FinanceSQLSyntaxFilter / FinanceSQLRepairRefiner
- End-to-end FinanceText2SQLPipeline (link → generate → filter → repair) with YAML config
- 30-sample golden eval set (easy/medium/hard: aggregates, joins, YoY growth, window functions, subqueries) + expansion script
- sqlglot-based syntax validation with regex fallback; error-driven LLM SQL repair
- Data-card template + dataset card, COMPLIANCE.md (JR/T 0197 mapping, schema red-lines, disclaimer)
- Offline tests (framework stubs, no LLM/sqlglot needed), CI on ubuntu/windows × py3.10–3.12

## Request
- [ ] List in "Awesome Works" → Community Extensions as:
  `dataflow-finance-text2sql — Finance Text2SQL · Schema Linking · Syntax Filter · SQL Repair. Maintainer: DingNova`

## Compliance
- Apache-2.0; templated/mock schemas only; see COMPLIANCE.md
```

## 4. ⚠️ compare URL 生成命令（Windows bash）

```bash
python - <<'PY'
from urllib.parse import quote
base = "https://github.com/OpenDCAI/DataFlow/compare/main...DingNova:dataflow-finance:feat/finance-text2sql-extension?expand=1"
title = quote("feat: add dataflow-finance-text2sql community extension (finance question -> SQL)")
body = quote(open("SUBMISSION.md", encoding="utf-8").read().split("```markdown")[1].split("```")[0])
print(f"{base}&title={title}&body={body}")
PY
```

## 5. 合并后动作

1. ⚠️ 在 PR 评论或社区群申请 Awesome Works 收录；
2. ⚠️ PyPI 发布（需个人 token）：`python -m build && twine upload dist/*`；
3. 更新 CHANGELOG 与数据卡版本号。
