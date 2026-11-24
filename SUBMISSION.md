# 提交物料包（人工执行用）

> 向 OpenDCAI/DataFlow 主仓库提交 PR 与申请 Awesome Works 收录所需物料（dataflow-finance-eval）。
> 所有代码/文档已就绪，需要人工执行的只有标 ⚠️ 的步骤。

## 1. 提交前检查清单

- [ ] 如需在 PyPI/PR 中展示邮箱，给 `pyproject.toml` 的 authors 补 email（可选，用 GitHub noreply 邮箱）
- [ ] 本地验证：`pip install -e ".[dev]" && pytest`（离线测试，不需要 open-dataflow）
- [ ] 如已装 open-dataflow + sqlglot，开启 use_ast=True 跑一次 AST 比较冒烟

## 2. ⚠️ Fork → Push → PR 三步（沿用既定手动流程）

1. **Fork**（共用 fork，已完成）：fork 统一使用 `DingNova/dataflow-finance`（OpenDCAI/DataFlow 的唯一 fork，同一账号只能 fork 一次，四个金融子包共用此仓库，各自使用独立分支）。
2. **Push**：
   ```bash
   cd dataflow-finance-eval
   git remote add fork https://github.com/DingNova/dataflow-finance.git
   # 若 fork remote 已存在（指向旧地址），改用：git remote set-url fork https://github.com/DingNova/dataflow-finance.git
   git push -f fork feat/finance-eval-extension
   # ⚠️ 必须 -f：分支历史需嫁接到上游 main（与 PR #594 同构），远端旧分支为独立历史，普通 push 会被拒绝
   ```
   （本地已完成 `git init -b feat/finance-eval-extension` 与初始提交）
3. **Create PR**：compare URL 生成命令见第 4 节。

## 3. ⚠️ PR 标题与描述（预填版）

**Title**: `feat: add dataflow-finance-eval community extension (QA / SQL / hallucination eval suite)`

**Body**（可直接粘贴）:

```markdown
## Summary
- New community extension `dataflow-finance-eval`: finance task evaluation suite
- 4 operators: FinanceQASampleEvaluator / FinanceSQLSampleEvaluator / FinanceHallucinationSampleEvaluator / EvalReportGenerator
- End-to-end FinanceEvalPipeline (qa → sql → hallucination → report) with YAML config
- Deterministic QA eval (normalized + numeric-tolerance matching) and SQL eval (normalized text; optional sqlglot AST) — no LLM needed
- LLM hallucination detection (numeric consistency / citation grounding / causal support) with explicit fail-closed policy
- 30-sample labeled eval set; deterministic gold fields cross-checked against evaluator logic and guarded by unit test
- Natural evaluation downstream of dataflow-finance-qa / dataflow-finance-text2sql
- Data-card template + dataset card, COMPLIANCE.md (methodology limits, JR/T 0197 mapping)
- Offline tests (framework stubs, no LLM/sqlglot needed), CI on ubuntu/windows × py3.10–3.12

## Request
- [ ] List in "Awesome Works" → Community Extensions as:
  `dataflow-finance-eval — Finance Eval · QA Accuracy · SQL Match · Hallucination Detection · Reporting. Maintainer: DingNova`

## Compliance
- Apache-2.0; simulated data only; see COMPLIANCE.md
```

## 4. ⚠️ compare URL 生成命令（Windows bash）

```bash
python - <<'PY'
from urllib.parse import quote
base = "https://github.com/OpenDCAI/DataFlow/compare/main...DingNova:dataflow-finance:feat/finance-eval-extension?expand=1"
title = quote("feat: add dataflow-finance-eval community extension (QA / SQL / hallucination eval suite)")
body = quote(open("SUBMISSION.md", encoding="utf-8").read().split("```markdown")[1].split("```")[0])
print(f"{base}&title={title}&body={body}")
PY
```

## 5. 合并后动作

1. ⚠️ 在 PR 评论或社区群申请 Awesome Works 收录；
2. ⚠️ PyPI 发布（需个人 token）：`python -m build && twine upload dist/*`；
3. 更新 CHANGELOG 与数据卡版本号。
