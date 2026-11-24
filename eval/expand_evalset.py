"""LLM 辅助扩充评测集：30 条人工金标 → 目标规模（默认 200 条）。

用法：
    python eval/expand_evalset.py --target 200 --output eval/finance_text2sql_eval_set_expanded.jsonl

说明：
- 扩充样本由 LLM 基于 schema 模板生成，产出后必须人工抽检 ≥20%
  （重点检查 gold_sql 可执行性与语义一致性），抽检要求见数据卡。
- 仅使用模板化 schema，不包含任何真实机构敏感数据。
"""
import argparse
import json
import random
import sys
from pathlib import Path

SCHEMA_TEMPLATES = [
    "表 revenue(company TEXT, year INT, revenue NUMERIC)",
    "表 revenue(company TEXT, year INT, revenue NUMERIC); 表 cost(company TEXT, year INT, cost NUMERIC)",
    "表 asset(bank TEXT, year INT, npl_ratio NUMERIC)",
    "表 loans(bank TEXT, year INT, deposits NUMERIC, loans NUMERIC)",
    "表 research(industry TEXT, company TEXT, year INT)",
    "表 mktcap(code TEXT, date TEXT, mktcap NUMERIC)",
]

QUESTION_PATTERNS = [
    "查询{year}年{entity}的{metric}",
    "统计{year}年{metric}排名前{n}的{entity}",
    "计算{entity}{year}年{metric}的同比增速",
    "查询{year}年{metric}高于行业平均的{entity}列表",
]

COMPANIES = ["甲公司", "乙公司", "丙公司", "丁公司"]
METRICS = ["营业收入", "净利润", "不良贷款率", "存贷比", "总资产"]


def build_prompt(sample):
    return (
        "请基于以下金融 Text2SQL 评测样本的风格，生成一个新样本。\n"
        f"示例：{json.dumps(sample, ensure_ascii=False)}\n"
        '只输出一个 JSON 对象：{"question": ..., "schema": ..., "gold_sql": ..., '
        '"difficulty": "easy|medium|hard"}，不要输出其他内容。'
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=Path, default=Path(__file__).parent / "finance_text2sql_eval_set.jsonl")
    parser.add_argument("--target", type=int, default=200)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed-num", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed_num)
    output = args.output or args.seed.with_name(args.seed.stem + "_expanded.jsonl")

    with open(args.seed, encoding="utf-8") as f:
        seed_rows = [json.loads(line) for line in f if line.strip()]

    expanded = list(seed_rows)
    n_needed = args.target - len(expanded)
    if n_needed <= 0:
        print(f"seed already >= target ({len(expanded)}), nothing to do")
        return

    try:
        from dataflow.serving import APILLMServing_request  # noqa: F401
    except ImportError:
        print(
            "[offline-fallback] 未安装 open-dataflow，仅生成结构占位样本骨架"
            "（gold_sql 为空，必须由人工/LLM 补齐并抽检后才能使用）。",
            file=sys.stderr,
        )
        for i in range(n_needed):
            tpl = random.choice(SCHEMA_TEMPLATES)
            pattern = random.choice(QUESTION_PATTERNS)
            row = {
                "id": f"t2s-ext-{i + 1:03d}",
                "difficulty": random.choice(["easy", "medium", "hard"]),
                "question": pattern.format(
                    year=random.choice([2022, 2023, 2024]),
                    entity=random.choice(COMPANIES),
                    metric=random.choice(METRICS),
                    n=random.choice([3, 5, 10]),
                ),
                "schema": tpl,
                "gold_sql": "",  # TODO(expansion): fill via LLM then manual spot-check
            }
            expanded.append(row)
    else:
        raise NotImplementedError(
            "在线扩充需接入实际 LLM serving：请实现 generate_from_input 调用并 "
            "逐条解析 JSONL 输出；产出后人工抽检 ≥20%。"
        )

    with open(output, "w", encoding="utf-8") as f:
        for row in expanded:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"written {len(expanded)} rows -> {output}")
    print("REMINDER: 人工抽检 >=20% 后方可用于评测。")


if __name__ == "__main__":
    main()
