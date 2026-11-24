"""LLM 辅助扩充评测集：30 条人工金标 → 目标规模（默认 200 条）。

用法：
    python eval/expand_evalset.py --target 200 --output eval/finance_eval_sample_set_expanded.jsonl

说明：
- 扩充样本由 LLM 参照金标风格生成，产出后必须人工抽检 ≥20%
  （重点核对 expected_qa_correct / expected_sql_correct / gold_grounded 标注）。
- 评测文本均为模拟撰写，不含真实机构与真实市场数据。
"""
import argparse
import json
import sys
from pathlib import Path

SCENARIOS = [
    "数值完全一致的正样本",
    "数值失真的幻觉负样本",
    "虚构引用的幻觉负样本",
    "SQL 聚合函数错误的负样本",
    "SQL 语义等价正样本",
    "原文信息不足导致的超范围回答负样本",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=Path, default=Path(__file__).parent / "finance_eval_sample_set.jsonl")
    parser.add_argument("--target", type=int, default=200)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed-num", type=int, default=42)
    args = parser.parse_args()

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
            "（answer/sql 为空，必须由人工/LLM 补齐并抽检后才能使用）。",
            file=sys.stderr,
        )
        for i in range(n_needed):
            row = {
                "id": f"ev-ext-{i + 1:03d}",
                "difficulty": "medium" if i % 2 else "hard",
                "text": "",
                "question": "",
                "answer": "",  # TODO(expansion): fill then manual spot-check
                "gold_answer": "",
                "sql": "",  # TODO(expansion): fill then manual spot-check
                "gold_sql": "",
                "gold_grounded": None,
                "expected_qa_correct": None,
                "expected_sql_correct": None,
                "_scenario_hint": SCENARIOS[i % len(SCENARIOS)],
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
