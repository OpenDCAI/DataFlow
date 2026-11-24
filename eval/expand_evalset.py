"""LLM 辅助扩充评测集：30 条人工金标 → 目标规模（默认 200 条）。

用法：
    python eval/expand_evalset.py --target 200 --output eval/finance_compliance_eval_set_expanded.jsonl

说明：
- 扩充样本由 LLM 参照金标风格生成，产出后必须人工抽检 ≥20%
  （重点核对 expected_compliant / expected_risk 标注准确性）。
- 评测文本均为模拟撰写，不包含真实机构名称、真实个人与真实账号信息。
"""
import argparse
import json
import sys
from pathlib import Path

DOC_TYPES = ["research", "announcement", "marketing"]
SCENARIOS = [
    "收益承诺类违规样本",
    "荐股诱导类违规样本",
    "夸大宣传类违规样本",
    "含手机号/邮箱等敏感信息的合规样本",
    "常规合规研报表述",
    "常规合规公告表述",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=Path, default=Path(__file__).parent / "finance_compliance_eval_set.jsonl")
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
            "（text 为空，必须由人工/LLM 补齐并抽检后才能使用）。",
            file=sys.stderr,
        )
        for i in range(n_needed):
            row = {
                "id": f"comp-ext-{i + 1:03d}",
                "doc_type": DOC_TYPES[i % len(DOC_TYPES)],
                "expected_compliant": None,  # TODO(expansion): fill then manual spot-check
                "expected_risk": None,  # TODO(expansion): fill then manual spot-check
                "contains_sensitive": None,  # TODO(expansion): fill then manual spot-check
                "text": "",
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
