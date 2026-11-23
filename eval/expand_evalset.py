"""用 LLM 把金标评测集扩充到目标规模（默认 200 条）。

用法：
  export DF_API_KEY=sk-xxx
  python eval/expand_evalset.py --target 200 --out eval/finance_qa_eval_set_full.jsonl

注意：扩充产物属于合成金标，必须经过人工抽检（见 COMPLIANCE.md 第 4 节）
后方可用于正式评测发布。
"""
import argparse
import json
import os
import random

TYPES = ["factual", "numeric", "causal"]
DIFFICULTIES = ["easy", "medium", "hard"]
SCENARIOS = [
    "上市公司年度报告片段", "券商研究报告片段", "银行财务摘要", "基金定期报告片段",
    "保险公司经营摘要", "招股说明书节选", "业绩快报片段", "债券募集说明书节选",
]

SYSTEM_PROMPT = (
    "你是金融评测集构造专家。根据指定的场景、题型与难度，构造一条金融问答金标样本。\n"
    "要求：\n"
    "1. source_text 为 2-4 句自洽的模拟金融文本（标注为模拟数据，不使用真实公司名，用甲/乙/丙等）；\n"
    "2. 问题必须且只能由 source_text 回答；\n"
    "3. numeric 题必须给出计算过程；\n"
    '4. 仅输出一个 JSON 对象：{"qtype":..., "difficulty":..., "scenario":..., '
    '"source_text":..., "question":..., "answer":...}'
)


def expand(target: int, seed_path: str, out_path: str, llm_serving=None) -> None:
    """seed_path 中的样本原样保留（金标），新增样本由 LLM 合成。"""
    import pandas as pd

    with open(seed_path, encoding="utf-8") as f:
        seed = [json.loads(l) for l in f if l.strip()]

    records = [dict(r) for r in seed]
    if llm_serving is None:
        # 未提供 LLM 时：基于种子样本做扰动扩充占位（需 LLM 替换后人工审核）
        rng = random.Random(42)
        while len(records) < target:
            base = dict(rng.choice(seed))
            base["id"] = f"finqa-syn-{len(records):03d}"
            base["notes"] = "LLM 占位样本：需用 expand_evalset.py --llm 重新生成并人工审核"
            records.append(base)
    else:
        inputs, metas = [], []
        while len(records) + len(metas) < target:
            qtype = TYPES[len(metas) % 3]
            difficulty = DIFFICULTIES[(len(metas) // 3) % 3]
            scenario = SCENARIOS[(len(metas) // 5) % len(SCENARIOS)]
            metas.append((qtype, difficulty, scenario))
            inputs.append(
                f"场景：{scenario}；题型：{qtype}；难度：{difficulty}。请构造一条样本。"
            )
        responses = llm_serving.generate_from_input(
            user_inputs=inputs, system_prompt=SYSTEM_PROMPT
        )
        start = len(records)
        for i, resp in enumerate(responses):
            try:
                obj = json.loads(resp.strip().strip("`"))
                obj["id"] = f"finqa-syn-{start + i:03d}"
                records.append(obj)
            except json.JSONDecodeError:
                print(f"[warn] skip malformed sample #{i}")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    pd.DataFrame(records).to_json(out_path, orient="records", lines=True, force_ascii=False)
    print(f"Wrote {len(records)} samples to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=os.path.join(os.path.dirname(__file__), "finance_qa_eval_set.jsonl"))
    parser.add_argument("--target", type=int, default=200)
    parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "finance_qa_eval_set_full.jsonl"))
    args = parser.parse_args()
    expand(args.target, args.seed, args.out)
