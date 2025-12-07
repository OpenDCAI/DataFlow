from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow  import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC

from dataflow.prompts.agenticrag import (
InferenceCheckPrompt,
ComparisonCheckPrompt,
ReasoningPrompt,
ComparisonReasoningPrompt,
SingleHopPrompt,
MultihopInferencePrompt,
MultihopComparisonPrompt,
EssEqPrompt,
)
from dataflow.core.prompt import prompt_restrict
from typing import List
import pandas as pd
import json
import requests
from itertools import combinations

def _clean_json_block(item: str) -> str:
        return item.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

@prompt_restrict(
    InferenceCheckPrompt,
    ComparisonCheckPrompt,
    ReasoningPrompt,
    ComparisonReasoningPrompt,
    SingleHopPrompt,
    MultihopInferencePrompt,
    MultihopComparisonPrompt,
    EssEqPrompt,
)
@OPERATOR_REGISTRY.register()
class MultiHopRAGVerifier(OperatorABC):
    def __init__(
        self,
        llm_serving: LLMServingABC = None,
    ):
        self.logger = get_logger()
        self.llm_serving = llm_serving

    @staticmethod
    def get_desc(lang: str = "zh"):
        return "MultiHopRAG 验证算子：对 multi_hop_data 中每个候选进行多步验证并返回合格的数据。" if lang == "zh" else "Verifier for MultiHop RAG."

    def run(self, storage: DataFlowStorage):
        df = storage.read("dataframe")
        
        rows = []
        for row in df.itertuples(index=False):
            current_data = row._asdict()
            hop_num = max(int(k.split('_')[1]) for k in current_data.keys() if k.startswith("hop_"))
            rows.append(current_data)

        if not rows:
            final_df = pd.DataFrame([])
            out_file = storage.write(final_df)
            self.logger.info(f"MultiHop candidates saved to {out_file} (valid rows: {len(final_df)})")
            return

        # ---- Phase 1: build check prompts (InferenceCheckPrompt / ComparisonCheckPrompt) ----
        check_prompts = []
        check_meta = []
        for i, current_data in enumerate(rows):
            hop_num = max(int(k.split('_')[1]) for k in current_data.keys() if k.startswith("hop_"))
            hop_key = f"hop_{hop_num}"
            qa_type = current_data[hop_key]["qa_type"]
            final_question = current_data[hop_key]["final_question"]
            final_answer = current_data[hop_key]["final_answer"]

            if qa_type == 'inference':
                check_prompt = InferenceCheckPrompt().build_prompt(
                    Question1=current_data[f"hop_{hop_num-1}"]["final_question"],
                    Answer1=current_data[f"hop_{hop_num-1}"]["final_answer"],
                    Document1=current_data[f"hop_{hop_num-1}"]["doc"],
                    Question2=current_data[hop_key]["question"],
                    Answer2=current_data[hop_key]["answer"],
                    Document2=current_data[hop_key]["doc"],
                    Final_question=final_question,
                    Final_answer=final_answer,
                    qa_type=qa_type,
                )
            else:
                check_prompt = ComparisonCheckPrompt().build_prompt(
                    Question1=current_data[f"hop_{hop_num-1}"]["final_question"],
                    Answer1=current_data[f"hop_{hop_num-1}"]["final_answer"],
                    Document1=current_data[f"hop_{hop_num-1}"]["doc"],
                    Question2=current_data[hop_key]["question"],
                    Answer2=current_data[hop_key]["answer"],
                    Document2=current_data[hop_key]["doc"],
                    Final_question=final_question,
                    Final_answer=final_answer,
                    qa_type=qa_type,
                )

            check_prompts.append(check_prompt)
            check_meta.append({
                "row_idx": i,
                "hop_num": hop_num,
                "hop_key": hop_key,
                "qa_type": qa_type,
                "final_question": final_question,
                "final_answer": final_answer,
                "current_data": current_data,
            })

        check_outputs = self.llm_serving.generate_from_input(check_prompts) if check_prompts else []
        parsed_checks = []
        for out in check_outputs:
            cleaned = _clean_json_block(out)
            parsed_checks.append(json.loads(cleaned))

        passed_after_check = []
        for idx, check_result in enumerate(parsed_checks):
            meta = check_meta[idx]
            if not check_result:
                continue
            if str(check_result.get("valid", "")).lower() != 'true':
                continue
            passed_after_check.append({
                "row_idx": meta["row_idx"],
                "hop_num": meta["hop_num"],
                "hop_key": meta["hop_key"],
                "qa_type": meta["qa_type"],
                "final_question": meta["final_question"],
                "final_answer": meta["final_answer"],
                "current_data": meta["current_data"],
            })

        if not passed_after_check:
            final_df = pd.DataFrame([])
            out_file = storage.write(final_df)
            self.logger.info(f"MultiHop candidates saved to {out_file} (valid rows: {len(final_df)})")
            return

        # ---- Phase 2: reasoning prompts (one per passed row) ----
        reasoning_prompts = []
        reasoning_meta = []
        print("passed_after_check: ", len(passed_after_check))
        for item in passed_after_check:
            qa_type = item["qa_type"]
            final_question = item["final_question"]
            if qa_type == 'inference':
                r_prompt = ReasoningPrompt().build_prompt(problem=final_question)
            else:
                r_prompt = ComparisonReasoningPrompt().build_prompt(problem=final_question)
            reasoning_prompts.append(r_prompt)
            reasoning_meta.append(item)

        reasoning_outputs = self.llm_serving.generate_from_input(reasoning_prompts) if reasoning_prompts else []
        
        # ---- Phase 3: judge reasoning results with EssEq (llm_judge) ----
        judge_prompts = []
        judge_meta = []
        for idx, reasoning_result in enumerate(reasoning_outputs):
            meta = reasoning_meta[idx]
            judge_prompt = EssEqPrompt().build_prompt(
                question=meta["final_question"],
                golden_answer=meta["final_answer"],
                other_answer=reasoning_result,
            )
            judge_prompts.append(judge_prompt)
            judge_meta.append(meta)

        judge_outputs = self.llm_serving.generate_from_input(judge_prompts) if judge_prompts else []
        parsed_judges = []
        for out in judge_outputs:
            cleaned = _clean_json_block(out)
            parsed_judges.append(json.loads(cleaned))

        passed_after_reasoning = []
        for idx, judge_res in enumerate(parsed_judges):
            meta = judge_meta[idx]
            if judge_res.get("answer_score", 0) >= 1:
                continue
            passed_after_reasoning.append(meta)

        if not passed_after_reasoning:
            final_df = pd.DataFrame([])
            out_file = storage.write(final_df)
            self.logger.info(f"MultiHop candidates saved to {out_file} (valid rows: {len(final_df)})")
            return

        # ---- Phase 4: for每个候选构建所有 combos 的 singlehop prompts 并批量调用 ----
        singlehop_prompts = []
        singlehop_meta = []
        print("passed_after_reasoning: ", len(passed_after_reasoning))
        for meta in passed_after_reasoning:
            current_data = meta["current_data"]
            current_full_docs = [current_data[k]["doc"] for k in current_data.keys()][:-1]
            for combo in combinations(current_full_docs, max(1, len(current_full_docs)-1)):
                if len(combo) == 1:
                    combo_type = "single_doc"
                    combo_docs = combo[0]
                else:
                    combo_type = f"{len(combo)}_docs_combination"
                    combo_docs = "\n\n".join(combo)
                single_prompt = SingleHopPrompt().build_prompt(
                    Document=combo_docs,
                    Question=meta["final_question"],
                )
                singlehop_prompts.append(single_prompt)
                singlehop_meta.append({
                    "row_meta": meta,
                    "combo_type": combo_type,
                })

        single_outputs = self.llm_serving.generate_from_input(singlehop_prompts) if singlehop_prompts else []

        # ---- Phase 5: judge all singlehop results with EssEq in batch ----
        single_judge_prompts = []
        single_judge_meta = []
        for idx, single_out in enumerate(single_outputs):
            meta = singlehop_meta[idx]
            row_meta = meta["row_meta"]
            judge_prompt = EssEqPrompt().build_prompt(
                question=row_meta["final_question"],
                golden_answer=row_meta["final_answer"],
                other_answer=single_out,
            )
            single_judge_prompts.append(judge_prompt)
            single_judge_meta.append(meta)

        single_judge_outputs = self.llm_serving.generate_from_input(single_judge_prompts) if single_judge_prompts else []
        parsed_single_judges = []
        for out in single_judge_outputs:
            cleaned = _clean_json_block(out)
            parsed_single_judges.append(json.loads(cleaned))

        row_fail_map = {}
        for idx, judge_res in enumerate(parsed_single_judges):
            meta = single_judge_meta[idx]
            row_idx = meta["row_meta"]["row_idx"]
            if judge_res.get("answer_score", 0) >= 1:
                row_fail_map[row_idx] = True

        passed_after_combos = []
        for meta in passed_after_reasoning:
            if row_fail_map.get(meta["row_idx"], False):
                continue
            passed_after_combos.append(meta)

        if not passed_after_combos:
            final_df = pd.DataFrame([])
            out_file = storage.write(final_df)
            self.logger.info(f"MultiHop candidates saved to {out_file} (valid rows: {len(final_df)})")
            return

        # ---- Phase 6: 构建 multihop prompts 并批量调用 ----
        multihop_prompts = []
        multihop_meta = []
        print("Passed after combos: ", len(passed_after_combos))
        for meta in passed_after_combos:
            current_data = meta["current_data"]
            hop_num = meta["hop_num"]
            Data = []
            for h in range(1, hop_num):
                info = current_data[f"hop_{h}"]
                Data.append(
                    f"Question{h}: {info['question']}\n"
                    f"Answer{h}: {info['answer']}\n"
                    f"Supporting Document{h}: {info['doc']}"
                )
            if meta["qa_type"] == 'inference':
                Data.append(
                    f"Question{hop_num}: {current_data[f'hop_{hop_num}']['question']}\n"
                    f"Supporting Document{hop_num}: {current_data[f'hop_{hop_num}']['doc']}"
                )
                m_prompt = MultihopInferencePrompt().build_prompt(
                    Data="\n".join(Data),
                    FinalQuestion=meta["final_question"],
                )
            else:
                Data.append(
                    f"Question{hop_num}: {current_data[f'hop_{hop_num}']['question']}\n"
                    f"Answer{hop_num}: {current_data[f'hop_{hop_num}']['answer']}\n"
                    f"Supporting Document{hop_num}: {current_data[f'hop_{hop_num}']['doc']}"
                )
                m_prompt = MultihopComparisonPrompt().build_prompt(
                    Data="\n".join(Data),
                    FinalQuestion=meta["final_question"],
                )
            multihop_prompts.append(m_prompt)
            multihop_meta.append(meta)

        multihop_outputs = self.llm_serving.generate_from_input(multihop_prompts) if multihop_prompts else []

        # ---- Phase 7: judge multihop results with EssEq ----
        final_judge_prompts = []
        final_judge_meta = []
        for idx, m_out in enumerate(multihop_outputs):
            meta = multihop_meta[idx]
            judge_prompt = EssEqPrompt().build_prompt(
                question=meta["final_question"],
                golden_answer=meta["final_answer"],
                other_answer=m_out,
            )
            final_judge_prompts.append(judge_prompt)
            final_judge_meta.append(meta)

        final_judge_outputs = self.llm_serving.generate_from_input(final_judge_prompts) if final_judge_prompts else []
        parsed_final_judges = []
        for out in final_judge_outputs:
            cleaned = _clean_json_block(out)
            parsed_final_judges.append(json.loads(cleaned))

        verified_rows = []
        for idx, judge_res in enumerate(parsed_final_judges):
            meta = final_judge_meta[idx]
            if judge_res.get("answer_score", 0) < 1:
                continue
            verified_rows.append(meta["current_data"])

        final_df = pd.DataFrame(verified_rows)
        out_file = storage.write(final_df)
        self.logger.info(f"MultiHop candidates saved to {out_file} (valid rows: {len(final_df)})")