from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow  import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC

from dataflow.prompts.agenticrag import (
AtomicQAGeneratorPrompt,
MergeAtomicQAPrompt,
RefineAnswerPrompt,
MoreOptionalAnswersPrompt,
)
from dataflow.core.prompt import prompt_restrict

import pandas as pd
import json
import string
import re
from collections import Counter
from typing import List
import requests
import time
from tqdm import tqdm

def _clean_json_block(item: str) -> str:
        return item.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

def normalize_answer(s: str) -> str:
    if s.strip() in ["A", "B", "C", "D", "E"]:
        return s.strip().upper()

    def remove_articles(text):
        return re.sub(r"\b(a|an|the|do|does|is|are|was|were|of|under|in|at|on|with|by|for|from|about)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


@prompt_restrict(
    AtomicQAGeneratorPrompt,
    MergeAtomicQAPrompt,
    RefineAnswerPrompt,
    MoreOptionalAnswersPrompt,
)
@OPERATOR_REGISTRY.register()
class MultiHopRAGGenerator(OperatorABC):
    def __init__(
        self,
        llm_serving: LLMServingABC = None,
        retriever_url: str = None,
    ):
        self.logger = get_logger()
        self.llm_serving = llm_serving
        self.retriever_url = retriever_url

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return "MultiHopRAG 生成算子：根据原始 QA 的 A 做检索；对检索到的文档生成 atomic QA; 合成 multi-hop 候选问题并输出。"
        return "Generator for MultiHop RAG: retrieve by initial answer, generate atomic QA, merge into multi-hop candidates."

    def _validate_dataframe(self, dataframe: pd.DataFrame):
        required = [f"hop_{self.input_hop}"]
        forbidden = [self.output_key]
        missing = [k for k in required if k not in dataframe.columns]
        conflict = [k for k in forbidden if k in dataframe.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        if conflict:
            raise ValueError(f"The following columns already exist and would be overwritten: {conflict}")

    def retrieve_docs(self, query: str, original_docs: List[str], now_hop: int, topk: int=10) -> List[str]:
        if not self.retriever_url:
            return []
        response = requests.post(
            self.retriever_url,
            json={"query": query, "topk": topk + now_hop},
            timeout=60
        )
        data = response.json()
        all_docs = [doc.get("contents", "") for doc in data.get("results", [])]
        unique_docs = []
        for d in all_docs:
            if any(d.strip() == od.strip() for od in original_docs):
                continue
            if d not in unique_docs:
                unique_docs.append(d)
        filter_docs = [d for d in unique_docs if "(number)" not in d and "(decade)" not in d]
        return filter_docs[:topk]

    def _safe_json_load(self, text: str, stage: str):
        """
        Safely load JSON from LLM output.
        Return None if parsing fails.
        """
        if not text or not text.strip():
            self.logger.warning(f"[{stage}] Empty LLM output")
            return None

        cleaned = _clean_json_block(text)
        if not cleaned or not cleaned.strip():
            self.logger.warning(f"[{stage}] Empty cleaned JSON")
            return None

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            self.logger.warning(
                f"[{stage}] JSON decode failed: {e} | content: {cleaned[:200]}"
            )
            return None

    def run(
        self, 
        storage: DataFlowStorage, 
        input_hop: int, 
        input_question_key: str = "question", 
        input_answer_key: str = "answer", 
        input_doc_key: str = "doc",
        input_topk: int = 3,
        input_per_doc_qa: int = 1,
    ):
        self.input_hop = input_hop
        self.input_question_key = input_question_key
        self.input_answer_key = input_answer_key
        self.input_doc_key = input_doc_key
        self.output_key = f"hop_{input_hop + 1}"
        self.topk = input_topk
        self.per_doc_qa = input_per_doc_qa

        df = storage.read("dataframe")
        self._validate_dataframe(df)
        
        # step0: collect rows that match input_hop
        rows = []
        for row in df.itertuples(index=False):
            current_data = row._asdict()
            hop_num = max(int(k.split('_')[1]) for k in current_data.keys() if k.startswith("hop_"))
            if hop_num != input_hop:
                self.logger.error(f"Inconsistent input_hop: {input_hop} vs hop_num: {hop_num}")
                continue
            rows.append(current_data)

        # ---- Phase 1: build atomic prompts for ALL rows/docs and call model in batch ----
        atomic_prompts = []
        atomic_meta = []
        for i, current_data in tqdm(enumerate(rows), total=len(rows), desc="Generating atomic QA prompts"):
            hop_num = input_hop
            hop_key = f"hop_{hop_num}"
            now_question = current_data[hop_key][input_question_key]
            now_answer = current_data[hop_key][input_answer_key]
            docs = [current_data[k][input_doc_key] for k in current_data.keys() if input_doc_key in current_data.get(k, {})]
            retrieved_docs = self.retrieve_docs(now_answer, docs, now_hop=hop_num, topk=self.topk)
            for doc in retrieved_docs:
                atomic_qa_prompt = AtomicQAGeneratorPrompt().build_prompt(
                    gen_qa_num=self.per_doc_qa,
                    input_doc=doc
                )
                atomic_prompts.append(atomic_qa_prompt)
                atomic_meta.append({
                    "row_idx": i,
                    "doc": doc,
                    "hop_num": hop_num,
                    "orig_docs": docs,
                    "hop_key": hop_key,
                })
        if not atomic_prompts:
            final_df = pd.DataFrame([])
            out_file = storage.write(final_df)
            self.logger.info(f"MultiHop candidates saved to {out_file} (valid rows: {len(final_df)})")
            return

        atomic_outputs = self.llm_serving.generate_from_input(atomic_prompts)
        parsed_atomic = []
        for out in atomic_outputs:
            obj = self._safe_json_load(out, stage="atomic_qa")
            if obj is None:
                continue
            parsed_atomic.append(obj)

        # ---- Phase 2: build merge prompts for ALL atomic qas and call model in batch ----
        merge_prompts = []
        merge_meta = []  # map to row and carry mid_question/mid_answer/doc
        for idx, atomic in enumerate(parsed_atomic):
            meta = atomic_meta[idx]
            for qa in atomic:
                mid_question, mid_answer = qa['question'], qa['answer']
                current_data = rows[meta["row_idx"]]
                hop_num = meta["hop_num"]
                data_str = "\n".join(
                    f"Hop_{h}:\nQuestion: {current_data[f'hop_{h}']['final_question']}\nAnswer: {current_data[f'hop_{h}']['final_answer']}\nDocument: {current_data[f'hop_{h}']['doc']}"
                    for h in range(1, hop_num + 1)
                )
                merge_prompt = MergeAtomicQAPrompt().build_prompt(
                    Data=data_str,
                    New_question=mid_question,
                    New_answer=mid_answer,
                    New_document=meta["doc"],
                )
                merge_prompts.append(merge_prompt)
                merge_meta.append({
                    "row_idx": meta["row_idx"],
                    "hop_num": hop_num,
                    "hop_key": meta["hop_key"],
                    "mid_question": mid_question,
                    "mid_answer": mid_answer,
                    "doc": meta["doc"],
                    "current_data": rows[meta["row_idx"]],
                })

        if not merge_prompts:
            final_df = pd.DataFrame([])
            out_file = storage.write(final_df)
            self.logger.info(f"MultiHop candidates saved to {out_file} (valid rows: {len(final_df)})")
            return

        merge_outputs = self.llm_serving.generate_from_input(merge_prompts)
        parsed_merges = []
        for out in merge_outputs:
            obj = self._safe_json_load(out, stage="merge_qa")
            if obj is None:
                continue
            parsed_merges.append(obj)

        print("parsed_merges: ", len(parsed_merges))

        # ---- Phase 3: filter merges and build refine prompts ----
        refine_prompts = []
        refine_meta = []
        for idx, merge_result in enumerate(parsed_merges):
            meta = merge_meta[idx]
            if not merge_result:
                continue
            if merge_result.get("type") == "inference" and normalize_answer(merge_result.get("final_answer", "")) != normalize_answer(meta["mid_answer"]):
                continue
 
            refine_prompt = RefineAnswerPrompt().build_prompt(
                question=merge_result["final_question"],
                original_answer=merge_result["final_answer"],
            )
            refine_prompts.append(refine_prompt)
            refine_meta.append({
                "merge_result": merge_result,
                "row_idx": meta["row_idx"],
                "hop_num": meta["hop_num"],
                "doc": meta["doc"],
                "mid_question": meta["mid_question"],
                "mid_answer": meta["mid_answer"],
                "current_data": meta["current_data"],
            })

        if not refine_prompts:
            final_df = pd.DataFrame([])
            out_file = storage.write(final_df)
            self.logger.info(f"MultiHop candidates saved to {out_file} (valid rows: {len(final_df)})")
            return

        refine_outputs = self.llm_serving.generate_from_input(refine_prompts)
        parsed_refines = []
        valid_refine_meta = []
        for out, meta in zip(refine_outputs, refine_meta):
            obj = self._safe_json_load(out, stage="refine_answer")
            if obj is None:
                continue
            parsed_refines.append(obj)
            valid_refine_meta.append(meta)
        refine_meta = valid_refine_meta

        # ---- Phase 4: build optional prompts for ALL refines and batch call ----
        opt_prompts = []
        opt_meta = []
        for i, refine_result in enumerate(parsed_refines):
            meta = refine_meta[i]
            refined_answer = refine_result.get("refined_answer")
            opt_prompt = MoreOptionalAnswersPrompt().build_prompt(refined_answer=refined_answer)
            opt_prompts.append(opt_prompt)
            opt_meta.append({
                "refine_result": refine_result,
                "merge_result": meta["merge_result"],
                "row_idx": meta["row_idx"],
                "hop_num": meta["hop_num"],
                "doc": meta["doc"],
                "mid_question": meta["mid_question"],
                "mid_answer": meta["mid_answer"],
                "current_data": meta["current_data"],
            })

        if not opt_prompts:
            final_df = pd.DataFrame([])
            out_file = storage.write(final_df)
            self.logger.info(f"MultiHop candidates saved to {out_file} (valid rows: {len(final_df)})")
            return

        opt_outputs = self.llm_serving.generate_from_input(opt_prompts)
        parsed_opts = []
        valid_opt_meta = []
        for out, meta in zip(opt_outputs, opt_meta):
            obj = self._safe_json_load(out, stage="optional_answer")
            if obj is None:
                continue
            parsed_opts.append(obj)
            valid_opt_meta.append(meta)
        opt_meta = valid_opt_meta

        # ---- Phase 5: assemble new_rows from opt results and corresponding meta ----
        new_rows = []
        for i, opt_result in enumerate(parsed_opts):
            meta = opt_meta[i]
            merge_result = meta["merge_result"]
            refine_result = meta["refine_result"]
            current_data = meta["current_data"]

            new_hop_key = f"hop_{meta['hop_num'] + 1}"
            new_row = current_data.copy()
            new_row[new_hop_key] = {
                "question": meta["mid_question"],
                "answer": meta["mid_answer"],
                "doc": meta["doc"],
                "final_question": merge_result["final_question"],
                "final_answer": refine_result["refined_answer"],
                "optional_answers": opt_result,
                "qa_type": merge_result["type"],
            }
            new_rows.append(new_row)

        final_df = pd.DataFrame(new_rows)
        out_file = storage.write(final_df)
        self.logger.info(f"MultiHop candidates saved to {out_file} (valid rows: {len(final_df)})")