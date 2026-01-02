import asyncio
import json
import re
import string
import pandas as pd
from itertools import combinations
from typing import Any, List, Dict

# --- Dataflow 核心组件 ---
from dataflow.utils.storage import FileStorage
from dataflow.serving import APILLMServing_request
from dataflow.serving import FlashRAGServing 

# --- 算子导入 ---
from dataflow.operators.core_text import (
    PandasOperator, 
    PromptTemplatedGenerator, 
    RetrievalGenerator, 
    GeneralFilter
)

# --- Prompt 模板导入 ---
from dataflow.prompts.core_text import StrFormatPrompt
from dataflow.prompts.agenticrag import (
    AtomicQAGeneratorPrompt,
    MergeAtomicQAPrompt,
    RefineAnswerPrompt,
    MoreOptionalAnswersPrompt,
    # Verifier Prompts
    InferenceCheckPrompt,
    ComparisonCheckPrompt,
    ReasoningPrompt,
    ComparisonReasoningPrompt,
    SingleHopPrompt,
    MultihopInferencePrompt,
    MultihopComparisonPrompt,
    EssEqPrompt,
)

# ==========================================
# 1. 通用辅助函数
# ==========================================

def _clean_json_block(item: Any) -> str:
    if not isinstance(item, str): return "{}"
    return item.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

def safe_parse_json(text: Any) -> Any:
    try:
        return json.loads(_clean_json_block(text))
    except:
        return {} 

def normalize_answer(s: str) -> str:
    if not isinstance(s, str): return ""
    if s.strip() in ["A", "B", "C", "D", "E"]:
        return s.strip().upper()
    def remove_articles(text):
        return re.sub(r"\b(a|an|the|do|does|is|are|was|were|of|under|in|at|on|with|by|for|from|about)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(s.lower())))

# ==========================================
# 2. 生成阶段辅助函数
# ==========================================

def prepare_retrieval_query(df: pd.DataFrame, hop_key: str) -> pd.DataFrame:
    df['retrieval_query'] = df[hop_key].apply(lambda x: x.get('answer', '') if isinstance(x, dict) else '')
    return df

def format_history_string(row: pd.Series, current_hop: int) -> str:
    history = []
    for h in range(1, current_hop + 1):
        k = f"hop_{h}"
        if k in row and isinstance(row[k], dict):
            data = row[k]
            q = data.get('final_question', data.get('question', ''))
            a = data.get('final_answer', data.get('answer', ''))
            d = data.get('doc', '')
            history.append(f"Hop_{h}:\nQuestion: {q}\nAnswer: {a}\nDocument: {d}")
    return "\n".join(history)

def filter_inference_logic(df: pd.DataFrame) -> pd.Series:
    def check_row(row):
        merge_data = row.get('merge_data')
        if not isinstance(merge_data, dict): return False
        m_type = merge_data.get('type', '')
        final_ans = merge_data.get('final_answer', '')
        mid_ans = row.get('mid_answer', '')
        if m_type == "inference":
            return normalize_answer(str(final_ans)) == normalize_answer(str(mid_ans))
        return True
    return df.apply(check_row, axis=1)

def pack_next_hop(row: pd.Series) -> dict:
    return {
        "question": row.get("mid_question"),
        "answer": row.get("mid_answer"),
        "doc": row.get("doc_content"),
        "final_question": row.get("merge_data", {}).get("final_question"),
        "final_answer": row.get("refine_data", {}).get("refined_answer"),
        "optional_answers": row.get("opt_data"),
        "qa_type": row.get("merge_data", {}).get("type"),
    }

# ==========================================
# 3. 验证阶段辅助函数
# ==========================================

def build_check_prompt_str(row: pd.Series, current_hop: int) -> str:
    hop_key = f"hop_{current_hop + 1}"
    prev_hop_key = f"hop_{current_hop}"
    curr = row[hop_key]
    prev = row[prev_hop_key]
    qa_type = curr.get("qa_type", "inference")
    
    kwargs = {
        "Question1": prev.get("final_question"),
        "Answer1": prev.get("final_answer"),
        "Document1": prev.get("doc"),
        "Question2": curr.get("question"),
        "Answer2": curr.get("answer"),
        "Document2": curr.get("doc"),
        "Final_question": curr.get("final_question"),
        "Final_answer": curr.get("final_answer"),
        "qa_type": qa_type,
    }
    
    if qa_type == 'inference':
        return InferenceCheckPrompt().build_prompt(**kwargs)
    else:
        return ComparisonCheckPrompt().build_prompt(**kwargs)

def build_reasoning_prompt_str(row: pd.Series, current_hop: int) -> str:
    hop_key = f"hop_{current_hop + 1}"
    curr = row[hop_key]
    qa_type = curr.get("qa_type", "inference")
    question = curr.get("final_question")
    
    if qa_type == 'inference':
        return ReasoningPrompt().build_prompt(problem=question)
    else:
        return ComparisonReasoningPrompt().build_prompt(problem=question)

def generate_doc_combinations(row: pd.Series, current_hop: int) -> List[str]:
    all_docs = []
    for h in range(1, current_hop + 1):
        k = f"hop_{h}"
        if k in row and isinstance(row[k], dict):
            all_docs.append(row[k].get("doc", ""))
    
    combos_str = []
    for combo in combinations(all_docs, max(1, len(all_docs)-1)):
        if len(combo) == 1:
            combos_str.append(combo[0])
        else:
            combos_str.append("\n\n".join(combo))
    return combos_str

def build_multihop_prompt_str(row: pd.Series, current_hop: int) -> str:
    total_hops = current_hop + 1
    hop_key = f"hop_{total_hops}"
    curr = row[hop_key]
    qa_type = curr.get("qa_type", "inference")
    
    Data = []
    for h in range(1, total_hops):
        info = row[f"hop_{h}"]
        Data.append(f"Question{h}: {info['question']}\nAnswer{h}: {info['answer']}\nSupporting Document{h}: {info['doc']}")
    
    last_info = row[f"hop_{total_hops}"]
    if qa_type == 'inference':
        Data.append(f"Question{total_hops}: {last_info['question']}\nSupporting Document{total_hops}: {last_info['doc']}")
        return MultihopInferencePrompt().build_prompt(
            Data="\n".join(Data),
            FinalQuestion=curr.get("final_question")
        )
    else:
        Data.append(f"Question{total_hops}: {last_info['question']}\nAnswer{total_hops}: {last_info['answer']}\nSupporting Document{total_hops}: {last_info['doc']}")
        return MultihopComparisonPrompt().build_prompt(
            Data="\n".join(Data),
            FinalQuestion=curr.get("final_question")
        )

# ==========================================
# 4. Pipeline 类
# ==========================================

class MultiHopRAGPipeline:
    def __init__(self, input_hop: int = 1, gen_qa_num: int = 3, topk: int = 3):
        self.input_hop = input_hop
        self.hop_key = f"hop_{input_hop}"
        self.next_hop_key = f"hop_{input_hop + 1}"
        self.gen_qa_num = gen_qa_num
        self.topk = topk

        # 1. 初始化
        self.storage = FileStorage(
            first_entry_file_name="../example_data/AgenticRAGPipeline/multihop_pipeline_data.jsonl",
            cache_path=f"./agenticRAG_multihop_cache",
            file_name_prefix=f"multihop_input_hop{input_hop}",
            cache_type="jsonl",
        )

        self.llm_serving = APILLMServing_request(
            api_url="https://api.openai.com/v1/chat/completions",
            model_name="o4-mini",
            max_workers=50
        )

        self.retriever_serving = FlashRAGServing(
            retrieval_method="e5",  # name of your retrieval_method.
            retrieval_model_path="/path/to/retrieval/model",  # name or path of the retrieval model. 
            index_path="/path/to/index", # path to the indexed file
            corpus_path="/path/to/corpus",  # path to corpus in '.jsonl' format that store the documents
            faiss_gpu=False, # whether use gpu to hold index
            gpu_id="",
            max_workers=1,
            topk=self.topk
        )

        # =======================================================
        # Generator Operators (生成阶段)
        # =======================================================

        self.op_prep_retrieval = PandasOperator(process_fn=[
            lambda df: prepare_retrieval_query(df, self.hop_key)
        ])

        self.op_retrieval = RetrievalGenerator(
            llm_serving=self.retriever_serving,
        )

        self.op_explode_docs = PandasOperator(process_fn=[
            lambda df: df.explode('retrieved_docs').reset_index(drop=True),
            lambda df: df.rename(columns={'retrieved_docs': 'doc_content'}),
            lambda df: df.dropna(subset=['doc_content']),
            lambda df: df.assign(gen_qa_num=self.gen_qa_num)
        ])

        # Step: Atomic QA
        self.op_atomic_gen = PromptTemplatedGenerator(
            llm_serving=self.llm_serving,
            prompt_template=StrFormatPrompt(f_str_template=AtomicQAGeneratorPrompt().prompt)
        )

        self.op_parse_atomic = PandasOperator(process_fn=[
            lambda df: df.assign(atomic_list=df['atomic_raw'].apply(safe_parse_json)),
            lambda df: df.explode('atomic_list').reset_index(drop=True).dropna(subset=['atomic_list']),
            lambda df: df.join(pd.json_normalize(df['atomic_list']).add_prefix('mid_'))
        ])

        # Step: Merge QA
        self.op_prep_merge = PandasOperator(process_fn=[
            lambda df: df.assign(history_str=df.apply(lambda r: format_history_string(r, self.input_hop), axis=1))
        ])
        
        # 使用字段映射: {Data} <- history_str, {New_question} <- mid_question ...
        self.op_merge_gen = PromptTemplatedGenerator(
            llm_serving=self.llm_serving,
            prompt_template=StrFormatPrompt(f_str_template=MergeAtomicQAPrompt().prompt)
        )

        self.op_parse_merge = PandasOperator(process_fn=[
            lambda df: df.assign(merge_data=df['merge_raw'].apply(safe_parse_json))
        ])
        
        self.op_filter = GeneralFilter(filter_rules=[filter_inference_logic])

        # Step: Refine
        self.op_prep_refine = PandasOperator(process_fn=[
            lambda df: df.assign(
                merge_final_q=df['merge_data'].apply(lambda x: x.get('final_question', '')),
                merge_final_a=df['merge_data'].apply(lambda x: x.get('final_answer', ''))
            )
        ])
        
        self.op_refine_gen = PromptTemplatedGenerator(
            llm_serving=self.llm_serving,
            prompt_template=StrFormatPrompt(f_str_template=RefineAnswerPrompt().prompt)
        )

        self.op_parse_refine = PandasOperator(process_fn=[
            lambda df: df.assign(refine_data=df['refine_raw'].apply(safe_parse_json)),
            lambda df: df.assign(temp_refined_a=df.apply(lambda r: r['refine_data'].get('refined_answer', '') if isinstance(r['refine_data'], dict) else '', axis=1))
        ])

        # Step: Optional Answers
        self.op_optional_gen = PromptTemplatedGenerator(
            llm_serving=self.llm_serving,
            prompt_template=StrFormatPrompt(f_str_template=MoreOptionalAnswersPrompt().prompt)
        )

        self.op_finalize = PandasOperator(process_fn=[
            lambda df: df.assign(opt_data=df['opt_raw'].apply(safe_parse_json)),
            lambda df: df.assign(**{self.next_hop_key: df.apply(pack_next_hop, axis=1)}),
            lambda df: df.assign(row_id=range(len(df)))
        ])

        # =======================================================
        # Verifier Operators (验证阶段)
        # =======================================================
        
        # --- Phase 1: Check (Dynamic) ---
        self.op_verify_prep_check = PandasOperator(process_fn=[
            lambda df: df if df.empty else df.assign(check_prompt_str=df.apply(lambda r: build_check_prompt_str(r, self.input_hop), axis=1))
        ])
        
        self.op_verify_gen_check = PromptTemplatedGenerator(
            llm_serving=self.llm_serving,
            prompt_template=StrFormatPrompt(f_str_template="{input_str}"),
        )
        
        self.op_verify_filter_check = PandasOperator(process_fn=[
            lambda df: df if df.empty else df.assign(check_json=df['check_raw'].apply(safe_parse_json)),
            lambda df: df[df['check_json'].apply(lambda x: str(x.get('valid', '')).lower() == 'true')]
        ])

        # --- Phase 2: Reasoning (Dynamic) ---
        self.op_verify_prep_reasoning = PandasOperator(process_fn=[
            lambda df: df if df.empty else df.assign(reason_prompt_str=df.apply(lambda r: build_reasoning_prompt_str(r, self.input_hop), axis=1)),
            lambda df: df if df.empty else df.assign(target_ans=df[self.next_hop_key].apply(lambda x: x.get('final_answer') if isinstance(x, dict) else ''))
        ])

        self.op_verify_gen_reasoning = PromptTemplatedGenerator(
            llm_serving=self.llm_serving,
            prompt_template=StrFormatPrompt(f_str_template="{input_str}"),
        )

        # --- Phase 3: Judge Reasoning (Static) ---
        self.op_verify_judge_reasoning = PromptTemplatedGenerator(
            llm_serving=self.llm_serving,
            prompt_template=StrFormatPrompt(f_str_template=EssEqPrompt().prompt)
        )

        self.op_verify_filter_reasoning = PandasOperator(process_fn=[
            lambda df: df if df.empty else df.assign(judge_reason_json=df['judge_reason_raw'].apply(safe_parse_json)),
            lambda df: df[df['judge_reason_json'].apply(lambda x: x.get('answer_score', 0) < 1)]
        ])

        # --- Phase 4: Shortcut Check (Static) ---
        self.op_verify_prep_shortcut = PandasOperator(process_fn=[
            lambda df: df if df.empty else df.assign(doc_combos=df.apply(lambda r: generate_doc_combinations(r, self.input_hop), axis=1)),
            lambda df: df.explode('doc_combos').reset_index(drop=True).dropna(subset=['doc_combos'])
        ])

        self.op_verify_gen_shortcut = PromptTemplatedGenerator(
            llm_serving=self.llm_serving,
            prompt_template=StrFormatPrompt(f_str_template=SingleHopPrompt().prompt)
        )

        # --- Phase 5: Judge Shortcut (Static) ---
        self.op_verify_judge_shortcut = PromptTemplatedGenerator(
            llm_serving=self.llm_serving,
            prompt_template=StrFormatPrompt(f_str_template=EssEqPrompt().prompt)
        )

        self.op_verify_filter_shortcut = PandasOperator(process_fn=[
            lambda df: df if df.empty else df.assign(sc_judge_json=df['sc_judge_raw'].apply(safe_parse_json)),
            lambda df: df if df.empty else df.assign(sc_score=df['sc_judge_json'].apply(lambda x: x.get('answer_score', 0))),
            lambda df: df if df.empty else df.groupby('row_id').filter(lambda x: x['sc_score'].max() < 1),
            lambda df: df.drop_duplicates(subset=['row_id'])
        ])

        # --- Phase 6: Final Verification (Dynamic) ---
        self.op_verify_prep_final = PandasOperator(process_fn=[
            lambda df: df if df.empty else df.assign(final_prompt_str=df.apply(lambda r: build_multihop_prompt_str(r, self.input_hop), axis=1))
        ])

        self.op_verify_gen_final = PromptTemplatedGenerator(
            llm_serving=self.llm_serving,
            prompt_template=StrFormatPrompt(f_str_template="{input_str}"),
        )

        # --- Phase 7: Final Judge (Static) ---
        self.op_verify_judge_final = PromptTemplatedGenerator(
            llm_serving=self.llm_serving,
            prompt_template=StrFormatPrompt(f_str_template=EssEqPrompt().prompt)
        )

        self.op_verify_filter_final = PandasOperator(process_fn=[
            lambda df: df if df.empty else df.assign(final_judge_json=df['final_judge_raw'].apply(safe_parse_json)),
            lambda df: df if df.empty else df[df['final_judge_json'].apply(lambda x: x.get('answer_score', 0) >= 1)],
            lambda df: df[[col for col in df.columns if col.startswith("hop_")]]
        ])


    async def forward(self):
        # --- Generation Phase ---
        self.op_prep_retrieval.run(self.storage.step())
        await self.op_retrieval.run(self.storage.step(), input_key="retrieval_query", output_key="retrieved_docs")
        self.op_explode_docs.run(self.storage.step())
        
        # Atomic (Standardized)
        self.op_atomic_gen.run(
            self.storage.step(), 
            output_key='atomic_raw', 
            gen_qa_num='gen_qa_num', 
            input_doc='doc_content'
        )
        self.op_parse_atomic.run(self.storage.step())
        
        # Merge (Standardized)
        self.op_prep_merge.run(self.storage.step())
        self.op_merge_gen.run(
            self.storage.step(), 
            output_key='merge_raw', 
            Data='history_str', 
            New_question='mid_question', 
            New_answer='mid_answer', 
            New_document='doc_content'
        )
        self.op_parse_merge.run(self.storage.step())
        self.op_filter.run(self.storage.step())
        
        # Refine (Standardized)
        self.op_prep_refine.run(self.storage.step())
        self.op_refine_gen.run(
            self.storage.step(), 
            output_key='refine_raw', 
            question='merge_final_q', 
            original_answer='merge_final_a'
        )
        self.op_parse_refine.run(self.storage.step())
        
        # Optional (Standardized)
        self.op_optional_gen.run(
            self.storage.step(), 
            output_key='opt_raw', 
            refined_answer='temp_refined_a'
        )
        
        self.op_finalize.run(self.storage.step())

        # --- Verification Phase ---
        print("--- Starting Verification ---")
        
        # 1. Check (Dynamic)
        self.op_verify_prep_check.run(self.storage.step())
        self.op_verify_gen_check.run(self.storage.step(), output_key='check_raw', input_str='check_prompt_str')
        self.op_verify_filter_check.run(self.storage.step())

        # 2. Reasoning (Dynamic)
        self.op_verify_prep_reasoning.run(self.storage.step())
        self.op_verify_gen_reasoning.run(self.storage.step(), output_key='reasoning_raw', input_str='reason_prompt_str')
        
        # 3. Judge Reasoning (Standardized)
        self.op_verify_judge_reasoning.run(
            self.storage.step(), 
            output_key='judge_reason_raw', 
            question='merge_final_q', 
            golden_answer='target_ans', 
            other_answer='reasoning_raw'
        )
        self.op_verify_filter_reasoning.run(self.storage.step())

        # 4. Shortcut Check (Standardized)
        self.op_verify_prep_shortcut.run(self.storage.step())
        self.op_verify_gen_shortcut.run(
            self.storage.step(), 
            output_key='sc_raw', 
            Document='doc_combos', 
            Question='merge_final_q'
        )

        # 5. Judge Shortcut (Standardized)
        self.op_verify_judge_shortcut.run(
            self.storage.step(), 
            output_key='sc_judge_raw', 
            question='merge_final_q', 
            golden_answer='target_ans', 
            other_answer='sc_raw'
        )
        self.op_verify_filter_shortcut.run(self.storage.step())

        # 6. Final Gen (Dynamic)
        self.op_verify_prep_final.run(self.storage.step())
        self.op_verify_gen_final.run(self.storage.step(), output_key='final_raw', input_str='final_prompt_str')

        # 7. Final Judge (Standardized)
        self.op_verify_judge_final.run(
            self.storage.step(), 
            output_key='final_judge_raw', 
            question='merge_final_q', 
            golden_answer='target_ans', 
            other_answer='final_raw'
        )
        self.op_verify_filter_final.run(self.storage.step())

        await self.retriever_serving.cleanup()
        print(f"Pipeline Hop {self.input_hop} -> {self.input_hop + 1} Completed and Verified.")

if __name__ == "__main__":
    pipeline = MultiHopRAGPipeline(input_hop=1, gen_qa_num=3, topk=3)
    asyncio.run(pipeline.forward())