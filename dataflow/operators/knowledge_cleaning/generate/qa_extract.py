#!/usr/bin/env python3
"""QA Extractor - 提取QA对并转换为Alpaca格式"""

import json
from pathlib import Path
from typing import Optional, List
from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage
from dataflow import get_logger
import pandas as pd

@OPERATOR_REGISTRY.register()
class QAExtractor(OperatorABC):
    """
    从QA_pairs字段提取问答对，转换为Alpaca微调格式

    Input:  QA_pairs (nested structure)
    Output: instruction, input, output (Alpaca format)
    """

    def __init__(
            self,
            output_json_file: Optional[str] = None,
            input_instruction: Optional[str] = "Please answer the following question based on the provided information."
    ):
        self.logger = get_logger()
        self.output_json_file = output_json_file
        self.instruction = input_instruction

    @staticmethod
    def get_desc(lang: str = "zh"):
        """获取算子描述"""
        if lang == "zh":
            return (
                "QA对提取器 - 将嵌套的QA_pairs转换为Alpaca微调格式\n\n"
                "核心功能:\n"
                "从结构化的QA对数据中提取问答内容，自动整合推理步骤和支持事实，\n"
                "输出符合Stanford Alpaca标准的instruction-input-output格式。\n\n"
                "初始化参数:\n"
                "• output_json_file: 输出JSON文件路径 (可选，不指定则只更新DataFrame)\n"
                "• input_instruction: 统一的指令前缀 (默认: 'Please answer the following question...')\n\n"
                "运行参数 (input_key):\n"
                "• input_qa_key: QA对的字段名 (默认: 'QA_pairs')\n"
                "• instruction: 本次运行的具体指令内容 (若不填则使用默认指令)\n"
                "输出字段:\n"
                "• output_instruction_key: prompt对应的key (若不填则使用默认字段)\n"
                "• output_question_key: 问题字段 (若不填则使用默认字段)\n"
                "• output_answer_key: 答案字段 (若不填则使用默认字段)\n\n"
                "适用场景: 知识库QA微调、领域问答模型训练"
                "instruct input output专门适用于llamafactory"
            )
        else:  # English
            return (
                "QA Extractor - Convert nested QA_pairs to Alpaca fine-tuning format\n\n"
                "Core Function:\n"
                "Extract question-answer pairs from structured data, automatically integrate\n"
                "reasoning steps and supporting facts, output in Stanford Alpaca standard\n"
                "instruction-input-output format.\n\n"
                "Initialization Parameters:\n"
                "• output_json_file: Output JSON path (optional, skip to only update DataFrame)\n"
                "• input_instruction: Unified instruction prefix (default: 'Please answer...')\n\n"
                "Runtime Parameters (input_key):\n"
                "• input_qa_key: Field name for QA pairs (default: 'QA_pairs')\n"
                "• instruction: Specific instruction for this run (uses default if not provided)\n"
                "Output Fields:\n"
                "• output_instruction_key: Question as instruction (optional, skip to only update DataFrame)\n"
                "• output_question_key: Context information (optional, skip to only update DataFrame)\n"
                "• output_answer_key: Answer (optional, skip to only update DataFrame)\n\n"
                "Use Cases: Knowledge base QA fine-tuning, domain-specific Q&A training"
            )

    def _parse_fields(self, input_key: Optional[str]) -> Optional[List[str]]:
        """解析要包含的字段"""
        if input_key is None:
            return None  # 包含所有
        if isinstance(input_key, list):
            return input_key
        if isinstance(input_key, str):
            return [f.strip() for f in input_key.split(',') if f.strip()] if input_key.strip() else []
        return None

    def _extract_qa(self, row, fields: Optional[List[str]], 
                   key_inst: str, key_q: str, key_a: str) -> List[dict]:
        """核心提取逻辑"""
        qa_data = row.get(self.qa_key)
        if not qa_data:return []

        # 兼容性处理
        qa_list = qa_data.get('qa_pairs', []) if isinstance(qa_data, dict) else qa_data
        if not isinstance(qa_list, list):
            qa_list = [qa_list] if isinstance(qa_list, dict) else []

        results = []
        # 默认字段
        default_fields = ['question', 'reasoning_steps', 'supporting_facts']
        fields = fields if fields is not None else default_fields

        for qa in qa_list:
            if not isinstance(qa, dict):continue

            question = qa.get('question', '').strip()
            answer = qa.get('answer', '').strip()
            if not question or not answer:continue
            # 方便下游使用reasoning 暂时删除
            # # --- 1. 构建 Context (input内容) ---
            # parts = []
            # for field in fields:
            #     if field == 'question':
            #         parts.append(f"Question: {question}")
            #     elif field == 'reasoning_steps' and qa.get('reasoning_steps'):
            #         if parts: parts.append("")
            #         parts.append("Reasoning Process:")
            #         for i, step in enumerate(qa['reasoning_steps'], 1):
            #             text = step.get('step', step) if isinstance(step, dict) else str(step)
            #             if text: parts.append(f"{i}. {text}")
            #     elif field == 'supporting_facts' and qa.get('supporting_facts'):
            #         if parts: parts.append("")
            #         parts.append("Supporting Information:")
            #         for fact in qa['supporting_facts']:
            #             text = fact.get('fact', fact) if isinstance(fact, dict) else str(fact)
            #             if text: parts.append(f"- {text}")
            #     elif field in qa and qa[field]:
            #         if parts: parts.append("")
            #         parts.append(f"{field}: {qa[field]}")

            # --- 2. 组装结果字典 (使用动态 Key) ---
            item = {
                key_inst: self.instruction,  # Instruction 列
                key_q: question,     # Question/Input 列 (内容由 fields 决定)
                key_a: answer                # Answer/Output 列
            }
            results.append(item)
        return results

    def _load_from_files(self, df):
        """从chunk文件加载QA数据"""
        path_keys = ['enhanced_chunk_path', 'cleaned_chunk_path', 'chunk_path']
        path_col = next((k for k in path_keys if k in df.columns), None)

        if not path_col:
            raise ValueError(f"需要这些字段之一: {path_keys}")

        rows = []
        for _, row in df.iterrows():
            file_path = row[path_col]
            if not file_path or not Path(file_path).exists():
                continue

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    chunks = json.load(f)
                    chunks = chunks if isinstance(chunks, list) else [chunks]

                    for chunk in chunks:
                        if self.qa_key in chunk:
                            rows.append({
                                self.qa_key: chunk[self.qa_key],
                                'source_file': file_path
                            })
            except Exception as e:
                self.logger.error(f"加载失败 {file_path}: {e}")

        if not rows:
            raise ValueError("未找到有效QA数据")

        return pd.DataFrame(rows)

    def run(
            self,
            storage: DataFlowStorage,
            input_qa_key: str = "QA_pairs",
            output_instruction_key:Optional[str] = "instruction",
            output_question_key: Optional[str] = "input",
            output_answer_key: Optional[str] = "output"
    ) -> List[str]:
        """提取QA对"""
        is_modified = False
        modified_details = []
        self.qa_key = input_qa_key
        
        if output_question_key != "question":
            is_modified = True
            modified_details.append(f"output_question_key -> '{output_question_key}'")
            
        if output_answer_key != "answer":
            is_modified = True
            modified_details.append(f"output_answer_key -> '{output_answer_key}'")

        if is_modified:
            self.logger.warning(
                f"\n{'='*20} ⚠️ Configuration Change Warning {'='*20}\n"
                f"Detected changes in output field names: {', '.join(modified_details)}\n\n"
                f"Please note the following based on your downstream tasks:\n"
                f"1. [SFT / LLaMA-Factory]: If you plan to use LLaMA-Factory directly, DO NOT modify the default keys arbitrarily. "
                f"Otherwise, you must manually update the column mapping in 'dataset_info.json'.\n"
                f"2. [Pretraining / Downstream]: If connecting to 'ReasoningPretrainFormatConvertGenerator', "
                f"ensure you specify the corresponding read keys (input_read_key_question/answer) in the downstream operator.\n"
                f"{'='*66}"
            )
        self.logger.info("Start extract QA from QA pairs")

        df = storage.read(output_type="dataframe")

        # 如果没有QA_pairs，从文件加载
        if self.qa_key not in df.columns:
            df = self._load_from_files(df)
        

        # 提取所有QA对
        fields = self._parse_fields(None)
        
        all_qas = []
        for _, row in df.iterrows():
            qas = self._extract_qa(
                row, 
                fields, 
                key_inst=output_instruction_key,
                key_q=output_question_key, 
                key_a=output_answer_key
            )
            all_qas.extend(qas)

        self.logger.info(f"Extracted {len(all_qas)} QA pairs")

        if not all_qas:
            self.logger.warning("No QA pairs found!")
            return [output_instruction_key, output_question_key, output_answer_key]

        # 保存JSON（可选）
        if self.output_json_file:
            output_path = Path(self.output_json_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(all_qas, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Saved to {output_path}")

        # 写回storage
        storage.write(pd.DataFrame(all_qas))

        return [output_instruction_key, output_question_key, output_answer_key]