#!/usr/bin/env python3
"""QA Extractor - 提取QA对并转换为Alpaca格式"""

import json
from pathlib import Path
from typing import Optional, List
from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage
from dataflow import get_logger


@OPERATOR_REGISTRY.register()
class QAExtractor(OperatorABC):
    """
    从QA_pairs字段提取问答对，转换为Alpaca微调格式

    Input:  QA_pairs (nested structure)
    Output: instruction, input, output (Alpaca format)
    """

    def __init__(
            self,
            qa_key: str = "QA_pairs",
            output_json_file: Optional[str] = None,
            instruction: str = "Please answer the following question based on the provided information."
    ):
        self.logger = get_logger()
        self.qa_key = qa_key
        self.output_json_file = output_json_file
        self.instruction = instruction

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
                "• qa_key: QA对的字段名 (默认: 'QA_pairs')\n"
                "• output_json_file: 输出JSON文件路径 (可选，不指定则只更新DataFrame)\n"
                "• instruction: 统一的指令前缀 (默认: 'Please answer the following question...')\n\n"
                "运行参数 (input_key):\n"
                "• None - 包含所有字段 (question + reasoning_steps + supporting_facts)\n"
                "• '' - 空字符串，不包含额外上下文\n"
                "• 'reasoning_steps' - 只包含推理步骤\n"
                "• 'question,reasoning_steps' - 逗号分隔多个字段\n"
                "• ['question', 'supporting_facts'] - 列表格式\n\n"
                "输出字段:\n"
                "• instruction: 问题指令\n"
                "• input: 上下文信息 (根据input_key动态拼接)\n"
                "• output: 答案\n\n"
                "适用场景: 知识库QA微调、领域问答模型训练"
            )
        else:  # English
            return (
                "QA Extractor - Convert nested QA_pairs to Alpaca fine-tuning format\n\n"
                "Core Function:\n"
                "Extract question-answer pairs from structured data, automatically integrate\n"
                "reasoning steps and supporting facts, output in Stanford Alpaca standard\n"
                "instruction-input-output format.\n\n"
                "Initialization Parameters:\n"
                "• qa_key: Field name for QA pairs (default: 'QA_pairs')\n"
                "• output_json_file: Output JSON path (optional, skip to only update DataFrame)\n"
                "• instruction: Unified instruction prefix (default: 'Please answer...')\n\n"
                "Runtime Parameters (input_key):\n"
                "• None - Include all fields (question + reasoning_steps + supporting_facts)\n"
                "• '' - Empty string, no additional context\n"
                "• 'reasoning_steps' - Only reasoning steps\n"
                "• 'question,reasoning_steps' - Comma-separated fields\n"
                "• ['question', 'supporting_facts'] - List format\n\n"
                "Output Fields:\n"
                "• instruction: Question as instruction\n"
                "• input: Context information (dynamically assembled by input_key)\n"
                "• output: Answer\n\n"
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

    def _extract_qa(self, row, fields: Optional[List[str]] = None) -> List[dict]:
        """从单行提取QA对"""
        qa_data = row.get(self.qa_key)
        if not qa_data:
            return []

        # 支持嵌套结构
        qa_list = qa_data.get('qa_pairs', []) if isinstance(qa_data, dict) else qa_data
        if not isinstance(qa_list, list):
            return []

        results = []
        default_fields = ['question', 'reasoning_steps', 'supporting_facts']
        fields = fields if fields is not None else default_fields

        for qa in qa_list:
            if not isinstance(qa, dict):
                continue

            question = qa.get('question', '').strip()
            answer = qa.get('answer', '').strip()
            if not question or not answer:
                continue

            # 构建input
            parts = []
            for field in fields:
                if field == 'question':
                    parts.append(f"Question: {question}")

                elif field == 'reasoning_steps' and qa.get('reasoning_steps'):
                    if parts:
                        parts.append("")
                    parts.append("Reasoning Process:")
                    for i, step in enumerate(qa['reasoning_steps'], 1):
                        text = step.get('step', step) if isinstance(step, dict) else str(step)
                        if text:
                            parts.append(f"{i}. {text}")

                elif field == 'supporting_facts' and qa.get('supporting_facts'):
                    if parts:
                        parts.append("")
                    parts.append("Supporting Information:")
                    for fact in qa['supporting_facts']:
                        text = fact.get('fact', fact) if isinstance(fact, dict) else str(fact)
                        if text:
                            parts.append(f"- {text}")

                elif field in qa and qa[field]:
                    if parts:
                        parts.append("")
                    parts.append(f"{field}: {qa[field]}")

            results.append({
                'instruction': self.instruction,
                'input': "\n".join(parts),
                'output': answer
            })

        return results

    def _load_from_files(self, df):
        """从chunk文件加载QA数据"""
        import pandas as pd

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
            input_key: Optional[str] = None,
            output_key: Optional[str] = None
    ) -> List[str]:
        """提取QA对"""
        import pandas as pd

        self.logger.info("开始提取QA对...")

        df = storage.read(output_type="dataframe")

        # 如果没有QA_pairs，从文件加载
        if self.qa_key not in df.columns:
            df = self._load_from_files(df)

        # 提取所有QA对
        fields = self._parse_fields(input_key)
        all_qas = []
        for _, row in df.iterrows():
            all_qas.extend(self._extract_qa(row, fields))

        self.logger.info(f"提取了 {len(all_qas)} 个QA对")

        if not all_qas:
            self.logger.warning("未提取到QA对!")
            return ['instruction', 'input', 'output']

        # 保存JSON（可选）
        if self.output_json_file:
            output_path = Path(self.output_json_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(all_qas, f, indent=2, ensure_ascii=False)
            self.logger.info(f"已保存到 {output_path}")

        # 写回storage
        storage.write(pd.DataFrame(all_qas))

        return ['instruction', 'input', 'output']