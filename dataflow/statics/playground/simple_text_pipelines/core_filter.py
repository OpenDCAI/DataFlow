"""
本文件主要作用：
    - 用于对SFT训练数据进行自动化质量过滤、精炼与扩写，最终输出高质量SFT训练集。
    - 主要流程包括：初始打分、初筛、精炼/扩写、终筛、整理输出。
    - 支持多阶段Prompted LLM打分与数据增强，适用于大规模数据自动清洗与提升。

使用说明：
    直接运行本文件，将自动读取指定的原始数据文件，经过多轮筛选与增强，最终输出高质量SFT数据到cache目录。
    需根据实际环境配置好cache路径与API参数。

"""

import pandas as pd
from dataflow.operators.core_text import PromptedGenerator
from dataflow.operators.core_text import GeneralFilter
from dataflow.utils.storage import FileStorage
from dataflow.serving import APILLMServing_request

input_key="raw_content" #指定输入的字段

def GetFilterInitScorerPrompt():
    system_prompt = (
            "You are a data quality assessor performing initial content evaluation. "
            "Your task is to score content based on information density and basic quality on a scale of 1-5.\n\n"
            "Scoring Guidelines:\n"
            "• 1: Meaningless/empty content - should be filtered out\n"
            "• 2: Minimal useful information - needs significant refinement\n"
            "• 3: Basic information present - needs moderate refinement\n"
            "• 4: Good information quality - suitable for expansion\n"
            "• 5: High-quality, rich information - ideal for expansion\n\n"
            "Focus on: information completeness, factual accuracy, and basic coherence. "
            "This score determines processing path: 1=filter, 2-3=refine, 4-5=expand.\n\n"
            "Respond with only the integer score (1-5)."
        )

    user_prompt = (
        f"Score this content for information quality and completeness (1-5):\n\n"
        f"Output format: single integer 1-5, no other text"
    )
    return system_prompt + "\n\n" + user_prompt

def GetFilterRefinerPrompt():
    system_prompt = (
        "You are a specialized content optimization expert tasked with transforming medium-quality data into clear, valuable information. "
        "Focus on enhancing clarity, precision, and information density while preserving core meaning.\n\n"
        "Core Refinement Principles:\n"
        "• Transform vague expressions into specific, actionable statements\n"
        "• Eliminate redundant phrases while retaining essential meaning\n"
        "• Reorganize content for optimal logical progression\n"
        "• Correct linguistic errors and enhance vocabulary precision\n"
        "• Bridge information gaps through contextual inference\n"
        "• Strengthen connections between related concepts\n"
        "• Preserve factual integrity and original author intent\n\n"
        "STRICT JSON OUTPUT PROTOCOL:\n"
        "1. Input format: JSON object containing multiple data fields\n"
        "2. Output requirement: Valid JSON with IDENTICAL field structure\n"
        "3. Field preservation: NO addition, deletion, or modification of keys\n"
        "4. Format compliance: RAW JSON only - no markdown, comments, or explanatory text\n"
        "5. Character handling: Proper escaping of quotes and special characters\n"
        "6. Response boundary: Begin with '{' and terminate with '}'\n"
        "7. Content validation: Each refined value must be clear and actionable\n\n"
        "Processing Example:\n"
        "Input: {\"concept\": \"somewhat unclear idea\", \"details\": \"repetitive explanation\"}\n"
        "Output: {\"concept\": \"precise, well-defined concept\", \"details\": \"streamlined, focused explanation\"}\n\n"
        "Execute refinement and output the enhanced JSON structure immediately."
    )

    user_prompt = (
        f"Refine this medium-quality JSON content to improve clarity and information value. "
        f"Respond with ONLY the refined JSON object (no code blocks, no explanations, just the JSON):\n\n"
        f"Output format: single valid JSON object only"
    )
    return system_prompt + "\n\n" + user_prompt

def GetFilterRewriterPrompt():
    system_prompt = (
        "You are an advanced content enrichment specialist working with premium-quality source materials. "
        "Your mission is to amplify already excellent content by incorporating depth, comprehensive context, and actionable insights.\n\n"
        "Strategic Expansion Framework:\n"
        "• Integrate practical examples and real-world implementation scenarios\n"
        "• Provide comprehensive background context and foundational knowledge\n"
        "• Incorporate multiple analytical perspectives and considerations\n"
        "• Elaborate on downstream effects and broader implications\n"
        "• Enrich with supporting evidence and detailed explanations\n"
        "• Establish connections to related domains and interdisciplinary insights\n"
        "• Embed actionable guidance and implementation strategies\n"
        "• Maintain superior organizational structure and narrative flow\n\n"
        "MANDATORY JSON OUTPUT SPECIFICATIONS:\n"
        "1. Input structure: JSON object with established field architecture\n"
        "2. Output mandate: JSON object maintaining exact key-value structure\n"
        "3. Field integrity: Absolute preservation of original field names and hierarchy\n"
        "4. Format restrictions: Pure JSON output without markdown, annotations, or commentary\n"
        "5. Content expansion: Significantly enrich each field with additional value\n"
        "6. Syntax compliance: Validate JSON formatting with proper character escaping\n"
        "7. Response parameters: Direct JSON output from '{' to '}'\n\n"
        "Transformation Pattern:\n"
        "Input: {\"topic\": \"basic concept\", \"description\": \"simple explanation\"}\n"
        "Output: {\"topic\": \"comprehensive concept with context, examples, and applications...\", "
        "\"description\": \"detailed explanation with multiple perspectives, implications, and practical guidance...\"}\n\n"
        "Proceed with comprehensive content expansion and deliver the enriched JSON structure."
    )

    user_prompt = (
        f"Expand the content in this JSON object with additional depth, examples, and context. "
        f"Respond with ONLY the expanded JSON object (no code blocks, no explanations, just the JSON):\n\n"
        f"Output format: single valid JSON object only"
    )
    return system_prompt + "\n\n" + user_prompt

def GetFilterFinalScorerPrompt():
    system_prompt = (
        "You are a senior content quality evaluator performing final assessment. "
        "Rate the processed content on a 1-5 scale for overall excellence and utility.\n\n"
        "Final Scoring Criteria:\n"
        "• 1: Poor - significant quality issues remain\n"
        "• 2: Below standard - limited value or clarity problems\n"
        "• 3: Acceptable - meets basic requirements but unremarkable\n"
        "• 4: High quality - well-crafted, informative, and valuable\n"
        "• 5: Excellent - exceptional content that serves as exemplary reference\n\n"
        "Evaluate holistically:\n"
        "- Information completeness and accuracy\n"
        "- Clarity and readability\n"
        "- Practical value and usefulness\n"
        "- Structure and logical flow\n"
        "- Overall contribution to knowledge/understanding\n\n"
        "Only content scoring 4-5 will be retained for final dataset. "
        "Be appropriately selective to ensure high standards.\n\n"
        "Respond with only the integer score (1-5)."
    )

    user_prompt = (
        f"Provide a final quality score (1-5) for this processed content:\n\n"
        f"Output format: single integer 1-5, no other text"
    )
    return system_prompt + "\n\n" + user_prompt


class CoreFilterPipeline:
    """
    SFT数据自动过滤与增强主流程
    """
    def __init__(self):
        self.storage = FileStorage(
            first_entry_file_name="",
            cache_path="./dataflow_cache",
            file_name_prefix="core_filter",
            cache_type="jsonl",
        )
        llm_serving = APILLMServing_request(
            api_url="http://your-api-url/v1/chat/completions", 
            model_name="gpt-4o",
            max_workers=100
        )
        self.need_data_num = 200  # 目标数据量

        # 各阶段Prompted LLM与过滤器
        self.init_scorer = PromptedGenerator(
            llm_serving=llm_serving,
            system_prompt=GetFilterInitScorerPrompt()
        )
        self.init_filter = GeneralFilter([lambda df: df['init_score'] > 1])
        self.refiner = PromptedGenerator(
            llm_serving=llm_serving,
            system_prompt=GetFilterRefinerPrompt()
        )
        self.rewriter = PromptedGenerator(
            llm_serving=llm_serving,
            system_prompt=GetFilterRewriterPrompt()
        )
        self.final_scorer = PromptedGenerator(
            llm_serving=llm_serving,
            system_prompt=GetFilterFinalScorerPrompt()
        )
        self.final_filter = GeneralFilter([lambda df: df['final_score'] >= 4])
    
    def wrap_data_into_input_key(self):
        df = self.storage.step().read("dataframe")
        df[input_key] = df.apply(lambda row: row.to_dict(), axis=1)
        df = df[[input_key]]
        self.storage.write(df)

    def get_final_data(self):
        """
        整理最终数据，将refine/rewrite字段内容拆分为标准SFT格式
        """
        df = self.storage.step().read("dataframe")
        sft_rows = []
        invalid_format_count = 0
        for _, row in df.iterrows():
            # 处理rewrite字段
            if pd.notnull(row.get("rewrite", None)) and str(row["rewrite"]).strip():
                try:
                    sample = eval(row["rewrite"]) if isinstance(row["rewrite"], str) else row["rewrite"]
                    if (
                        isinstance(sample, dict)
                        and all(k in sample for k in row[input_key].keys())
                    ):
                        sft_rows.append(sample)
                    else:
                        invalid_format_count += 1
                except Exception:
                    invalid_format_count += 1
                sft_rows.append(row[input_key])
            # 处理refine字段
            if pd.notnull(row.get("refine", None)) and str(row["refine"]).strip():
                try:
                    sample = eval(row["refine"]) if isinstance(row["refine"], str) else row["refine"]
                    if (
                        isinstance(sample, dict)
                        and all(k in sample for k in row[input_key].keys())
                    ):
                        sft_rows.append(sample)
                    else:
                        invalid_format_count += 1
                except Exception:
                    invalid_format_count += 1
        print(f"Filtered {invalid_format_count} rows with invalid format")
        sft_df = pd.DataFrame(sft_rows)
        self.storage.write(sft_df)

    def forward(self):
        """
        主流程：初筛-精炼/扩写-终筛-输出
        """
        #封装数据,可选
        self.wrap_data_into_input_key()
        # 初始打分
        self.init_scorer.run(
            storage=self.storage.step(),
            input_key=input_key,
            output_key="init_score"
        )
        # 初筛
        self.init_filter.run(
            storage=self.storage.step(),
        )

        df = self.storage.step().read("dataframe")
        df_refine = df[(df['init_score'] > 1) & (df['init_score'] <= 3)].copy()
        df_rewrite = df[df['init_score'] > 3].copy()

        # 精炼一般质量数据
        if not df_refine.empty:
            self.storage.write(df_refine)
            self.refiner.run(
                storage=self.storage.step(),
                input_key=input_key,
                output_key="refine"
            )
            self.final_scorer.run(
                storage=self.storage.step(),
                input_key="refine",
                output_key="final_score"
            )
            self.final_filter.run(
                storage=self.storage.step()
            )
            df_refine = self.storage.step().read("dataframe")
        else:
            df_refine = pd.DataFrame()

        # 扩写高质量数据（如数量不足）
        if not df_rewrite.empty and len(df) < self.need_data_num:
            self.storage.write(df_rewrite)
            self.rewriter.run(
                storage=self.storage.step(),
                input_key=input_key,
                output_key="rewrite"
            )
            self.final_scorer.run(
                storage=self.storage.step(),
                input_key="rewrite",
                output_key="final_score"
            )
            self.final_filter.run(
                storage=self.storage.step()
            )
            df_rewrite = self.storage.step().read("dataframe")
        else:
            df_rewrite = pd.DataFrame()

        # 合并最终数据并输出
        df_final = pd.concat([df_refine, df_rewrite], ignore_index=True)
        self.storage.write(df_final)
        self.get_final_data()

if __name__ == "__main__":
    model = CoreFilterPipeline()
    model.forward()
