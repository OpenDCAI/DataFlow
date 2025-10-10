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
from dataflow.prompts.core_filter import *

input_key="raw_content" #指定输入的字段

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
            system_prompt=get_filter_init_scorer_system_prompt() + get_filter_init_scorer_user_prompt(),
        )
        self.init_filter = GeneralFilter([lambda df: df['init_score'] > 1])
        self.refiner = PromptedGenerator(
            llm_serving=llm_serving,
            system_prompt=get_filter_refiner_system_prompt() + get_filter_init_scorer_user_prompt(),
        )
        self.rewriter = PromptedGenerator(
            llm_serving=llm_serving,
            system_prompt=get_filter_rewriter_system_prompt() + get_filter_rewriter_user_prompt(),
        )
        self.final_scorer = PromptedGenerator(
            llm_serving=llm_serving,
            system_prompt=get_filter_final_scorer_system_prompt() + get_filter_final_scorer_user_prompt(),
        )
        self.final_filter = GeneralFilter([lambda df: df['final_score'] >= 4])
    
    def wrap_data_into_input_key(self):
        # 读取self.storage中的文件，并封装进"input_key"字段，value为文件内容
        df = self.storage.step().read("dataframe")
        # 将每一行的数据整体作为raw_content字段的值
        df[input_key] = df.apply(lambda row: row.to_dict(), axis=1)
        # 只保留raw_content字段
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
