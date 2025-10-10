
"""
本文件用于从零合成SFT训练数据，主要流程包括：
1. 通过PromptedGenerator生成原始数据（raw_generation）；
2. 对原始数据进行重写（rewrite），丰富多样性；
3. 对原始和重写数据分别打分（score_raw, score_rewrite）；
4. 过滤低分样本，仅保留高质量数据；
5. 最终整理为SFT训练所需的标准格式。

使用方法：
直接运行本文件即可自动完成上述流程，最终结果会保存在指定的cache目录下。
"""
from dataflow.operators.core_text import RandomDomainKnowledgeRowGenerator
from dataflow.operators.core_text import PromptedGenerator
from dataflow.operators.core_text import GeneralFilter
from dataflow.utils.storage import FileStorage
from dataflow.serving import APILLMServing_request
from dataflow.prompts.core_sft_from_scratch import *
import pandas as pd


class CoreSftFromScratchPipeline:
    def __init__(self):
        self.storage = FileStorage(
            first_entry_file_name="",
            cache_path="./dataflow_cache",  
            file_name_prefix="core_sft_from_scratch_step",
            cache_type="jsonl",
        )
        self.model_cache_dir = './core_sft_cache'
        self.generated_samples_num = 10

        llm_serving = APILLMServing_request(
            api_url="http://your-api-url/v1/chat/completions", 
            model_name="gpt-4o",
            max_workers=100
        )

        self.generator = RandomDomainKnowledgeRowGenerator(
            llm_serving=llm_serving,
            system_prompt=get_sft_from_scratch_generator_system_prompt(),
            user_prompt=get_sft_from_scratch_generator_user_prompt()
        )
        self.rewriter = PromptedGenerator(
            llm_serving=llm_serving,
            system_prompt=get_sft_from_scratch_rewriter_system_prompt() + get_sft_from_scratch_rewriter_user_prompt(),
        )
        self.scorer = PromptedGenerator(
            llm_serving=llm_serving,
            system_prompt=get_sft_from_scratch_scorer_system_prompt() + get_sft_from_scratch_scorer_user_prompt(),
        )
        self.filter = GeneralFilter([
            lambda df: df['score_raw'] > 3,
            lambda df: df['score_rewrite'] > 3
        ])

    def get_final_data(self):
        df = self.storage.step().read("dataframe")
        sft_rows = []
        for _, row in df.iterrows():
            for col in ["raw_generation", "rewrite"]:
                value = row.get(col, None)
                if pd.notnull(value) and str(value).strip():
                    try:
                        # 假设每条内容是单行JSON
                        sample = eval(value) if isinstance(value, str) else value
                        # 只保留格式正确的数据
                        if (
                            isinstance(sample, dict)
                            and all(k in sample for k in ["instruction", "input", "output", "domain"])
                        ):
                            sft_rows.append(sample)
                    except Exception:
                        continue
        sft_df = pd.DataFrame(sft_rows)
        self.storage.write(sft_df)

    def forward(self):
        # 生成
        self.generator.run(
            storage=self.storage.step(),
            generation_num=self.generated_samples_num // 2,
            output_key="raw_generation"
        )
        # 重写，丰富
        self.rewriter.run(
            storage=self.storage.step(),
            input_key="raw_generation",
            output_key="rewrite"
        )
        # 给数据进行打分
        self.scorer.run(
            storage=self.storage.step(),
            input_key="raw_generation",
            output_key="score_raw"
        )
        self.scorer.run(
            storage=self.storage.step(),
            input_key="rewrite",
            output_key="score_rewrite"
        )
        # 过滤
        self.filter.run(
            storage=self.storage.step()
        )
        self.get_final_data()


if __name__ == "__main__":
    model = CoreSftFromScratchPipeline()
    model.forward()
