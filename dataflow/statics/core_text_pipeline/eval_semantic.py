"""
本文件用于以语义匹配的方式做Benchmark评测，主要流程包括：
1. 通过AnswerGenerator生成模型回答（generated_cot）；
2. 使用BenchEvaluator进行语义比较评估（semantic）；
3. 评估结果会保存在指定的eval_result_path路径下。

使用方法：
1. 设置您自己的提示模板（DIY_PROMPT_ANSWER）；
2. 配置您要评估的模型路径（model_paths）；
3. 指定评估数据集路径和缓存路径（Benchs）；
4. 运行本文件即可自动完成评估流程，最终结果会保存在指定路径下。
"""
from dataflow.operators.reasoning import ReasoningAnswerGenerator
from dataflow.operators.core_text import BenchDatasetEvaluator
from dataflow.serving import APILLMServing_request, LocalModelLLMServing_vllm
from dataflow.prompts.reasoning.diy import DiyAnswerGeneratorPrompt
from dataflow.utils.storage import FileStorage
from dataflow.core import LLMServingABC

import os

""" 3 steps for evaluating your own model using personal bench """
    # Step 1, your own prompt
DIY_PROMPT_ANSWER ="""<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n
<|im_start|>user\n{question}<|im_end|>\n
<|im_start|>assistant\n"""

    # Step 2, your own model path, support multi-model path
model_paths = [
    "Qwen/Qwen2.5-7B-Instruct",
]

prefixs = [os.path.basename(path) for path in model_paths]

for prefix, model_path in zip(prefixs, model_paths):
    # Step 3, your own bench path, prefix is according to model_path
    Benchs = [
        {
            "file_path": "../example_data/core_text_data/bench_eval_data.jsonl",
            "cache_path": f"../bench_result/{prefix}_test"
        },
    ]
###################################################

class SemanticBenchEvalPipeline():
    def __init__(self, first_entry_file_name: str, cache_path: str,  llm_serving: LLMServingABC = None, llm_serving_judger: LLMServingABC = None):
        self.storage = FileStorage(
            first_entry_file_name=first_entry_file_name,
            cache_path=cache_path,
            file_name_prefix="bench_cache_step",
            cache_type="jsonl",
        )
        
        self.answer_generator_step1 = ReasoningAnswerGenerator(
            llm_serving=llm_serving,
            prompt_template=DiyAnswerGeneratorPrompt(DIY_PROMPT_ANSWER)
        )

        self.eval_step2 = BenchDatasetEvaluator(
            compare_method="semantic",
            llm_serving=llm_serving_judger,
            prompt_template=None, # using default prompt
            eval_result_path="../eval_result.json",
        )
        
    def forward(self):

        self.answer_generator_step1.run(
            storage = self.storage.step(),
            input_key = "question", 
            output_key = "generated_cot"
        ),
        self.eval_step2.run(
            storage=self.storage.step(), 
            input_question_key="question", 
            input_test_answer_key="generated_cot",
            input_gt_answer_key="answer"
          )

if __name__ == "__main__":        
        # use vllm as LLM serving
        llm_serving = LocalModelLLMServing_vllm(
            hf_model_name_or_path=model_path, # set to your own model path
            vllm_tensor_parallel_size=1,
            vllm_max_tokens=512,
        )
        llm_serving_judger = APILLMServing_request(
                api_url="your api url",
                model_name="your model name",
                max_workers=30
        )
        
        for bench in Benchs:
            pl = SemanticBenchEvalPipeline(
                first_entry_file_name=bench["file_path"],
                cache_path=bench["cache_path"],
                llm_serving=llm_serving,
                llm_serving_judger=llm_serving_judger
            )
            pl.forward()

