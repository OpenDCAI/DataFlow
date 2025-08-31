from dataflow.operators.reasoning import (
    AnswerGenerator,
    AnswerGroundTruthFilter,
    BenchCalculator
)
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
        {   # every data must include: "question", "answer"
            "file_path": "../example_data/BenchEvalPipeline/test.jsonl",
            "cache_path": f"../bench_result/{prefix}_test"
        },
    ]
###################################################

class GeneralReasoningPipeline():
    def __init__(self, first_entry_file_name: str, cache_path: str,  llm_serving: LLMServingABC = None):
        self.storage = FileStorage(
            first_entry_file_name=first_entry_file_name,
            cache_path=cache_path,
            file_name_prefix="bench_cache_step",
            cache_type="jsonl",
        )
        
        self.answer_generator_step1 = AnswerGenerator(
            llm_serving=llm_serving,
            prompt_template=DiyAnswerGeneratorPrompt(DIY_PROMPT_ANSWER)
        )

        self.eval_step2 = AnswerGroundTruthFilter(compare_method="math_verify")
        
        self.eval_step3 = BenchCalculator()
        
    def forward(self):

        self.answer_generator_step1.run(
            storage = self.storage.step(),
            input_key = "question", 
            output_key = "generated_cot"
        ),
        self.eval_step2.run(
            storage=self.storage.step(), 
            input_test_answer_key="generated_cot",
            input_gt_answer_key="answer"
          )
        self.eval_step3.run(
            storage=self.storage.step(),
            cache_path=self.storage.cache_path,
        )

if __name__ == "__main__":        
        # use vllm as LLM serving
        llm_serving = LocalModelLLMServing_vllm(
            hf_model_name_or_path=model_path, # set to your own model path
            vllm_tensor_parallel_size=1,
            vllm_max_tokens=512,
        )
        
        for bench in Benchs:
            pl = GeneralReasoningPipeline(
                first_entry_file_name=bench["file_path"],
                cache_path=bench["cache_path"],
                llm_serving=llm_serving
            )
            pl.forward()

