from dataflow.operators.core_text import BenchDatasetEvaluator
from dataflow.operators.reasoning import ReasoningAnswerGenerator
from dataflow.utils.storage import FileStorage
from dataflow.serving import APILLMServing_request, LocalModelLLMServing_vllm
from dataflow.core import LLMServingABC
from dataflow.prompts.reasoning.diy import (
    DiyAnswerGeneratorPrompt,
)
    
DIY_PROMPT_ANSWER = """Please output the answer."""

class BenchEvalPipeline():
    def __init__(self, llm_serving_generator: LLMServingABC = None, llm_serving_judger: LLMServingABC = None):
        
        self.storage = FileStorage(
            first_entry_file_name="../example_data/core_text_data/bench_eval_data_2.jsonl",
            cache_path="./cache_local",
            file_name_prefix="dataflow_cache_step",
            cache_type="jsonl",
        )

        # use API server as LLM serving
        self.llm_serving_judger = APILLMServing_request(
                    api_url="https://api.openai.com/v1/chat/completions",
                    model_name="gpt-4o",
                    max_workers=30
        )
        
        self.evaluator_step = BenchDatasetEvaluator(
            eval_result_path="./cache_local/eval_result/eval_result.jsonl",
            compare_method="semantic", # or match
            llm_serving=self.llm_serving_judger,
            prompt_template = None # you can diy your judger prompt in dataflow.prompts.reasoning.general.AnswerJudgePrompt
        )
        
    def forward(self):
        self.evaluator_step.run(
            storage = self.storage.step(),
            input_test_answer_key="model_answer",
            input_gt_answer_key="golden_label"
        )

if __name__ == "__main__":
    pl = BenchEvalPipeline()
    pl.forward()
