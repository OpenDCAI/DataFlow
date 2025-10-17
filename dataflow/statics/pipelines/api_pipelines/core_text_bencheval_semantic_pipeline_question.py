from dataflow.operators.core_text import BenchDatasetEvaluatorQuestion
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
            first_entry_file_name="../example_data/core_text_data/bench_eval_data.jsonl",
            cache_path="./cache_local",
            file_name_prefix="dataflow_cache_step",
            cache_type="jsonl",
        )

        self.llm_serving_generator = LocalModelLLMServing_vllm(
            hf_model_name_or_path="Qwen/Qwen2.5-7B-Instruct", # set to your own model path
            vllm_tensor_parallel_size=1,
            vllm_max_tokens=2048,
        )
        
        # use API server as LLM serving
        self.llm_serving_judger = APILLMServing_request(
                    api_url="https://api.openai.com/v1/chat/completions",
                    model_name="gpt-4o",
                    max_workers=30
        )

        self.answer_generator_step1 = ReasoningAnswerGenerator(
            llm_serving=self.llm_serving_generator,
            prompt_template=DiyAnswerGeneratorPrompt(DIY_PROMPT_ANSWER)
        )
        
        self.evaluator_step2 = BenchDatasetEvaluatorQuestion(
            eval_result_path="./cache_local/eval_result/eval_result.jsonl",
            compare_method="semantic", # or match
            llm_serving=self.llm_serving_judger,
            prompt_template = None # you can diy your judger prompt in dataflow.prompts.reasoning.general.AnswerJudgePrompt
        )
        
    def forward(self):
        self.answer_generator_step1.run(
            storage = self.storage.step(),
            input_key = "instruction", 
            output_key = "generated_cot"
        )
        self.evaluator_step2.run(
            storage = self.storage.step(),
            input_test_answer_key="generated_cot",
            input_gt_answer_key="golden_answer",
            input_question_key="instruction",
        )

if __name__ == "__main__":
    pl = BenchEvalPipeline()
    pl.forward()
