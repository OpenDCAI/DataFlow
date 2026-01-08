from dataflow.operators.core_text import BenchAnswerGenerator, UnifiedBenchDatasetEvaluator
from dataflow.prompts.core_text import FormatStrPrompt
from dataflow.core.prompt import DIYPromptABC
from dataflow.utils.storage import FileStorage
from dataflow.serving import LocalModelLLMServing_vllm, APILLMServing_request
from dataflow.core import LLMServingABC

"""
all types:
"key1_text_score",
"key2_qa",
"key2_q_ma",
"key3_q_choices_a",
"key3_q_choices_as",
"key3_q_a_rejected",
"""

EVAL_TYPE = "key2_qa"

class UnifiedBenchEvalPipeline():
    def __init__(self, llm_serving_generator: LLMServingABC = None, llm_serving_judger: LLMServingABC = None):
        
        self.storage = FileStorage(
            first_entry_file_name="../example_data/core_text_data/unified_bench_eval_type2.jsonl",
            cache_path="./cache_local",
            file_name_prefix="dataflow_cache_step",
            cache_type="jsonl",
        )

        self.llm_serving_generator = LocalModelLLMServing_vllm(
            hf_model_name_or_path="/mnt/DataFlow/models/Qwen2.5-7B-Instruct", # set to your own model path
            vllm_tensor_parallel_size=1,
            vllm_max_tokens=2048,
        )

        # use API server as LLM serving
        self.llm_serving_judger = APILLMServing_request(
                    api_url="https://api.openai.com/v1/chat/completions",
                    model_name="gpt-4o",
                    max_workers=5
        )

        self.generation_prompt_template = FormatStrPrompt(
            f_str_template="Question: {question}\nAnswer:",
        )
        
        self.answer_generator_step1 = BenchAnswerGenerator(
            llm_serving=self.llm_serving_generator,
            eval_type=EVAL_TYPE,
            prompt_template=self.generation_prompt_template,
            allow_overwrite=False,
            force_generate=False,
        )
        
        self.evaluator_step2 = UnifiedBenchDatasetEvaluator(
            eval_result_path="./cache_local/eval_result/eval_result.jsonl",
            llm_serving=self.llm_serving_judger,
            eval_type=EVAL_TYPE,
            prompt_template=None,
            use_semantic_judge=True,
            metric_type=None,           # use default metric
        )
        
    def forward(self):
        self.answer_generator_step1.run(
            storage=self.storage.step(),
            input_question_key="question",
            input_target_key="golden_label",
            input_context_key=None,
            output_key="generated_ans",
        )

        self.evaluator_step2.run(
            storage=self.storage.step(),
            input_question_key="question",
            input_target_key="golden_label",
            input_context_key=None,
            input_pred_key="generated_ans",
        )

if __name__ == "__main__":
    pl = UnifiedBenchEvalPipeline()
    pl.forward()
