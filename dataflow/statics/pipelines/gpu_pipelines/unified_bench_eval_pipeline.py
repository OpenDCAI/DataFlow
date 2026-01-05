from dataflow.operators.core_text import BenchAnswerGenerator, UnifiedBenchDatasetEvaluator
from dataflow.utils.storage import FileStorage
from dataflow.serving import LocalModelLLMServing_vllm
from dataflow.core import LLMServingABC
    
DIY_PROMPT_ANSWER = """Please output the answer."""

class UnifiedBenchEvalPipeline():
    def __init__(self, llm_serving_generator: LLMServingABC = None, llm_serving_judger: LLMServingABC = None):
        
        self.storage = FileStorage(
            first_entry_file_name="../example_data/core_text_data/bench_eval_data.jsonl",
            cache_path="./cache_local",
            file_name_prefix="dataflow_cache_step",
            cache_type="jsonl",
        )

        self.llm_serving_generator = LocalModelLLMServing_vllm(
            hf_model_name_or_path="/mnt/DataFlow/scy/Model/Qwen2.5-7B-Instruct", # set to your own model path
            vllm_tensor_parallel_size=1,
            vllm_max_tokens=2048,
        )

        self.answer_generator_step1 = BenchAnswerGenerator(
            llm_serving=self.llm_serving_generator,
            prompt_template=None,
            allow_overwrite=False,
            force_generate=False,
        )
        
        self.evaluator_step2 = UnifiedBenchDatasetEvaluator(
            eval_result_path="./cache_local/eval_result/eval_result.jsonl",
            llm_serving=self.llm_serving_generator,
            eval_type="key1_text_score",
            prompt_template=None,
            use_semantic_judge=False,
            metric_type=None,           # use default metric
        )
        
    def forward(self):
        self.answer_generator_step1.run(
            storage=self.storage.step(),
            keys_map={"text": "text"},
            context_key=None,
            output_key="generated_ans",
        )
        """
        all types:
        "key1_text_score",
        "key2_qa",
        "key2_q_ma",
        "key3_q_choices_a",
        "key3_q_choices_as",
        "key3_q_a_rejected",
        """
        self.evaluator_step2.run(
            storage=self.storage.step(),
            keys_map={"text": "text"},
            context_key=None,
            input_pred_key="generated_ans",

        )

if __name__ == "__main__":
    pl = UnifiedBenchEvalPipeline()
    pl.forward()
