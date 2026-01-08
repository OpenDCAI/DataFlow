from dataflow.pipeline.Pipeline import PipelineABC
from dataflow.operators.core_text import BenchAnswerGenerator, UnifiedBenchDatasetEvaluator
from dataflow.prompts.core_text import FormatStrPrompt
from dataflow.utils.storage import FileStorage
from dataflow.serving import LocalModelLLMServing_vllm
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

EVAL_TYPE = "key3_q_choices_as"

class UnifiedBenchEvalPipeline(PipelineABC):
    def __init__(self, llm_serving_generator: LLMServingABC = None, llm_serving_judger: LLMServingABC = None):
        super().__init__()
        
        self.storage = FileStorage(
            first_entry_file_name="../example_data/core_text_data/unified_bench_eval_type5.jsonl",
            cache_path="./cache_local",
            file_name_prefix="dataflow_cache_step",
            cache_type="jsonl",
        )

        self.llm_serving_generator = LocalModelLLMServing_vllm(
            hf_model_name_or_path="Qwen/Qwen2.5-7B-Instruct", # set to your own model path
            vllm_tensor_parallel_size=1,
            vllm_max_tokens=2048,
        )

        self.generation_prompt_template = FormatStrPrompt(
            f_str_template="{context}Question:\n{question}\n\nChoices:\n{choices_text}\n\nAnswer:",
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
            llm_serving=self.llm_serving_generator,
            eval_type=EVAL_TYPE,
            prompt_template=None,
            use_semantic_judge=False,
            metric_type=None,           # use default metric
        )
        
    def forward(self):
        self.answer_generator_step1.run(
            storage=self.storage.step(),
            input_context_key="context",
            input_question_key="question",
            input_choices_key="choices",
            output_key="generated_ans",
        )

        self.evaluator_step2.run(
            storage=self.storage.step(),
            input_context_key="context",
            input_question_key="question",
            input_choices_key="choices",
            input_labels_key="labels",  
            input_pred_key="generated_ans",
        )

if __name__ == "__main__":
    pl = UnifiedBenchEvalPipeline()
    pl.compile()
    pl.forward()
