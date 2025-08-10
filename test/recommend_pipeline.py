import pytest
from dataflow.operators.generate.Reasoning.question_generator import QuestionGenerator
from dataflow.operators.filter.Reasoning.question_filter import QuestionFilter
from dataflow.operators.generate.Reasoning.question_difficulty_classifier import QuestionDifficultyClassifier
from dataflow.utils.storage import FileStorage
from dataflow.serving import APILLMServing_request, LocalModelLLMServing_vllm, LocalModelLLMServing_sglang


class RecommendPipeline():
    def __init__(self):

        # -------- FileStorage (请根据需要修改参数) --------
        self.storage = FileStorage(
            first_entry_file_name="/mnt/h_h_public/lh/lz/DataFlow/dataflow/example/ReasoningPipeline/pipeline_math_short.json",
            cache_path="./cache_local",
            file_name_prefix="dataflow_cache_step",
            cache_type="jsonl",
        )


        # -------- LLM Serving (Remote) --------
        llm_serving = APILLMServing_request(
            api_url="http://123.129.219.111:3000/v1/chat/completions",
            key_name_of_api_key='DF_API_KEY',
            model_name="gpt-4o",
            max_workers=100,
        )
        # For local models, uncomment below
        # llm_serving = LocalModelLLMServing_vllm(
        #     hf_model_name_or_path="/mnt/public/model/huggingface/Qwen2.5-7B-Instruct",
        #     vllm_tensor_parallel_size=1,
        #     vllm_max_tokens=8192,
        #     hf_local_dir="local",
        # )

        self.questiongenerator = QuestionGenerator(num_prompts=1, llm_serving=llm_serving, prompt_template="")
        self.questionfilter = QuestionFilter(system_prompt="You are a helpful assistant.", llm_serving=llm_serving, prompt_template="")
        self.questiondifficultyclassifier = QuestionDifficultyClassifier(llm_serving=llm_serving)

    def forward(self):
        self.questiongenerator.run(
            storage=self.storage.step(), input_key=""
        )
        self.questionfilter.run(
            storage=self.storage.step(), input_key="math_problem"
        )
        self.questiondifficultyclassifier.run(
            storage=self.storage.step(), input_key="", output_key="difficulty_score"
        )


if __name__ == "__main__":
    pipeline = RecommendPipeline()
    pipeline.forward()
