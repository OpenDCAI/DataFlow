from dataflow.operators.core_text import PromptedFilter
from dataflow.serving import LocalModelLLMServing, APILLMServing_request
from dataflow.utils.storage import FileStorage

class GPT_evaluator():
    def __init__(self):
        self.storage = FileStorage(
            first_entry_file_name="/Users/lianghao/Desktop/dataflow_code/test_dataflow/example_data/core_text_data/eval_data.json",
            cache_path="./cache_1",
            file_name_prefix="math_QA",
            cache_type="json",
        )
        self.llm_serving = APILLMServing_request(
                api_url="http://123.129.219.111:3000/v1/chat/completions",
                model_name="gpt-4o",
                max_workers=10
        )
        self.prompt_evaluator = PromptedFilter(
            llm_serving = self.llm_serving, 
        )

    def forward(self):
        # Initial filters
        self.prompt_evaluator.run(
            storage = self.storage.step(),
            input_key = "conversations",
            output_key = "eval_dim_1",
        )


if __name__ == "__main__":
    # This is the entry point for the pipeline

    model = GPT_evaluator()
    model.forward()
