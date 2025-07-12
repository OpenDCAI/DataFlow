from dataflow.operators.generate import PromptedGenerator
from dataflow.serving import LocalModelLLMServing, APILLMServing_request
from dataflow.utils.storage import FileStorage

class GPT_generator():
    def __init__(self):
        self.storage = FileStorage(
            first_entry_file_name="../../dataflow/example/GeneralTextPipeline/math_100.jsonl",
            cache_path="./cache",
            file_name_prefix="math_QA",
            cache_type="jsonl",
        )
        self.model_cache_dir = './dataflow_cache'
        self.llm_serving = APILLMServing_request(
                api_url="http://123.129.219.111:3000/v1/chat/completions",
                model_name="gpt-4o",
                max_workers=50
        )
        self.prompt_generator = PromptedGenerator(llm_serving = self.llm_serving)        

    def forward(self):
        # Initial filters
        self.prompt_generator.run(
            storage = self.storage.step(),
            system_prompt = "Please solve this math problem.",
            input_key = "problem",
        )


if __name__ == "__main__":
    # This is the entry point for the pipeline

    model = GPT_generator()
    model.forward()
