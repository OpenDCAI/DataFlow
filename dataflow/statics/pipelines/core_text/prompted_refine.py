from dataflow.operators.core_text import PromptedRefiner
from dataflow.serving import APILLMServing_request
from dataflow.utils.storage import FileStorage

class GPT_generator():
    def __init__(self):
        self.storage = FileStorage(
            first_entry_file_name="../example_data/GeneralTextPipeline/abbreviation.jsonl",
            cache_path="./cache",
            file_name_prefix="math_QA",
            cache_type="jsonl",
        )
        self.model_cache_dir = './dataflow_cache'
        self.llm_serving = APILLMServing_request(
                api_url="https://api.openai.com/v1/chat/completions",
                model_name="gpt-4o",
                max_workers=10
        )
        self.prompt_refiner = PromptedRefiner(
            llm_serving = self.llm_serving, 
            system_prompt = "Please rewrite this sentence into a better one.", # System prompt for math problem solving
        )

    def forward(self):
        # Initial filters
        self.prompt_refiner.run(
            storage = self.storage.step(),
            input_key = "raw_content",
        )


if __name__ == "__main__":
    # This is the entry point for the pipeline

    model = GPT_generator()
    model.forward()
