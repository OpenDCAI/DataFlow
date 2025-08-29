from dataflow.operators.core_text import PairedPromptedGenerator
from dataflow.serving import APILLMServing_request
from dataflow.utils.storage import FileStorage

class GPT_generator():
    def __init__(self):
        self.storage = FileStorage(
            first_entry_file_name="../example_data/core_text_data/pair_math_data.json",
            cache_path="./cache",
            file_name_prefix="math_QA",
            cache_type="jsonl",
        )
        self.model_cache_dir = './dataflow_cache'
        self.llm_serving = APILLMServing_request(
                api_url="http://123.129.219.111:3000/v1/chat/completions",
                model_name="gpt-4o",
                max_workers=2
        )
        self.prompt_generator = PairedPromptedGenerator(
            llm_serving = self.llm_serving, 
            system_prompt = "Please use the two given math conditions to produce a challenging math question. Only output the question.", 
        )

    def forward(self):
        # Initial filters
        self.prompt_generator.run(
            storage = self.storage.step(),
            input_key_1 = "condition_1",
            input_key_2 = "condition_2",
            output_key = "output_key"
        )


if __name__ == "__main__":
    # This is the entry point for the pipeline

    model = GPT_generator()
    model.forward()
