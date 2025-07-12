from dataflow.operators.generate import PromptedGenerator
from dataflow.serving import LocalModelLLMServing_vllm
from dataflow.utils.storage import FileStorage

class Qwen_generator():
    def __init__(self):
        self.storage = FileStorage(
            first_entry_file_name="../example_data/GeneralTextPipeline/translation.jsonl",
            cache_path="./cache",
            file_name_prefix="translation",
            cache_type="jsonl",
        )
        self.model_cache_dir = './dataflow_cache'
        self.llm_serving = LocalModelLLMServing_vllm(
            hf_model_name_or_path="Qwen2.5-7B-Instruct", # set to your own model path
            vllm_tensor_parallel_size=1,
            vllm_max_tokens=8192,
        )
        self.prompt_generator = PromptedGenerator(
            llm_serving = self.llm_serving, 
            system_prompt = "Please translate to Chinese.", # System prompt for translation
        )

    def forward(self):
        # Initial filters
        self.prompt_generator.run(
            storage = self.storage.step(),
            input_key = "raw_content",
        )

if __name__ == "__main__":
    # This is the entry point for the pipeline

    model = Qwen_generator()
    model.forward()
