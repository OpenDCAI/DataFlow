from dataflow.operators.general_text import SFTGeneratorSeed
from dataflow.utils.storage import FileStorage
from dataflow.serving import APILLMServing_request

class TextPipeline():
    def __init__(self):
        self.storage = FileStorage(
            first_entry_file_name="../example_data/GeneralTextPipeline/pt_input.jsonl",
            cache_path="./cache",
            file_name_prefix="dataflow_cache_step",
            cache_type="jsonl",
        )
        self.model_cache_dir = './dataflow_cache'
        self.num_generated_samples = 3
        llm_serving = APILLMServing_request(
                api_url="https://api.openai.com/v1/chat/completions",
                model_name="gpt-4o",
                max_workers=100
        )
        self.generator = SFTGeneratorSeed(llm_serving=llm_serving, custom_prompt="Try to make the question suitable for middle school students.")

    def forward(self):
        self.generator.run(
            storage=self.storage.step()
        )

model = TextPipeline()
model.forward()
