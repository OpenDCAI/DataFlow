 
from dataflow.operators.general_text import AlpagasusFilter
from dataflow.operators.general_text import CondorGenerator
from dataflow.operators.general_text import CondorRefiner
from dataflow.utils.storage import FileStorage
from dataflow.serving import APILLMServing_request

class TextPipeline():
    def __init__(self):
        self.storage = FileStorage(
            first_entry_file_name="",
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
        self.generator = CondorGenerator(llm_serving=llm_serving, num_samples=self.num_generated_samples)
        self.refiner = CondorRefiner(llm_serving=llm_serving)
        self.alpagasus_filter = AlpagasusFilter(min_score=3,max_score=5,llm_serving=llm_serving)

    def forward(self):
        self.generator.run(
            storage=self.storage.step()
        )
        self.refiner.run(
            storage=self.storage.step(),
            input_instruction_key='instruction',
            input_output_key='output'
        )
        self.alpagasus_filter.run(
            storage=self.storage.step(),
            input_instruction_key='instruction',
            input_input_key="input",
            input_output_key='output'
        )

model = TextPipeline()
model.forward()
