from dataflow.operators.conversations import ConsistentChatGenerator
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
        serving = APILLMServing_request(
            api_url="http://123.129.219.111:3000/v1/chat/completions",
            model_name="gpt-4o",
            max_workers=100
        )
        self.model_cache_dir = './dataflow_cache'
        self.processor = ConsistentChatGenerator(llm_serving=serving, num_dialogs_per_intent=5)

    def forward(self):
        self.processor.run(
            storage=self.storage.step()
        )

if __name__ == "__main__":
    # This is a test entry point for the TextPipeline
    # It will run the forward method of the TextPipeline class
    # to process the data and generate the output.
    print("Running TextPipeline...")
    model = TextPipeline()
    model.forward()