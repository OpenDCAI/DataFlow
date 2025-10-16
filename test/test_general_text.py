from dataflow.operators.text_pt import PerplexityFilter
from dataflow.utils.storage import FileStorage
class TextPipeline():
    def __init__(self):
        self.storage = FileStorage(
            first_entry_file_name="./dataflow/example/GeneralTextPipeline/pt_input.jsonl",
            cache_path="./cache",
            file_name_prefix="dataflow_cache_step",
            cache_type="jsonl",
        )
        self.model_cache_dir = './dataflow_cache'
        self.processor = PerplexityFilter(min_score=20, model_name='gpt2')

    def forward(self):
        self.processor.run(
            storage=self.storage.step(),
            input_key='raw_content'
        )

if __name__ == "__main__":
    # This is a test entry point for the TextPipeline
    # It will run the forward method of the TextPipeline class
    # to process the data and generate the output.
    print("Running TextPipeline...")
    model = TextPipeline()
    model.forward()