from dataflow.operators.filter import WordNumberFilter
from dataflow.utils.storage import FileStorage


class SFTTextPipeline():
    
    def __init__(self):
        
        self.storage = FileStorage(
            first_entry_file_name="../example_data/GeneralTextPipeline/sft_input.jsonl",
            cache_path="./cache",
            file_name_prefix="dataflow_cache_step",
            cache_type="jsonl",
        )
        
        self.model_cache_dir = './dataflow_cache'
        self.word_number_filter_step1 = WordNumberFilter(
            min_words=20,
            max_words=1000
        )
        
    def forward(self):
        
        self.word_number_filter_step1.run(
            storage=self.storage.step(),
            input_key="output",
        )

if __name__ == "__main__":
    # This is the entry point for the pipeline
    pipeline = SFTTextPipeline()
    pipeline.forward()