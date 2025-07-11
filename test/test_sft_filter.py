from dataflow.operators.filter import (
    WordNumberFilter,
    SuperfilteringFilter,
    DeitaQualityFilter,
    InstagFilter
)
from dataflow.utils.storage import FileStorage


class SFTTextPipeline():
    
    def __init__(self):
        
        self.storage = FileStorage(
            first_entry_file_name="./dataflow/example/GeneralTextPipeline/sft_input.jsonl",
            cache_path="./cache",
            file_name_prefix="dataflow_cache_step",
            cache_type="jsonl",
        )
        
        self.model_cache_dir = './dataflow_cache'
        self.word_number_filter_step1 = WordNumberFilter(
            min_words=20,
            max_words=1000
        )
        self.super_filtering_filter_step2 = SuperfilteringFilter(
            min_score=0.5,
            max_score=1.0,
            model_cache_dir=self.model_cache_dir
        )
        self.deita_quality_filter_step3 = DeitaQualityFilter(
            min_score=2.5,
            max_score=10000,
            max_length=512,
            model_cache_dir=self.model_cache_dir
        )
        
        self.instag_filter_step4 = InstagFilter(
            min_score=2,
            max_score=10000,
            model_cache_dir=self.model_cache_dir,
            max_new_tokens=1024
        )
        
    def forward(self):
        
        self.word_number_filter_step1.run(
            storage=self.storage.step(),
            input_key="output",
        )
        
        self.super_filtering_filter_step2.run(
            storage=self.storage.step(),
            input_instruction_key='instruction',
            input_input_key=None,
            input_output_key='output'
        )
        
        self.deita_quality_filter_step3.run(
            storage=self.storage.step(),
            input_instruction_key='instruction',
            input_output_key='output'
        )
        
        self.instag_filter_step4.run(
            storage=self.storage.step(),
            input_instruction_key='instruction'
        )
if __name__ == "__main__":
    # This is the entry point for the pipeline
    pipeline = SFTTextPipeline()
    pipeline.forward()