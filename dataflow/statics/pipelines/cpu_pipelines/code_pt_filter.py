from dataflow.operators.code import (
    AutogenFilter,
    CodeLengthFilter,
    CodeTextQualityFilter,
    DataPatternFilter,
    FileTypeFilter,
    DocQualityFilter,
    ScoreFilter,
)

from dataflow.utils.storage import FileStorage

class PTCodeFilter_CPUPipeline():
    def __init__(self):
        self.storage = FileStorage(
            first_entry_file_name="../example_data/CodePipeline/code_input.jsonl",
            cache_path="./cache",
            file_name_prefix="dataflow_cache_step",
            cache_type="jsonl",
        )
        
        # Initialize code filtering operators
        self.autogen_filter_step1 = AutogenFilter()
        
        self.length_filter_step2 = CodeLengthFilter()
        
        self.text_quality_filter_step3 = CodeTextQualityFilter()
        
        self.pattern_filter_step4 = DataPatternFilter()
        
        self.doc_quality_filter_step5 = DocQualityFilter()
        
        self.file_type_filter_step6 = FileTypeFilter()
        
        # Score filter (optional)
        self.score_filter_step7 = ScoreFilter()
    
    def forward(self):
        # Step 1: Auto-generated code filtering
        self.autogen_filter_step1.run(
            storage=self.storage.step(),
            input_lines_key="lines",
            output_pass_key="autogen_filter_pass"
        )
        
        # Step 2: Code length filtering
        self.length_filter_step2.run(
            storage=self.storage.step(),
            input_lines_key="lines",
            input_language_key="language",
            output_pass_key="length_filter_pass"
        )
        
        # Step 3: Code text quality filtering
        self.text_quality_filter_step3.run(
            storage=self.storage.step(),
            input_text_key="text",
            input_language_key="language",
            output_pass_key="text_quality_pass"
        )
        
        # Step 4: Data pattern filtering (Base64, hex, etc.)
        self.pattern_filter_step4.run(
            storage=self.storage.step(),
            input_text_key="text",
            output_pass_key="pattern_filter_pass"
        )
        
        # Step 5: Document quality filtering
        self.doc_quality_filter_step5.run(
            storage=self.storage.step(),
            thresholds={
                'min_num_chars': 100,
                'max_num_chars': 100000,
                'min_num_words': 10,
                'max_num_words': 50000,
                'max_frac_duplicate_lines': 0.3,
                'max_frac_duplicate_2gram': 0.3,
                'max_frac_duplicate_3gram': 0.3,
                'max_frac_duplicate_4gram': 0.3,
                'max_frac_duplicate_5gram': 0.3,
                'max_frac_curly_bracket': 0.1,
                'max_frac_all_caps_words': 0.3,
                'min_entropy_unigram': 2.0,
            }
        )
        
        # Step 6: File type filtering
        self.file_type_filter_step6.run(
            storage=self.storage.step(),
            input_dataframe_key="dataframe",
            output_dataframe_key="filtered_dataframe"
        )
        
        # Step 7: Score filtering (optional, requires quality_score field in data)
        # self.score_filter_step7.run(
        #     storage=self.storage.step(),
        #     input_score_key="quality_score",
        #     score_threshold=8,
        #     filter_method="greater_equal"
        # )

if __name__ == "__main__":
    # This is the entry point for the pipeline
    model = PTCodeFilter_CPUPipeline()
    model.forward()
