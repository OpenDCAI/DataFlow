from dataflow.operators.filter import (
    AnswerFormatterFilter,
    AnswerGroundTruthFilter,
    AnswerNgramFilter,
)
from dataflow.utils.storage import FileStorage

class ReasoningPipeline():
    def __init__(self):

        self.storage = FileStorage(
            first_entry_file_name="../example_data/ReasoningPipeline/pipeline_math_short.json",
            cache_path="./cache_local",
            file_name_prefix="dataflow_cache_step",
            cache_type="jsonl",
        )
    
        self.answer_format_filter_step1 = AnswerFormatterFilter()
        
        self.answer_groundtruth_filter_step2 = AnswerGroundTruthFilter()
        
        self.answer_ngram_filter_step3 = AnswerNgramFilter(
            min_score = 0.1,
            max_score = 1.0,
            ngrams = 5
        )
        
    def forward(self):
        self.answer_format_filter_step1.run(
            storage = self.storage.step(),
            input_key = "output",
        )
        
        self.answer_groundtruth_filter_step2.run(
            storage = self.storage.step(),
            test_answer_key = "output",
            gt_answer_key =  "golden_answer"
        )
        
        self.answer_ngram_filter_step3.run(
            storage = self.storage.step(),
            question_key = "instruction",
            answer_key = "output"
        )

if __name__ == "__main__":
    model = ReasoningPipeline()
    model.forward()
