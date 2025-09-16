from dataflow.operators.reasoning import (
    ReasoningAnswerFormatterFilter,
    ReasoningAnswerGroundTruthFilter,
    ReasoningAnswerNgramFilter,
)
from dataflow.utils.storage import FileStorage

class Reasoning_CPUPipeline():
    def __init__(self):

        self.storage = FileStorage(
            first_entry_file_name="../example_data/ReasoningPipeline/pipeline_math_short.json",
            cache_path="./cache_local",
            file_name_prefix="dataflow_cache_step",
            cache_type="jsonl",
        )
    
        self.answer_format_filter_step1 = ReasoningAnswerFormatterFilter()
        
        self.answer_groundtruth_filter_step2 = ReasoningAnswerGroundTruthFilter()
        
        self.answer_ngram_filter_step3 = ReasoningAnswerNgramFilter(
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
            input_test_answer_key = "output",
            input_gt_answer_key =  "golden_answer"
        )
        
        self.answer_ngram_filter_step3.run(
            storage = self.storage.step(),
            input_question_key = "instruction",
            input_answer_key = "output"
        )

if __name__ == "__main__":
    model = Reasoning_CPUPipeline()
    model.forward()
