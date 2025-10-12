from dataflow.operators.reasoning import (
    ReasoningTokenDatasetEvaluator
)
from dataflow.utils.storage import FileStorage

class Reasoning_CPUPipeline():
    def __init__(self):

        self.storage = FileStorage(
            first_entry_file_name="/mnt/DataFlow/scy/DataFlow/dataflow/example/ReasoningPipeline/pipeline_math_short.json",
            cache_path="./cache_local",
            file_name_prefix="dataflow_cache_step",
            cache_type="jsonl",
        )
    
        self.answer_format_filter_step1 = ReasoningTokenDatasetEvaluator(model_name_or_path="/mnt/DataFlow/models/Qwen2.5-0.5B-Instruct")
        
    def forward(self):
        self.answer_format_filter_step1.run(
            storage = self.storage.step(),
            input_question_key="instruction",
            input_answer_key="output"
        )
        

if __name__ == "__main__":
    model = Reasoning_CPUPipeline()
    model.forward()
