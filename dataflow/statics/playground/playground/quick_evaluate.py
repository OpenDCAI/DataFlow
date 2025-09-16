from dataflow.operators.general_text import MetaScorer
from dataflow.serving import APILLMServing_request
from dataflow.utils.storage import FileStorage

class QuickEvaluatePipeline():
    def __init__(self):
        self.storage = FileStorage(
            first_entry_file_name="../example_data/GeneralTextPipeline/pt_input.jsonl",
            cache_path="./cache",
            file_name_prefix="dataflow_cache_step_langc",
            cache_type="jsonl",
        )
        self.llm_serving = APILLMServing_request(
            api_url="http://123.129.219.111:3000/v1/chat/completions",
            model_name="gpt-4o"
        )
        self.meta_scorer = MetaScorer(llm_serving=self.llm_serving)
        
        
    def forward(self):
        self.meta_scorer.run(
            self.storage.step(),
            input_key='raw_content'
        )
        
if __name__ == "__main__":
    pipeline = QuickEvaluatePipeline()
    pipeline.forward()