from dataflow.operators.filter import (
    LLMLanguageFilter,
)
from dataflow.operators.eval import MetaScorer
from dataflow.serving import APILLMServing_request
from dataflow.utils.storage import FileStorage

class LLMFilterandEvaluatePipeline():
    def __init__(self):
        self.storage = FileStorage(
            first_entry_file_name="./dataflow/example/GeneralTextPipeline/pt_input.jsonl",
            cache_path="./cache",
            file_name_prefix="dataflow_cache_step_langc",
            cache_type="jsonl",
        )
        self.llm_serving = APILLMServing_request(
            api_url="http://123.129.219.111:3000/v1/chat/completions",
            model_name="gpt-4o"
        )
        self.llm_language_filter = LLMLanguageFilter(
            llm_serving=self.llm_serving,
            allowed_languages=['en']
            )
        self.meta_scorer = MetaScorer(llm_serving=self.llm_serving)
        
        
    def forward(self):
        self.llm_language_filter.run(
            self.storage.step(),
            input_key='raw_content'
        )
        self.meta_scorer.run(
            self.storage.step(),
            input_key='raw_content'
        )
        
if __name__ == "__main__":
    pipeline = LLMFilterandEvaluatePipeline()
    pipeline.forward()