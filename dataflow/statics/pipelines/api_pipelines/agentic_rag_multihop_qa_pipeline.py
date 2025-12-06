import pandas as pd
from dataflow.operators.agentic_rag import MultiHopRAGVerifier
from dataflow.operators.agentic_rag import MultiHopRAGGenerator

from dataflow.utils.storage import FileStorage
from dataflow.serving import APILLMServing_request
from dataflow.core import LLMServingABC

class AgenticRAGMultihopQA_APIPipeline():

    def __init__(self, llm_serving=None):

        self.storage = FileStorage(
            first_entry_file_name="../example_data/AgenticRAGPipeline/multihop_pipeline_data.jsonl",
            cache_path="./agenticRAG_multihop_cache",
            file_name_prefix="agentic_rag_multihop",
            cache_type="jsonl",
        )

        self.llm_serving = APILLMServing_request(
            api_url="https://api.openai.com/v1/chat/completions",
            model_name="o4-mini",
            max_workers=500
        )

        self.task_step1 = MultiHopRAGGenerator(
            llm_serving=self.llm_serving,
            retriever_url="your_retriever_url"
        )

        self.task_step2 = MultiHopRAGVerifier(
            llm_serving=self.llm_serving
        )
        
    def forward(self):

        self.task_step1.run(
            storage = self.storage.step(),
            input_hop = 1,
        )

        self.task_step2.run(
            storage=self.storage.step(),
        )

if __name__ == "__main__":
    model = AgenticRAGMultihopQA_APIPipeline()
    model.forward()
