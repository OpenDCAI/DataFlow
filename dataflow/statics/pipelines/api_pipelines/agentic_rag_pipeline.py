import pandas as pd
from dataflow.operators.agentic_rag import AgenticRAGQAF1SampleEvaluator

from dataflow.operators.agentic_rag import (
    AgenticRAGAtomicTaskGenerator,
    AgenticRAGDepthQAGenerator,
    AgenticRAGWidthQAGenerator
)

from dataflow.utils.storage import FileStorage
from dataflow.serving import APILLMServing_request
from dataflow.core import LLMServingABC

class AgenticRAGEval_APIPipeline():

    def __init__(self, llm_serving=None):

        self.storage = FileStorage(
            first_entry_file_name="../example_data/AgenticRAGPipeline/eval_test_data.jsonl",
            cache_path="./agenticRAG_eval_cache",
            file_name_prefix="agentic_rag_eval",
            cache_type="jsonl",
        )

        self.llm_serving = APILLMServing_request(
            api_url="https://api.openai.com/v1/chat/completions",
            model_name="gpt-4o-mini",
            max_workers=500
        )

        self.task_step1 = AgenticRAGAtomicTaskGenerator(
            llm_serving=self.llm_serving
        )

        self.task_step2 = AgenticRAGQAF1SampleEvaluator()
        
    def forward(self):

        self.task_step1.run(
            storage = self.storage.step(),
            input_key = "contents",
        )

        self.task_step2.run(
            storage=self.storage.step(),
            output_key="F1Score",
            input_prediction_key="refined_answer",
            input_ground_truth_key="golden_doc_answer"
        )

if __name__ == "__main__":
    model = AgenticRAGEval_APIPipeline()
    model.forward()
