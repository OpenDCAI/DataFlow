import pandas as pd
from dataflow.operators.eval import F1Scorer

from dataflow.operators.generate import (
    AtomicTaskGenerator,
    DepthQAGenerator,
    WidthQAGenerator
)

from dataflow.utils.storage import FileStorage
from dataflow.serving import APILLMServing_request, LocalModelLLMServing
from dataflow.core import LLMServingABC

class AgenticRAGEvalPipeline():

    def __init__(self, llm_serving=None):

        self.storage = FileStorage(
            first_entry_file_name="../dataflow/example/AgenticRAGPipeline/eval_test_data.jsonl",
            cache_path="./agenticRAG_eval_cache",
            file_name_prefix="agentic_rag_eval",
            cache_type="jsonl",
        )

        llm_serving = APILLMServing_request(
            api_url="https://api.openai.com/v1/chat/completions", 
            model_name="gpt-4o-mini",
            max_workers=500
        )

        self.task_step1 = AtomicTaskGenerator(
            llm_serving=llm_serving
        )

        self.task_step2 = F1Scorer()
        
    def forward(self):

        self.task_step1.run(
            storage = self.storage.step(),
            input_key = "contents",
        )

        self.task_step2.run(
            storage=self.storage.step(),
            output_key="F1Score",
            prediction_key="refined_answer",
            ground_truth_key="golden_doc_answer"
        )

if __name__ == "__main__":
    model = AgenticRAGEvalPipeline()
    model.forward()
