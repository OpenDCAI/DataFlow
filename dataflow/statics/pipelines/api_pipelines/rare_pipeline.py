from dataflow.operators.generate import (
    Doc2Query,
    BM25HardNeg,
    ReasonDistill,
)
from dataflow.utils.storage import FileStorage
from dataflow.serving import APILLMServing_request

class RAREPipeline():
    def __init__(self):

        self.storage = FileStorage(
            first_entry_file_name="../example_data/AgenticRAGPipeline/pipeline_small_chunk.json",
            cache_path="./cache_local",
            file_name_prefix="dataflow_cache_step",
            cache_type="json",
        )

        # use API server as LLM serving
        llm_serving = APILLMServing_request(
                api_url="https://api.openai.com/v1/chat/completions",
                model_name="gpt-4o",
                max_workers=1
        )

        self.doc2query_step1 = Doc2Query(llm_serving)

        self.bm25hardneg_step2 = BM25HardNeg()

        self.reasondistill_step3 = ReasonDistill(llm_serving)
        
    def forward(self):

        self.doc2query_step1.run(
            storage = self.storage.step(),
            input_key = "text",
        )

        self.bm25hardneg_step2.run(
            storage = self.storage.step(),
            input_question_key = "question",
            input_text_key = "text",
            output_negatives_key = "hard_negatives",
        )

        self.reasondistill_step3.run(
            storage= self.storage.step(),
            input_text_key = "text",
            input_question_key = "question",
            input_scenario_key = "scenario",
            input_hardneg_key = "hard_negatives",
            output_key= "reasoning",
        )
        
if __name__ == "__main__":
    model = RAREPipeline()
    model.forward()
