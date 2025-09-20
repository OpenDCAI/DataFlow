from dataflow.operators.rare import (
    Doc2Query,
    BM25HardNeg,
    ReasonDistill,
)
from dataflow.utils.storage import FileStorage
from dataflow.serving.api_llm_serving_request import APILLMServing_request
from dataflow.serving.local_model_llm_serving import LocalModelLLMServing_vllm

class RAREPipeline():
    def __init__(self):

        self.storage = FileStorage(
            first_entry_file_name="./dataflow/example/RAREPipeline/pipeline_small_chunk.json",
            cache_path="./cache_local",
            file_name_prefix="dataflow_cache_step",
            cache_type="json",
        )

        # 使用 API 服务器作为 LLM 服务，可以修改为LocalModelLLMServing_vllm以使用本地模型
        llm_serving = APILLMServing_request(
                api_url="https://api.openai.com/v1/chat/completions",
                key_name_of_api_key="OPENAI_API_KEY",
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