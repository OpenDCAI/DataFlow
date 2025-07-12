from dataflow.operators.generate import (
    Doc2Query,
    BM25HardNeg,
    ReasonDistill,
)
from dataflow.utils.storage import FileStorage
from dataflow.serving import LocalModelLLMServing_vllm

class RAREPipeline():
    def __init__(self):

        self.storage = FileStorage(
            first_entry_file_name="../example_data/AgenticRAGPipeline/pipeline_small_chunk.json",
            cache_path="./cache_local",
            file_name_prefix="dataflow_cache_step",
            cache_type="json",
        )

        # use local model as LLM serving
        llm_serving = LocalModelLLMServing_vllm(
            hf_model_name_or_path="LLama3.1-70B-Instruct", # set to your own model path
            vllm_tensor_parallel_size=1,
            vllm_max_tokens=1024*8,
            vllm_max_model_len=1024*24,
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
            input_text_key="text",
            input_question_key="question",
            output_negatives_key="hard_negatives",
        )

        self.reasondistill_step3.run(
            storage = self.storage.step(),
            input_text_key="text",
            input_question_key="question",
            input_scenario_key="scenario",
            input_hardneg_key="hard_negatives",
            output_key="reasoning",
        )
        
if __name__ == "__main__":
    model = RAREPipeline()
    model.forward()
