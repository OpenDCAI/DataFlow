from dataflow.operators.knowledge_cleaning import (
    KBCChunkGenerator,
    FileOrURLToMarkdownConverterAPI,
    FileOrURLToMarkdownConverterFlash,
    FileOrURLToMarkdownConverterLocal,
    KBCTextCleaner,
    # KBCMultiHopQAGenerator,
)
from dataflow.operators.core_text import Text2MultiHopQAGenerator
from dataflow.utils.storage import FileStorage
from dataflow.serving import APILLMServing_request

class KBCleaningPDF_APIPipeline():
    def __init__(self):

        self.storage = FileStorage(
            first_entry_file_name="../example_data/KBCleaningPipeline/kbc_test_1.jsonl",
            cache_path="./.cache/api",
            file_name_prefix="knowledge_cleaning_step",
            cache_type="json",
        )

        self.llm_serving = APILLMServing_request(
                api_url="https://api.openai.com/v1/chat/completions",
                model_name="gpt-4o",
                max_workers=100
        )

        # ------------case1: use MinerU official API (by default) ------------
        # by default we use API provided by MinerU official
        # https://mineru.net/apiManage/docs
        self.knowledge_cleaning_step1 = FileOrURLToMarkdownConverterAPI(
            intermediate_dir="../example_data/KBCleaningPipeline/API/",
            mineru_backend="vlm",  # vlm or pipeline
            api_key=None  # !!! place your api key here or set environment variable MINERU_API_KEY!!!
        )
        # ------------case2: use Flash-MinerU inference locally with GPU ------------
        # https://github.com/OpenDCAI/Flash-MinerU
        # self.knowledge_cleaning_step1 = FileOrURLToMarkdownConverterFlash(
        #     intermediate_dir="../example_data/KBCleaningPipeline/flash/",
        #     mineru_model_path="<your Model Path>/MinerU2.5-2509-1.2B",  # !!! place your local model path here !!!
        #     # https://huggingface.co/opendatalab/MinerU2.5-2509-1.2B.
        #     batch_size=4, # batchsize per vllm worker
        #     replicas=1,   # num of vllm workers
        #     num_gpus_per_replica=0.5, # for ray to schedule vllm workers to GPU, can be float, e.g. 0.5 means each worker uses half GPU, 1 means each worker uses whole GPU
        #     engine_gpu_util_rate_to_ray_cap=0.9 # actuall GPU utilization for each worker; acturall memory per worker= num_gpus_per_replica * engine_gpu_util_rate_to_ray_cap; this is to avoid OOM, you can set it to 0.9 or 0.8 to leave some buffer for other processes on GPU
        # )

        # ------------case3: use MinerU official inference locally (much slower than other two) ------------
        # self.knowledge_cleaning_step1 = FileOrURLToMarkdownConverterLocal(
        #     intermediate_dir="../example_data/KBCleaningPipeline/local/",
        #     mineru_backend="vlm-local-engine",
        #     # https://huggingface.co/opendatalab/MinerU2.5-2509-1.2B.
        #     mineru_model_path="<your Model Path>/MinerU2.5-2509-1.2B",
        # )

        self.knowledge_cleaning_step2 = KBCChunkGenerator(
            split_method="token",
            chunk_size=512,
            tokenizer_name="Qwen/Qwen2.5-7B-Instruct",
        )

        self.knowledge_cleaning_step3 = KBCTextCleaner(
            llm_serving=self.llm_serving,
            lang="en"
        )

        self.knowledge_cleaning_step4 = Text2MultiHopQAGenerator(
            llm_serving=self.llm_serving,
            lang="en",
            num_q = 5
        )

    def forward(self):
        extracted=self.knowledge_cleaning_step1.run(
            storage=self.storage.step(),
            input_key="source",
            output_key="text_path"
        )
        
        self.knowledge_cleaning_step2.run(
            storage=self.storage.step(),
            input_key="text_path",
            output_key="raw_chunk"
        )

        self.knowledge_cleaning_step3.run(
            storage=self.storage.step(),
            input_key="raw_chunk",
            output_key="cleaned_chunk"
        )
        self.knowledge_cleaning_step4.run(
            storage=self.storage.step(),
            input_key="cleaned_chunk",
            output_key="QA_pairs",
            output_meta_key="QA_metadata"
        )
        
if __name__ == "__main__":
    model = KBCleaningPDF_APIPipeline()
    model.forward()