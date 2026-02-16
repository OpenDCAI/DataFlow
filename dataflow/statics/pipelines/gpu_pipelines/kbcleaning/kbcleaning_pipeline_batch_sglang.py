from dataflow.operators.knowledge_cleaning import (
    KBCChunkGeneratorBatch,
    FileOrURLToMarkdownConverterFlash,
    KBCTextCleanerBatch,
    KBCMultiHopQAGeneratorBatch,
)
from dataflow.utils.storage import FileStorage
from dataflow.serving import LocalModelLLMServing_vllm, LocalModelLLMServing_sglang


class KBCleaning_batchSglang_GPUPipeline():
    def __init__(self):

        self.storage = FileStorage(
            first_entry_file_name="../../example_data/KBCleaningPipeline/kbc_test.jsonl",
            cache_path="./.cache/gpu",
            file_name_prefix="batch_cleaning_step",
            cache_type="json",
        )

        self.knowledge_cleaning_step1 = FileOrURLToMarkdownConverterFlash(
            intermediate_dir="../example_data/KBCleaningPipeline/flash/",
            mineru_model_path="<your Model Path>/MinerU2.5-2509-1.2B",  # !!! place your local model path here !!!
            # https://huggingface.co/opendatalab/MinerU2.5-2509-1.2B.
            batch_size=4, # batchsize per vllm worker
            replicas=1,   # num of vllm workers
            num_gpus_per_replica=0.5, # for ray to schedule vllm workers to GPU, can be float, e.g. 0.5 means each worker uses half GPU, 1 means each worker uses whole GPU
            engine_gpu_util_rate_to_ray_cap=0.9 # actuall GPU utilization for each worker; acturall memory per worker= num_gpus_per_replica * engine_gpu_util_rate_to_ray_cap; this is to avoid OOM, you can set it to 0.9 or 0.8 to leave some buffer for other processes on
        )
        self.knowledge_cleaning_step2 = KBCChunkGeneratorBatch(
            split_method="token",
            chunk_size=512,
            tokenizer_name="Qwen/Qwen2.5-7B-Instruct",
        )

    def forward(self):
        self.knowledge_cleaning_step1.run(
            storage=self.storage.step(),
        )

        self.knowledge_cleaning_step2.run(
            storage=self.storage.step(),
        )

        self.llm_serving = LocalModelLLMServing_sglang(
            hf_model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
            sgl_dp_size=1, # data parallel size
            sgl_tp_size=1, # tensor parallel size
            sgl_max_new_tokens=2048,
        )

        self.knowledge_cleaning_step3 = KBCTextCleanerBatch(
            llm_serving=self.llm_serving,
            lang="en"
        )

        self.knowledge_cleaning_step4 = KBCMultiHopQAGeneratorBatch(
            llm_serving=self.llm_serving,
            lang="en"
        )

        self.knowledge_cleaning_step3.run(
            storage=self.storage.step(),
        )
        self.knowledge_cleaning_step4.run(
            storage=self.storage.step(),
        )


if __name__ == "__main__":
    model = KBCleaning_batchSglang_GPUPipeline()
    model.forward()
