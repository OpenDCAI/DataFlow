import argparse
from pathlib import Path

from dataflow.operators.knowledge_cleaning import (
    KBCChunkGenerator,
    FileOrURLToMarkdownConverterFlash,
    KBCTextCleaner,
    QAExtractor
    # KBCMultiHopQAGenerator,
)
from dataflow.operators.core_text import Text2MultiHopQAGenerator
from dataflow.utils.storage import FileStorage
from dataflow.serving import LocalModelLLMServing_vllm


class KBCleaning_PDFvllm_GPUPipeline():
    def __init__(self, cache_base=None):
        if cache_base is None:
            first_entry_file_name = "../../example_data/KBCleaningPipeline/kbc_test.jsonl"
            storage_cache_path = "./.cache/gpu"
            mineru_intermediate_dir = "../example_data/KBCleaningPipeline/flash/"
            qa_output_file = "./.cache/data/qa.json"
        else:
            cache_path = Path(cache_base).expanduser()
            if not cache_path.is_absolute():
                cache_path = Path.cwd() / cache_path
            cache_path = cache_path.resolve()
            first_entry_file_name = str(cache_path / ".cache" / "pdf_list.jsonl")
            storage_cache_path = str(cache_path / ".cache" / "gpu")
            mineru_intermediate_dir = str(cache_path / ".cache" / "gpu" / "pdf2model_kbc" / "flash")
            qa_output_file = str(cache_path / ".cache" / "data" / "qa.json")

        self.storage = FileStorage(
            first_entry_file_name=first_entry_file_name,
            cache_path=storage_cache_path,
            file_name_prefix="knowledge_cleaning_step_vllm_engine",
            cache_type="json",
        )

        # Faster backend by Flash-MinerU
        self.knowledge_cleaning_step1 = FileOrURLToMarkdownConverterFlash(
            intermediate_dir=mineru_intermediate_dir,
            mineru_model_path="<your Model Path>/MinerU2.5-2509-1.2B",  # !!! place your local model path here !!!
            # https://huggingface.co/opendatalab/MinerU2.5-2509-1.2B.
            batch_size=4, # batchsize per vllm worker
            replicas=1,   # num of vllm workers
            num_gpus_per_replica=0.5, # for ray to schedule vllm workers to GPU, can be float, e.g. 0.5 means each worker uses half GPU, 1 means each worker uses whole GPU
            engine_gpu_util_rate_to_ray_cap=0.9 # actuall GPU utilization for each worker; acturall memory per worker= num_gpus_per_replica * engine_gpu_util_rate_to_ray_cap; this is to avoid OOM, you can set it to 0.9 or 0.8 to leave some buffer for other processes on
        )

        self.knowledge_cleaning_step2 = KBCChunkGenerator(
            split_method="token",
            chunk_size=512,
            tokenizer_name="Qwen/Qwen2.5-7B-Instruct",
        )

        self.extract_format_qa_step5 = QAExtractor(
            output_json_file=qa_output_file,
        )

    def forward(self):
        self.knowledge_cleaning_step1.run(
            storage=self.storage.step(),
            # input_key=
            # output_key=
        )
        
        self.knowledge_cleaning_step2.run(
            storage=self.storage.step(),
            # input_key=
            # output_key=
        )

        self.llm_serving = LocalModelLLMServing_vllm(
            hf_model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
            vllm_max_tokens=2048,
            vllm_tensor_parallel_size=4,
            vllm_gpu_memory_utilization=0.6,
            vllm_repetition_penalty=1.2
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

        self.knowledge_cleaning_step3.run(
            storage=self.storage.step(),
            # input_key=
            # output_key=
        )
        self.knowledge_cleaning_step4.run(
            storage=self.storage.step(),
            # input_key=
            # output_key=
        )

        self.extract_format_qa_step5.run(
            storage=self.storage.step(),
            input_qa_key="Qa_pairs",
            output_instruction_key="instruction",
            output_question_key="input",
            output_answer_key="output"            
        )


def main():
    parser = argparse.ArgumentParser(description="PDF2Model KBC Step-2 pipeline")
    parser.add_argument("--cache", default=None, help="PDF2Model cache base. Omit for legacy standalone example paths.")
    args = parser.parse_args()

    model = KBCleaning_PDFvllm_GPUPipeline(cache_base=args.cache)
    model.forward()


if __name__ == "__main__":
    main()
