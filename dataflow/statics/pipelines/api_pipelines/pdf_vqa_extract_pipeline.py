import argparse
from pathlib import Path

from dataflow.operators.knowledge_cleaning import FileOrURLToMarkdownConverterFlash, FileOrURLToMarkdownConverterAPI

from dataflow.serving import APILLMServing_request
from dataflow.utils.storage import FileStorage
from dataflow.operators.pdf2vqa import MinerU2LLMInputOperator, LLMOutputParser, QA_Merger, PDF_Merger, VQAFormatter
from dataflow.operators.core_text import ChunkedPromptedGenerator

from dataflow.pipeline import PipelineABC
from dataflow.prompts.pdf2vqa import QAExtractPrompt

from pypdf import PdfWriter
    
class PDF_VQA_extract_optimized_pipeline(PipelineABC):
    def __init__(self, cache_base=None):
        super().__init__()

        if cache_base is None:
            first_entry_file_name = "./example_data/PDF2VQAPipeline/vqa_extract_test.jsonl"
            storage_cache_path = "./cache"
            vqa_output_dir = "./cache"
            mineru_intermediate_dir = "../example_data/PDF2VQAPipeline/flash/"
            parser_intermediate_dir = "intermediate"
            qa_output_file = "./.cache/data/qa.json"
        else:
            cache_path = Path(cache_base).expanduser()
            if not cache_path.is_absolute():
                cache_path = Path.cwd() / cache_path
            cache_path = cache_path.resolve()
            first_entry_file_name = str(cache_path / ".cache" / "pdf_list.jsonl")
            storage_cache_path = str(cache_path / ".cache" / "gpu")
            vqa_output_dir = str(cache_path / ".cache" / "gpu" / "pdf2model_vqa")
            mineru_intermediate_dir = str(cache_path / ".cache" / "gpu" / "pdf2model_vqa" / "flash")
            parser_intermediate_dir = str(cache_path / ".cache" / "gpu" / "pdf2model_vqa" / "intermediate")
            qa_output_file = str(cache_path / ".cache" / "data" / "qa.json")

        self.storage = FileStorage(
            first_entry_file_name=first_entry_file_name,
            cache_path=storage_cache_path,
            file_name_prefix="vqa",
            cache_type="jsonl",
        )
        
        self.llm_serving = APILLMServing_request(
            api_url="http://123.129.219.111:3000/v1/chat/completions",
            key_name_of_api_key="DF_API_KEY",
            model_name="gemini-2.5-pro",
            max_workers=100,
        )
        
        self.vqa_extract_prompt = QAExtractPrompt()
        
        self.pdf_merger = PDF_Merger(output_dir=vqa_output_dir)
        
        # self.mineru_executor = FileOrURLToMarkdownConverterAPI(intermediate_dir = "intermediate")

        # Faster backend by Flash-MinerU
        self.mineru_executor = FileOrURLToMarkdownConverterFlash(
            intermediate_dir=mineru_intermediate_dir,
            mineru_model_path="<your Model Path>/MinerU2.5-2509-1.2B",  # !!! place your local model path here !!!
            # https://huggingface.co/opendatalab/MinerU2.5-2509-1.2B.
            batch_size=4, # batchsize per vllm worker
            replicas=1,   # num of vllm workers
            num_gpus_per_replica=0.5, # for ray to schedule vllm workers to GPU, can be float, e.g. 0.5 means each worker uses half GPU, 1 means each worker uses whole GPU
            engine_gpu_util_rate_to_ray_cap=0.9 # actuall GPU utilization for each worker; acturall memory per worker= num_gpus_per_replica * engine_gpu_util_rate_to_ray_cap; this is to avoid OOM, you can set it to 0.9 or 0.8 to leave some buffer for other processes on
        )

        self.input_formatter = MinerU2LLMInputOperator()
        self.vqa_extractor = ChunkedPromptedGenerator(
            llm_serving=self.llm_serving,
            system_prompt = self.vqa_extract_prompt.build_prompt(),
            max_chunk_len=128000,
        )
        self.llm_output_parser = LLMOutputParser(output_dir=vqa_output_dir, intermediate_dir=parser_intermediate_dir)
        self.qa_merger = QA_Merger(output_dir=vqa_output_dir, strict_title_match=False)

        self.vqa_format_converter = VQAFormatter(output_json_file=qa_output_file,)

    def forward(self):
        self.pdf_merger.run(
            storage=self.storage.step(),
            input_pdf_list_key="input_pdf_paths",
            input_name_key="name",
            output_pdf_path_key="merged_pdf_path",
        )
        self.mineru_executor.run(
            storage=self.storage.step(),
            input_key="merged_pdf_path",
            output_key="vqa_markdown_path",
        )
        self.input_formatter.run(
            storage=self.storage.step(),
            input_markdown_path_key="vqa_markdown_path",
            output_converted_layout_key="converted_vqa_layout_path",
        )
        self.vqa_extractor.run(
            storage=self.storage.step(),
            input_path_key="converted_vqa_layout_path",
            output_path_key="extracted_llm_vqa_path",
        )
        self.llm_output_parser.run(
            storage=self.storage.step(),
            input_response_path_key="extracted_llm_vqa_path",
            input_converted_layout_path_key="converted_vqa_layout_path",
            input_name_key="name",
            output_qalist_path_key="extracted_vqa_path",
        )
        self.qa_merger.run(
            storage=self.storage.step(),
            input_qalist_path_key="extracted_vqa_path",
            input_name_key="name",
            output_merged_qalist_path_key="output_merged_vqalist_path",
            output_merged_md_path_key="output_merged_md_path",
            output_qa_item_key="vqa_pair",
        )
        self.vqa_format_converter.run(
            storage=self.storage.step(),
            input_qa_item_key="vqa_pair",
            output_messages_key="messages",
            output_images_key="images",
        )


def main():
    parser = argparse.ArgumentParser(description="PDF2Model VQA Step-2 pipeline")
    parser.add_argument("--cache", default=None, help="PDF2Model cache base. Omit for legacy standalone example paths.")
    args = parser.parse_args()

    pipeline = PDF_VQA_extract_optimized_pipeline(cache_base=args.cache)
    pipeline.compile()
    pipeline.forward()


if __name__ == "__main__":
    main()
