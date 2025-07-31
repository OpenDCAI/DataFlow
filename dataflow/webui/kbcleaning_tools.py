from dataflow.operators.generate import (
    CorpusTextSplitterBatch,
    FileOrURLToMarkdownConverterBatch,
    KnowledgeCleanerBatch,
    MultiHopQAGeneratorBatch,
)
from dataflow.utils.storage import FileStorage
from dataflow.serving import LocalModelLLMServing_vllm, APILLMServing_request
from dataflow.webui.kb_sglang_server import SGLangServer
import pandas as pd
import os
from typing import Literal
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KBCleaning_Tools():
    def __init__(self):
        pass

    def multi_pdf_or_url_to_json(self, pdf_or_url_list: list[str], cache_path: str = "./.cache", file_name: str = "pdf_or_urls.jsonl"):
        output_list = []
        for pdf_or_url in pdf_or_url_list:
            output_list.append({"raw_content": pdf_or_url})
        os.makedirs(cache_path, exist_ok=True)
        pd.DataFrame(output_list).to_json(os.path.join(cache_path, file_name), orient="records", lines=True, force_ascii=False)
        file_path = os.path.join(cache_path, file_name)
        return file_path

    def _get_storage(self, file_name: str):
        storage = FileStorage(
            first_entry_file_name=file_name,
            cache_path="./.cache/gpu",
            file_name_prefix="batch_cleaning_step",
            cache_type="json",
        )
        return storage
    
    def _get_llm_serving(self,
                        serving_type: Literal["vllm", "sglang", "api"],
                        model_name_or_path: str,
                        max_tokens: int,
                        tensor_parallel_size: int,
                        api_url: str,
                        api_key: str,
                        max_workers: int,
                        temperature: float
                        ):
        if serving_type == "vllm":
            logger.info(f"Using VLLM to serve the model {model_name_or_path}, pipeline parallel is unavailable")
            llm_serving = LocalModelLLMServing_vllm(
                hf_model_name_or_path=model_name_or_path,
                vllm_max_tokens=max_tokens,
                vllm_tensor_parallel_size=tensor_parallel_size,
                vllm_gpu_memory_utilization=0.6,
                vllm_repetition_penalty=1.2
            )
        elif serving_type == "sglang":
            logger.info(f"Using SGLang to serve the model {model_name_or_path}, max_tokens refer to sgl_max_new_tokens")
            llm_serving = SGLangServer(
                model_path=model_name_or_path,
                tp=tensor_parallel_size,
                max_total_tokens=max_tokens,
                max_workers=max_workers,
                temperature=temperature,
            )
        elif serving_type == "api":
            logger.info(f"Using API to serve the model {model_name_or_path}, max_tokens refer to max_new_tokens")
            llm_serving = APILLMServing_request(
                api_url=api_url,
                key_name_of_api_key="DF_API_KEY",
                model_name=model_name_or_path,
                max_workers=max_workers
            )
        return llm_serving

    def run_pipeline(self,
                     pdf_or_url_list: list[str],
                     serving_type: Literal["vllm", "sglang", "api"],
                     model_name: str,
                     max_tokens: int,
                     tensor_parallel_size: int,
                     api_url: str,
                     api_key: str,
                     max_workers: int,
                     temperature: float,
                     ):
        file_path = self.multi_pdf_or_url_to_json(pdf_or_url_list)
        storage = self._get_storage(file_path)
        knowledge_cleaning_step1 = FileOrURLToMarkdownConverterBatch(
            intermediate_dir="./.cache/raw",
            lang="en",
            mineru_backend="vlm-sglang-engine",
        )
        if serving_type == "api":
            tokenizer_name = "Qwen/Qwen2.5-7B-Instruct"
        else:
            tokenizer_name = model_name
        knowledge_cleaning_step2 = CorpusTextSplitterBatch(
            split_method="token",
            chunk_size=512,
            tokenizer_name=tokenizer_name,
        )
        knowledge_cleaning_step1.run(storage=storage.step())
        knowledge_cleaning_step2.run(storage=storage.step())
        llm_serving = self._get_llm_serving(
            serving_type=serving_type,
            model_name_or_path=model_name,
            max_tokens=max_tokens,
            tensor_parallel_size=tensor_parallel_size,
            api_url=api_url,
            api_key=api_key,
            max_workers=max_workers,
            temperature=temperature,
        )
        knowledge_cleaning_step3 = KnowledgeCleanerBatch(
            llm_serving=llm_serving,
            lang="en",
        )
        knowledge_cleaning_step3.run(storage=storage.step())
        knowledge_cleaning_step4 = MultiHopQAGeneratorBatch(
            llm_serving=llm_serving,
            lang="en",
        )
        knowledge_cleaning_step4.run(storage=storage.step())
        
        return
        



