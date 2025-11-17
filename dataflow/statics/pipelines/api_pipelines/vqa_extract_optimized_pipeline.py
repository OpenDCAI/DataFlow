import os
import sys

from dataflow.serving import APILLMServing_request
from dataflow.utils.storage import FileStorage
from dataflow.operators.vqa import VQAExtractor

class VQA_extract_optimized_pipeline:
    def __init__(self):
        self.storage = FileStorage(
            first_entry_file_name="./dataflow/example/VQA/vqa_test.jsonl",
            cache_path="./cache",
            file_name_prefix="vqa",
            cache_type="jsonl",
        )
        
        self.llm_serving = APILLMServing_request(
            api_url="http://123.129.219.111:3000/v1/chat/completions",
            key_name_of_api_key="DF_API_KEY",
            model_name="gemini-2.5-pro",
            max_workers=100,
        )
        
        self.vqa_extractor = VQAExtractor(
            llm_serving=self.llm_serving,
            mineru_backend='vlm-vllm-engine',
            max_chunk_len=128000
        )
        
    def forward(self):
        # 单一算子：包含预处理、QA提取、后处理的所有功能
        self.vqa_extractor.run(
            storage=self.storage.step(),
            input_question_pdf_path_key="question_pdf_path",
            input_answer_pdf_path_key="answer_pdf_path",
            input_pdf_path_key="pdf_path",  # 支持 interleaved 模式
            input_subject_key="subject",
            output_dir_key="output_dir",
            output_jsonl_key="output_jsonl_path",
        )



if __name__ == "__main__":
    # jsonl中每一行包含question_pdf_path, answer_pdf_path, subject (math, physics, chemistry, ...), output_dir
    # 如果question和answer在同一份pdf中，请将question_pdf_path和answer_pdf_path设置为相同的路径，会自动切换为interleaved模式
    pipeline = VQA_extract_optimized_pipeline()
    pipeline.forward()