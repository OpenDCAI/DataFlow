from dataflow.operators.knowledge_cleaning import FileOrURLToMarkdownConverterFlash

from dataflow.serving import APILLMServing_request
from dataflow.utils.storage import FileStorage
from dataflow.operators.pdf2vqa import MinerU2LLMInputOperator, LLMOutputParser, QA_Merger
from dataflow.operators.core_text import ChunkedPromptedGenerator

from dataflow.pipeline import PipelineABC
from dataflow.prompts.pdf2vqa import QAExtractPrompt

class PDF_VQA_extract_optimized_pipeline(PipelineABC):
    def __init__(self):
        super().__init__()
        self.storage = FileStorage(
            first_entry_file_name="./example_data/PDF2VQAPipeline/vqa_extract_test.jsonl",
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
        
        self.vqa_extract_prompt = QAExtractPrompt()
        
        self.mineru_executor = FileOrURLToMarkdownConverterFlash(intermediate_dir = "intermediate", mineru_backend="vlm-vllm-engine")
        self.input_formatter = MinerU2LLMInputOperator()
        self.vqa_extractor = ChunkedPromptedGenerator(
            llm_serving=self.llm_serving,
            system_prompt = self.vqa_extract_prompt.build_prompt(),
            max_chunk_len=128000,
        )
        self.llm_output_question_parser = LLMOutputParser(mode="question", output_dir="./cache", intermediate_dir="intermediate")
        self.llm_output_answer_parser = LLMOutputParser(mode="answer", output_dir="./cache", intermediate_dir="intermediate")
        self.qa_merger = QA_Merger(output_dir="./cache", strict_title_match=False)
    def forward(self):
        # 目前的处理逻辑是：MinerU处理问题-MinerU处理答案-格式化问题文本-格式化答案文本-问题文本输入LLM-答案文本输入LLM-解析问题输出-解析答案输出-合并问答对
        # 由于问答对可能来自同一份pdf，也有可能来自不同pdf，而dataflow目前不支持分支，因此这里只能将question和answer的pdf都进行一次处理，
        # 即使是同一份pdf也会被处理两次，最后再合并问答对。
        # 未来会再思考如何优化这个流程，避免重复处理同一份pdf，提升性能。
        
        self.mineru_executor.run(
            storage=self.storage.step(),
            input_key="question_pdf_path",
            output_key="question_markdown_path",
        )
        self.mineru_executor.run(
            storage=self.storage.step(),
            input_key="answer_pdf_path",
            output_key="answer_markdown_path",
        )
        self.input_formatter.run(
            storage=self.storage.step(),
            input_markdown_path_key="question_markdown_path",
            output_converted_layout_key="converted_question_layout_path",
        )
        self.input_formatter.run(
            storage=self.storage.step(),
            input_markdown_path_key="answer_markdown_path",
            output_converted_layout_key="converted_answer_layout_path",
        )
        self.vqa_extractor.run(
            storage=self.storage.step(),
            input_path_key="converted_question_layout_path",
            output_path_key="vqa_extracted_questions_path",
        )
        self.vqa_extractor.run(
            storage=self.storage.step(),
            input_path_key="converted_answer_layout_path",
            output_path_key="vqa_extracted_answers_path",
        )
        self.llm_output_question_parser.run(
            storage=self.storage.step(),
            input_response_path_key="vqa_extracted_questions_path",
            input_converted_layout_path_key="converted_question_layout_path",
            input_name_key="name",
            output_qalist_path_key="extracted_questions_path",
        )
        self.llm_output_answer_parser.run(
            storage=self.storage.step(),
            input_response_path_key="vqa_extracted_answers_path",
            input_converted_layout_path_key="converted_answer_layout_path",
            input_name_key="name",
            output_qalist_path_key="extracted_answers_path",
        )
        self.qa_merger.run(
            storage=self.storage.step(),
            input_question_qalist_path_key="extracted_questions_path",
            input_answer_qalist_path_key="extracted_answers_path",
            input_name_key="name",
            output_merged_qalist_path_key="output_merged_qalist_path",
            output_merged_md_path_key="output_merged_md_path",
            output_qa_item_key="qa_pair",
        )



if __name__ == "__main__":
    # jsonl中每一行包含question_pdf_path, answer_pdf_path, name (math1, math2, physics1, chemistry1, ...)
    # 如果question和answer在同一份pdf中，请将question_pdf_path和answer_pdf_path设置为相同的路径，会自动切换为interleaved模式
    pipeline = PDF_VQA_extract_optimized_pipeline()
    pipeline.compile()
    pipeline.forward()