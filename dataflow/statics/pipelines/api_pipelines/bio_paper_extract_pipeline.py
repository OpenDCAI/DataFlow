import os
import pandas as pd

from dataflow.utils.storage import FileStorage
from dataflow.serving import PaperDownloaderServing
from dataflow.serving.api_llm_serving_request import APILLMServing_request
from dataflow.operators.bio_paper_extract import (
    PaperDownloaderGenerator,
    PaperParsingGenerator,
    PaperInfoExtractGenerator,
)
from dataflow.prompts.bio_paper_extract import (
    BioPaperInfoExtractPrompt,
    BioPaperInfoExtractPrompt5,
    BioPaperInfoExtractPrompt6,
    BioPaperInfoExtractPrompt7,
    BioPaperInfoExtractPrompt8,
    BioPaperInfoExtractPrompt10,
)


class BioPaperExtract_APIPipeline:

    def __init__(self):
        self.storage = FileStorage(
            first_entry_file_name="../example_data/bio_paper_extract_cache/bio_paper_extract_step1.jsonl",
            cache_path="./bio_paper_extract_cache",
            file_name_prefix="bio_paper_extract2",
            cache_type="jsonl",
        )

        self.llm_serving = APILLMServing_request(
            api_url="https://api.openai.com/v1/chat/completions",
            model_name="gemini-2.5-pro",
            max_workers=200,
        )

        self.paper_serving = PaperDownloaderServing(
            unpaywall_email="your@email.com",
            entrez_email="your@email.com",
            entrez_api_key="your_api_key",
        )

        self.downloader_op = PaperDownloaderGenerator(
            paper_serving=self.paper_serving,
        )

        self.parser_op = PaperParsingGenerator(
            host = "your_uniparser_server_host", # Uniparser server host
            max_workers=5,
        )

        self.info_extract_op = PaperInfoExtractGenerator(
            llm_serving=self.llm_serving,
            prompt_template=BioPaperInfoExtractPrompt6(),
        )

    def forward(self):
        # Step 1: Download papers
        self.downloader_op.run(
            storage=self.storage.step(),
            input_key="id",
            input_mode_key="input_mode",
            output_key="download_status",
            output_pdf_path="pdf_path",
            output_download_dir="./downloaded_papers",
        )

        # Step 2: Parse PDFs to markdown
        # Reads pdf_path from dataframe and outputs md_path
        # Uses multi-threading for parallel processing
        self.parser_op.run(
            storage=self.storage.step(),
            input_pdf_path_key="pdf_path",
            output_md_path="md_path",
            output_dir="./test_input",
        )

        # Step 3: Extract structured info from markdowns using LLM
        # Reads md_path from dataframe and outputs info_json_path
        self.info_extract_op.run(
            storage=self.storage.step(),
            input_paper_id_key="id",
            input_markdown_path_key="md_path",
            output_dir="./test_output",
            output_json_path_key="info_json_path",
        )

if __name__ == "__main__":
    pipeline = BioPaperExtract_APIPipeline()
    pipeline.forward()