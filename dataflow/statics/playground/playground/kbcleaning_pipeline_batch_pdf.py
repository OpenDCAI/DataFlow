from dataflow.operators.generate import (
    PDFExtractor,
)
from dataflow.utils.storage import FileStorage
from dataflow.serving import LocalModelLLMServing_vllm
import os
import pandas as pd

class PDFCleaningPipeline():
    def __init__(self):

        self.storage = FileStorage(
            first_entry_file_name="../example_data/KBCleaningPipeline/kbc_test_pdf.jsonl",
            #first_entry_file_name="../example_data/GeneralTextPipeline/translation.jsonl",
            cache_path="./.cache/gpu",
            file_name_prefix="pdf_cleaning_step",
            cache_type="jsonl",
        )

        self.knowledge_cleaning_step1 = PDFExtractor(
            intermediate_dir="../example_data/KBCleaningPipeline/raw/",
            lang="en",
        )

    def forward(self):
        extracted=self.knowledge_cleaning_step1.run(
            storage=self.storage.step(),
            input_key = "raw_content",
        )
        
        
if __name__ == "__main__":
    model = PDFCleaningPipeline()
    model.forward()