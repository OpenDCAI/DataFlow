from dataflow.operators.knowledge_cleaning import (
    KBCChunkGenerator,
    FileOrURLToMarkdownConverterAPI,
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

        self.knowledge_cleaning_step1 = FileOrURLToMarkdownConverterAPI(
            intermediate_dir="../example_data/KBCleaningPipeline/raw/",
            lang="en",
            mineru_backend="vlm",  # vlm or pipeline
        )

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
            # input_key=,
            # output_key=,
        )
        
        self.knowledge_cleaning_step2.run(
            storage=self.storage.step(),
            # input_key=,
            # output_key=,
        )

        self.knowledge_cleaning_step3.run(
            storage=self.storage.step(),
            # input_key=,
            # output_key=,
        )
        self.knowledge_cleaning_step4.run(
            storage=self.storage.step(),
            # input_key=,
            # output_key=,
        )
        
if __name__ == "__main__":
    model = KBCleaningPDF_APIPipeline()
    model.forward()