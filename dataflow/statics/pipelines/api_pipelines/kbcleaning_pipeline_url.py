from dataflow.operators.knowledge_cleaning import (
    KBCChunkGenerator,
    FileOrURLToMarkdownConverter,
    KBCTextCleaner,
    KBCMultiHopQAGenerator,
)
from dataflow.utils.storage import FileStorage
from dataflow.serving import APILLMServing_request

class KBCleaningURL_APIPipeline():
    def __init__(self, url:str=None, raw_file:str=None):

        self.storage = FileStorage(
            first_entry_file_name="../example_data/KBCleaningPipeline/kbc_placeholder.json",
            cache_path="./.cache/api",
            file_name_prefix="url_cleaning_step",
            cache_type="json",
        )

        self.llm_serving = APILLMServing_request(
                api_url="https://api.openai.com/v1/chat/completions",
                model_name="gpt-4o",
                max_workers=100
        )

        self.knowledge_cleaning_step1 = FileOrURLToMarkdownConverter(
            intermediate_dir="../example_data/KBCleaningPipeline/raw/",
            lang="en",
            mineru_backend="vlm-sglang-engine",
            url = url,
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

        self.knowledge_cleaning_step4 = KBCMultiHopQAGenerator(
            llm_serving=self.llm_serving,
            lang="en"
        )

    def forward(self):
        extracted=self.knowledge_cleaning_step1.run(
            storage=self.storage,
        )
        
        self.knowledge_cleaning_step2.run(
            storage=self.storage.step(),
            input_file=extracted,
            output_key="raw_content",
        )

        self.knowledge_cleaning_step3.run(
            storage=self.storage.step(),
            input_key= "raw_content",
            output_key="cleaned",
        )

        self.knowledge_cleaning_step4.run(
            storage=self.storage.step(),
            input_key="cleaned",
            output_key="MultiHop_QA"
        )

if __name__ == "__main__":
    model = KBCleaningURL_APIPipeline(url="https://trafilatura.readthedocs.io/en/latest/quickstart.html")
    model.forward()

