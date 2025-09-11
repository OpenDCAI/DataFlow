from dataflow.operators.knowledge_cleaning import (
    KBCChunkGenerator,
    FileOrURLToMarkdownConverter,
)
from dataflow.utils.storage import FileStorage
class KBCleaning_CPUPipeline():
    def __init__(self, url:str=None, raw_file:str=None):

        self.storage = FileStorage(
            first_entry_file_name="../example_data/KBCleaningPipeline/kbc_placeholder.json",
            cache_path="./.cache/cpu",
            file_name_prefix="url_cleaning_step",
            cache_type="json",
        )

        self.knowledge_cleaning_step1 = FileOrURLToMarkdownConverter(
            intermediate_dir="../example_data/KBCleaningPipeline/raw/",
            lang="en",
            url = url,
        )

        self.knowledge_cleaning_step2 = KBCChunkGenerator(
            split_method="token",
            chunk_size=512,
            tokenizer_name="Qwen/Qwen2.5-7B-Instruct",
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

if __name__ == "__main__":
    model = KBCleaning_CPUPipeline(url="https://trafilatura.readthedocs.io/en/latest/quickstart.html")
    model.forward()

