from dataflow.operators.knowledge_cleaning import (
    KBCChunkGenerator,
    FileOrURLToMarkdownConverterAPI
)
from dataflow.utils.storage import FileStorage
class KBCleaning_CPUPipeline():
    def __init__(self):

        self.storage = FileStorage(
            first_entry_file_name="../example_data/KBCleaningPipeline/kbc_test_1.jsonl",
            cache_path="./.cache/cpu",
            file_name_prefix="url_cleaning_step",
            cache_type="json",
        )

        self.knowledge_cleaning_step1 = FileOrURLToMarkdownConverterAPI(
            intermediate_dir="../example_data/KBCleaningPipeline/raw/",
            mineru_backend="pipeline",
        )

        self.knowledge_cleaning_step2 = KBCChunkGenerator(
            split_method="token",
            chunk_size=512,
            tokenizer_name="Qwen/Qwen2.5-7B-Instruct",
        )

    def forward(self):
        self.knowledge_cleaning_step1.run(
            storage=self.storage.step(),
            # input_file=,
            # output_key=,
        )
        
        self.knowledge_cleaning_step2.run(
            storage=self.storage.step(),
            # input_file=,
            # output_key=,
        )

if __name__ == "__main__":
    model = KBCleaning_CPUPipeline()
    model.forward()

