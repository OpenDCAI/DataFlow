from dataflow.operators.generate.KnowledgeCleaning import (
    corpus_text_splitter,
    knowledge_extractor,
)
from dataflow.utils.storage import FileStorage
class KBCleaningPipeline():
    def __init__(self):

        self.storage = FileStorage(
            first_entry_file_name="../example_data/KBCleaningPipeline/kbc_placeholder.json",
            cache_path="./.cache/cpu",
            file_name_prefix="url_cleaning_step",
            cache_type="json",
        )

        self.knowledge_cleaning_step1 = knowledge_extractor(
            intermediate_dir="../example_data/KBCleaningPipeline/raw/",
            lang="en",
        )

        self.knowledge_cleaning_step2 = corpus_text_splitter(
            split_method="token",
            chunk_size=512,
            tokenizer_name="Qwen/Qwen2.5-7B-Instruct",
        )

    def forward(self, url:str=None, raw_file:str=None):
        extracted=self.knowledge_cleaning_step1.run(
            storage=self.storage,
            raw_file=raw_file,
            url=url,
        )
        
        self.knowledge_cleaning_step2.run(
            storage=self.storage.step(),
            input_file=extracted,
            output_key="raw_content",
        )

if __name__ == "__main__":
    model = KBCleaningPipeline()
    model.forward(url="https://trafilatura.readthedocs.io/en/latest/quickstart.html")

