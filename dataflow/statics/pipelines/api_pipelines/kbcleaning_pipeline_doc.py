from dataflow.operators.generate.KnowledgeCleaning import (
    corpus_text_splitter,
    knowledge_extractor,
    knowledge_cleaner,
    multihop_qa_generator,
)
from dataflow.utils.storage import FileStorage
from dataflow.serving import APILLMServing_request

class KBCleaningPipeline():
    def __init__(self):

        self.storage = FileStorage(
            first_entry_file_name="../example_data/KBCleaningPipeline/kbc_placeholder.json",
            cache_path="./.cache/api",
            file_name_prefix="doc_cleaning_step",
            cache_type="json",
        )

        api_llm_serving = APILLMServing_request(
                api_url="https://api.openai.com/v1/chat/completions",
                model_name="gpt-4o",
                max_workers=100
        )

        self.knowledge_cleaning_step1 = knowledge_extractor(
            intermediate_dir="../example_data/KBCleaningPipeline/raw/",
            lang="ch"
        )

        self.knowledge_cleaning_step2 = corpus_text_splitter(
            split_method="token",
            chunk_size=512,
            tokenizer_name="Qwen/Qwen2.5-7B-Instruct",
        )

        self.knowledge_cleaning_step3 = knowledge_cleaner(
            llm_serving=api_llm_serving,
            lang="ch"
        )

        self.knowledge_cleaning_step4 = multihop_qa_generator(
            llm_serving=api_llm_serving,
            lang="ch"
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
    model = KBCleaningPipeline()
    model.forward(raw_file="../example_data/KBCleaningPipeline/test.doc")

