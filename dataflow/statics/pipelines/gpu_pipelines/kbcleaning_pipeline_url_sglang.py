from dataflow.operators.generate import (
    CorpusTextSplitter,
    FileOrURLToMarkdownConverter,
    KnowledgeCleaner,
    MultiHopQAGenerator,
)
from dataflow.utils.storage import FileStorage
from dataflow.serving import LocalModelLLMServing_sglang

class KBCleaningPipeline():
    def __init__(self):

        self.storage = FileStorage(
            first_entry_file_name="../example_data/KBCleaningPipeline/kbc_placeholder.json",
            cache_path="./.cache/gpu",
            file_name_prefix="url_cleaning_step",
            cache_type="json",
        )

        self.knowledge_cleaning_step1 = FileOrURLToMarkdownConverter(
            intermediate_dir="../example_data/KBCleaningPipeline/raw/",
            lang="en",
            mineru_backend="vlm-sglang-engine",
        )

        self.knowledge_cleaning_step2 = CorpusTextSplitter(
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

        local_llm_serving = LocalModelLLMServing_sglang(
            hf_model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
            sgl_max_tokens=2048,
            sgl_tensor_parallel_size=4
        )

        self.knowledge_cleaning_step3 = KnowledgeCleaner(
            llm_serving=local_llm_serving,
            lang="en"
        )

        self.knowledge_cleaning_step4 = MultiHopQAGenerator(
            llm_serving=local_llm_serving,
            lang="en"
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
    model.forward(url="https://trafilatura.readthedocs.io/en/latest/quickstart.html")