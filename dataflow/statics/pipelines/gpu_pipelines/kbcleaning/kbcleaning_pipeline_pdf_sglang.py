from dataflow.operators.generate import (
    CorpusTextSplitter,
    FileOrURLToMarkdownConverter,
    KnowledgeCleaner,
    MultiHopQAGenerator,
)
from dataflow.utils.storage import FileStorage
from dataflow.serving import LocalModelLLMServing_vllm, LocalModelLLMServing_sglang

class KBCleaning_PDFSglang_GPUPipeline():
    def __init__(self):

        self.storage = FileStorage(
            first_entry_file_name="../../example_data/KBCleaningPipeline/kbc_placeholder.json",
            cache_path="./.cache/gpu",
            file_name_prefix="pdf_cleaning_step",
            cache_type="json",
        )

        self.knowledge_cleaning_step1 = FileOrURLToMarkdownConverter(
            intermediate_dir="../../example_data/KBCleaningPipeline/raw/",
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

        self.llm_serving = LocalModelLLMServing_sglang(
            hf_model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
            sgl_dp_size=1, # data parallel size
            sgl_tp_size=1, # tensor parallel size
            sgl_max_new_tokens=2048,
        )

        self.knowledge_cleaning_step3 = KnowledgeCleaner(
            llm_serving=self.llm_serving,
            lang="en"
        )

        self.knowledge_cleaning_step4 = MultiHopQAGenerator(
            llm_serving=self.llm_serving,
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
    model = KBCleaning_PDFSglang_GPUPipeline()
    model.forward(raw_file="../../example_data/KBCleaningPipeline/test.pdf")