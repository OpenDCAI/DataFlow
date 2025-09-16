from dataflow.operators.knowledge_cleaning import (
    CorpusTextSplitterBatch,
    FileOrURLToMarkdownConverterBatch,
    KnowledgeCleanerBatch,
    MultiHopQAGeneratorBatch,
)
from dataflow.utils.storage import FileStorage
from dataflow.serving import LocalModelLLMServing_vllm, LocalModelLLMServing_sglang


class KBCleaning_batchSglang_GPUPipeline():
    def __init__(self):

        self.storage = FileStorage(
            first_entry_file_name="../example_data/KBCleaningPipeline/kbc_playground.jsonl",
            cache_path="./.cache/gpu",
            file_name_prefix="batch_cleaning_step",
            cache_type="json",
        )

        self.knowledge_cleaning_step1 = FileOrURLToMarkdownConverterBatch(
            intermediate_dir="../example_data/KBCleaningPipeline/raw/",
            lang="en",
            mineru_backend="vlm-sglang-engine",
        )

        self.knowledge_cleaning_step2 = CorpusTextSplitterBatch(
            split_method="token",
            chunk_size=512,
            tokenizer_name="Qwen/Qwen2.5-7B-Instruct",
        )

    def forward(self):
        self.knowledge_cleaning_step1.run(
            storage=self.storage.step(),
        )

        self.knowledge_cleaning_step2.run(
            storage=self.storage.step(),
        )

        self.llm_serving = LocalModelLLMServing_sglang(
            hf_model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
            sgl_dp_size=1, # data parallel size
            sgl_tp_size=1, # tensor parallel size
            sgl_max_new_tokens=2048,
        )

        self.knowledge_cleaning_step3 = KnowledgeCleanerBatch(
            llm_serving=self.llm_serving,
            lang="en"
        )

        self.knowledge_cleaning_step4 = MultiHopQAGeneratorBatch(
            llm_serving=self.llm_serving,
            lang="en"
        )

        self.knowledge_cleaning_step3.run(
            storage=self.storage.step(),
        )
        self.knowledge_cleaning_step4.run(
            storage=self.storage.step(),
        )


if __name__ == "__main__":
    model = KBCleaning_batchSglang_GPUPipeline()
    model.forward()
