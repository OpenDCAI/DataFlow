from dataflow.operators.generate.KnowledgeCleaning import (
    corpus_text_splitter,
    knowledge_extractor,
    knowledge_cleaner,
    multihop_qa_generator,
)
from dataflow.utils.storage import FileStorage
from dataflow.serving import LocalModelLLMServing_vllm

class KBCleaningPipeline():
    def __init__(self):

        self.storage = FileStorage(
            first_entry_file_name="../example_data/KBCleaningPipeline/kbc_placeholder.json",
            cache_path="./.cache/gpu",
            file_name_prefix="pdf_cleaning_step",
            cache_type="json",
        )

        local_llm_serving = LocalModelLLMServing_vllm(
            hf_model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
            vllm_max_tokens=1024,
            vllm_tensor_parallel_size=4,
            vllm_gpu_memory_utilization=0.6,
            vllm_repetition_penalty=1.2
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

        self.knowledge_cleaning_step3 = knowledge_cleaner(
            llm_serving=local_llm_serving,
            lang="en"
        )

        self.knowledge_cleaning_step4 = multihop_qa_generator(
            llm_serving=local_llm_serving,
            lang="en"
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
    model.forward(raw_file="../example_data/KBCleaningPipeline/test.pdf")