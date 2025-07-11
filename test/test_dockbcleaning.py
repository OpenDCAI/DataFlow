import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from dataflow.operators.generate import (
    CorpusTextSplitter,
    KnowledgeExtractor,
    KnowledgeCleaner,
    MultiHopQAGenerator,
)
from dataflow.utils.storage import FileStorage
from dataflow.serving import LocalModelLLMServing

class KBCleaningPipeline():
    def __init__(self):

        self.storage = FileStorage(
            first_entry_file_name="dataflow/example/KBCleaningPipeline/kbc_placeholder.json",
            cache_path="./.cache",
            file_name_prefix="doc_cleaning_step",
            cache_type="json",
        )

        # api_llm_serving = APILLMServing_request(
        #         api_url="http://123.129.219.111:3000/v1/chat/completions",
        #         model_name="gpt-4o",
        #         max_workers=100
        # )

        local_llm_serving = LocalModelLLMServing(
            model_name_or_path="/data0/models/Qwen2.5-7B-Instruct",
            max_tokens=512,
            tensor_parallel_size=4,
            model_source="local",
            gpu_memory_utilization=0.6,
            repetition_penalty=1.2
        )

        self.knowledge_cleaning_step1 = KnowledgeExtractor(
            intermediate_dir="dataflow/example/KBCleaningPipeline/raw/"
        )

        self.knowledge_cleaning_step2 = CorpusTextSplitter(
            split_method="token",
            chunk_size=1024,
            tokenizer_name="/data0/hzy/RARE/model_base/Qwen2.5-3B-Instruct",
        )

        self.knowledge_cleaning_step3 = KnowledgeCleaner(
            llm_serving=local_llm_serving,
            lang="ch"
        )

        self.knowledge_cleaning_step4 = MultiHopQAGenerator(
            llm_serving=local_llm_serving,
            lang="ch"
        )

    def forward(self, url:str=None, raw_file:str=None):
        extracted=self.knowledge_cleaning_step1.run(
            storage=self.storage,
            raw_file=raw_file,
            url=url,
            lang="ch"
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
    model.forward(raw_file="/data0/hzy/DataFlow-Preview/dataflow/example/KBCleaningPipeline/test.doc")

