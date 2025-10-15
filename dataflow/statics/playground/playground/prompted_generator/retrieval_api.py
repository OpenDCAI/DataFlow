from dataflow.logger import get_logger
from dataflow.operators.core_text import RetrievalGenerator
from dataflow.utils.storage import FileStorage
from dataflow.serving import LightRAGServing
import asyncio

class RAG():
    def __init__(self, docs):
        self.storage = FileStorage(
            first_entry_file_name="/home/achewwa/Projects/DataFlow/example_data/GeneralTextPipeline/paperquestion.jsonl",
            cache_path="./cache",
            file_name_prefix="paper_question",
            cache_type="jsonl",
        )
        
        self.model_cache_dir = './dataflow_cache'
        self.llm_serving = asyncio.run(LightRAGServing.create(api_url="http://123.129.219.111:3000/v1", document_list=docs))
        self.retrieval_generator = RetrievalGenerator(
            llm_serving = self.llm_serving,
            system_prompt="Answer the question based on the text."
        )
    
    def forward(self):
        self.retrieval_generator.run(
            storage=self.storage.step()
        )

if __name__ == "__main__":
    doc = ["/home/achewwa/Projects/DataFlow/example_data/GeneralTextPipeline/test1.txt"]
    model = RAG(doc)
    model.forward()