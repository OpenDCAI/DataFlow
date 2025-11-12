from dataflow.logger import get_logger
from dataflow.operators.core_text import RetrievalGenerator
from dataflow.utils.storage import FileStorage
from dataflow.serving import LightRAGServing
import asyncio

class RAG():
    def __init__(self, docs):
        self.storage = FileStorage(
            first_entry_file_name="./paperquestion.jsonl",
            cache_path="./cache",
            file_name_prefix="paper_question",
            cache_type="jsonl",
        )
        
        self.model_cache_dir = './dataflow_cache'
        self.llm_serving = None
        self.retrieval_generator = None
        self.docs = docs

    async def initialize(self):
        self.llm_serving = await LightRAGServing.create(api_url="https://api.openai.com/v1", document_list=self.docs)
        self.retrieval_generator = RetrievalGenerator(
            llm_serving = self.llm_serving,
            system_prompt="Answer the question based on the text."
        )
    
    async def forward(self):
        await self.retrieval_generator.run(
            storage=self.storage.step()
        )

async def main():
    doc = ["./text1.txt"]
    model = RAG(doc)
    await model.initialize()
    await model.forward()

if __name__ == "__main__":
    asyncio.run(main())