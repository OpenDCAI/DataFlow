import os
import asyncio
from typing import List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor

from dataflow.core import LLMServingABC
from dataflow.logger import get_logger

from flashrag.config import Config
from flashrag.utils import get_retriever

class FlashRAGServing(LLMServingABC):
    def __init__(self,
                 config_path: str = "retriever_config.yaml",
                 llm_model_name: str = "gpt-4o",
                 max_workers: int = 1
                 ):
        
        self.logger = get_logger()
        self.config_path = config_path
        
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.retriever = None

    @classmethod
    async def create(cls, *args, **kwargs) -> "FlashRAGServing":
        instance = cls(*args, **kwargs)
        if instance.retriever is None:
            instance.logger.info(f"Initializing FlashRAG from {instance.config_path}...")
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, instance._init_retriever_sync)
            instance.logger.info("FlashRAG Retriever initialized.")
        return instance

    def _init_retriever_sync(self):
        config_dict = Config(self.config_path)
        self.retriever = get_retriever(config_dict)

    def start_serving(self):
        self.logger.info("FlashRAG Serving started and ready to handle requests.")

    async def cleanup(self):
        self.logger.info("Cleaning up FlashRAG resources...")
        self.executor.shutdown(wait=True)

    async def generate_from_input(self, user_inputs: List[str], system_prompt: str = "") -> List[str]:
        results = []
        for query in user_inputs:
            docs = await self.retrieve_for_api(query, topk=5)
            combined_text = "\n\n".join([d['contents'] for d in docs])
            results.append(combined_text)
        return results

    async def retrieve_for_api(self, query: str, topk: int) -> List[Dict[str, Any]]:
        loop = asyncio.get_event_loop()
        
        def run_search():
            return self.retriever.batch_search([query], topk, return_score=True)

        try:
            results, scores = await loop.run_in_executor(self.executor, run_search)
            
            docs_list = results[0]
            scores_list = scores[0]

            formatted_results = []
            for doc, score in zip(docs_list, scores_list):
                formatted_results.append({
                    "doc_id": doc.get("id", "N/A"),
                    "contents": doc.get("contents", ""),
                    "score": round(score, 4)
                })
            return formatted_results
        except Exception as e:
            self.logger.error(f"Error during retrieval: {e}")
            return []

    def load_model(self, model_name_or_path: str, **kwargs: Any):
        self.logger.info(f"load_model called with {model_name_or_path}, but FlashRAG uses config file.")
        pass