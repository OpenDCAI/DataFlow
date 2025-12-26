import asyncio
from typing import List, Any, Dict
from concurrent.futures import ThreadPoolExecutor

from dataflow.logger import get_logger
from flashrag.config import Config
from flashrag.utils import get_retriever
from dataflow.core import LLMServingABC

class FlashRAGServing(LLMServingABC):
    def __init__(self,
                 config_path: str = "retriever_config.yaml",
                 max_workers: int = 1,
                 topk: int = 2,
                 **kwargs
                 ):
        self.logger = get_logger()
        self.default_config_path = config_path
        self.topk = topk
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.retriever = None

    def load_model(self, model_name_or_path: str = None, **kwargs: Any):
        target_config_path = model_name_or_path if model_name_or_path else self.default_config_path
        self.logger.info(f"Loading FlashRAG retriever from: {target_config_path}")
        
        try:
            config_dict = Config(target_config_path)
            self.retriever = get_retriever(config_dict)
            self.logger.info("FlashRAG Retriever loaded successfully.")
        except Exception as e:
            self.logger.error(f"Failed to load FlashRAG retriever: {e}")
            raise e

    def start_serving(self):
        if self.retriever is None:
            self.logger.warning("FlashRAGServing started but retriever is NOT loaded. Call load_model() first.")
        else:
            self.logger.info("FlashRAGServing is ready to serve.")

    async def cleanup(self):
        self.logger.info("Cleaning up FlashRAGServing resources...")
        self.executor.shutdown(wait=True)

    async def generate_from_input(self, user_inputs: List[str], system_prompt: str = "") -> List[List[str]]:
        # 1. 懒加载检查
        if self.retriever is None:
            self.logger.warning("Retriever not loaded explicitly. Triggering lazy load...")
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(self.executor, lambda: self.load_model())


        topk = self.topk
        loop = asyncio.get_running_loop()
        def _run_sync_search():
            return self.retriever.batch_search(user_inputs, topk, return_score=True)

        try:
            results, scores = await loop.run_in_executor(self.executor, _run_sync_search)
        
            # results 结构: [ [doc1_obj, doc2_obj...], [doc1_obj, doc2_obj...] ]
            formatted_outputs = []
            
            for docs in results:
                docs_content_list = [doc.get('contents', '') for doc in docs]
                formatted_outputs.append(docs_content_list)

            # 返回 List[List[str]]
            return formatted_outputs

        except Exception as e:
            self.logger.error(f"Error during retrieval: {e}")
            return [[] for _ in user_inputs]