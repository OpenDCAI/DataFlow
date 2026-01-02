import asyncio
from typing import List, Any, Dict
from concurrent.futures import ThreadPoolExecutor

from dataflow.logger import get_logger
from flashrag.config import Config
from flashrag.utils import get_retriever
from dataflow.core import LLMServingABC

import asyncio
from typing import List, Any, Dict, Optional
from concurrent.futures import ThreadPoolExecutor

from dataflow.core import LLMServingABC
from dataflow.logger import get_logger

from flashrag.config import Config
from flashrag.utils import get_retriever

class FlashRAGServing(LLMServingABC):
    def __init__(self,
                 retrieval_method: str = "e5",
                 retrieval_model_path: str = None,
                 index_path: str = None,
                 corpus_path: str = None,
                 faiss_gpu: bool = False,
                 gpu_id: str = "",

                 max_workers: int = 1,
                 topk: int = 2,
                 **kwargs
                 ):
        """
        Args:
            retrieval_method: 检索方法名称 (e.g. "e5", "bm25")
            retrieval_model_path: 检索模型路径
            index_path: 索引文件路径
            corpus_path: 语料库路径 (.jsonl)
            faiss_gpu: 是否使用 GPU 进行 FAISS 检索
            gpu_id: 指定使用的 GPU ID
            max_workers: 线程池最大工作线程数
            topk: 默认检索数量
            **kwargs: 其他可能传递给 FlashRAG Config 的参数
        """
        self.logger = get_logger()
        
        if not retrieval_model_path or not index_path or not corpus_path:
             self.logger.warning("FlashRAGServing initialized without critical paths (model, index, or corpus). "
                                 "Ensure they are passed before loading.")

        self.rag_config_dict = {
            "retrieval_method": retrieval_method,
            "retrieval_model_path": retrieval_model_path,
            "index_path": index_path,
            "corpus_path": corpus_path,
            "faiss_gpu": faiss_gpu,
            "gpu_id": gpu_id,
        }
        
        # 将额外的 kwargs 也合并进去，以支持更高级的 FlashRAG 配置
        self.rag_config_dict.update(kwargs)

        self.topk = topk
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.retriever = None

    def load_model(self, model_name_or_path: str = None, **kwargs: Any):
        """
        加载 Retriever。
        注意：这里不再读取文件，而是使用 init 中保存的字典配置。
        model_name_or_path 参数保留是为了兼容基类接口，实际上不使用。
        """
        self.logger.info("Initializing FlashRAG Config from parameters...")
        
        try:
            # 使用 config_dict 初始化 FlashRAG Config
            # FlashRAG 的 Config 类通常接受 config_file_path 和 config_dict
            config = Config(config_dict=self.rag_config_dict)
            
            self.logger.info(f"Loading Retriever with method: {self.rag_config_dict.get('retrieval_method')}...")
            self.retriever = get_retriever(config)
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
                # 提取 content，保持 List[str] 结构以便后续 explode
                docs_content_list = [doc.get('contents', '') for doc in docs]
                formatted_outputs.append(docs_content_list)

            # 返回 List[List[str]]
            return formatted_outputs

        except Exception as e:
            self.logger.error(f"Error during retrieval: {e}")
            # 出错时返回空列表的列表，保持维度一致
            return [[] for _ in user_inputs]