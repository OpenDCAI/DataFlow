import torch
from sentence_transformers import SentenceTransformer
from tqdm.autonotebook import tqdm
import numpy as np
import logging

# 假设 LLMServingABC 和 get_logger 已经定义好了
# from your_abc_module import LLMServingABC 
# from your_logger_module import get_logger

# 为了让代码可运行，我们先简单定义一下
class LLMServingABC:
    def start_serving(self): pass
    def cleanup(self): pass

def get_logger():
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(__name__)


class LocalEmbeddingServing(LLMServingABC):
    """
    Use a local sentence-transformer model to generate embeddings.
    """
    def __init__(self, 
                 model_name: str = 'all-MiniLM-L6-v2',
                 device: str = None
                 ):
        """
        Initializes the local embedding model.
        
        Args:
            model_name (str): The name of the sentence-transformer model to load.
            device (str): The device to run the model on ('cuda', 'cpu'). 
                          If None, it will auto-detect.
        """
        self.logger = get_logger()
        self.model_name = model_name
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.logger.info(f"Loading local embedding model '{self.model_name}' onto device '{self.device}'...")
        # 加载本地模型
        self.model = SentenceTransformer(self.model_name, device=self.device)
        self.logger.info("Model loaded successfully.")

    def start_serving(self) -> None:
        self.logger.info("LocalEmbeddingServing: No separate service to start. Model is loaded in memory.")
        return

    def generate_embedding_from_input(self, 
                                      texts: list[str], 
                                      batch_size: int = 32
                                      ) -> list[list[float]]:
        """
        Generates embeddings for a list of texts.
        
        Args:
            texts (list[str]): A list of strings to be embedded.
            batch_size (int): The batch size for encoding.
            
        Returns:
            list[list[float]]: A list of embeddings.
        """
        self.logger.info(f"Generating embeddings for {len(texts)} texts...")
        
        # 使用 sentence-transformer 的 encode 方法，它非常高效
        embeddings = self.model.encode(
            texts, 
            batch_size=batch_size, 
            show_progress_bar=True, # 显示一个类似 tqdm 的进度条
            convert_to_numpy=True   # 结果为 numpy array
        )
        
        # 将 numpy array 转换为 list[list[float]]
        return embeddings.tolist()

    def cleanup(self):
        self.logger.info(f"Cleaning up resources for {self.model_name}")
        # 释放模型占用的内存
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        pass

    # --- 其他原有接口的适配 ---
    # 如果你的系统需要这些接口，可以保留，否则可以删除

    def generate_from_input(self, user_inputs: list[str], system_prompt: str = "") -> list[str]:
        self.logger.warning("generate_from_input is not supported by LocalEmbeddingServing.")
        return [None] * len(user_inputs)

    def generate_from_conversations(self, conversations: list[list[dict]]) -> list[str]:
        self.logger.warning("generate_from_conversations is not supported by LocalEmbeddingServing.")
        return [None] * len(conversations)
