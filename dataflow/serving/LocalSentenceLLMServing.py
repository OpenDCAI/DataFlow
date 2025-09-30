import os
import time
from typing import List
from dataflow.core import LLMServingABC
from dataflow import get_logger


class LocalEmbeddingServing(LLMServingABC):
    """
    Generate embeddings using local sentence-transformer models.
    Interface style aligns with APILLMServing_request and supports parallel computation through max_workers.
    """
    def __init__(self,
                 model_name: str = 'all-MiniLM-L6-v2',
                 device: str = None,
                 max_workers: int = 2,
                 max_retries: int = 3
                 ):
        """
        Initialize local embedding model.

        Args:
            model_name (str): Name of the sentence-transformer model to load.
            device (str): Preferred device type ('cpu' or 'cuda'). If None, will auto-detect.
            max_workers (int): Number of parallel workers.
                               - 1 (default): No parallelism, runs on single device.
                               - > 1: Enable parallel mode.
                                 - On CPU: Will enable multiprocessing mode (using all available cores).
                                 - On multi-GPU: Will use min(max_workers, local GPU count) GPUs for data parallelism.
            max_retries (int): Maximum number of retries when errors occur.
        """
        self.logger = get_logger()
        self.model_name = model_name
        self.max_workers = max_workers
        self.max_retries = max_retries

        self._model = None
        self._torch = None
        self._SentenceTransformer = None
        self._initialized = False
        
        self._device = device

    def _ensure_dependencies_available(self):
        if self._torch is not None and self._SentenceTransformer is not None:
            return
            
        try:
            import torch
            from sentence_transformers import SentenceTransformer
            self._torch = torch
            self._SentenceTransformer = SentenceTransformer
        except ImportError:
            raise ImportError(
                "The 'embedding' optional dependencies are required but not installed.\n"
                "Please run: pip install 'open-dataflow[vectorsql]'"
            )

    def _initialize_model(self):
        if self._initialized:
            return
            
        self._ensure_dependencies_available()
        
        self._execution_strategy = "single_device"
        self._target_devices = None

        device = self._device
        if device is None:
            device = 'cuda' if self._torch.cuda.is_available() else 'cpu'

        if self.max_workers > 1:
            if 'cpu' in device:
                cpu_count = os.cpu_count() or 1
                if cpu_count > 1:
                    self._execution_strategy = "cpu_parallel"
                    self.logger.info(f"CPU parallel mode enabled (max_workers > 1).")
                    self.logger.warning(f"Note: sentence-transformers will use all {cpu_count} available CPU cores, "
                                        "max_workers parameter only enables this mode, does not limit core count.")
                else:
                    self.logger.warning("max_workers > 1 but only 1 CPU core detected, falling back to single device mode.")

            elif 'cuda' in device:
                gpu_count = self._torch.cuda.device_count()
                if gpu_count > 1:
                    num_gpus_to_use = min(self.max_workers, gpu_count)
                    
                    if num_gpus_to_use > 1:
                        self._execution_strategy = "gpu_parallel"
                        self._target_devices = [f'cuda:{i}' for i in range(num_gpus_to_use)]
                        self.logger.info(f"Multi-GPU parallel mode enabled. Will use {num_gpus_to_use} GPUs (min({self.max_workers}, {gpu_count})).")
                        self.logger.info(f"Target devices: {self._target_devices}")
                    else:
                        self.logger.warning("max_workers > 1 but final GPU count for computation is 1, falling back to single device mode.")
                else:
                    self.logger.warning("max_workers > 1 but only 1 GPU detected, falling back to single device mode.")

        if self._execution_strategy == "gpu_parallel":
            self.primary_device = self._target_devices[0]
        else:
            self.primary_device = 'cuda:0' if 'cuda' in device and self._torch.cuda.is_available() else 'cpu'

        self.logger.info(f"Loading model '{self.model_name}' to primary device '{self.primary_device}'...")
        self._model = self._SentenceTransformer(self.model_name, device=self.primary_device)
        self.logger.info("Model loaded successfully.")
        
        self._initialized = True

    @property
    def model(self):
        if not self._initialized:
            self._initialize_model()
        return self._model

    def start_serving(self) -> None:
        self.logger.info("LocalEmbeddingServing: No need to start independent service, model is already in memory.")

    def generate_embedding_from_input(self,
                                      texts: List[str],
                                      batch_size: int = 32
                                      ) -> List[List[float]]:
        """
        Generate embeddings for a list of texts, including retry logic and parallel execution logic.
        """
        if not self._initialized:
            self._initialize_model()
            
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                self.logger.info(f"Starting to generate embeddings for {len(texts)} texts (attempt {attempt + 1}/{self.max_retries})...")

                if self._execution_strategy == "gpu_parallel":
                    pool = self.model.start_multi_process_pool(target_devices=self._target_devices)
                    embeddings = self.model.encode_multi_process(texts, pool=pool, batch_size=batch_size)
                    self.model.stop_multi_process_pool(pool)

                elif self._execution_strategy == "cpu_parallel":
                    pool = self.model.start_multi_process_pool()
                    embeddings = self.model.encode_multi_process(texts, pool=pool, batch_size=batch_size)
                    self.model.stop_multi_process_pool(pool)

                else: # Single device mode
                    embeddings = self.model.encode(texts, batch_size=batch_size, show_progress_bar=True)

                self.logger.info("Embedding generation successful.")
                return embeddings.tolist()

            except Exception as e:
                last_exception = e
                self.logger.error(f"Error occurred while generating embeddings (attempt {attempt + 1}): {e}")
                if attempt + 1 < self.max_retries:
                    wait_time = 2 ** attempt
                    self.logger.info(f"Will retry after {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    self.logger.error("All retries failed.")
                    raise last_exception

        return []

    def cleanup(self):
        if not self._initialized:
            self.logger.info("Model not initialized, nothing to clean up.")
            return
            
        self.logger.info(f"Cleaning up resources for model '{self.model_name}'...")
        if self._model is not None:
            del self._model
            self._model = None
            
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
            
        self._initialized = False
        self.logger.info("Cleanup completed.")

    def generate_from_input(self, user_inputs: List[str], system_prompt: str = "") -> List[str]:
        self.logger.warning("generate_from_input is not applicable for LocalEmbeddingServing.")
        return [None] * len(user_inputs)

    def generate_from_conversations(self, conversations: List[List[dict]]) -> List[str]:
        self.logger.warning("generate_from_conversations is not applicable for LocalEmbeddingServing.")
        return [None] * len(conversations)
