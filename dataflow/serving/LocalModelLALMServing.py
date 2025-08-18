import os
import torch
from dataflow import get_logger
from huggingface_hub import snapshot_download
from dataflow.core import LLMServingABC
from transformers import AutoProcessor
from typing import Optional, Union, List, Dict, Any

class LocalModelLALMServing_vllm(LLMServingABC):
    '''
    A class for generating text using vllm, with model from huggingface or local directory
    '''
    def __init__(self, 
                 hf_model_name_or_path: str = None,
                 hf_cache_dir: str = None,
                 hf_local_dir: str = None,
                 vllm_tensor_parallel_size: int = 1,
                 vllm_temperature: float = 0.7,
                 vllm_top_p: float = 0.9,
                 vllm_max_tokens: int = 1024,
                 vllm_top_k: int = 40,
                 vllm_repetition_penalty: float = 1.0,
                 vllm_seed: int = 42,
                 vllm_max_model_len: int = None,
                 vllm_gpu_memory_utilization: float=0.9,
                 ):

        self.load_model(
            hf_model_name_or_path=hf_model_name_or_path,
            hf_cache_dir=hf_cache_dir,
            hf_local_dir=hf_local_dir,
            vllm_tensor_parallel_size=vllm_tensor_parallel_size,
            vllm_temperature=vllm_temperature, 
            vllm_top_p=vllm_top_p,
            vllm_max_tokens=vllm_max_tokens,
            vllm_top_k=vllm_top_k,
            vllm_repetition_penalty=vllm_repetition_penalty,
            vllm_seed=vllm_seed,
            vllm_max_model_len=vllm_max_model_len,
            vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
        )

    def load_model(self, 
                 hf_model_name_or_path: str = None,
                 hf_cache_dir: str = None,
                 hf_local_dir: str = None,
                 vllm_tensor_parallel_size: int = 1,
                 vllm_temperature: float = 0.7,
                 vllm_top_p: float = 0.9,
                 vllm_max_tokens: int = 1024,
                 vllm_top_k: int = 40,
                 vllm_repetition_penalty: float = 1.0,
                 vllm_seed: int = 42,
                 vllm_max_model_len: int = None,
                 vllm_gpu_memory_utilization: float=0.9,
                 ):
        self.logger = get_logger()
        if hf_model_name_or_path is None:
            raise ValueError("hf_model_name_or_path is required") 
        elif os.path.exists(hf_model_name_or_path):
            self.logger.info(f"Using local model path: {hf_model_name_or_path}")
            self.real_model_path = hf_model_name_or_path
        else:
            self.logger.info(f"Downloading model from HuggingFace: {hf_model_name_or_path}")
            self.real_model_path = snapshot_download(
                repo_id=hf_model_name_or_path,
                cache_dir=hf_cache_dir,
                local_dir=hf_local_dir,
            )
        # get the model name from the real_model_path
        self.model_name = os.path.basename(self.real_model_path)
        self.processor = AutoProcessor.from_pretrained(self.real_model_path, cache_dir=hf_cache_dir)
        print(f"Model name: {self.model_name}")


        # Import vLLM and set up the environment for multiprocessing
        # vLLM requires the multiprocessing method to be set to spawn
        try:
            from vllm import LLM,SamplingParams
        except:
            raise ImportError("please install vllm first like 'pip install open-dataflow[vllm]'")
        # Set the environment variable for vllm to use spawn method for multiprocessing
        # See https://docs.vllm.ai/en/v0.7.1/design/multiprocessing.html 
        os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = "spawn"
        
        self.sampling_params = SamplingParams(
            temperature=vllm_temperature,
            top_p=vllm_top_p,
            max_tokens=vllm_max_tokens,
            top_k=vllm_top_k,
            repetition_penalty=vllm_repetition_penalty,
            seed=vllm_seed
        )
        
        self.llm = LLM(
            model=self.real_model_path,
            tensor_parallel_size=vllm_tensor_parallel_size,
            max_model_len=vllm_max_model_len,
            gpu_memory_utilization=vllm_gpu_memory_utilization,
        )
        self.logger.success(f"Model loaded from {self.real_model_path} by vLLM backend")

    def generate_from_input(self,    
                            user_inputs: list[str], 
                            audio_inputs: list,
                        ) -> list[str]:   


        full_prompts = []
        for user_input, audio_input in zip(user_inputs, audio_inputs):
            full_prompts.append({
                'prompt': user_input,
                'multi_modal_data': {'audio': audio_input}
            })

        responses = self.llm.generate(full_prompts, self.sampling_params)
        return [output.outputs[0].text for output in responses]

    def cleanup(self):
        del self.llm
        import gc;
        gc.collect()
        torch.cuda.empty_cache()
    