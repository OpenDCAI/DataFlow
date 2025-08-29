import os
import torch
import contextlib
import time
from typing import Optional, Union, List, Dict, Any
from dataflow import get_logger
from huggingface_hub import snapshot_download
from dataflow.core import LLMServingABC
from transformers import AutoTokenizer

class LocalModelLLMServing_vllm(LLMServingABC):
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
                 vllm_seed: int = None,
                 vllm_max_model_len: int = None,
                 vllm_gpu_memory_utilization: float=0.9,
                 ):
        self.logger = get_logger()
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
        self.backend_initialized = False
        
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
        
        self.hf_model_name_or_path = hf_model_name_or_path
        self.hf_cache_dir = hf_cache_dir
        self.hf_local_dir = hf_local_dir
        self.vllm_tensor_parallel_size = vllm_tensor_parallel_size
        self.vllm_temperature = vllm_temperature
        self.vllm_top_p = vllm_top_p
        self.vllm_max_tokens = vllm_max_tokens
        self.vllm_top_k = vllm_top_k
        self.vllm_repetition_penalty = vllm_repetition_penalty
        self.vllm_seed = vllm_seed
        self.vllm_max_model_len = vllm_max_model_len
        self.vllm_gpu_memory_utilization = vllm_gpu_memory_utilization
        
    def start_serving(self):
        self.backend_initialized = True  
        self.logger = get_logger()
        if self.hf_model_name_or_path is None:
            raise ValueError("hf_model_name_or_path is required") 
        elif os.path.exists(self.hf_model_name_or_path):
            self.logger.info(f"Using local model path: {self.hf_model_name_or_path}")
            self.real_model_path = self.hf_model_name_or_path
        else:
            self.logger.info(f"Downloading model from HuggingFace: {self.hf_model_name_or_path}")
            self.real_model_path = snapshot_download(
                repo_id=self.hf_model_name_or_path,
                cache_dir=self.hf_cache_dir,
                local_dir=self.hf_local_dir,
            )

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
            temperature=self.vllm_temperature,
            top_p=self.vllm_top_p,
            max_tokens=self.vllm_max_tokens,
            top_k=self.vllm_top_k,
            repetition_penalty=self.vllm_repetition_penalty,
            seed=self.vllm_seed
        )
        
        self.llm = LLM(
            model=self.real_model_path,
            tensor_parallel_size=self.vllm_tensor_parallel_size,
            max_model_len=self.vllm_max_model_len,
            gpu_memory_utilization=self.vllm_gpu_memory_utilization,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.real_model_path, cache_dir=self.hf_cache_dir)
        self.logger.success(f"Model loaded from {self.real_model_path} by vLLM backend")
    
    def generate_from_input(self, 
                            user_inputs: list[str], 
                            system_prompt: str = "You are a helpful assistant"
                            ) -> list[str]:
        if not self.backend_initialized:
            self.start_serving()
        full_prompts = []
        for question in user_inputs:
            messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
            full_prompts.append(messages)
        full_template = self.tokenizer.apply_chat_template(
            full_prompts,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,  # Set to False to strictly disable thinking
        )
        responses = self.llm.generate(full_template, self.sampling_params)
        return [output.outputs[0].text for output in responses]

    def generate_embedding_from_input(self, texts: list[str]) -> list[list[float]]:
        if not self.backend_initialized:
            self.start_serving()
        outputs = self.llm.embed(texts)
        return [output.outputs.embedding for output in outputs]

    def cleanup(self):
        free_mem = torch.cuda.mem_get_info()[0]  # 返回可用显存（单位：字节）
        total_mem = torch.cuda.get_device_properties(0).total_memory
        self.logger.info(f"Free memory: {free_mem / (1024 ** 2):.2f} MB / {total_mem / (1024 ** 2):.2f} MB")
        self.logger.info("Cleaning up vLLM backend resources...")
        self.backend_initialized = False
        from vllm.distributed.parallel_state import (
            destroy_model_parallel,
            destroy_distributed_environment,
        )
        del self.llm.llm_engine
        del self.llm
        destroy_model_parallel()
        destroy_distributed_environment()
        with contextlib.suppress(AssertionError):
            torch.distributed.destroy_process_group()
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        import ray
        ray.shutdown()
        free_mem = torch.cuda.mem_get_info()[0]  # 返回可用显存（单位：字节）
        total_mem = torch.cuda.get_device_properties(0).total_memory

        self.logger.info(f"Free memory: {free_mem / (1024 ** 2):.2f} MB / {total_mem / (1024 ** 2):.2f} MB")
            
class LocalModelLLMServing_sglang(LLMServingABC):
    def __init__(
        self, 
        hf_model_name_or_path: str = None,
        hf_cache_dir: str = None,
        hf_local_dir: str = None,
        sgl_tp_size: int = 1, # tensor parallel size
        sgl_dp_size: int = 1, # data parallel size
        sgl_mem_fraction_static: float = 0.9,  # memory fraction for static memory allocation
        sgl_max_new_tokens: int = 2048, # maximum number of new tokens to generate
        sgl_stop: Optional[Union[str, List[str]]] = None,
        sgl_stop_token_ids: Optional[List[int]] = None,
        sgl_temperature: float = 1.0,
        sgl_top_p: float = 1.0,
        sgl_top_k: int = -1,
        sgl_min_p: float = 0.0,
        sgl_frequency_penalty: float = 0.0,
        sgl_presence_penalty: float = 0.0,
        sgl_repetition_penalty: float = 1.0,
        sgl_min_new_tokens: int = 0,
        sgl_n: int = 1,
        sgl_json_schema: Optional[str] = None,
        sgl_regex: Optional[str] = None,
        sgl_ebnf: Optional[str] = None,
        sgl_structural_tag: Optional[str] = None,
        sgl_ignore_eos: bool = False,
        sgl_skip_special_tokens: bool = True,
        sgl_spaces_between_special_tokens: bool = True,
        sgl_no_stop_trim: bool = False,
        sgl_custom_params: Optional[Dict[str, Any]] = None,
        sgl_stream_interval: Optional[int] = None,
        sgl_logit_bias: Optional[Dict[str, float]] = None,
    ):
        self.logger = get_logger()
        self.load_model(
            hf_model_name_or_path=hf_model_name_or_path,
            hf_cache_dir=hf_cache_dir,
            hf_local_dir=hf_local_dir,
            sgl_tp_size=sgl_tp_size,
            sgl_dp_size=sgl_dp_size,
            sgl_mem_fraction_static=sgl_mem_fraction_static,  # memory fraction for static
            sgl_max_new_tokens=sgl_max_new_tokens,
            sgl_stop=sgl_stop,
            sgl_stop_token_ids=sgl_stop_token_ids,
            sgl_temperature=sgl_temperature,
            sgl_top_p=sgl_top_p,
            sgl_top_k=sgl_top_k,
            sgl_min_p=sgl_min_p,
            sgl_frequency_penalty=sgl_frequency_penalty,
            sgl_presence_penalty=sgl_presence_penalty,
            sgl_repetition_penalty=sgl_repetition_penalty,
            sgl_min_new_tokens=sgl_min_new_tokens,
            sgl_n=sgl_n,
            sgl_json_schema=sgl_json_schema,
            sgl_regex=sgl_regex,
            sgl_ebnf=sgl_ebnf,
            sgl_structural_tag=sgl_structural_tag,
            sgl_ignore_eos=sgl_ignore_eos,
            sgl_skip_special_tokens=sgl_skip_special_tokens,
            sgl_spaces_between_special_tokens=sgl_spaces_between_special_tokens,
            sgl_no_stop_trim=sgl_no_stop_trim,
            sgl_custom_params=sgl_custom_params,
            sgl_stream_interval=sgl_stream_interval,
            sgl_logit_bias=sgl_logit_bias,
        )
        self.backend_initialized = False
        
    def load_model(
        self, 
        hf_model_name_or_path:str = None,
        hf_cache_dir:str = None,
        hf_local_dir:str = None,
        sgl_tp_size: int = 1,
        sgl_dp_size: int = 1,
        sgl_mem_fraction_static: float = 0.9,  # memory fraction for static memory allocation
        sgl_max_new_tokens: int = 2048,
        sgl_stop: Optional[Union[str, List[str]]] = None,
        sgl_stop_token_ids: Optional[List[int]] = None,
        sgl_temperature: float = 1.0,
        sgl_top_p: float = 1.0,
        sgl_top_k: int = -1,
        sgl_min_p: float = 0.0,
        sgl_frequency_penalty: float = 0.0,
        sgl_presence_penalty: float = 0.0,
        sgl_repetition_penalty: float = 1.0,
        sgl_min_new_tokens: int = 0,
        sgl_n: int = 1,
        sgl_json_schema: Optional[str] = None,
        sgl_regex: Optional[str] = None,
        sgl_ebnf: Optional[str] = None,
        sgl_structural_tag: Optional[str] = None,
        sgl_ignore_eos: bool = False,
        sgl_skip_special_tokens: bool = True,
        sgl_spaces_between_special_tokens: bool = True,
        sgl_no_stop_trim: bool = False,
        sgl_custom_params: Optional[Dict[str, Any]] = None,
        sgl_stream_interval: Optional[int] = None,
        sgl_logit_bias: Optional[Dict[str, float]] = None,
    ):
        self.hf_model_name_or_path = hf_model_name_or_path
        self.hf_cache_dir = hf_cache_dir
        self.hf_local_dir = hf_local_dir
        self.sgl_tp_size = sgl_tp_size
        self.sgl_dp_size = sgl_dp_size
        self.sgl_mem_fraction_static = sgl_mem_fraction_static
        self.sgl_max_new_tokens = sgl_max_new_tokens
        self.sgl_stop = sgl_stop
        self.sgl_stop_token_ids = sgl_stop_token_ids
        self.sgl_temperature = sgl_temperature
        self.sgl_top_p = sgl_top_p
        self.sgl_top_k = sgl_top_k
        self.sgl_min_p = sgl_min_p
        self.sgl_frequency_penalty = sgl_frequency_penalty
        self.sgl_presence_penalty = sgl_presence_penalty
        self.sgl_repetition_penalty = sgl_repetition_penalty
        self.sgl_min_new_tokens = sgl_min_new_tokens
        self.sgl_n = sgl_n
        self.sgl_json_schema = sgl_json_schema
        self.sgl_regex = sgl_regex
        self.sgl_ebnf = sgl_ebnf
        self.sgl_structural_tag = sgl_structural_tag
        self.sgl_ignore_eos = sgl_ignore_eos
        self.sgl_skip_special_tokens = sgl_skip_special_tokens
        self.sgl_spaces_between_special_tokens = sgl_spaces_between_special_tokens
        self.sgl_no_stop_trim = sgl_no_stop_trim
        self.sgl_custom_params = sgl_custom_params
        self.sgl_stream_interval = sgl_stream_interval
        self.sgl_logit_bias = sgl_logit_bias
    
    def start_serving(self):
        self.backend_initialized = True
        self.logger = get_logger()
        if self.hf_model_name_or_path is None:
            raise ValueError("hf_model_name_or_path is required") 
        elif os.path.exists(self.hf_model_name_or_path):
            self.logger.info(f"Using local model path: {self.hf_model_name_or_path}")
            self.real_model_path = self.hf_model_name_or_path
        else:
            self.logger.info(f"Downloading model from HuggingFace: {self.hf_model_name_or_path}")
            self.real_model_path = snapshot_download(
                repo_id=self.hf_model_name_or_path,
                cache_dir=self.hf_cache_dir,
                local_dir=self.hf_local_dir,
            )
        
        # import sglang and set up the environment for multiprocessing
        try:
            import sglang as sgl
        except ImportError:
            raise ImportError("please install sglang first like 'pip install open-dataflow[sglang]'")
        self.llm = sgl.Engine(
            model_path=self.real_model_path,
            tp_size=self.sgl_tp_size,
            dp_size=self.sgl_dp_size,
            mem_fraction_static=self.sgl_mem_fraction_static,  # memory fraction for static memory allocation
        )
        self.sampling_params = {
            "max_new_tokens": self.sgl_max_new_tokens,
            "stop": self.sgl_stop,
            "stop_token_ids": self.sgl_stop_token_ids,
            "temperature": self.sgl_temperature,
            "top_p": self.sgl_top_p,
            "top_k": self.sgl_top_k,
            "min_p": self.sgl_min_p,
            "frequency_penalty": self.sgl_frequency_penalty,
            "presence_penalty": self.sgl_presence_penalty,
            "repetition_penalty": self.sgl_repetition_penalty,
            "min_new_tokens": self.sgl_min_new_tokens,
            "n": self.sgl_n,
            "json_schema": self.sgl_json_schema,
            "regex": self.sgl_regex,
            "ebnf":self.sgl_ebnf,
            "structural_tag": self.sgl_structural_tag,
            "ignore_eos": self.sgl_ignore_eos,
            "skip_special_tokens": self.sgl_skip_special_tokens,
            "spaces_between_special_tokens": self.sgl_spaces_between_special_tokens,
            "no_stop_trim": self.sgl_no_stop_trim,
            "custom_params": self.sgl_custom_params,
            "stream_interval":self.sgl_stream_interval,
            "logit_bias": self.sgl_logit_bias,
        }
        # remove all keys equal to None
        self.sampling_params = {k: v for k, v in self.sampling_params.items() if v is not None}

        self.tokenizer = AutoTokenizer.from_pretrained(self.real_model_path, cache_dir=self.hf_cache_dir)
        self.logger.success(f"Model loaded from {self.real_model_path} by SGLang backend")

    def generate_from_input(self,
                            user_inputs: list[str], 
                            system_prompt: str = "You are a helpful assistant"
                            ) -> list[str]:
        if not self.backend_initialized:
            self.start_serving()
        full_prompts = []
        for question in user_inputs:
            messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
            full_prompts.append(messages)
        full_template = self.tokenizer.apply_chat_template(
            full_prompts,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,  # Set to False to strictly disable thinking
        )
        try: 
            responses = self.llm.generate(full_template, self.sampling_params)
        except Exception as e:
            self.logger.error(f"Error during Sglang Backend generation, please check your parameters.: {e}")
            raise e

        return [output['text'] for output in responses]
    
    def generate_embedding_from_input(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError("SGLang backend does not support embedding generation yet. If you have experience with SGLang, please contribute to this feature in Pull Request.")
        # if not self.backend_initialized:
            # self.start_serving()
            # self.llm.
        # outputs = self.llm.embed(texts)
        # return [output['embedding'] for output in outputs]
    
    def cleanup(self):
        self.logger.info("Cleaning up SGLang backend resources...")
        self.backend_initialized = False
        self.llm.shutdown()
        del self.llm
        import gc;
        gc.collect()
        torch.cuda.empty_cache()
        