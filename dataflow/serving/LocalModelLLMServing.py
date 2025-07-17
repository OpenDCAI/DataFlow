import os
import torch
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
        self.tokenizer = AutoTokenizer.from_pretrained(self.real_model_path, cache_dir=hf_cache_dir)
        self.logger.success(f"Model loaded from {self.real_model_path} by vLLM backend")
    
    def generate_from_input(self, 
                            user_inputs: list[str], 
                            system_prompt: str = "You are a helpful assistant"
                            ) -> list[str]:
        full_prompts = [system_prompt + '\n' + question for question in user_inputs]
        responses = self.llm.generate(full_prompts, self.sampling_params)
        return [output.outputs[0].text for output in responses]

    def generate_embedding_from_input(self, texts: list[str]) -> list[list[float]]:
        outputs = self.llm.embed(texts)
        return [output.outputs.embedding for output in outputs]

    def cleanup(self):
        del self.llm
        import gc;
        gc.collect()
        torch.cuda.empty_cache()
    
class LocalModelLLMServing_sglang(LLMServingABC):
    def __init__(
        self, 
        hf_model_name_or_path: str = None,
        hf_cache_dir: str = None,
        hf_local_dir: str = None,
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
        **kwargs
    ):
        self.load_model(
            hf_model_name_or_path=hf_model_name_or_path,
            hf_cache_dir=hf_cache_dir,
            hf_local_dir=hf_local_dir,
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
            **kwargs
        )
    def load_model(
        self, 
        hf_model_name_or_path,
        hf_cache_dir,
        hf_local_dir,
        sgl_max_new_tokens: int = 128,
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
        **kwargs
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
        
        # import sglang and set up the environment for multiprocessing
        try:
            import sglang as sgl
        except ImportError:
            raise ImportError("please install sglang first like 'pip install open-dataflow[sglang]'")
        self.llm = sgl.Engine(
            model_path=self.real_model_path,
        )
        self.sampling_params = {
            "max_new_tokens": sgl_max_new_tokens,
            "stop": sgl_stop,
            "stop_token_ids": sgl_stop_token_ids,
            "temperature": sgl_temperature,
            "top_p": sgl_top_p,
            "top_k": sgl_top_k,
            "min_p": sgl_min_p,
            "frequency_penalty": sgl_frequency_penalty,
            "presence_penalty": sgl_presence_penalty,
            "repetition_penalty": sgl_repetition_penalty,
            "min_new_tokens": sgl_min_new_tokens,
            "n": sgl_n,
            "json_schema": sgl_json_schema,
            "regex": sgl_regex,
            "ebnf": sgl_ebnf,
            "structural_tag": sgl_structural_tag,
            "ignore_eos": sgl_ignore_eos,
            "skip_special_tokens": sgl_skip_special_tokens,
            "spaces_between_special_tokens": sgl_spaces_between_special_tokens,
            "no_stop_trim": sgl_no_stop_trim,
            "custom_params": sgl_custom_params or {},
            "stream_interval": sgl_stream_interval or 0.1,  # default to 100ms
            "logit_bias": sgl_logit_bias or {},
            **kwargs
        }
        self.tokenizer = AutoTokenizer.from_pretrained(self.real_model_path, cache_dir=hf_cache_dir)
        self.logger.success(f"Model loaded from {self.real_model_path} by SGLang backend")

    def generate_from_input(self,
                            user_inputs: list[str], 
                            system_prompt: str = "You are a helpful assistant"
                            ) -> list[str]:
        full_prompts = [system_prompt + '\n' + question for question in user_inputs]
        try: 
            responses = self.llm.generate(full_prompts, self.sampling_params)
        except Exception as e:
            self.logger.error(f"Error during Sglang Backend generation, please check your parameters.: {e}")
            raise e

        return [output['text'] for output in responses]
    
    def cleanup(self):
        self.llm.shutdown()
        del self.llm
        import gc;
        gc.collect()
        torch.cuda.empty_cache()
        