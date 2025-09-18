import os
import torch
from dataflow import get_logger
from huggingface_hub import snapshot_download
from dataflow.core import LLMServingABC
from transformers import AutoProcessor
from typing import Optional, Union, List, Dict, Any, Tuple

import librosa
import requests
import numpy as np
from io import BytesIO

# 不重采样
DEFAULT_SR = None

def _read_audio_remote(path: str, sr: Optional[int] = DEFAULT_SR) -> Tuple[np.ndarray, int]:
    url = path
    resp = requests.get(url, stream=True)

    audio_bytes = BytesIO(resp.content)
    y, sr = librosa.load(audio_bytes, sr=sr)
    return y, sr

def _read_audio_local(path: str, sr: Optional[int] = DEFAULT_SR) -> Tuple[np.ndarray, int]:
    return librosa.load(path, sr=sr, mono=True)

def _read_audio_bytes(data: bytes, sr: Optional[int] = DEFAULT_SR) -> Tuple[np.ndarray, int]:
    return librosa.load(BytesIO(data), sr=sr, mono=True)

def _read_audio_base64(b64: str, sr: Optional[int] = DEFAULT_SR) -> Tuple[np.ndarray, int]:
    header, b64data = b64.split(",", 1)
    data = base64.b64decode(b64data)
    return _read_audio_bytes(data, sr=sr)

def process_audio_info(
    conversations: List[dict] | List[List[dict]],   # 这个conversation对应的是vllm中的messages列表(对应的是conversation_to_message函数的message)
    sampling_rate: Optional[int]
) -> Tuple[
    Optional[List[np.ndarray]],
    Optional[List[int]],
    Optional[List[str]]
]:
    """
    类似于 vision 的 process_vision_info，从 message 列表中提取音频输入。
    支持三种格式输入：
      - 本地或 http(s) URL 路径（通过 librosa 接口处理）
      - base64 编码 (data:audio/…;base64,…)
      - 直接传入 bytes 对象
    返回二元组：
      - audio_arrays: 解码后的 waveform (List[np.ndarray])
      - sample_rates: 采样率列表 (List[int])
    """
    if isinstance(conversations, list) and conversations and isinstance(conversations[0], dict):
        # 单条 conversaion
        conversations = [conversations]     # conversations被统一为List[List[dict]]

    audio_arrays = []
    sampling_rates = []

    for conv in conversations:
        for msg in conv:
            if not isinstance(msg.get("content"), list):
                continue
            for ele in msg["content"]:
                if ele.get("type") != "audio":
                    continue
                aud = ele.get("audio")
                if isinstance(aud, str):
                    if aud.startswith("data:audio") and "base64," in aud:
                        arr, sr = _read_audio_base64(aud, sr=sampling_rate)
                        audio_arrays.append(arr)
                        sampling_rates.append(sr)
                    elif aud.startswith("http://") or aud.startswith("https://"):
                        # 使用 librosa 支持远程路径
                        arr, sr = _read_audio_remote(aud, sr=sampling_rate)
                        audio_arrays.append(arr)
                        sampling_rates.append(sr)
                    else:
                        # 本地路径
                        arr, sr = _read_audio_local(aud, sr=sampling_rate)
                        audio_arrays.append(arr)
                        sampling_rates.append(sr)
                elif isinstance(aud, (bytes, bytearray)):
                    arr, sr = _read_audio_bytes(bytes(aud), sr=sampling_rate)
                    audio_arrays.append(arr)
                    sampling_rates.append(sr)
                else:
                    raise ValueError(f"Unsupported audio type: {type(aud)}")

    if not audio_arrays:
        return None, None
    return audio_arrays, sampling_rates

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
        self.processor = AutoProcessor.from_pretrained(self.real_model_path, cache_dir=self.hf_cache_dir)
        self.logger.success(f"Model loaded from {self.real_model_path} by vLLM backend")

    def generate_from_input(self,    
                            user_inputs: list[str], 
                            system_prompt: str = "You are a helpful assistant",
                        ) -> list[str]:   
        if not self.backend_initialized:
            self.start_serving()

        messages = []
        for path_or_url in user_inputs:
            message = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": "请帮我把这段音频的文字翻译成中文"},
                        {"type": "audio", "audio": path_or_url} 
                    ]
                }
            ]
            messages.append(message)

        user_inputs = [self.processor.apply_chat_template(
            msg,
            tokenize=False,
            add_generation_prompt=True,
            add_audio_id = True
        ) for msg in messages]

        audio_arrays, sampling_rates = process_audio_info(conversations=messages, sampling_rate=16000) 
        audio_inputs = [(audio_array, sampling_rate) for audio_array, sampling_rate in zip(audio_arrays, sampling_rates)]

        prompts = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,  # Set to False to strictly disable thinking
        )

        mm_entries = [
            {
                "prompt": prompt,
                "multi_modal_data": {"audio": (audio_array, sampling_rate)}
            }
            for prompt, audio_array, sampling_rate in zip(prompts, audio_arrays, sampling_rates)
        ]

        responses = self.llm.generate(mm_entries, self.sampling_params)
        return [output.outputs[0].text for output in responses]

    def cleanup(self):
        del self.llm
        import gc;
        gc.collect()
        torch.cuda.empty_cache()
    