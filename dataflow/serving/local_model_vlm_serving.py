import os
import torch
import contextlib
from typing import List, Optional, Union, Dict, Any
from PIL import Image
import re

from dataflow import get_logger
from huggingface_hub import snapshot_download
from dataflow.core import LLMServingABC
from transformers import AutoTokenizer

class LocalVLMServing_vllm(LLMServingABC):
    """
    Client for serving a Vision-Language Model (VLM) locally using vLLM.
    Combines the interface of APIVLMServing with the backend efficiency of vLLM.
    """
    def __init__(self, 
                 hf_model_name_or_path: str = None,
                 hf_cache_dir: str = None,
                 hf_local_dir: str = None,
                 vllm_tensor_parallel_size: int = 1,
                 vllm_temperature: float = 0.2, # Lower temperature is usually better for VLM tasks
                 vllm_top_p: float = 0.9,
                 vllm_max_tokens: int = 1024,
                 vllm_max_model_len: int = None,
                 vllm_top_k: int = 40,
                 vllm_repetition_penalty: float = 1.0,
                 vllm_seed: int = None,
                 vllm_gpu_memory_utilization: float = 0.9,
                 vllm_limit_mm_per_prompt: int = 1, # Specific to VLMs: max images per prompt
                 trust_remote_code: bool = True,
                 enable_thinking:bool =True,  # Set to False to strictly disable thinking
                 batch_size: int = 128
                 ):
        """
        Initialize the Local VLM Serving client.
        """
        self.logger = get_logger()
        QWEN_VL_PATTERN = re.compile(r'Qwen-VL|Qwen[0-9\.]+-VL')
        # 报Warning显示目前经过测试的VLM主要是QwenVL
        if hf_model_name_or_path and not QWEN_VL_PATTERN.search(hf_model_name_or_path):
            self.logger.warning(
                "Model Compatibility Alert: LocalVLMServing_vllm is primarily tested with Qwen-VL models "
                "(e.g., Qwen-VL-Chat, Qwen2.5-VL-Chat). Other VLMs may require additional adjustments "
                "for correct functionality."
            )
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
            vllm_limit_mm_per_prompt=vllm_limit_mm_per_prompt,
            trust_remote_code=trust_remote_code,
            enable_thinking=enable_thinking
        )
        self.backend_initialized = False
        self.batch_size = batch_size

    def load_model(self, 
                 hf_model_name_or_path: str = None,
                 hf_cache_dir: str = None,
                 hf_local_dir: str = None,
                 vllm_tensor_parallel_size: int = 1,
                 vllm_temperature: float = 0.2,
                 vllm_top_p: float = 0.9,
                 vllm_max_tokens: int = 1024,
                 vllm_max_model_len: int = None,
                 vllm_top_k: int = 40,
                 vllm_repetition_penalty: float = 1.0,
                 vllm_seed: int = None,
                 vllm_gpu_memory_utilization: float = 0.9,
                 vllm_limit_mm_per_prompt: int = 1,
                 trust_remote_code: bool = True,
                 enable_thinking:bool =True,
                 ):
        
        self.hf_model_name_or_path = hf_model_name_or_path
        self.hf_cache_dir = hf_cache_dir
        self.hf_local_dir = hf_local_dir
        self.vllm_tensor_parallel_size = vllm_tensor_parallel_size
        self.vllm_temperature = vllm_temperature
        self.vllm_top_p = vllm_top_p
        self.vllm_max_tokens = vllm_max_tokens
        self.vllm_max_model_len = vllm_max_model_len
        self.vllm_top_k = vllm_top_k
        self.vllm_repetition_penalty = vllm_repetition_penalty
        self.vllm_seed = vllm_seed
        self.vllm_gpu_memory_utilization = vllm_gpu_memory_utilization
        self.vllm_limit_mm_per_prompt = vllm_limit_mm_per_prompt
        self.trust_remote_code = trust_remote_code
        self.enable_thinking = enable_thinking

    def start_serving(self):
        self.backend_initialized = True  
        self.logger = get_logger()
        
        # 1. Handle Model Path (HuggingFace or Local)
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

        # 2. Import vLLM and Setup Environment
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError("please install vllm first like 'pip install open-dataflow[vllm]'")
        
        # Set environment for multiprocessing compatibility
        os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = "spawn"
        
        self.sampling_params = SamplingParams(
            temperature=self.vllm_temperature,
            top_p=self.vllm_top_p,
            max_tokens=self.vllm_max_tokens,
            top_k=self.vllm_top_k,
            repetition_penalty=self.vllm_repetition_penalty,
            seed=self.vllm_seed
        )
        
        # 3. Initialize LLM Engine with VLM specific params
        self.llm = LLM(
            model=self.real_model_path,
            tensor_parallel_size=self.vllm_tensor_parallel_size,
            max_model_len=self.vllm_max_model_len,
            gpu_memory_utilization=self.vllm_gpu_memory_utilization,
            limit_mm_per_prompt={"image": self.vllm_limit_mm_per_prompt}, # Specific config for image limits
            trust_remote_code=self.trust_remote_code
        )
        
        # Load tokenizer for chat templating
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.real_model_path, 
            cache_dir=self.hf_cache_dir,
            trust_remote_code=self.trust_remote_code,
            enable_thinking=self.enable_thinking
        )
        self.logger.success(f"VLM Model loaded from {self.real_model_path} by vLLM backend")

    def _load_image(self, image_path: str) -> Image.Image:
        """
        Helper to load image from path using PIL. 
        Replaces API _encode_image_to_base64 logic with PIL loading for vLLM.
        """
        try:
            return Image.open(image_path).convert("RGB")
        except Exception as e:
            self.logger.error(f"Failed to load image at {image_path}: {e}")
            raise e
        
    def _run_batch_inference(self, vllm_inputs, batch_size):
        all_outputs = []
        # 按 batch_size 分批处理
        for i in range(0, len(vllm_inputs), batch_size):
            batch_inputs = vllm_inputs[i:i+batch_size]
            outputs = self.llm.generate(batch_inputs, sampling_params=self.sampling_params)
            all_outputs.extend([output.outputs[0].text for output in outputs])
        return all_outputs

    def generate_from_input_one_image(
        self,
        image_paths: List[str],
        text_prompts: List[str],
        system_prompt: str = "Describe the image in detail.",
        model: str = None, # Unused, kept for interface compatibility
        timeout: int = 1800 # Unused, kept for interface compatibility
    ) -> List[str]:
        """
        Batch process single-image chat requests concurrently using vLLM.
        Matches the signature of APIVLMServing_openai.generate_from_input_one_image.
        """
        if not self.backend_initialized:
            self.start_serving()

        if len(image_paths) != len(text_prompts):
            raise ValueError("`image_paths` and `text_prompts` must have the same length")

        inputs = []
        
        # Prepare inputs for vLLM
        for img_path, user_text in zip(image_paths, text_prompts):
            image = self._load_image(img_path)
            
            # Construct messages using standard chat format
            # Note: Specific VLMs might require specific placeholder tokens (e.g., <image>) 
            # but AutoTokenizer.apply_chat_template usually handles this if configured correctly.
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_text}
                ]}
            ]
            
            # Apply template to get the prompt string
            prompt = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # Construct vLLM input dictionary
            inputs.append({
                "prompt": prompt,
                "multi_modal_data": {
                    "image": image
                }
            })

        # Run Inference
        outputs = self.llm.generate(inputs, sampling_params=self.sampling_params)
        
        # Extract Text
        responses = []
        for output in outputs:
            responses.append(output.outputs[0].text)
            
        return responses

    def generate_from_input_multi_images(
        self,
        list_of_image_paths: List[List[str]],
        list_of_image_labels: List[List[str]],
        user_prompts: List[str],
        system_prompt: str = "Analyze the provided images.",
        model: str = None,  # 保持接口一致性，实际不使用
        timeout: int = 1800 # 保持接口一致性，实际不使用
    ) -> List[str]:
        """
        Batch process multi-image chat requests using vLLM.
        
        :param list_of_image_paths: Outer list is the batch, inner list contains paths for one request.
        :param list_of_image_labels: Corresponding labels/prompts for each image in the inner list.
        """
        if not self.backend_initialized:
            self.start_serving()

        if len(list_of_image_paths) != len(list_of_image_labels):
            raise ValueError("`list_of_image_paths` and `list_of_image_labels` must have the same length")

        vllm_inputs = []

        # 遍历每一个请求 (Batch Loop)
        for paths, labels, user_prompt in zip(list_of_image_paths, list_of_image_labels, user_prompts):
            if len(paths) != len(labels):
                raise ValueError("Inner lists of paths and labels must have the same length")
            
            # 1. 加载该请求下的所有图片
            # vLLM 支持传入 PIL Image 列表
            current_images = [self._load_image(p) for p in paths]
            
            # 检查是否超过了初始化时设定的最大图片数
            if len(current_images) > self.vllm_limit_mm_per_prompt:
                self.logger.warning(
                    f"Request contains {len(current_images)} images, but limit is {self.vllm_limit_mm_per_prompt}. "
                    "This might cause vLLM errors. Increase `vllm_limit_mm_per_prompt` in init."
                )

            # 2. 构建 User Content
            # 我们需要交替插入文本(Label)和图片占位符
            user_content = [{"type": "text", "text": user_prompt}]
            for label, _ in zip(labels, paths):
                # 插入标签文本（如果有）
                if label:
                    user_content.append({"type": "text", "text": f"{label}\n"})
                # 插入图片占位符
                user_content.append({"type": "image"})
            
            # 3. 构建完整的 Messages
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]

            # 4. 应用 Chat Template
            # 大多数现代 VLM 的 template 会自动处理多个 {"type": "image"} 
            # 并将其转换为类似 <image> <image> ... 的 token 序列
            prompt = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )

            # 5. 构建 vLLM 输入 payload
            # 当有多张图片时，multi_modal_data["image"] 应该是一个列表
            vllm_inputs.append({
                "prompt": prompt,
                "multi_modal_data": {
                    "image": current_images
                }
            })

        # 6. 执行批量推理
        all_outputs = self._run_batch_inference(vllm_inputs, batch_size=self.batch_size)
        return all_outputs
    
    def cleanup(self):
        """
        Clean up vLLM backend resources.
        Identical logic to LocalModelLLMServing_vllm.cleanup.
        """
        free_mem = torch.cuda.mem_get_info()[0]
        total_mem = torch.cuda.get_device_properties(0).total_memory
        self.logger.info(f"Free memory before cleanup: {free_mem / (1024 ** 2):.2f} MB / {total_mem / (1024 ** 2):.2f} MB")
        
        self.logger.info("Cleaning up vLLM VLM backend resources...")
        self.backend_initialized = False
        
        # vLLM Distributed Cleanup
        from vllm.distributed.parallel_state import (
            destroy_model_parallel,
            destroy_distributed_environment,
        )
        
        if hasattr(self, 'llm'):
            del self.llm.llm_engine
            del self.llm
            
        destroy_model_parallel()
        destroy_distributed_environment()
        
        with contextlib.suppress(AssertionError):
            torch.distributed.destroy_process_group()
            
        import gc
        import ray
        
        gc.collect()
        torch.cuda.empty_cache()
        ray.shutdown()
        
        free_mem = torch.cuda.mem_get_info()[0]
        self.logger.info(f"Free memory after cleanup: {free_mem / (1024 ** 2):.2f} MB")
        
    def generate_from_input(self, user_inputs: List[str], system_prompt: str = "Describe the image in detail.", json_schema: dict = None):
        """
        保持接口一致性，实际不使用
        """
        
        return self.generate_from_input_one_image(
            image_paths=user_inputs,
            text_prompts=[""] * len(user_inputs),
            system_prompt=system_prompt
        )