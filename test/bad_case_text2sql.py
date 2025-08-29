from dataflow.serving import LocalModelLLMServing
from transformers import AutoTokenizer, AutoConfig, XLMRobertaXLModel
a = LocalModelLLMServing(
    model_name_or_path="/mnt/public/model/huggingface/Qwen2.5-7B-Instruct",
    tensor_parallel_size=4,
    max_tokens=2048,
    max_model_len=2048,
    model_source="local"
)
