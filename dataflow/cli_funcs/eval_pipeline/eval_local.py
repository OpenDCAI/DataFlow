# eval_local.py - 包含算子配置的本地评估配置文件
"""DataFlow 本地评估配置 - 包含算子实例化"""

from pathlib import Path
from dataflow.operators.core_text import BenchDatasetEvaluator
from dataflow.serving import LocalModelLLMServing_vllm
from dataflow.utils.storage import FileStorage

# ============================================================================= 
# 基础配置参数
# =============================================================================

# 评估器配置（本地强模型作为裁判）
JUDGE_MODEL_CONFIG = {
    "model_path": "./Qwen2.5-14B-Instruct",  # 用更强的模型做裁判
    "tensor_parallel_size": 2,
    "max_tokens": 512,
    "gpu_memory_utilization": 0.8
}

# 被评估模型配置（与API模式相同）
TARGET_MODELS = {
    "auto_detect": True,
    "models": [
        # 当 auto_detect=False 时，手动指定要评估的模型
        # "Qwen/Qwen2.5-7B-Instruct",
        # "meta-llama/Llama-3-8B-Instruct",
        # "/path/to/local/model",
        # "./.cache/saves/text2model_cache_20241201_143022"
    ]
}

# 数据配置（与API模式相同）
DATA_CONFIG = {
    "input_file": "./.cache/data/qa.json",
    "output_dir": "./eval_results", 
    "question_key": "input",
    "reference_answer_key": "output"
}

EVALUATOR_RUN_CONFIG = {
    "input_test_answer_key": "model_generated_answer",  # 模型生成的答案字段名
    "input_gt_answer_key": "output",                    # 标准答案字段名（对应原始数据）
    "input_question_key": "input"                       # 问题字段名（对应原始数据）
}

# 评估配置（与API模式相同）
EVAL_CONFIG = {
    "compare_method": "semantic",  # "semantic" 或 "match"
    "batch_size": 8,
    "max_tokens": 512
}

# =============================================================================
# 算子实例化 - DataFlow 风格
# =============================================================================

def create_judge_serving():
    """创建本地评估器LLM服务"""
    model_path = JUDGE_MODEL_CONFIG["model_path"]
    
    # 检查本地模型路径（对于HuggingFace模型ID，跳过检查）
    if not model_path.startswith(("Qwen", "meta-llama", "microsoft", "google")) and not Path(model_path).exists():
        raise ValueError(f"模型路径不存在：{model_path}")
    
    return LocalModelLLMServing_vllm(
        hf_model_name_or_path=model_path,
        vllm_tensor_parallel_size=JUDGE_MODEL_CONFIG.get("tensor_parallel_size", 1),
        vllm_max_tokens=JUDGE_MODEL_CONFIG.get("max_tokens", 512),
        vllm_gpu_memory_utilization=JUDGE_MODEL_CONFIG.get("gpu_memory_utilization", 0.8)
    )

def create_evaluator(judge_serving, eval_result_path):
    """创建评估算子"""
    return BenchDatasetEvaluator(
        compare_method=EVAL_CONFIG["compare_method"],
        llm_serving=judge_serving,
        eval_result_path=eval_result_path
    )

def create_storage(data_file, cache_path):
    """创建存储算子"""  
    return FileStorage(
        first_entry_file_name=data_file,
        cache_path=cache_path,
        file_name_prefix="eval_result",
        cache_type="json"
    )

# =============================================================================
# 配置获取函数
# =============================================================================

def get_evaluator_config():
    """返回完整配置"""
    return {
        "JUDGE_MODEL_CONFIG": JUDGE_MODEL_CONFIG,
        "TARGET_MODELS": TARGET_MODELS,
        "DATA_CONFIG": DATA_CONFIG,
        "EVALUATOR_RUN_CONFIG": EVALUATOR_RUN_CONFIG,
        "EVAL_CONFIG": EVAL_CONFIG,
        "create_judge_serving": create_judge_serving,
        "create_evaluator": create_evaluator,
        "create_storage": create_storage
    }

# =============================================================================
# 如果直接运行此文件
# =============================================================================

if __name__ == "__main__":
    # 直接运行时的简单评估
    print("开始评估...")
    from dataflow.cli_funcs.cli_eval import run_evaluation
    
    config = get_evaluator_config()
    success = run_evaluation(config)
    
    if success:
        print("评估完成")
    else:
        print("评估失败")