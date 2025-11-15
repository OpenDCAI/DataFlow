# eval_local.py - 本地评估配置文件
"""DataFlow Local Evaluation Configuration - Enhanced Version"""

from pathlib import Path
from dataflow.serving import LocalModelLLMServing_vllm
from dataflow.utils.storage import FileStorage
from dataflow.operators.core_text import BenchDatasetEvaluatorQuestion


# =============================================================================
# Fair Evaluation Prompt Template
# =============================================================================

class FairAnswerJudgePrompt:
    """Fair answer evaluation prompt template with English prompts"""

    def build_prompt(self, question, answer, reference_answer):
        prompt = f"""You are an expert evaluator assessing answer quality for academic questions.

            **Question:**
            {question}

            **Answer to Evaluate:**
            {answer}

            **Evaluation Instructions:**
            Judge this answer based on:
            1. **Factual Accuracy**: Is the information correct?
            2. **Completeness**: Does it address the key aspects of the question?
            3. **Relevance**: Is it directly related to what was asked?
            4. **Academic Quality**: Is the reasoning sound and appropriate?

            **Important Guidelines:**
            - Focus on content correctness, not writing style
            - A good answer may be longer, shorter, or differently structured
            - Accept different valid approaches or explanations
            - Judge based on whether the answer demonstrates correct understanding
            - Consider partial credit for answers that are mostly correct

            **Reference Answer (for context only):** {reference_answer}

            **Output Format:**
            Return your judgment in JSON format:
            {{"judgement_result": true}} if the answer is factually correct and adequately addresses the question
            {{"judgement_result": false}} if the answer contains significant errors or fails to address the question

            **Your Judgment:**"""
        return prompt


# =============================================================================
# Configuration Parameters
# =============================================================================

# Judge Model Configuration (local strong model as judge)
JUDGE_MODEL_CONFIG = {
    "model_path": "./Qwen2.5-7B-Instruct",  # 用更强的模型做裁判
    "hf_cache_dir": "",
    "hf_local_dir": "",
    "vllm_tensor_parallel_size": 4,
    "vllm_temperature": 0.9,
    "vllm_top_p": 0.9,
    "vllm_max_tokens": 512,
    "vllm_repetition_penalty": 0.8,
    "vllm_seed": None,
    "vllm_max_model_len": None,
    "vllm_gpu_memory_utilization": 0.9
}

# Target Models Configuration (字典格式 - 必需)
TARGET_MODELS = [
    {
        "name": "qwen_3b",  # 模型名称（可选，默认使用路径最后一部分）
        "path": "./Qwen2.5-3B-Instruct",  # 模型路径（必需）

        # 大模型可以用不同的参数
        "vllm_tensor_parallel_size": 4,
        "vllm_temperature": 0.1,
        "vllm_top_p": 0.9,
        "vllm_max_tokens": 2048,
        "vllm_repetition_penalty": 1.0,
        "vllm_seed": None,
        "vllm_gpu_memory_utilization": 0.9,
    },
    {
        "name": "qwen_7b",
        "path": "./Qwen2.5-7B-Instruct",
        # 大模型可以用不同的参数
        "vllm_tensor_parallel_size": 4,
        "vllm_temperature": 0.1,
        "vllm_top_p": 0.9,
        "vllm_max_tokens": 2048,
        "vllm_repetition_penalty": 1.0,
        "vllm_seed": None,
        "vllm_gpu_memory_utilization": 0.9,

        # 可以为每个模型自定义提示词
        "answer_prompt": """please answer the following question:"""

    },

]

# Bench Configuration
BENCH_CONFIG = [
    {
        "name": "math_bench",  # bench名称
        "input_file": "./.cache/data/qa.json",  # 数据文件
        "question_key": "input",  # 问题字段名
        "reference_answer_key": "output",  # 答案字段名
        "output_dir": "./eval_results/math_bench",  # 输出目录
    },
    {
        "name": "reasoning_bench",
        "input_file": "/mnt/DataFlow/fyl/workspace/nt.json",
        "question_key": "input",
        "reference_answer_key": "output",
        "output_dir": "./eval_results/reasoning_bench",
    },
    # {
    #     "name": "code_bench",
    #     "input_file": "./.cache/data/code_qa.json",
    #     "question_key": "problem",
    #     "reference_answer_key": "solution",
    #     "output_dir": "./eval_results/code_bench",
    # },

]

# Evaluator Run Configuration (parameters passed to BenchDatasetEvaluator.run)
EVALUATOR_RUN_CONFIG = {
    "input_test_answer_key": "model_generated_answer",  # 模型生成的答案字段名
    "input_gt_answer_key": "output",  # 标准答案字段名（对应原始数据）
    "input_question_key": "input"  # 问题字段名（对应原始数据）
}

# Evaluation Configuration
EVAL_CONFIG = {
    "compare_method": "semantic",  # "semantic" 语义匹配 或 "match" 字段完全匹配
}


# =============================================================================
# Component Creation Functions - DataFlow Style
# =============================================================================

def create_judge_serving():
    """创建本地评估器LLM服务"""
    model_path = JUDGE_MODEL_CONFIG["model_path"]

    # Enhanced model path validation
    if not model_path.startswith(("Qwen", "meta-llama", "microsoft", "google", "huggingface.co")):
        model_path_obj = Path(model_path)
        if not model_path_obj.exists():
            raise FileNotFoundError(f"Local model path does not exist: {model_path}")

        # Check for required model files
        required_files = ["config.json"]
        missing_files = [f for f in required_files if not (model_path_obj / f).exists()]
        if missing_files:
            raise ValueError(f"Missing required model files in {model_path}: {missing_files}")

    # Enhanced VLLM configuration
    vllm_config = {
        "hf_model_name_or_path": model_path,
        "hf_cache_dir": JUDGE_MODEL_CONFIG.get("hf_cache_dir"),
        "hf_local_dir": JUDGE_MODEL_CONFIG.get("hf_local_dir"),
        "vllm_tensor_parallel_size": JUDGE_MODEL_CONFIG.get("vllm_tensor_parallel_size", 1),
        "vllm_temperature": JUDGE_MODEL_CONFIG.get("vllm_temperature", 0.9),
        "vllm_top_p": JUDGE_MODEL_CONFIG.get("vllm_top_p", 0.9),
        "vllm_max_tokens": JUDGE_MODEL_CONFIG.get("max_tokens", 512),
        "vllm_repetition_penalty": JUDGE_MODEL_CONFIG.get("vllm_repetition_penalty", 1.0),
        "vllm_seed": JUDGE_MODEL_CONFIG.get("vllm_seed", None),
        "vllm_max_model_len": JUDGE_MODEL_CONFIG.get("vllm_max_model_len", None),
        "vllm_gpu_memory_utilization": JUDGE_MODEL_CONFIG.get("gpu_memory_utilization", 0.9)
    }

    # Add optional VLLM parameters if they exist
    optional_params = ["dtype", "trust_remote_code", "enforce_eager", "disable_log_stats"]
    for param in optional_params:
        if param in JUDGE_MODEL_CONFIG:
            vllm_config[f"vllm_{param}"] = JUDGE_MODEL_CONFIG[param]

    return LocalModelLLMServing_vllm(**vllm_config)


def create_evaluator(judge_serving, eval_result_path):
    """创建评估算子"""
    return BenchDatasetEvaluatorQuestion(
        compare_method=EVAL_CONFIG["compare_method"],
        llm_serving=judge_serving,
        prompt_template=FairAnswerJudgePrompt(),
        eval_result_path=eval_result_path
    )


def create_storage(data_file, cache_path, bench_name="eval_result"):
    """创建存储算子"""
    return FileStorage(
        first_entry_file_name=data_file,
        cache_path=cache_path,
        file_name_prefix=bench_name,
        cache_type="json"
    )


# =============================================================================
# Main Configuration Function
# =============================================================================

def get_evaluator_config():
    """返回完整配置"""
    return {
        "JUDGE_MODEL_CONFIG": JUDGE_MODEL_CONFIG,
        "TARGET_MODELS": TARGET_MODELS,
        "BENCH_CONFIG": BENCH_CONFIG,
        "EVALUATOR_RUN_CONFIG": EVALUATOR_RUN_CONFIG,
        "EVAL_CONFIG": EVAL_CONFIG,
        "create_judge_serving": create_judge_serving,
        "create_evaluator": create_evaluator,
        "create_storage": create_storage
    }


# =============================================================================
# Direct Execution Support
# =============================================================================

if __name__ == "__main__":
    # 直接运行时的简单评估
    print("Starting local evaluation...")
    from dataflow.cli_funcs.cli_eval import run_evaluation

    try:
        config = get_evaluator_config()
        success = run_evaluation(config)

        if success:
            print("Local evaluation completed successfully")
        else:
            print("Local evaluation failed")
    except Exception as e:
        print(f"Evaluation error: {e}")
        import traceback

        traceback.print_exc()