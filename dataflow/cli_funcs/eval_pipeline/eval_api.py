# eval_api.py - API评估配置文件
"""DataFlow API Evaluation Configuration - Enhanced Version"""

import os
from dataflow.operators.core_text import BenchDatasetEvaluator
from dataflow.serving import APILLMServing_request
from dataflow.utils.storage import FileStorage


# =============================================================================
# Fair Evaluation Prompt Template
# =============================================================================

class FairAnswerJudgePrompt:
    """Fair answer evaluation prompt template with English prompts"""

    # 默认评估模型提示词 该prompt为评估模型的提示词，请勿与被评估模型提示词混淆
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
# 参数设置
# =============================================================================

# Judge Model Configuration (API model as judge)
# 评估模型设置
JUDGE_MODEL_CONFIG = {
    "model_name": "gpt-4o-mini",
    "api_url": "http://123.129.219.111:3000/v1/chat/completions",  # 请求URL 必填 / request (required)
    "api_key_env": "DF_API_KEY",  # api_key 必填 / api_key (required)
    "max_workers": 3,
    "max_retries": 5,
}

# Target Models Configuration (List format - required, each element is a dict)
# 被评估模型设置 (列表格式 - 必需，每个元素是字典)
TARGET_MODELS = [
    {
        "name": "qwen_3b",  # 模型名称（可选，默认使用路径最后一部分） / Model name (optional, uses the last part of the path by default)
        "path": "./Qwen2.5-3B-Instruct",  # 模型路径（必需） / Model path (required)

        # ===== 答案生成的模型加载参数（可选）=====
        "tensor_parallel_size": 1,  # GPU并行数量 / Number of GPU parallels
        "max_tokens": 1024,  # 最大生成token数 / Maximum number of generated tokens
        "gpu_memory_utilization": 0.8,  # GPU显存利用率 / GPU memory utilization
    },
    {
        "name": "qwen_7b",
        "path": "./Qwen2.5-7B-Instruct",

        # 大模型可以用不同的参数
        "tensor_parallel_size": 2,
        "max_tokens": 2048,
        "gpu_memory_utilization": 0.9,

        # 可以为每个模型自定义提示词 不写就为默认模板 即build_prompt函数中的prompt
        # 默认被评估模型提示词
        # 再次提示:该prompt为被评估模型的提示词，请勿与评估模型提示词混淆！！！
        # You can customize prompts for each model. If not specified, defaults to the template in build_prompt function.
        # Default prompt for evaluated models
        # IMPORTANT: This is the prompt for models being evaluated, NOT for the judge model!!!
        "answer_prompt": """please answer the following question:"""  # 这里不要使用{question} / do not code {question} here
    },

    # 添加更多模型...
    # {
    #     "name": "llama_8b",
    #     "path": "meta-llama/Llama-3-8B-Instruct",
    #     "tensor_parallel_size": 2
    # }
]

# Data Configuration
DATA_CONFIG = {
    "input_file": "./.cache/data/qa.json",  # 输入数据文件
    "output_dir": "./eval_results",  # 输出目录
    "question_key": "input",  # 原始数据中的问题字段
    "reference_answer_key": "output"  # 原始数据中的参考答案字段
}

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
# Component Creation Functions
# =============================================================================

def create_judge_serving():
    """创建评估器LLM服务（API模式）"""
    api_key_env = JUDGE_MODEL_CONFIG["api_key_env"]
    if api_key_env not in os.environ:
        raise ValueError(f"Environment variable {api_key_env} is not set. "
                         f"Please set it with your API key.")

    api_key = os.environ[api_key_env]
    if not api_key.strip():
        raise ValueError(f"Environment variable {api_key_env} is empty. "
                         f"Please provide a valid API key.")

    return APILLMServing_request(
        api_url=JUDGE_MODEL_CONFIG["api_url"],
        key_name_of_api_key=api_key_env,
        model_name=JUDGE_MODEL_CONFIG["model_name"],
        max_workers=JUDGE_MODEL_CONFIG.get("max_workers", 10),
        max_retries=JUDGE_MODEL_CONFIG.get("max_retries", 5)
    )


def create_evaluator(judge_serving, eval_result_path):
    """创建评估算子"""
    return BenchDatasetEvaluator(
        compare_method=EVAL_CONFIG["compare_method"],
        llm_serving=judge_serving,
        prompt_template=FairAnswerJudgePrompt(),
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
# Main Configuration Function
# =============================================================================

def get_evaluator_config():
    # 返回完整配置
    # Return complete configuration
    return {
        "JUDGE_MODEL_CONFIG": JUDGE_MODEL_CONFIG,  # 评估模型设置映射
        "TARGET_MODELS": TARGET_MODELS,  # 被评估模型设置映射
        "DATA_CONFIG": DATA_CONFIG,  # 数据设置映射
        "EVAL_CONFIG": EVAL_CONFIG,  # 评估模式设置映射
        "EVALUATOR_RUN_CONFIG": EVALUATOR_RUN_CONFIG,  # 评估数据集字段映射
        "create_judge_serving": create_judge_serving,
        "create_evaluator": create_evaluator,
        "create_storage": create_storage
    }


# =============================================================================
# Direct Execution Support
# 直接运行评估
# =============================================================================

if __name__ == "__main__":
    # 直接运行时的简单评估
    # Simple evaluation when run directly
    print("Starting API evaluation...")
    from dataflow.cli_funcs.cli_eval import run_evaluation

    try:
        config = get_evaluator_config()
        success = run_evaluation(config)

        if success:
            print("API evaluation completed successfully")
        else:
            print("API evaluation failed")
    except Exception as e:
        print(f"Evaluation error: {e}")
        import traceback

        traceback.print_exc()