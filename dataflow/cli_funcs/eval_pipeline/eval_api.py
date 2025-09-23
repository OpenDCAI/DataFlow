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

# Judge Model Configuration (API model as judge)
JUDGE_MODEL_CONFIG = {
    "model_name": "gpt-4o-mini",
    "api_url": "http://OPENAI_URL/v1/chat/completions",  # 请求URL 必填
    "api_key_env": "DF_API_KEY",  # api_key 必填
    "max_workers": 3,
    "max_retries": 5,
    "timeout": 60  # 添加超时配置
}

# Target Models Configuration
TARGET_MODELS = {
    "auto_detect": True,  # 自动检测本地训练的模型
    "models": [
        # 当 auto_detect=False 时，手动指定要评估的模型
        # "Qwen/Qwen2.5-7B-Instruct",
        # "meta-llama/Llama-3-8B-Instruct",
        # "/path/to/local/model",
        # "./.cache/saves/text2model_cache_20241201_143022"
    ]
}

# Data Configuration
DATA_CONFIG = {
    "input_file": "./.cache/data/qa.json",
    "output_dir": "./eval_results",
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
    "compare_method": "semantic",  # "semantic" 或 "match"
    "batch_size": 8,
    "max_tokens": 512
}


# =============================================================================
# Component Creation Functions
# =============================================================================

def create_judge_serving():
    """创建评估器LLM服务"""
    api_key_env = JUDGE_MODEL_CONFIG["api_key_env"]
    if api_key_env not in os.environ:
        raise ValueError(f"Environment variable {api_key_env} is not set. Please set it with your API key.")

    api_key = os.environ[api_key_env]
    if not api_key.strip():
        raise ValueError(f"Environment variable {api_key_env} is empty. Please provide a valid API key.")

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
    """返回完整配置"""
    return {
        "JUDGE_MODEL_CONFIG": JUDGE_MODEL_CONFIG,
        "TARGET_MODELS": TARGET_MODELS,
        "DATA_CONFIG": DATA_CONFIG,
        "EVAL_CONFIG": EVAL_CONFIG,
        "EVALUATOR_RUN_CONFIG": EVALUATOR_RUN_CONFIG,
        "create_judge_serving": create_judge_serving,
        "create_evaluator": create_evaluator,
        "create_storage": create_storage
    }


# =============================================================================
# Direct Execution Support
# =============================================================================

if __name__ == "__main__":
    # 直接运行时的简单评估
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