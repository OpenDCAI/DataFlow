# eval_api.py - 评估配置文件
"""DataFlow API评估配置 - 简化版本"""

import os
from dataflow.operators.core_text import BenchDatasetEvaluator
from dataflow.serving import APILLMServing_request
from dataflow.utils.storage import FileStorage

# =============================================================================
# 公平评估Prompt
# =============================================================================

class FairAnswerJudgePrompt:
    """公平的答案评判提示词模板"""
    
    def build_prompt(self, question, answer, reference_answer):
        prompt = f"""
You are an expert evaluator assessing answer quality for academic questions.

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

**Reference (for context only):** {reference_answer}

Return your judgment in JSON format:
{{"judgement_result": true}} if the answer is factually correct and adequately addresses the question
{{"judgement_result": false}} if the answer contains significant errors or fails to address the question

Your judgment:
"""
        return prompt

# =============================================================================
# 配置参数
# =============================================================================

# 评估器配置（API模型作为裁判）
JUDGE_MODEL_CONFIG = {
    "model_name": "gpt-4o-mini",
    "api_url": "http://123.129.219.111:3000/v1/chat/completions",
    "api_key_env": "DF_API_KEY",
    "max_workers": 3,
    "max_retries": 5
}

# 被评估模型配置
TARGET_MODELS = {
    "auto_detect": True,    # 自动检测本地训练的模型
    "models": [
        # 当 auto_detect=False 时，手动指定要评估的模型
        # "Qwen/Qwen2.5-7B-Instruct",
        # "meta-llama/Llama-3-8B-Instruct",
        # "/path/to/local/model",
    ]
}

# 数据配置
DATA_CONFIG = {
    "input_file": "./.cache/data/qa.json",
    "output_dir": "./eval_results",
    "question_key": "input",                    # 原始数据中的问题字段
    "reference_answer_key": "output"            # 原始数据中的参考答案字段
}

# 评估算子参数配置（传给BenchDatasetEvaluator.run的参数）
EVALUATOR_RUN_CONFIG = {
    "input_test_answer_key": "model_generated_answer",  # 模型生成的答案字段名
    "input_gt_answer_key": "output",                    # 标准答案字段名（对应原始数据）
    "input_question_key": "input"                       # 问题字段名（对应原始数据）
}

# 评估配置
EVAL_CONFIG = {
    "compare_method": "semantic",  # "semantic" 或 "match"
    "batch_size": 8,
    "max_tokens": 512
}

# =============================================================================
# 算子创建函数
# =============================================================================

def create_judge_serving():
    """创建评估器LLM服务"""
    api_key_env = JUDGE_MODEL_CONFIG["api_key_env"]
    if api_key_env not in os.environ:
        raise ValueError(f"环境变量 {api_key_env} 未设置")
    
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
# 主配置函数
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
# 直接运行支持
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