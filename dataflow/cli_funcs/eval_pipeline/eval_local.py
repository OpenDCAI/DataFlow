from dataflow.operators.core_text import BenchDatasetEvaluator
from dataflow.serving import APILLMServing_request, LocalModelLLMServing_vllm
from dataflow.utils.storage import FileStorage
import os

llm_serving = LocalModelLLMServing_vllm(
    hf_model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
    vllm_tensor_parallel_size=1,
    vllm_max_tokens=512,
)


# 创建存储
storage = FileStorage(
    first_entry_file_name="your_data.jsonl",
    cache_path="./cache",
    file_name_prefix="eval_result",
    cache_type="jsonl",
)

# 创建评估器
evaluator = BenchDatasetEvaluator(
    compare_method="semantic",  # 或 "match"
    llm_serving=llm_serving,  # 语义匹配模式需要
    eval_result_path="./eval_result.json",
)

# 运行评估
evaluator.run(
    storage=storage.step(),
    input_test_answer_key="generated_answer",  # 模型生成的答案字段
    input_gt_answer_key="ground_truth",       # 标准答案字段
    input_question_key="question"             # 问题字段（语义匹配需要）
)