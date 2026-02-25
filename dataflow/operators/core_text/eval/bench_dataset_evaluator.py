from dataflow.utils.reasoning.AnswerExtraction import StringCleaner, UnitTextManager, AnswerExtractor
from dataflow.prompts.model_evaluation.general import AnswerJudgePrompt
from dataflow.core.prompt import DIYPromptABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import LLMServingABC
from dataflow.core import OperatorABC
from dataflow.core.prompt import prompt_restrict

from math_verify import parse, verify
from dataflow import get_logger
from typing import Literal, Union
import pandas as pd
import numpy as np
import time
import os
import re

@prompt_restrict(
    AnswerJudgePrompt
)

@OPERATOR_REGISTRY.register()
class BenchDatasetEvaluator(OperatorABC):
    def __init__(self,
                 eval_result_path: str = None,
                 compare_method: Literal["match", "semantic"] = "match",
                 system_prompt: str = "You are a helpful assistant specialized in evaluating answer correctness.",
                 llm_serving: LLMServingABC = None,
                 prompt_template: Union[AnswerJudgePrompt, DIYPromptABC] = AnswerJudgePrompt):
        
        if eval_result_path is None:
            timestamp = int(time.time())
            eval_result_path = f"result_bencheval/BenchDatasetEvaluator_result_{timestamp}.json"

        self.eval_result_path = eval_result_path
        self.compare_method = compare_method
        self.empty_responses_count = 0

        if compare_method == "match":
            self.compare = self.math_verify_compare
            unit_manager = UnitTextManager()
            string_cleaner = StringCleaner(unit_manager)
            self.answer_extractor = AnswerExtractor(string_cleaner)
        else:
            if prompt_template is None:
                prompt_template = AnswerJudgePrompt()
            self.prompt_template = prompt_template
            self.system_prompt = system_prompt
            self.llm_serving = llm_serving

        self.logger = get_logger()

    def math_verify_compare(self, answer, ground_truth):
        try:
            return verify(parse(str(ground_truth)), parse(str(answer)))
        except Exception:
            try:
                return verify(parse(ground_truth), parse(answer))
            except Exception:
                return False

    def ResolveResponse(self, response):
        if response is None or (isinstance(response, str) and response.strip() == ''):
            self.empty_responses_count += 1
            return False
        try:
            pattern = re.compile(r'"judgement_result"\s*:\s*(true|false)', re.IGNORECASE)
            match = pattern.search(response)
            result_value = None
            if match:
                result_value = match.group(1).lower()
            else:
                if "true" in response.lower():
                    result_value = "true"
                else:
                    result_value = "false"
            return result_value == "true"
        except Exception as e:
            self.logger.error(f"Response format error: {response}. Error: {e}")
            return False

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "该算子用于对比预测答案与标准答案的匹配度，支持两种评估模式：\n\n"
                "1. 字符串匹配（match）：使用数学验证方法比较答案，适用于有明确答案的问题\n"
                "2. 语义匹配（semantic）：使用LLM评估语义相似度，仅输入预测答案与标准答案\n\n"
                "输入参数：\n"
                "- input_test_answer_key：预测答案字段名\n"
                "- input_gt_answer_key：标准答案字段名\n"
                "- compare_method：比较方法（match/semantic）\n\n"
                "输出参数：\n"
                "- answer_match_result：匹配结果（True/False）\n"
                "- 统计结果将保存到指定的eval_result_path路径\n"
            )
        else:
            return (
                "This operator compares predicted answers against ground truth using two evaluation modes:\n\n"
                "1. String Matching (match): Mathematical verification for exact answers.\n"
                "2. Semantic Matching (semantic): LLM-based evaluation comparing predicted vs ground truth answers only.\n\n"
                "Input Parameters:\n"
                "- input_test_answer_key: Predicted answer field\n"
                "- input_gt_answer_key: Ground truth field\n"
                "- compare_method: Comparison method (match/semantic)\n\n"
                "Output Parameters:\n"
                "- answer_match_result: Boolean match result\n"
                "- Statistics are saved to the specified eval_result_path\n"
            )

    def check_column(self, required_columns: list[str], dataframe: pd.DataFrame):
        for column in required_columns:
            if column not in dataframe.columns:
                self.logger.error(f"Required column '{column}' not found in dataframe")
                return False
        return True

    def statistic(self, file_name_prefix: str, dataframe: pd.DataFrame, compare_method: str):
        total_samples = len(dataframe)
        valid_samples = len(dataframe) - self.empty_responses_count
        matched_samples = sum(dataframe['answer_match_result'])
        accuracy = matched_samples / valid_samples if valid_samples > 0 else 0

        stats = {
            "bench_name_or_prefix": file_name_prefix,
            "total_samples": total_samples,
            "valid_samples": valid_samples,
            "matched_samples": matched_samples,
            "accuracy": float(accuracy),
            "empty_responses_count": self.empty_responses_count,
            "compare_method": compare_method
        }

        stats_df = pd.DataFrame([stats])
        os.makedirs(os.path.dirname(self.eval_result_path), exist_ok=True)
        stats_df.to_json(self.eval_result_path, orient="records", force_ascii=False, indent=2)
        self.logger.success(f"Statistics saved to {self.eval_result_path}")

        return stats_df

    def run(self,
            storage: DataFlowStorage,
            input_test_answer_key: str = "generated_cot",
            input_gt_answer_key: str = "golden_answer") -> list:

        dataframe = storage.read("dataframe")
        dataframe['answer_match_result'] = False
        answers = dataframe[input_test_answer_key]
        ground_truths = dataframe[input_gt_answer_key]

        if self.compare_method == "match":
            if not self.check_column(
                required_columns=[input_test_answer_key, input_gt_answer_key],
                dataframe=dataframe
            ):
                return [input_test_answer_key, input_gt_answer_key]

            for i in range(len(answers)):
                final_answer = self.answer_extractor.extract_answer(answers[i], None)
                dataframe.at[i, 'answer_match_result'] = self.compare(final_answer, ground_truths[i])

            storage.write(dataframe)
            self.statistic(storage.file_name_prefix, dataframe, self.compare_method)
            return [input_test_answer_key, input_gt_answer_key, 'answer_match_result']

        else:
            if not self.check_column(
                required_columns=[input_test_answer_key, input_gt_answer_key],
                dataframe=dataframe
            ):
                return [input_test_answer_key, input_gt_answer_key]

            empty_reference_mask = dataframe[input_gt_answer_key].isna() | (dataframe[input_gt_answer_key] == '')
            valid_rows = dataframe[~empty_reference_mask]
            skipped_count = len(dataframe[empty_reference_mask])

            if len(valid_rows) == 0:
                self.logger.warning("No valid reference answers found. All samples skipped.")
                storage.write(dataframe)
                return [input_test_answer_key, input_gt_answer_key, 'answer_match_result']

            # 仅用预测答案与标准答案构建Prompt
            inputs = [
                self.prompt_template.build_prompt(
                    answer=row[input_test_answer_key],
                    reference_answer=row[input_gt_answer_key]
                )
                for _, row in valid_rows.iterrows()
            ]

            responses = self.llm_serving.generate_from_input(
                user_inputs=inputs,
                system_prompt=self.system_prompt
            )
            results = [self.ResolveResponse(response) for response in responses]

            for i, idx in enumerate(valid_rows.index):
                dataframe.at[idx, 'answer_match_result'] = results[i]

            storage.write(dataframe)
            self.statistic(storage.file_name_prefix, dataframe, self.compare_method)

            self.empty_responses_count = 0
            return [input_test_answer_key, input_gt_answer_key, 'answer_match_result']
