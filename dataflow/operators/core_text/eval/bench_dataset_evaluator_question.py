from dataflow.utils.reasoning.AnswerExtraction import StringCleaner, UnitTextManager, AnswerExtractor
from dataflow.prompts.model_evaluation.general import AnswerJudgePromptQuestion, AnswerJudgeMultipleQuestionsPrompt
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
import time
import os
import re
import json5

@prompt_restrict(
    AnswerJudgePromptQuestion,
    AnswerJudgeMultipleQuestionsPrompt
)

@OPERATOR_REGISTRY.register()
class BenchDatasetEvaluatorQuestion(OperatorABC):
    def __init__(self,
                eval_result_path: str = None,
                compare_method: Literal["match", "semantic"] = "match",
                system_prompt: str = "You are a helpful assistant specialized in evaluating answer correctness.",
                llm_serving: LLMServingABC = None,
                prompt_template: Union[AnswerJudgePromptQuestion, AnswerJudgeMultipleQuestionsPrompt, DIYPromptABC] = AnswerJudgePromptQuestion,
                support_subquestions: bool = False,
                keep_all_samples: bool = False,
                ):
        
        if eval_result_path is None:
            timestamp = int(time.time())
            eval_result_path = f"result_bencheval/BenchDatasetEvaluator_result_{timestamp}.json"
    
        self.eval_result_path = eval_result_path
        self.compare_method = compare_method
        self.empty_responses_count = 0  # 添加空响应计数器
        self.keep_all_samples = keep_all_samples
        self.support_subquestions = support_subquestions
        
        if compare_method == "match":
            self.compare = self.math_verify_compare
            unit_manager = UnitTextManager()
            string_cleaner = StringCleaner(unit_manager)
            self.answer_extractor = AnswerExtractor(string_cleaner)
        else:
            if prompt_template is None:
                prompt_template = AnswerJudgePromptQuestion() if not support_subquestions else AnswerJudgeMultipleQuestionsPrompt()
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
        # 检查空响应
        if not self.support_subquestions:
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
                    # 备用解析逻辑，检查响应中是否包含true或false
                    if "true" in response.lower():
                        result_value = "true"
                    else:
                        result_value = "false"
                if result_value == "true":
                    return True
                else:
                    return False
            except Exception as e:
                self.logger.error(f"Response format error: {response}. Error: {e}")
                return False
        
        if self.support_subquestions:
            # 如果支持子问题，假设response是一个列表, 返回正确的数量/总数
            correct_num = 0
            total_num = 0
            try:
                response = json5.loads(response, strict=False)  # 使用json5解析，允许更宽松的格式
                judgement = response.get("judgement", [])
            except Exception as e:
                self.logger.error(f"Response JSON parse error: {response}. Error: {e}")
                self.empty_responses_count += 1
                return "0/0"
            for resp in judgement:
                if isinstance(resp, bool): 
                    if resp is True:
                        correct_num += 1
                        total_num += 1
                    elif resp is False:
                        total_num += 1
                    elif resp.lower() == "empty":
                        continue  # 不计入总数
                elif isinstance(resp, str):
                    if resp.lower() == "true":
                        correct_num += 1
                        total_num += 1
                    elif resp.lower() == "false":
                        total_num += 1
                    elif resp.lower() == "empty":
                        continue  # 不计入总数
                    
            return f"{correct_num}/{total_num}"
            
    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "该算子用于对比预测答案与标准答案的匹配度，支持两种评估模式：\n\n"
                "1. 字符串匹配（match）：使用数学验证方法比较答案，适用于有明确答案的问题\n"
                "2. 语义匹配（semantic）：使用LLM评估答案的语义相似度，适用于开放性问题\n\n"
                "输入参数：\n"
                "- input_test_answer_key：预测答案字段名\n"
                "- input_gt_answer_key：标准答案字段名\n"
                "- input_question_key：问题字段名（语义匹配模式下必需）\n"
                "- compare_method：比较方法（match/semantic）\n"
                "- keep_all_samples：参考答案全部为空时是否保留输入行\n\n"
                "输出参数：\n"
                "- answer_match_result：匹配结果（True/False）\n"
                "- 统计结果将保存到指定的eval_result_path路径\n"
            )
        elif lang == "en":
            return (
                "This operator compares predicted answers against ground truth using two evaluation modes:\n\n"
                "1. String Matching (match): Uses mathematical verification to compare answers, suitable for questions with definitive answers\n"
                "2. Semantic Matching (semantic): Uses LLM to evaluate semantic similarity, suitable for open-ended questions\n\n"
                "Input Parameters:\n"
                "- input_test_answer_key: Predicted answer field\n"
                "- input_gt_answer_key: Ground truth field\n"
                "- input_question_key: Question field (required for semantic mode)\n"
                "- compare_method: Comparison method (match/semantic)\n"
                "- keep_all_samples: Preserve input rows when all reference answers are missing\n\n"
                "Output Parameters:\n"
                "- answer_match_result: Matching result (True/False)\n"
                "- Statistics will be saved to the specified eval_result_path\n"
            )
        else:
            return "BenchEvaluator performs answer validation using string matching or semantic comparison"
        
    def check_column(self, required_columns: list[str], dataframe: pd.DataFrame):
        for column in required_columns:
            if column not in dataframe.columns:
                self.logger.error(f"Required column '{column}' not found in dataframe")
                return False
        return True
            
    def statistic(self, file_name_prefix: str, dataframe: pd.DataFrame, compare_method: Literal["match", "semantic"]):
        total_samples = len(dataframe)
        valid_samples = len(dataframe) - self.empty_responses_count
        matched_samples = sum(dataframe['answer_match_result'])
        accuracy = matched_samples / valid_samples if valid_samples > 0 else 0
        
        # 创建统计信息字典
        stats = {
            "bench_name_or_prefix": file_name_prefix,
            "total_samples": total_samples,
            "valid_samples": valid_samples,
            "matched_samples": matched_samples,
            "accuracy": float(accuracy),  # 确保可以被JSON序列化
            "empty_responses_count": self.empty_responses_count,
            "compare_method": compare_method
        }
        
        if self.support_subquestions:
            total_subquestions = dataframe['total_subquestions'].sum()
            correct_subquestions = dataframe['correct_answer_num'].sum()
            subquestion_accuracy = correct_subquestions / total_subquestions if total_subquestions > 0 else 0
            stats.update({
                "total_subquestions": int(total_subquestions),
                "correct_subquestions": int(correct_subquestions),
                "subquestion_accuracy": float(subquestion_accuracy)
            })
        
        # 将字典转换为DataFrame
        stats_df = pd.DataFrame([stats])
        
        # 直接将统计信息写入到self.eval_result_path
        os.makedirs(os.path.dirname(self.eval_result_path), exist_ok=True)
        stats_df.to_json(self.eval_result_path, orient="records", force_ascii=False, indent=2)
        self.logger.success(f"Statistics saved to {self.eval_result_path}")
        
        return stats_df

    def _get_required_columns(
        self,
        input_test_answer_key: str,
        input_gt_answer_key: str,
        input_question_key: str,
    ) -> list[str]:
        required_columns = [input_test_answer_key, input_gt_answer_key]
        if self.compare_method == "semantic":
            required_columns.append(input_question_key)
        return required_columns

    def _run_match_evaluation(
        self,
        storage: DataFlowStorage,
        dataframe: pd.DataFrame,
        required_columns: list[str],
    ) -> list[str]:
        for row_index in dataframe.index:
            answer = dataframe.at[row_index, self.test_answer_key]
            ground_truth = dataframe.at[row_index, self.gt_answer_key]
            final_answer = self.answer_extractor.extract_answer(answer, None)
            dataframe.at[row_index, "answer_match_result"] = self.compare(
                final_answer,
                ground_truth,
            )

        storage.write(dataframe)
        self.statistic(storage.file_name_prefix, dataframe, self.compare_method)
        return required_columns + ["answer_match_result"]

    def _build_semantic_inputs(self, valid_rows: pd.DataFrame) -> list[str]:
        return [
            self.prompt_template.build_prompt(
                question=row[self.question_key],
                answer=row[self.test_answer_key],
                reference_answer=row[self.gt_answer_key],
            )
            for _, row in valid_rows.iterrows()
        ]

    def _handle_missing_reference_answers(
        self,
        storage: DataFlowStorage,
        dataframe: pd.DataFrame,
        required_columns: list[str],
        skipped_count: int,
    ) -> list[str]:
        self.logger.warning(
            "No valid samples with reference answers found. All samples skipped."
        )
        output_dataframe = (
            dataframe if self.keep_all_samples else dataframe.iloc[0:0].copy()
        )
        output_file = storage.write(output_dataframe)
        self.logger.info(
            f"Dataframe saved to {output_file}. Skipped {skipped_count} samples "
            "due to missing reference answers."
        )
        return required_columns + ["answer_match_result"]

    def _apply_subquestion_results(
        self,
        dataframe: pd.DataFrame,
        valid_rows: pd.DataFrame,
        responses: list[str],
        results: list[str],
    ) -> None:
        for row_index, response, result in zip(valid_rows.index, responses, results):
            correct_answer_count, total_subquestions = map(int, result.split("/"))
            dataframe.at[row_index, "correct_answer_num"] = correct_answer_count
            dataframe.at[row_index, "total_subquestions"] = total_subquestions
            dataframe.at[row_index, "answer_match_result"] = (
                correct_answer_count == total_subquestions and total_subquestions > 0
            )
            dataframe.at[row_index, "response_evaluation"] = response

    def _apply_semantic_results(
        self,
        dataframe: pd.DataFrame,
        valid_rows: pd.DataFrame,
        responses: list[str],
        results: list,
    ) -> None:
        if self.support_subquestions:
            self._apply_subquestion_results(dataframe, valid_rows, responses, results)
            return

        for row_index, result in zip(valid_rows.index, results):
            dataframe.at[row_index, "answer_match_result"] = result

    def _run_semantic_evaluation(
        self,
        storage: DataFlowStorage,
        dataframe: pd.DataFrame,
        required_columns: list[str],
    ) -> list[str]:
        empty_reference_mask = dataframe[self.gt_answer_key].isna() | (
            dataframe[self.gt_answer_key] == ""
        )
        valid_rows = dataframe[~empty_reference_mask]
        skipped_count = int(empty_reference_mask.sum())

        if valid_rows.empty:
            return self._handle_missing_reference_answers(
                storage,
                dataframe,
                required_columns,
                skipped_count,
            )

        inputs = self._build_semantic_inputs(valid_rows)
        responses = self.llm_serving.generate_from_input(
            user_inputs=inputs,
            system_prompt=self.system_prompt,
        )
        results = [self.ResolveResponse(response) for response in responses]
        self._apply_semantic_results(dataframe, valid_rows, responses, results)
        storage.write(dataframe)
        self.statistic(storage.file_name_prefix, dataframe, self.compare_method)
        self.empty_responses_count = 0
        return required_columns + ["answer_match_result"]

    def run(
            self,
            storage:DataFlowStorage,
            input_test_answer_key: str = "generated_cot",
            input_gt_answer_key: str = "golden_answer",
            input_question_key: str = None,
            ) -> list:

        self.test_answer_key = input_test_answer_key
        self.gt_answer_key = input_gt_answer_key
        self.question_key = input_question_key
        
        dataframe = storage.read("dataframe")
        required_columns = self._get_required_columns(
            input_test_answer_key,
            input_gt_answer_key,
            input_question_key,
        )
        if not self.check_column(required_columns, dataframe):
            return required_columns

        dataframe["answer_match_result"] = False
        if self.compare_method == "match":
            return self._run_match_evaluation(storage, dataframe, required_columns)
        return self._run_semantic_evaluation(storage, dataframe, required_columns)
