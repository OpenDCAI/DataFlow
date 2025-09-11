from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import LLMServingABC
from dataflow.prompts.reasoning.general import AnswerJudgePrompt
import re
import pandas as pd
import numpy as np

@OPERATOR_REGISTRY.register()
class ReasoningAnswerModelJudgeFilter(OperatorABC):
    def __init__(self,
                 system_prompt: str = "You are a helpful assistant specialized in evaluating answer correctness.",
                 llm_serving: LLMServingABC = None,
                 prompt_template = None,
                 keep_all_samples: bool = False,  # 新增参数，控制是否保留所有样本
                 ):

        self.logger = get_logger()
        
        if prompt_template is None:
            prompt_template = AnswerJudgePrompt()
        self.prompt_template = prompt_template
        self.system_prompt = system_prompt
        self.llm_serving = llm_serving
        self.empty_responses_count = 0  # 添加空响应计数器
        self.keep_all_samples = keep_all_samples  # 保存参数
        
    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "该算子用于对答案进行正确性评判，通过比较当前答案与参考答案的语义一致性，判断答案是否正确。"
                "调用大语言模型进行语义理解和判断，最终返回每个答案是否正确的二分类结果。\n"
                "输入参数：\n"
                "- system_prompt：系统提示词，用于定义模型行为\n"
                "- llm_serving：LLM服务对象，需实现LLMServingABC接口\n"
                "- prompt_template：提示模板对象，用于构建评判提示词\n"
                "- keep_all_samples：是否保留所有样本，默认为False（仅保留正确答案）\n"
                "- question_key：问题字段名，默认为'question'\n"
                "- answer_key：当前答案字段名，默认为'answer'\n"
                "- reference_key：参考答案字段名，默认为'reference_answer'\n"
                "输出参数：\n"
                "- DataFrame，包含原始数据和判断结果（answer_match_result字段）\n"
                "- 如果keep_all_samples为False，则仅保留判断结果为True的行\n"
                "- 返回包含输入字段名的列表，用于后续算子引用"
            )
        elif lang == "en":
            return (
                "This operator evaluates the correctness of answers by comparing the semantic consistency between "
                "the current answer and the reference answer. It uses a large language model for semantic understanding "
                "and judgment, ultimately returning a binary classification result for each answer.\n"
                "Input Parameters:\n"
                "- system_prompt: System prompt to define model behavior\n"
                "- llm_serving: LLM serving object implementing LLMServingABC interface\n"
                "- prompt_template: Prompt template object for constructing evaluation prompts\n"
                "- keep_all_samples: Whether to keep all samples, default is False (only keep correct answers)\n"
                "- question_key: Field name for questions, default is 'question'\n"
                "- answer_key: Field name for current answers, default is 'answer'\n"
                "- reference_key: Field name for reference answers, default is 'reference_answer'\n\n"
                "Output Parameters:\n"
                "- DataFrame containing original data and judgment results (answer_match_result field)\n"
                "- If keep_all_samples is False, only rows with True judgment results are retained\n"
                "- List containing input field names for subsequent operator reference"
            )
        else:
            return (
                "AnswerJudge evaluates answer correctness by comparing semantic consistency with reference answers using LLM."
            )
    
    def ResolveResponse(self, response):
        # 检查空响应
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
            
    def run(self, storage: DataFlowStorage, input_question_key: str = "question", input_answer_key: str = "answer", input_reference_key: str = "reference_answer") -> list:
        self.question_key = input_question_key
        self.answer_key = input_answer_key
        self.reference_key = input_reference_key
        
        dataframe = storage.read("dataframe")
        
        # 检查必要的列是否存在
        required_columns = [input_question_key, input_answer_key, input_reference_key]
        for column in required_columns:
            if column not in dataframe.columns:
                self.logger.error(f"Required column '{column}' not found in dataframe")
                return required_columns
        
        # 检查参考答案是否为空或不存在
        empty_reference_mask = dataframe[input_reference_key].isna() | (dataframe[input_reference_key] == '')
        skipped_rows = dataframe[empty_reference_mask]
        valid_rows = dataframe[~empty_reference_mask]
        
        # 记录跳过的行数
        skipped_count = len(skipped_rows)
        
        # 初始化结果列，默认所有行为False
        dataframe['answer_match_result'] = False
        
        if len(valid_rows) == 0:
            self.logger.warning("No valid samples with reference answers found. All samples skipped.")
            if self.keep_all_samples:
                output_file = storage.write(dataframe)  # 保留所有行，但answer_match_result都为False
            else:
                output_file = storage.write(pd.DataFrame(columns=dataframe.columns))  # 不保留任何行
            self.logger.info(f"Dataframe saved to {output_file}. Skipped {skipped_count} samples due to missing reference answers.")
            return required_columns + ['answer_match_result']
        
        # 只对有参考答案的行构建提示词并调用LLM
        inputs = [self.prompt_template.build_prompt(
            question=row[input_question_key],
            answer=row[input_answer_key],
            reference_answer=row[input_reference_key]
        ) for _, row in valid_rows.iterrows()]
        
        responses = self.llm_serving.generate_from_input(user_inputs=inputs, system_prompt=self.system_prompt)
        results = [self.ResolveResponse(response) for response in responses]
        
        # 创建结果掩码，与valid_rows长度相同
        result_mask = np.array(results, dtype=bool)
        
        # 更新有效行的answer_match_result
        valid_indices = valid_rows.index
        for i, idx in enumerate(valid_indices):
            dataframe.at[idx, 'answer_match_result'] = results[i]
        
        # 根据keep_all_samples决定是否保留所有样本
        if self.keep_all_samples:
            # 保留所有样本，包括不匹配的和没有参考答案的
            final_dataframe = dataframe
        else:
            # 只保留匹配的样本
            final_dataframe = dataframe[dataframe['answer_match_result'] == True]
        
        output_file = storage.write(final_dataframe)
        
        # 记录统计信息
        total_samples = len(dataframe)
        valid_samples = len(valid_rows)
        matched_samples = sum(results)
        accuracy = matched_samples / valid_samples if valid_samples > 0 else 0
        
        self.logger.info(f"Processed answers saved to {output_file}.")
        self.logger.info(f"Total samples: {total_samples}, Valid samples: {valid_samples}, Skipped samples: {skipped_count}")
        self.logger.info(f"Matched answers: {matched_samples}, Accuracy: {accuracy:.2%}")
        self.logger.info(f"Output samples: {len(final_dataframe)}")
        
        # 记录空响应数量并重置计数器
        if self.empty_responses_count > 0:
            self.logger.error(f"Found {self.empty_responses_count} empty responses during evaluation.")
        self.empty_responses_count = 0
        
        return required_columns + ['answer_match_result']