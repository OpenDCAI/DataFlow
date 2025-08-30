import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger

from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC
from dataflow.operators.core_text import PromptedEvaluator

@OPERATOR_REGISTRY.register()
class PromptedFilter(OperatorABC):
    '''
    Answer Generator is a class that generates answers for given questions.
    '''
    def __init__(self, llm_serving: LLMServingABC, system_prompt: str = "Please evaluate the quality of this data on a scale from 1 to 5."):
        self.logger = get_logger()
        self.llm_serving = llm_serving
        self.prompted_evaluator = PromptedEvaluator(llm_serving, system_prompt)
    
    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "PromptedFilter 使用内置的 PromptedEvaluator 对输入数据进行数值化打分，"
                "并根据指定的分数区间（min_score 到 max_score，闭区间）筛选出符合条件的样本。"
                "默认情况下打分范围是 1–5，但用户可以通过 system_prompt 自定义其他评分规则。\n"
                "\n输入参数：\n"
                "- llm_serving：LLM 服务对象，需实现 LLMServingABC 接口\n"
                "- system_prompt：系统提示词，定义评估规范（可选，默认 "
                "'Please evaluate the quality of this data on a scale from 1 to 5.'）\n"
                "- input_key：待评估文本所在列名（默认 'raw_content'）\n"
                "- output_key：写回打分结果的列名（默认 'eval'，若已存在将被覆盖）\n"
                "- min_score：筛选的最小分（默认 5）\n"
                "- max_score：筛选的最大分（默认 5）\n"
                "\n输出参数：\n"
                "- 过滤后的 DataFrame（仅保留分数位于 [min_score, max_score] 的行）\n"
                "- 返回 output_key 以供后续算子引用\n"
                "\n备注：\n"
                "- 默认打分区间是 1–5，但可根据实际 prompt 改变。"
            )
        elif lang == "en":
            return (
                "PromptedFilter leverages PromptedEvaluator to assign numeric scores to input data, "
                "and filters rows whose scores fall within [min_score, max_score] (inclusive). "
                "By default, the scoring scale is 1–5, but this can be customized through system_prompt.\n"
                "\nInput Parameters:\n"
                "- llm_serving: LLM serving object implementing LLMServingABC\n"
                "- system_prompt: System prompt defining the evaluation criteria "
                "(default: 'Please evaluate the quality of this data on a scale from 1 to 5.')\n"
                "- input_key: Column name containing the text to evaluate (default 'raw_content')\n"
                "- output_key: Column name to store the score (default 'eval'; overwritten if it exists)\n"
                "- min_score: Minimum score for filtering (default 5)\n"
                "- max_score: Maximum score for filtering (default 5)\n"
                "\nOutput:\n"
                "- Filtered DataFrame (rows with scores in [min_score, max_score])\n"
                "- Returns output_key for downstream operators\n"
                "\nNote:\n"
                "- Default scoring range is 1–5, but can vary depending on the system_prompt."
            )
        else:
            return "PromptedFilter scores rows via PromptedEvaluator and filters by a configurable score range (default 1–5)."


    def run(self, storage: DataFlowStorage, input_key: str = "raw_content", output_key: str = "eval", min_score = 5, max_score = 5):
        self.logger.info("Running PromptGenerator...")

        # Load the raw dataframe from the input file
        dataframe = storage.read('dataframe')
        self.logger.info(f"Loading, number of rows: {len(dataframe)}")

        # Create a list to hold all generated questions and answers
        generated_outputs = self.prompted_evaluator.eval(dataframe, input_key)

        # Add the generated content back to the dataframe
        dataframe[output_key] = generated_outputs
        filtered_dataframe = dataframe[(dataframe[output_key] >= min_score) & (dataframe[output_key] <= max_score)]
        # Save the updated dataframe to the output file
        output_file = storage.write(filtered_dataframe)
        return output_key
