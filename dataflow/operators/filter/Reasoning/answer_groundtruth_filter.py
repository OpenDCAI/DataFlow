from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.reasoning.AnswerExtraction import StringCleaner, UnitTextManager, AnswerExtractor
from dataflow.utils.storage import DataFlowStorage
from dataflow import get_logger
from dataflow.core import OperatorABC
from typing import Literal
from math_verify import parse, verify
import pandas as pd

@OPERATOR_REGISTRY.register()
class AnswerGroundTruthFilter(OperatorABC):
    def __init__(self,
                compare_method: Literal["math_verify", "exact"] = "math_verify"):
        
        name2compare = {
            'exact': self.exact_compare,
            'math_verify': self.math_verify_compare
        }
        self.compare = name2compare[compare_method]
        unit_manager = UnitTextManager()
        string_cleaner = StringCleaner(unit_manager)
        self.answer_extractor = AnswerExtractor(string_cleaner)
        
        self.logger = get_logger()

    def exact_compare(self, answer, ground_truth):
        return str(answer) == str(ground_truth)
    
    def math_verify_compare(self, answer, ground_truth):
        try:
            return verify(parse(str(ground_truth)), parse(str(answer)))
        except:
            try:
                return verify(parse(ground_truth), parse(answer))
            except:
                return False

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "该算子用于对比预测答案与标准答案的匹配度，支持精确匹配和数学验证两种方式。\n\n"
                "输入参数：\n"
                "- test_answer_key：预测答案字段名\n"
                "- gt_answer_key：标准答案字段名\n"
                "- compare_method：比较方法（exact/math_verify）\n\n"
                "输出参数：\n"
                "- 匹配成功返回1，否则返回0"
            )
        elif lang == "en":
            return (
                "This operator compares predicted answers against ground truth using exact or mathematical verification.\n\n"
                "Input Parameters:\n"
                "- test_answer_key: Predicted answer field\n"
                "- gt_answer_key: Ground truth field\n"
                "- compare_method: Comparison method (exact/math_verify)\n\n"
                "Output Parameters:\n"
                "- Returns 1 for matches, 0 otherwise"
            )
        else:
            return "AnswerGroundTruthFilter performs answer validation"
        
    def run(
            self,
            storage:DataFlowStorage,
            test_answer_key: str = "generated_cot",
            gt_answer_key: str = "golden_answer"
            ) -> list:
        
        self.test_answer_key = test_answer_key
        self.gt_answer_key = gt_answer_key
        
        dataframe = storage.read("dataframe")
        output = []
        answers = dataframe[self.test_answer_key]
        ground_truths = dataframe[self.gt_answer_key]
        for i in range(len(answers)):
            final_answer =  self.answer_extractor.extract_answer(answers[i], None)
            if self.compare(final_answer, ground_truths[i]):
                output.append(dataframe.iloc[i])
        output = pd.DataFrame(output)
        
        output_file = storage.write(output)
        self.logger.info(f"Filtered data saved to {output_file}")
        
        return [test_answer_key, gt_answer_key]