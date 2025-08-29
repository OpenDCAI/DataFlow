import pandas as pd
from tqdm import tqdm
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from math_verify import parse, verify, LatexExtractionConfig

@OPERATOR_REGISTRY.register()
class AnswerJudger_MathVerify(OperatorABC):
    def __init__(self, config: dict):
        
        self.logger = get_logger()

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "该算子通过符号计算验证答案正确性，执行数学表达式解析和等价性验证。\n\n"
                "输入参数：\n"
                "- answer_key：待验证答案字段名\n"
                "- gt_key：标准答案字段名\n"
                "- tolerance：数值容差阈值\n"
                "- symbolic_check：是否启用符号验证\n\n"
                "输出参数：\n"
                "- result_key：验证结果字段（True/False）"
            )
        elif lang == "en":
            return (
                "This operator verifies answer correctness through symbolic computation, "
                "performing mathematical expression parsing and equivalence checking.\n\n"
                "Input Parameters:\n"
                "- answer_key: Answer field to verify\n"
                "- gt_key: Ground truth field name\n"
                "- tolerance: Numerical tolerance threshold\n"
                "- symbolic_check: Enable symbolic verification\n\n"
                "Output Parameters:\n"
                "- result_key: Verification result field (True/False)"
            )
        else:
            return "AnswerJudger_MathVerify validates mathematical answer correctness."

    def _validate_dataframe(self, dataframe: pd.DataFrame):
        required_keys = [self.input_key, self.answer_key, self.gt_key]
        forbidden_keys = []

        missing = [k for k in required_keys if k not in dataframe.columns]
        conflict = [k for k in forbidden_keys if k in dataframe.columns]

        if missing:
            self.logger.error(f"Missing required column(s): {missing}")
        if conflict:
            self.logger.error(f"The following column(s) already exist and would be overwritten: {conflict}")
        missing_keys = [key for key in required_keys if key not in dataframe.columns]

        if missing_keys:
            self.logger.error(f"The following required columns are missing from the dataframe: {missing_keys}")

    def run(
            self,
            storage:DataFlowStorage,
            input_key: str = "instruction",
            answer_key: str = "student_answer",
            gt_key: str = "correct_answer",
            result_key: str = "result",
            ) -> list:
        
        self.input_key = input_key
        self.answer_key = answer_key
        self.gt_key = gt_key
        self.result_key = result_key
        
        dataframe = storage.read("dataframe")
        self.logger.info(f"Found {len(dataframe)} rows in the dataframe")
        self._validate_dataframe(dataframe)

        results = []
        for answer, gt in tqdm(zip(dataframe[self.answer_key], dataframe[self.gt_key]), total=len(dataframe), desc='processed'):
            results.append(float(verify(parse(answer), parse(gt))) > 0)
        dataframe[self.result_key] = results

        output_file = storage.write(dataframe)
        self.logger.info(f"Results saved to {self.output_file}")

        return [input_key, answer_key, gt_key, result_key]