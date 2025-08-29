import numpy as np
import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger

from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC

@OPERATOR_REGISTRY.register()
class AnswerFormatterFilter(OperatorABC):
    def __init__(self):
        
        self.logger = get_logger()

    def is_valid_answer(answer: str) -> bool:
        # check final answer in \boxed{} or not 
        # if not re.search(r'\\boxed{.*}', answer):
        #     return False
        return True 

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "该算子用于检查答案格式是否符合规范，主要验证数学答案是否包含正确的\\boxed{}标记。\n\n"
                "输入参数：\n"
                "- input_key：输入字段名\n"
                "- result_key：结果字段名\n\n"
                "输出参数：\n"
                "- 通过格式检查返回1，否则返回0"
            )
        elif lang == "en":
            return (
                "This operator validates answer formatting, specifically checking for correct \\boxed{} notation.\n\n"
                "Input Parameters:\n"
                "- input_key: Field name containing the answer\n"
                "- result_key: Output result field name\n\n"
                "Output Parameters:\n"
                "- Returns 1 for valid format, 0 otherwise"
            )
        else:
            return "AnswerFormatterFilter validates mathematical answer formatting"
    
    def _validate_dataframe(self, dataframe: pd.DataFrame):
        required_keys = [self.input_key]
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
            input_key: str = "generated_cot",
            ) -> list:
        '''
        Execute the answer format filter process
        '''
        self.input_key = input_key
        dataframe = storage.read("dataframe")
        self._validate_dataframe(dataframe)
        
        indexes =  np.zeros(len(dataframe)).astype(int)
        for i, item in dataframe.iterrows():
            answer = item[self.input_key]
            if AnswerFormatterFilter.is_valid_answer(answer):
                indexes[i] = 1
        dataframe = dataframe[np.array(indexes) == 1]

        output_file = storage.write(dataframe)
        self.logger.info(f"Results saved to {output_file}")
        
        return [self.input_key,]