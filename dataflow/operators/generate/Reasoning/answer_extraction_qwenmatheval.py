import pandas as pd
from tqdm import tqdm
import logging
import re
from word2number import w2n
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.reasoning.AnswerExtraction import StringCleaner, UnitTextManager, AnswerExtractor

# The main class to manage the entire extraction process
@OPERATOR_REGISTRY.register()
class AnswerExtraction_QwenMathEval(OperatorABC):
    """
    A class to handle the process of extracting answers from a dataset.
    """

    def __init__(self, dataset_name:str = None):
        """
        Initializes the AnswerExtraction_QwenMathEval class.
        """
        self.logger = get_logger()
        self.data_name = dataset_name
        # Initialize helpers
        unit_manager = UnitTextManager()
        string_cleaner = StringCleaner(unit_manager)
        self.answer_extractor = AnswerExtractor(string_cleaner)

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "该算子用于从数学问题回答中提取规范化答案表达式，进行字符串清洗、单位处理和格式标准化。\n\n"
                "输入参数：\n"
                "- input_key：输入数据字段名\n"
                "- answer_key：原始答案字段名\n"
                "- output_key：处理后的答案字段名\n"
                "- unit_texts：需要过滤的单位文本列表\n\n"
                "输出参数：\n"
                "- output_key：标准化后的数学表达式字段"
            )
        elif lang == "en":
            return (
                "This operator extracts and normalizes mathematical expressions from answers, "
                "performing string cleaning, unit processing and format standardization.\n\n"
                "Input Parameters:\n"
                "- input_key: Input data field name\n"
                "- answer_key: Raw answer field name\n"
                "- output_key: Processed answer field name\n"
                "- unit_texts: List of unit texts to filter\n\n"
                "Output Parameters:\n"
                "- output_key: Standardized mathematical expression field"
            )
        else:
            return "AnswerExtraction_QwenMathEval performs mathematical answer normalization and standardization."

    def run(self, storage: DataFlowStorage, response_key:str = "pseudo_correct_solution_example", extraction_key:str = "extraction"):
        """
        Executes the answer extraction process.
        """
        self.response_key, self.extraction_key = response_key, extraction_key
        raw_dataframe = storage.read("dataframe")
        key_list = raw_dataframe.columns.to_list()
        if self.response_key not in key_list:
            raise ValueError(f"response_key: {self.response_key} not found in dataframe columns.")

        self.logger.info(f"Found {len(raw_dataframe)} rows.")
        extractions = [
            self.answer_extractor.extract_answer(resp, self.data_name)
            for resp in tqdm(raw_dataframe[self.response_key], desc='Processing')
        ]
        raw_dataframe[self.extraction_key] = extractions

        output_file = storage.write(raw_dataframe)
        self.logger.info(f"Extracted answers saved to {output_file}")
        
        return [extraction_key]
