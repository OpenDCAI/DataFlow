from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
import pandas as pd


@OPERATOR_REGISTRY.register()
class ReasoningDifficultyDatasetEvaluator(OperatorABC):
    def __init__(self):
        self.logger = get_logger()
        self.logger.info(f'Initializing {self.__class__.__name__}...')
        self.logger.info(f'{self.__class__.__name__} initialized.')
        self.information_name = "Difficulty Information"

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "该算子用于统计数据集中的难度信息，计算不同难度级别的样本数量分布。"
                "它统计每个难度级别的样本数量，并返回难度分布的统计结果。\n"
                "输入参数：\n"
                "- input_diffulty_key：难度分数字段名，默认为'difficulty_score'\n"
                "输出参数：\n"
                "- 返回包含难度统计信息的字典，难度级别作为键，值为该难度级别的样本数量"
            )
        elif lang == "en":
            return (
                "This operator analyzes difficulty distribution in the dataset, calculating the number of samples at different difficulty levels. "
                "It counts samples at each difficulty level and returns statistical results of difficulty distribution.\n"
                "Input Parameters:\n"
                "- input_diffulty_key: Field name for difficulty score, default is 'difficulty_score'\n\n"
                "Output Parameters:\n"
                "- Returns a dictionary containing difficulty statistics, with difficulty levels as keys and sample counts as values"
            )
        else:
            return (
                "DifficultyInfo analyzes and reports the distribution of difficulty levels in the dataset."
            )
    
    def get_category_info(self, samples, input_diffulty_key="difficulty_score"):
        diffultys = [sample.get(input_diffulty_key, 'null') for sample in samples]
        diffultys_count = pd.Series(diffultys).value_counts().to_dict()
        self.logger.info(f"Difficulty information: {diffultys_count}")
        return diffultys_count



        
        
    
    def run(self,storage: DataFlowStorage, input_diffulty_key: str = "difficulty_score"):
        self.input_diffulty_key = input_diffulty_key
        dataframe = storage.read("dataframe")
        if self.input_diffulty_key not in dataframe.columns:
            self.logger.error(f"Input key {self.input_diffulty_key} not found in dataframe columns.")
            return {}
        samples = dataframe.to_dict(orient='records')
        category_info = self.get_category_info(samples, self.input_diffulty_key)
        return category_info
        