import pandas as pd
import numpy as np
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.operators.general_text import LexicalDiversitySampleEvaluator

@OPERATOR_REGISTRY.register()
class LexicalDiversityFilter(OperatorABC):
    def __init__(self, min_scores: dict = {'mtld': 50, 'hdd': 0.8}, max_scores: dict = {'mtld': 99999, 'hdd': 1.0}):
        
        self.min_scores = min_scores
        self.max_scores = max_scores
        if not self.min_scores.keys() == self.max_scores.keys():
            raise ValueError("min_scores and max_scores must have the same keys")
        self.logger = get_logger()
        self.logger.info(f"Initializing {self.__class__.__name__} with min_scores: {self.min_scores} and max_scores: {self.max_scores}...")  
        self.metric_name_map = {
            'hdd': 'LexicalDiversityHD-DScore',
            'mtld': 'LexicalDiversityMTLDScore',
        }
        self.scorer = LexicalDiversitySampleEvaluator()
    
    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "基于LexicalDiversityScorer打分器的得分对数据进行过滤。使用MTLD（移动平均类型-令牌比）和HDD（超几何分布多样性）两种方法计算词汇多样性，高分代表更丰富的词汇使用。\n"
                "输入参数：\n"
                "- min_scores：各指标的最小阈值字典，包含'mtld'和'hdd'\n"
                "- max_scores：各指标的最大阈值字典，包含'mtld'和'hdd'\n"
                "输出参数：\n"
                "- 过滤后的DataFrame，仅保留词汇多样性在指定范围内的文本\n"
                "- 返回包含各指标标签字段名的列表"
            )
        else:
            return (
                "Filter data using scores from the LexicalDiversityScorer. Measure lexical diversity using MTLD (Moving-Average Type-Token Ratio) and HDD (Hypergeometric Distribution Diversity) methods; higher scores indicate more diverse vocabulary usage.\n"
                "Input Parameters:\n"
                "- min_scores: Dictionary of minimum thresholds for each metric, containing 'mtld' and 'hdd'\n"
                "- max_scores: Dictionary of maximum thresholds for each metric, containing 'mtld' and 'hdd'\n\n"
                "Output Parameters:\n"
                "- Filtered DataFrame containing only texts with lexical diversity within specified range\n"
                "- List containing label field names for each metric"
            )
        
    def run(self, storage: DataFlowStorage, input_key: str, output_keys = ['mtld', 'hdd']):
        self.input_key = input_key
        self.output_keys = output_keys
        if not list(self.min_scores.keys()) == output_keys:
            raise ValueError("min_scores and output_keys must have the same keys")  
        self.logger.info(f"Running {self.__class__.__name__} with input_key: {self.input_key} and output_keys: {self.output_keys}...")
        dataframe = storage.read("dataframe")
        scores = self.scorer.eval(dataframe, self.input_key)
        results = np.ones(len(dataframe), dtype=int)
        for _label in self.output_keys:
            min_score = self.min_scores[_label]
            max_score = self.max_scores[_label]
            label = self.metric_name_map[_label]
            dataframe[label] = pd.DataFrame(scores)[label]
            metric_scores = np.array(dataframe[label])
            metric_filter = (min_score <= metric_scores) & (metric_scores <= max_score)
            nan_filter = np.isnan(metric_scores)
            metric_filter = metric_filter | nan_filter    
            results = results & metric_filter.astype(int)
            self.logger.debug(f"Filtered by {_label}, {np.sum(results)} data remained")
            dataframe[f"{label}_label"] = metric_filter.astype(int)
        filtered_dataframe = dataframe[results == 1]
        output_file = storage.write(filtered_dataframe)
        self.logger.info(f"Filtering completed. Total records passing filter: {len(filtered_dataframe)}.")
        return [f"{label}_label" for label in self.output_keys]
