from dataflow.operators.eval import InstagScorer
from dataflow.core import OperatorABC
import numpy as np
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.utils import get_logger
from dataflow.utils.storage import DataFlowStorage

@OPERATOR_REGISTRY.register()
class InstagFilter(OperatorABC):

    def __init__(self, min_score=0.0, max_score=1.0, model_cache_dir='./dataflow_cache', device='cuda', max_new_tokens=1024):
        self.logger = get_logger()
        self.min_score = min_score
        self.max_score = max_score
        
        # Initialize the scorer
        self.scorer = InstagScorer(
            model_cache_dir=model_cache_dir,
            device=device,
            max_new_tokens=max_new_tokens
        )
        self.logger.info(f"Initializing {self.__class__.__name__} with min_score = {self.min_score} and max_score = {self.max_score}...")

    @staticmethod
    def get_desc(lang: str = "zh"):
        return "使用Instag评分器过滤掉低标签数量数据" if lang == "zh" else "Filter out data with low tag counts using the Instag scorer."

    def run(self, storage: DataFlowStorage, input_instruction_key: str = 'instruction', output_key: str = 'InstagScore'):
        self.input_instruction_key = input_instruction_key
        self.output_key = output_key
        dataframe = storage.read("dataframe")
        self.logger.info(f"Running {self.__class__.__name__}...")

        scores = self.scorer.eval(dataframe, self.input_instruction_key)
        dataframe[self.output_key] = scores
        filtered_dataframe = dataframe[(dataframe[self.output_key] >= self.min_score) & (dataframe[self.output_key] <= self.max_score)]
        storage.write(filtered_dataframe)

        self.logger.info(f"Filtering completed. Total records passing filter: {len(filtered_dataframe)}.")

        return [self.output_key]
