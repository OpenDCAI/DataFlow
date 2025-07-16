from dataflow.operators.eval import DeitaQualityScorer
from dataflow.core import OperatorABC
import numpy as np
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.utils import get_logger
from dataflow.utils.storage import DataFlowStorage

@OPERATOR_REGISTRY.register()
class DeitaQualityFilter(OperatorABC):
    def __init__(self, min_score=2.5, max_score=10000.0, device='cuda', model_cache_dir='./dataflow_cache', max_length=512):
        self.logger = get_logger()
        self.min_score = min_score
        self.max_score = max_score
        self.scorer = DeitaQualityScorer(
            device=device,
            model_cache_dir=model_cache_dir,
            max_length=max_length,
        )
        
        self.logger.info(f"Initializing {self.__class__.__name__} with min_score = {self.min_score} and max_score = {self.max_score}...")

    @staticmethod
    def get_desc(lang: str = "zh"):
        return "基于DeitaQualityScorer打分器的得分对数据进行过滤。基于 Llama 模型的 Deita 指令质量评估器，高分表示指令质量较高。" if lang == "zh" else "Filter data using scores from the DeitaQualityScorer. Evaluate instruction quality using Llama-based Deita model."

    def run(self, storage: DataFlowStorage, input_instruction_key: str = 'instruction', input_output_key : str = 'output', output_key: str = "DeitaQualityScore"):
        self.input_instruction_key = input_instruction_key
        self.input_output_key = input_output_key
        self.output_key = output_key
        dataframe = storage.read("dataframe")
        self.logger.info(f"Running {self.__class__.__name__}...")

        scores = self.scorer.eval(dataframe, input_instruction_key, input_output_key)
        dataframe[self.output_key] = scores
        filtered_dataframe = dataframe[(dataframe[self.output_key] >= self.min_score) & (dataframe[self.output_key] <= self.max_score)]
        storage.write(filtered_dataframe)
        self.logger.info(f"Filtering completed. Total records passing filter: {len(filtered_dataframe)}.")
        return [self.output_key]
