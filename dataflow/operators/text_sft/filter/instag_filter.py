from dataflow.operators.text_sft import InstagSampleEvaluator
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
        self.scorer = InstagSampleEvaluator(
            model_cache_dir=model_cache_dir,
            device=device,
            max_new_tokens=max_new_tokens
        )
        self.logger.info(f"Initializing {self.__class__.__name__} with min_score = {self.min_score} and max_score = {self.max_score}...")

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return "基于InstagScorer打分器的过滤算子。使用预训练的Instag模型对指令进行分析，返回标签的数量来评估指令的内容多样性。参数包括模型缓存目录(model_cache_dir)、计算设备(device)和最大新生成标记数(max_new_tokens)。过滤范围由min_score和max_score参数控制，标签越多表示内容多样性越大。"
        else:
            return "Filter operator based on InstagScorer. Uses pre-trained Instag model to analyze instructions, returning the number of tags to evaluate content diversity. Parameters include model cache directory (model_cache_dir), computing device (device), and maximum new tokens (max_new_tokens). Filter range is controlled by min_score and max_score parameters, with more tags indicating greater content diversity."

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
