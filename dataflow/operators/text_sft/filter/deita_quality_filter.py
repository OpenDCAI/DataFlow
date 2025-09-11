from dataflow.operators.text_sft import DeitaQualitySampleEvaluator
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
        self.scorer = DeitaQualitySampleEvaluator(
            device=device,
            model_cache_dir=model_cache_dir,
            max_length=max_length,
        )
        
        self.logger.info(f"Initializing {self.__class__.__name__} with min_score = {self.min_score} and max_score = {self.max_score}...")

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "基于DeitaQualityScorer打分器的得分对数据进行过滤。使用基于Llama模型的Deita指令质量评估器，评估指令的质量高低。\n\n"
                "初始化参数：\n"
                "- min_score: 最低分数阈值，默认为2.5\n"
                "- max_score: 最高分数阈值，默认为10000.0\n"
                "- device: 运行设备，默认为'cuda'\n"
                "- model_cache_dir: 模型缓存目录，默认为'./dataflow_cache'\n"
                "- max_length: 最大序列长度，默认为512\n\n"
                "运行参数：\n"
                "- input_instruction_key: 输入指令字段名，默认为'instruction'\n"
                "- input_output_key: 输入输出字段名，默认为'output'\n"
                "- output_key: 输出分数字段名，默认为'DeitaQualityScore'\n\n"
                "评分标准：1-6分，分数越高表示指令质量越高\n"
                "过滤逻辑：保留分数在[min_score, max_score]范围内的数据"
            )
        else:
            return (
                "Filter data using scores from the DeitaQualityScorer. Evaluate instruction quality using Llama-based Deita model.\n\n"
                "Initialization Parameters:\n"
                "- min_score: Minimum score threshold, default is 2.5\n"
                "- max_score: Maximum score threshold, default is 10000.0\n"
                "- device: Running device, default is 'cuda'\n"
                "- model_cache_dir: Model cache directory, default is './dataflow_cache'\n"
                "- max_length: Maximum sequence length, default is 512\n\n"
                "Run Parameters:\n"
                "- input_instruction_key: Input instruction field name, default is 'instruction'\n"
                "- input_output_key: Input output field name, default is 'output'\n"
                "- output_key: Output score field name, default is 'DeitaQualityScore'\n\n"
                "Scoring Standard: 1-6 points, higher score indicates higher instruction quality\n"
                "Filter Logic: Keep data with scores in [min_score, max_score] range"
            )

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
