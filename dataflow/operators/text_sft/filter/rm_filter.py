import numpy as np
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.operators.text_sft import RMSampleEvaluator

@OPERATOR_REGISTRY.register()
class RMFilter(OperatorABC):

    def __init__(self, min_score: float = 0.2, max_score: float = 0.8, device='cuda', model_cache_dir='./dataflow_cache'):
        self.logger = get_logger()
        self.min_score = min_score
        self.max_score = max_score
        self.scorer = RMSampleEvaluator(device=device, model_cache_dir=model_cache_dir)
        self.logger.info(f"Initializing {self.__class__.__name__} with min_score = {self.min_score}, max_score = {self.max_score}")
    
    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "基于RMScorer打分器的得分对数据进行过滤。使用基于人类偏好数据训练的奖励模型对文本质量进行评分，高分代表质量较高。\n"
                "奖励模型能够评估文本的相关性、有用性、无害性等人类偏好指标，可用于筛选符合人类价值观的高质量文本。\n"
                "输入参数：\n"
                "- min_score：保留样本的最小奖励分数阈值，默认为0.2\n"
                "- max_score：保留样本的最大奖励分数阈值，默认为0.8\n"
                "- device：模型运行设备，默认为'cuda'\n"
                "- model_cache_dir：模型缓存目录，默认为'./dataflow_cache'\n"
                "- input_instruction_key：指令字段名，默认为'instruction'\n"
                "- input_output_key：输出字段名，默认为'output'\n"
                "输出参数：\n"
                "- 过滤后的DataFrame，仅保留奖励分数在[min_score, max_score]范围内的样本\n"
                "- 返回包含奖励分数字段名的列表，用于后续算子引用"
            )
        elif lang == "en":
            return (
                "Filter data using scores from the RMScorer. Quality scoring using reward model trained with human preference data, where higher scores indicate better quality.\n"
                "Reward model evaluates human preference metrics such as relevance, helpfulness, and harmlessness, useful for filtering high-quality text aligned with human values.\n"
                "Input Parameters:\n"
                "- min_score: Minimum reward score threshold for retaining samples, default is 0.2\n"
                "- max_score: Maximum reward score threshold for retaining samples, default is 0.8\n"
                "- device: Model running device, default is 'cuda'\n"
                "- model_cache_dir: Model cache directory, default is './dataflow_cache'\n"
                "- input_instruction_key: Instruction field name, default is 'instruction'\n"
                "- input_output_key: Output field name, default is 'output'\n"
                "Output Parameters:\n"
                "- Filtered DataFrame containing only samples with reward scores within [min_score, max_score] range\n"
                "- List containing reward score field name for subsequent operator reference"
            )
        else:
            return "Filter data based on quality scores from human preference-trained reward model."
        
    def run(self, storage: DataFlowStorage, input_instruction_key: str = 'instruction', input_output_key: str = 'output', output_key: str = 'RMScore'):
        self.input_instruction_key = input_instruction_key
        self.input_output_key = input_output_key
        self.output_key = output_key
        dataframe = storage.read("dataframe")
        self.logger.info(f"Running {self.__class__.__name__} with input_instruction_key = {self.input_instruction_key}, intput_output_key = {self.input_output_key}, output_key = {self.output_key}...")
        scores = np.array(self.scorer.eval(dataframe, self.input_instruction_key, self.input_output_key))
        dataframe[self.output_key] = scores
        filtered_dataframe = dataframe[(scores >= self.min_score) & (scores <= self.max_score)]
        storage.write(filtered_dataframe)
        self.logger.info(f"Filtering completed. Total records passing filter: {len(filtered_dataframe)}.")

        return [self.output_key]