import numpy as np
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.operators.text_pt import PerplexitySampleEvaluator

@OPERATOR_REGISTRY.register()
class PerplexityFilter(OperatorABC):
    def __init__(self, min_score: float = 10.0, max_score: float = 500.0, model_name: str = 'gpt2', device='cuda'):
        self.logger = get_logger()
        self.min_score = min_score
        self.max_score = max_score
        self.scorer = PerplexitySampleEvaluator(model_name=model_name, device=device)
        self.logger.info(f"Initializing {self.__class__.__name__} with min_score = {self.min_score} and max_score = {self.max_score}")
    
    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "基于PerplexityScorer打分器的得分对数据进行过滤。基于Huggingface模型计算文本的困惑度，困惑度越低，文本的流畅性和可理解性越高。\n"
                "输入参数：\n"
                "- min_score：最小困惑度阈值\n"
                "- max_score：最大困惑度阈值\n"
                "- model_name：Huggingface模型路径或名称\n"
                "- device：模型运行设备\n"
                "输出参数：\n"
                "- 过滤后的DataFrame，仅保留困惑度在指定范围内的文本\n"
                "- 返回包含困惑度得分字段名的列表"
            )
        else:
            return (
                "Filter data using scores from the PerplexityScorer. Uses Huggingface model to calculate text perplexity; lower scores indicate better fluency and understandability.\n"
                "Input Parameters:\n"
                "- min_score: Minimum perplexity threshold\n"
                "- max_score: Maximum perplexity threshold\n"
                "- model_name: Huggingface model path or name\n"
                "- device: Model device\n\n"
                "Output Parameters:\n"
                "- Filtered DataFrame containing only texts with perplexity within specified range\n"
                "- List containing perplexity score field name"
            )
        
    def run(self, storage: DataFlowStorage, input_key: str, output_key: str = 'PerplexityScore'):
        self.input_key = input_key
        self.output_key = output_key
        dataframe = storage.read("dataframe")
        self.logger.info(f"Running {self.__class__.__name__} with input_key = {self.input_key} and output_key = {self.output_key}...")

        # 获取得分并过滤
        scores = np.array(self.scorer.eval(dataframe, self.input_key))
        dataframe[self.output_key] = scores
        filtered_dataframe = dataframe[(scores >= self.min_score) & (scores <= self.max_score)]

        # 将过滤后的数据框写回存储
        storage.write(filtered_dataframe)

        self.logger.info(f"Filtering completed. Total records passing filter: {len(filtered_dataframe)}.")

        return [self.output_key]
