import numpy as np
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.operators.general_text import PerspectiveSampleEvaluator
from dataflow.serving import PerspectiveAPIServing

@OPERATOR_REGISTRY.register()
class PerspectiveFilter(OperatorABC):
    def __init__(self, min_score: float = 0.0, max_score: float = 0.5):
        self.logger = get_logger()
        self.logger.info(f"Initializing {self.__class__.__name__} with min_score = {min_score} and max_score = {max_score}")
        self.min_score = min_score
        self.max_score = max_score
        self.serving = PerspectiveAPIServing(max_workers=10)
        self.scorer = PerspectiveSampleEvaluator(serving=self.serving)
    
    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "基于PerspectiveScorer打分器的得分对数据进行过滤使用Perspective API评估文本的毒性，返回毒性概率，得分越高表明文本毒性越高。\n"
                "输入参数：\n"
                "- min_score：最小毒性得分阈值\n"
                "- max_score：最大毒性得分阈值\n"
                "输出参数：\n"
                "- 过滤后的DataFrame，仅保留毒性得分在指定范围内的文本\n"
                "- 返回包含毒性得分字段名的列表"
            )
        else:
            return (
                "Filter data using scores from the PerspectiveScorer. Assess text toxicity using Perspective API; higher scores indicate more toxicity.\n"
                "Input Parameters:\n"
                "- min_score: Minimum toxicity score threshold\n"
                "- max_score: Maximum toxicity score threshold\n\n"
                "Output Parameters:\n"
                "- Filtered DataFrame containing only texts with toxicity score within specified range\n"
                "- List containing toxicity score field name"
            )
        
    def run(self, storage: DataFlowStorage, input_key: str, output_key: str = 'PerspectiveScore'):
        self.input_key = input_key
        self.output_key = output_key
        dataframe = storage.read("dataframe")
        self.logger.info(f"Running {self.__class__.__name__} with input_key = {self.input_key} and output_key = {self.output_key}...")

        # Get the scores for filtering
        scores = np.array(self.scorer.eval(dataframe, self.input_key))

        dataframe[self.output_key] = scores
        metric_filter = (scores >= self.min_score) & (scores <= self.max_score)
        nan_filter = np.isnan(scores)
        metric_filter = metric_filter | nan_filter    
        filtered_dataframe = dataframe[metric_filter]

        # Write the filtered dataframe back to storage
        storage.write(filtered_dataframe)

        self.logger.info(f"Filtering completed. Total records passing filter: {len(filtered_dataframe)}.")

        return [self.output_key]
