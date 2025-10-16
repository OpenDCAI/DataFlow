from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.operators.general_text import NgramSampleEvaluator

@OPERATOR_REGISTRY.register()
class NgramFilter(OperatorABC):

    def __init__(self, min_score=0.8, max_score=1, ngrams=5):
        self.logger = get_logger()
        self.min_score = min_score
        self.max_score = max_score
        self.scorer = NgramSampleEvaluator(ngrams)
        self.logger.info(f"Initializing {self.__class__.__name__} with min_scores: {self.min_score} and max_scores: {self.max_score}...")  
    
    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "基于NgramScorer打分器的得分对数据进行过滤。计算文本中n-gram的重复比例，得分越高表示重复比例越低，文本冗余度越小。\n"
                "输入参数：\n"
                "- min_score：最小n-gram得分阈值\n"
                "- max_score：最大n-gram得分阈值\n"
                "- ngrams：n-gram的n值\n"
                "输出参数：\n"
                "- 过滤后的DataFrame，仅保留n-gram得分在指定范围内的文本\n"
                "- 返回包含n-gram得分字段名的列表"
            )
        else:
            return (
                "Filter data using scores from the NgramScorer. Evaluate text redundancy via n-gram repetition ratio; higher score means lower repetition and less text redundancy.\n"
                "Input Parameters:\n"
                "- min_score: Minimum n-gram score threshold\n"
                "- max_score: Maximum n-gram score threshold\n"
                "- ngrams: n value for n-gram\n\n"
                "Output Parameters:\n"
                "- Filtered DataFrame containing only texts with n-gram score within specified range\n"
                "- List containing n-gram score field name"
            )

    def run(self, storage: DataFlowStorage, input_key: str, output_key: str='NgramScore'):
        self.input_key = input_key
        self.output_key = output_key
        self.logger.info(f"Running {self.__class__.__name__} with input_key: {self.input_key} and output_key: {self.output_key}...")
        dataframe = storage.read("dataframe")
        scores = self.scorer.eval(dataframe, self.input_key)
        dataframe[self.output_key] = scores
        filtered_dataframe = dataframe[(dataframe[self.output_key] >= self.min_score) & (dataframe[self.output_key] <= self.max_score)]
        output_file = storage.write(filtered_dataframe)
        self.logger.info(f"Filtering completed. Total records passing filter: {len(filtered_dataframe)}.")
        return [self.output_key]
        
        