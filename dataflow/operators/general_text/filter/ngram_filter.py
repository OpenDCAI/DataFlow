from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.operators.general_text import NgramSampleEvaluator
from typing import Literal  # 记得在文件顶部导入

@OPERATOR_REGISTRY.register()
class NgramFilter(OperatorABC):

    def __init__(self, min_score=0.8, max_score=1, ngrams=5, language: Literal['zh', 'en', 'auto'] = 'en'):
        if language not in ['zh', 'en', 'auto']:
            raise ValueError(f"Unsupported language: '{language}'. Supported options are: ['zh', 'en', 'auto'].")
        self.logger = get_logger()
        self.min_score = min_score
        self.max_score = max_score
        self.scorer = NgramSampleEvaluator(ngrams, language)
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
                "- language：处理语言；'zh' 使用字粒度切分，'en' 使用空格分词，'auto' 根据每条文本是否包含汉字自动选择\n"
                "- input_key：输入文本字段名（必填）\n"
                "- output_key：写入n-gram得分的输出字段名，默认为'NgramScore'\n"
                "输出参数：\n"
                "- 过滤后的DataFrame，仅保留n-gram得分在[min_score, max_score]范围内（含边界）的文本；"
                "每行新增output_key字段，值为0到1之间的浮点得分（唯一n-gram数/n-gram总数；空文本或token数少于ngrams时为0.0），该字段保留在输出中\n"
                "- 返回包含output_key的列表"
            )
        else:
            return (
                "Filter data using scores from the NgramScorer. Evaluate text redundancy via n-gram repetition ratio; higher score means lower repetition and less text redundancy.\n"
                "Input Parameters:\n"
                "- min_score: Minimum n-gram score threshold\n"
                "- max_score: Maximum n-gram score threshold\n"
                "- ngrams: n value for n-gram\n"
                "- language: Processing language. 'zh' uses character-level splitting, 'en' uses whitespace splitting, and 'auto' selects per sample based on the presence of Han characters.\n"
                "- input_key: Input text field name (required)\n"
                "- output_key: Name of the output field that receives the n-gram score, default is 'NgramScore'\n\n"
                "Output Parameters:\n"
                "- Filtered DataFrame containing only texts with n-gram score within [min_score, max_score] (inclusive); "
                "each row gets a new output_key field holding a float score in 0-1 (unique n-grams / total n-grams; 0.0 for empty text or fewer tokens than ngrams), which is kept in the output\n"
                "- List containing output_key"
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
