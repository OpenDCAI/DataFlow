from dataflow.operators.text_pt import PairQualSampleEvaluator
import numpy as np
from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.utils import get_logger
from dataflow.utils.storage import DataFlowStorage

@OPERATOR_REGISTRY.register()
class PairQualFilter(OperatorABC):
    def __init__(self, min_score=0, max_score=10000, model_cache_dir='./dataflow_cache', lang='en'):
        self.logger = get_logger()
        self.min_score = min_score
        self.max_score = max_score
        
        self.scorer = PairQualSampleEvaluator(model_cache_dir=model_cache_dir, lang=lang)
        self.filter_name = 'PairQualFilter'

        self.logger.info(f"Initializing {self.filter_name} with min_score = {self.min_score}, max_score = {self.max_score}...")

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "基于PairQualScorer打分器的得分对数据进行过滤。基于BGE模型，使用GPT对文本成对比较打分后训练而成的双语文本质量评分器，得分越高表示质量越高。\n"
                "输入参数：\n"
                "- min_score：最小质量得分阈值（包含），默认为0\n"
                "- max_score：最大质量得分阈值（包含），默认为10000\n"
                "- model_cache_dir：模型缓存目录路径\n"
                "- lang：文本语言类型，'en'或'zh'，默认为'en'\n"
                "- input_key：输入文本字段名，无默认值，必须指定\n"
                "- output_key：写入质量得分的输出字段名，默认为'PairQualScore'\n"
                "输出参数：\n"
                "- 过滤后的DataFrame，仅保留 min_score <= 质量得分 <= max_score 的文本；"
                "每行新增output_key字段，值为该行的浮点质量得分，该字段保留在输出中\n"
                "- 返回包含output_key的列表"
            )
        else:
            return (
                "Filter data using scores from the PairQualScorer. Bilingual text quality scorer trained on GPT pairwise comparison annotations using BGE model; higher scores indicate better quality.\n"
                "Input Parameters:\n"
                "- min_score: Minimum quality score threshold (inclusive), default is 0\n"
                "- max_score: Maximum quality score threshold (inclusive), default is 10000\n"
                "- model_cache_dir: Model cache directory path\n"
                "- lang: Text language type, 'en' or 'zh', default is 'en'\n"
                "- input_key: Input text field name; no default, must be given\n"
                "- output_key: Name of the output field that receives the quality score, default is 'PairQualScore'\n\n"
                "Output Parameters:\n"
                "- Filtered DataFrame containing only texts with min_score <= quality score <= max_score; "
                "each row gets a new output_key field holding its float quality score, which is kept in the output\n"
                "- List containing output_key"
            )

    def eval(self, dataframe, input_key):
        self.logger.info(f"Start evaluating {self.filter_name}...")
        
        # Get the scores using the scorer
        scores = self.scorer.eval(dataframe, input_key)

        # Return the scores for filtering
        return np.array(scores)

    def run(self, storage: DataFlowStorage, input_key: str, output_key: str='PairQualScore'):
        self.input_key = input_key
        self.output_key = output_key
        dataframe = storage.read("dataframe")
        self.logger.info(f"Running {self.filter_name} with input_key = {self.input_key} and output_key = {self.output_key}...")
        scores = np.array(self.scorer.eval(dataframe, input_key))
        dataframe[self.output_key] = scores
        filtered_dataframe = dataframe[(scores >= self.min_score) & (scores <= self.max_score)]
        storage.write(filtered_dataframe)
        self.logger.info(f"Filtering completed. Total records passing filter: {len(filtered_dataframe)}.")
        return [self.output_key]
