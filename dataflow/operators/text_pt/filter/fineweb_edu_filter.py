from dataflow.operators.text_pt import FineWebEduSampleEvaluator
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.utils import get_logger

@OPERATOR_REGISTRY.register()
class FineWebEduFilter(OperatorABC):
    def __init__(self, min_score: float = 2.5, max_score: float = 10000, model_cache_dir: str = './dataflow_cache', device: str = 'cuda'):
        self.min_score = min_score
        self.max_score = max_score
        self.logger = get_logger()
        self.scorer = FineWebEduSampleEvaluator(model_cache_dir=model_cache_dir, device=device)
        self.filter_name = 'FineWebEduFilter'
        self.logger.info(f"Initializing {self.filter_name} with min_score = {self.min_score}, max_score = {self.max_score}, "
                         f"device = {device}, model_cache_dir = {model_cache_dir}")

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "基于FineWebEduScorer打分器的得分对数据进行过滤。Fineweb-Edu是一个用于评估文本教育价值的分类器。\n\n"
                "初始化参数：\n"
                "- min_score: 最低分数阈值，默认为2.5\n"
                "- max_score: 最高分数阈值，默认为10000\n"
                "- model_cache_dir: 模型缓存目录，默认为'./dataflow_cache'\n"
                "- device: 运行设备，默认为'cuda'\n\n"
                "运行参数：\n"
                "- input_key: 输入文本字段名\n"
                "- output_key: 输出分数字段名，默认为'FinewebEduScore'\n\n"
                "评分标准：0-5分，分数越高表示文本具有越高的教育价值\n"
                "过滤逻辑：保留分数在[min_score, max_score]范围内的数据"
            )
        else:
            return (
                "Filter data using scores from the FineWebEduScorer. Fineweb-Edu is a classifier for evaluating educational value of text.\n\n"
                "Initialization Parameters:\n"
                "- min_score: Minimum score threshold, default is 2.5\n"
                "- max_score: Maximum score threshold, default is 10000\n"
                "- model_cache_dir: Model cache directory, default is './dataflow_cache'\n"
                "- device: Running device, default is 'cuda'\n\n"
                "Run Parameters:\n"
                "- input_key: Input text field name\n"
                "- output_key: Output score field name, default is 'FinewebEduScore'\n\n"
                "Scoring Standard: 0-5 points, higher score indicates more educational content\n"
                "Filter Logic: Keep data with scores in [min_score, max_score] range"
            )

    def run(self, storage: DataFlowStorage, input_key: str, output_key: str='FinewebEduScore'):
        self.input_key = input_key
        self.output_key = output_key
        dataframe = storage.read("dataframe")
        self.logger.info(f"Running {self.filter_name}...")
        scores = self.scorer.eval(dataframe, input_key)
        dataframe[self.output_key] = scores
        filtered_dataframe = dataframe[(scores >= self.min_score) & (scores <= self.max_score)]
        storage.write(filtered_dataframe)
        self.logger.info(f"Filtering completed. Total records passing filter: {len(filtered_dataframe)}.")
        return [self.output_key]
