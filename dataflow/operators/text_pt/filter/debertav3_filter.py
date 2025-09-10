import numpy as np
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.operators.text_pt import DebertaV3SampleEvaluator

@OPERATOR_REGISTRY.register()
class DebertaV3Filter(OperatorABC):

    def __init__(self, allowed_scores : list = ['Medium', 'High'], model_name='nvidia/quality-classifier-deberta', model_cache_dir='./dataflow_cache', device='cuda', batch_size=16):
        self.logger = get_logger()
        self.allowed_scores = allowed_scores
        self.scorer = DebertaV3SampleEvaluator(
            model_name=model_name,
            model_cache_dir=model_cache_dir,
            device=device,
            batch_size=batch_size,
        )
        self.logger.info(f"Initializing {self.__class__.__name__} with allowed_scores = {self.allowed_scores}...")
    
    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "基于DebertaV3Scorer打分器的得分对数据进行过滤。使用Nvidia Deberta V3模型的质量分类器评估文本质量。\n\n"
                "初始化参数：\n"
                "- allowed_scores: 允许通过的分数列表，默认为['Medium', 'High']\n"
                "- model_name: 模型名称，默认为'nvidia/quality-classifier-deberta'\n"
                "- model_cache_dir: 模型缓存目录，默认为'./dataflow_cache'\n"
                "- device: 运行设备，默认为'cuda'\n"
                "- batch_size: 批处理大小，默认为16\n\n"
                "运行参数：\n"
                "- input_key: 输入文本字段名\n"
                "- output_key: 输出分数字段名，默认为'Debertav3Score'\n\n"
                "过滤逻辑：保留分类结果在allowed_scores列表中的数据"
            )
        else:
            return (
                "Filter data using scores from the DebertaV3Scorer. Evaluate text quality using Nvidia Deberta V3 model-based quality classifier.\n\n"
                "Initialization Parameters:\n"
                "- allowed_scores: List of allowed scores, default is ['Medium', 'High']\n"
                "- model_name: Model name, default is 'nvidia/quality-classifier-deberta'\n"
                "- model_cache_dir: Model cache directory, default is './dataflow_cache'\n"
                "- device: Running device, default is 'cuda'\n"
                "- batch_size: Batch size, default is 16\n\n"
                "Run Parameters:\n"
                "- input_key: Input text field name\n"
                "- output_key: Output score field name, default is 'Debertav3Score'\n\n"
                "Filter Logic: Keep data with classification results in allowed_scores list"
            )
        
    def run(self, storage: DataFlowStorage, input_key: str, output_key: str = 'Debertav3Score'):
        self.input_key = input_key
        self.output_key = output_key
        dataframe = storage.read("dataframe")
        scores = self.scorer.eval(dataframe, self.input_key)
        dataframe[self.output_key] = scores
        labels = np.array([1 if score in self.allowed_scores else 0 for score in scores])
        filtered_dataframe = dataframe[labels == 1]
        output_file = storage.write(filtered_dataframe)
        self.logger.info(f"Filtering completed. Total records passing filter: {len(filtered_dataframe)}.")
        return [self.output_key]
        
        