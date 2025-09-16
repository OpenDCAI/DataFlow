import numpy as np
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.operators.general_text import PresidioSampleEvaluator

@OPERATOR_REGISTRY.register()
class PresidioFilter(OperatorABC):

    def __init__(self, min_score: int = 0, max_score: int = 5, lang='en', device='cuda', model_cache_dir='./dataflow_cache'):
        self.logger = get_logger()
        self.min_score = min_score
        self.max_score = max_score
        self.scorer = PresidioSampleEvaluator(lang=lang, device=device, model_cache_dir=model_cache_dir)
        self.logger.info(f"Initializing {self.__class__.__name__} with min_score = {self.min_score} and max_score = {self.max_score}")
    
    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "基于PresidioScorer打分器的得分对数据进行过滤。使用Microsoft Presidio模型识别文本中的私人实体(PII)，返回PII信息个数。\n"
                "支持识别姓名、邮箱、电话号码、身份证号等多种敏感信息类型，可用于数据隐私保护和合规性检查。\n"
                "输入参数：\n"
                "- min_score：保留样本的最小PII数量阈值，默认为0\n"
                "- max_score：保留样本的最大PII数量阈值，默认为5\n"
                "- lang：文本语言，默认为'en'\n"
                "- device：模型运行设备，默认为'cuda'\n"
                "- model_cache_dir：模型缓存目录，默认为'./dataflow_cache'\n"
                "输出参数：\n"
                "- 过滤后的DataFrame，仅保留PII数量在[min_score, max_score]范围内的样本\n"
                "- 返回包含输出字段名的列表，用于后续算子引用"
            )
        elif lang == "en":
            return (
                "Filter data using scores from the PresidioScorer. Detect personally identifiable information (PII) entities in text using Microsoft Presidio model and return the count of detected PII items.\n"
                "Supports recognition of multiple sensitive information types including names, emails, phone numbers, and IDs for data privacy protection and compliance checks.\n"
                "Input Parameters:\n"
                "- min_score: Minimum PII count threshold for retaining samples, default is 0\n"
                "- max_score: Maximum PII count threshold for retaining samples, default is 5\n"
                "- lang: Text language, default is 'en'\n"
                "- device: Model running device, default is 'cuda'\n"
                "- model_cache_dir: Model cache directory, default is './dataflow_cache'\n"
                "Output Parameters:\n"
                "- Filtered DataFrame containing only samples with PII count within [min_score, max_score] range\n"
                "- List containing output field name for subsequent operator reference"
            )
        else:
            return "Filter data based on PII detection results using Microsoft Presidio model."
        
    def run(self, storage: DataFlowStorage, input_key: str, output_key: str = 'PresidioScore'):
        self.input_key = input_key
        self.output_key = output_key
        dataframe = storage.read("dataframe")
        self.logger.info(f"Running {self.__class__.__name__}...")

        # Get the scores for filtering
        scores = np.array(self.scorer.eval(dataframe, self.input_key))

        dataframe[self.output_key] = scores
        filtered_dataframe = dataframe[(scores >= self.min_score) & (scores <= self.max_score)]

        # Write the filtered dataframe back to storage
        storage.write(filtered_dataframe)

        self.logger.info(f"Filtering completed. Total records passing filter: {len(filtered_dataframe)}.")

        return [self.output_key]