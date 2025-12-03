import re
from tqdm import tqdm
import pandas as pd
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger

@OPERATOR_REGISTRY.register()
class NgramSampleEvaluator(OperatorABC):
    
    def __init__(self, ngrams=5, language: str = 'en'):
        self.logger = get_logger()
        self.logger.info(f'Initializing {self.__class__.__name__}...')
        self.ngrams = ngrams
        self.language = language
        self.score_name = 'NgramScore'
        self.logger.info(f'{self.__class__.__name__} initialized. Mode: {self.language}')
    
    @staticmethod
    def get_desc(lang: str = "en"):
        # 默认返回英文描述
        if lang == "zh":
            return (
                "计算文本中n-gram的重复比例，评估文本冗余度。\n"
                "支持中文（字级别）和英文（词级别）模式。\n"
                "初始化参数：\n"
                "- ngrams: n-gram长度，默认为5\n"
                "- language: 处理语言，'zh' 使用字粒度切分，其他使用空格分词，默认为 'en'\n"
                "输出参数：\n"
                "- NgramScore: n-gram重复比例得分（0到1之间，得分越高表示重复比例越低）"
            )
        else:
            return (
                "Evaluates text redundancy by calculating the n-gram repetition ratio.\n"
                "Supports Chinese (character-level) and English (word-level) modes.\n\n"
                "Initialization Parameters:\n"
                "- ngrams: Length of n-grams, default is 5.\n"
                "- language: Processing language. 'zh' for character-level splitting, others for whitespace splitting. Default is 'en'.\n\n"
                "Output Parameters:\n"
                "- NgramScore: N-gram repetition ratio score (0-1, higher score means less repetition/higher originality)."
            )
    
    def _score_func(self, sample):
        content = sample 
        if not content or not isinstance(content, str):
            return 0.0

        content = content.lower()
        # 移除标点符号
        content = re.sub(r'[^\w\s]', '', content)
        
        # --- 根据语言选择切分逻辑 ---
        if self.language == 'zh':
            # 中文模式：去除所有空格，按“字”切分
            content = re.sub(r'\s+', '', content)
            tokens = list(content) 
            join_char = ""
        else:
            # 默认/英文模式：按“空格”切分
            tokens = content.split()
            join_char = " "
        # ---------------------------

        if len(tokens) < self.ngrams:
            return 0.0

        # 生成 n-grams
        ngrams_list = [join_char.join(tokens[i:i + self.ngrams]) for i in range(len(tokens) - (self.ngrams - 1))]
        
        unique_ngrams = set(ngrams_list)
        total_ngrams = len(ngrams_list)
        unique_ngrams_count = len(unique_ngrams)

        repetition_score = unique_ngrams_count / total_ngrams if total_ngrams > 0 else 0.0
        return repetition_score

    def eval(self, dataframe: pd.DataFrame, input_key: str):
        self.logger.info(f"Evaluating {self.score_name} (Language: {self.language})...")
        scores = [self._score_func(sample) for sample in tqdm(dataframe[input_key], desc=f"NgramScorer ({self.language})")]
        self.logger.info("Evaluation complete!")
        return scores
    
    def run(self, storage: DataFlowStorage, input_key: str, output_key: str='NgramScore'):
        self.input_key = input_key
        self.output_key = output_key
        dataframe = storage.read("dataframe")
        scores = self.eval(dataframe, input_key)
        dataframe[self.output_key] = scores
        storage.write(dataframe)