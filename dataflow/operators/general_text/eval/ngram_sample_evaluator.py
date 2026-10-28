import re
from tqdm import tqdm
import pandas as pd
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from typing import Literal

_HAN_CHARACTER_PATTERN = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
    r"\U00020000-\U0002fa1f\U00030000-\U000323af]"
)

@OPERATOR_REGISTRY.register()
class NgramSampleEvaluator(OperatorABC):
    
    def __init__(self, ngrams: int = 5, language: Literal['zh', 'en', 'auto'] = 'en'):
        if language not in ['zh', 'en', 'auto']:
            raise ValueError(f"Unsupported language: '{language}'. Supported options are: ['zh', 'en', 'auto'].")
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
                "- language: 处理语言；'zh' 使用字粒度切分，'en' 使用空格分词，'auto' 根据每条文本是否包含汉字自动选择，默认为 'en'\n"
                "运行参数：\n"
                "- input_key: 输入文本字段名（必填）\n"
                "- output_key: 写入得分的输出字段名，默认为'NgramScore'\n"
                "输出参数：\n"
                "- output_key（默认NgramScore）: 每行新增该字段，值为n-gram重复比例得分（0到1之间的浮点数，得分越高表示重复比例越低；空文本或token数少于ngrams时为0.0），该字段保留在输出中；不过滤任何行\n"
                "- run()不返回字段名列表（返回None）"
            )
        else:
            return (
                "Evaluates text redundancy by calculating the n-gram repetition ratio.\n"
                "Supports Chinese (character-level) and English (word-level) modes.\n\n"
                "Initialization Parameters:\n"
                "- ngrams: Length of n-grams, default is 5.\n"
                "- language: Processing language. 'zh' uses character-level splitting, 'en' uses whitespace splitting, and 'auto' selects per sample based on the presence of Han characters. Default is 'en'.\n\n"
                "Run Parameters:\n"
                "- input_key: Input text field name (required).\n"
                "- output_key: Name of the output field that receives the score, default is 'NgramScore'.\n\n"
                "Output Parameters:\n"
                "- output_key (default NgramScore): New field added to every row, holding the n-gram repetition ratio score (float 0-1, higher score means less repetition/higher originality; 0.0 for empty text or fewer tokens than ngrams). The field is kept in the output; no rows are filtered.\n"
                "- run() returns None, not a list of field names."
            )
    
    def _score_func(self, sample):
        content = sample 
        if not content or not isinstance(content, str):
            return 0.0

        content = content.lower()
        # 移除标点符号
        content = re.sub(r'[^\w\s]', '', content)
        
        # Auto detection is opt-in so an explicit language mode remains predictable.
        use_character_tokens = self.language == 'zh' or (
            self.language == 'auto' and _HAN_CHARACTER_PATTERN.search(content)
        )
        if use_character_tokens:
            # 中文模式：去除所有空格，按“字”切分
            content = re.sub(r'\s+', '', content)
            tokens = list(content) 
            join_char = ""
        else:
            # 默认/英文模式：按“空格”切分
            tokens = content.split()
            join_char = " "
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
