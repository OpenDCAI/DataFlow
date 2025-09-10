import torch
from torch import nn
from transformers import BertModel, BertConfig, PreTrainedModel, AutoTokenizer
from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage
from tqdm import tqdm
from dataflow.utils.utils import get_logger
import numpy as np

@OPERATOR_REGISTRY.register()
class PairQualSampleEvaluator(OperatorABC):
    def __init__(self, model_cache_dir:str='./dataflow_cache', device="cuda", lang='en', max_length=512):
        self.logger = get_logger()
        self.logger.info(f'Initializing {self.__class__.__name__}...')
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_cache_dir = model_cache_dir
        self.lang = lang
        self.max_length = max_length
        self.score_name = 'PairQualScore'
        if lang not in ['en', 'zh']:
            raise ValueError("Invalid value for 'lang'. Only 'en' or 'zh' are allowed.")
        if self.lang == 'en':
            model = "zks2856/PairQual-Scorer-en"
            config = BertConfig.from_pretrained(model, cache_dir=self.model_cache_dir)
            self.model = BertForRegression_en.from_pretrained(model, config=config, trust_remote_code=True, cache_dir=self.model_cache_dir).to(self.device).eval()
            self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True, cache_dir=self.model_cache_dir)
        else:
            model = "zks2856/PairQual-Scorer-zh"
            config = BertConfig.from_pretrained(model, cache_dir=self.model_cache_dir)
            self.model = BertForRegression_zh.from_pretrained(model, config=config, trust_remote_code=True, cache_dir=self.model_cache_dir).to(self.device).eval()
            self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True, cache_dir=self.model_cache_dir)
        self.logger.info(f'{self.__class__.__name__} initialized.')

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "基于BGE模型和GPT成对比较数据训练的文本质量评分器，支持中英文输入。通过对文本进行单样本评估，返回0-1之间的质量分数，" 
                "分数越高表示文本质量越好。模型分为英文版本(zks2856/PairQual-Scorer-en)和中文版本(zks2856/PairQual-Scorer-zh)。\n" 
                "输入参数：\n" 
                "- text: 待评估的文本字符串\n" 
                "- lang: 语言类型，可选'en'或'zh'\n" 
                "输出参数：\n" 
                "- float: 0-1之间的质量分数，越高表示质量越好"
            )
        else:
            return (
                "Text quality scorer trained on BGE model and GPT pairwise comparison data, supporting bilingual input. Evaluate text through single-sample assessment, " 
                "returning a quality score between 0 and 1, where higher scores indicate better text quality. Models include English version (zks2856/PairQual-Scorer-en) and Chinese version (zks2856/PairQual-Scorer-zh).\n" 
                "Input parameters:\n" 
                "- text: Text string to be evaluated\n" 
                "- lang: Language type, optional 'en' or 'zh'\n" 
                "Output parameters:\n" 
                "- float: Quality score between 0 and 1, higher values indicate better quality"
            )

    def inference(self, input_text):
        inputs = self.tokenizer(input_text, return_tensors='pt', padding=True, truncation=True, max_length=self.max_length).to(self.device)
        with torch.no_grad():
            _, score = self.model(inputs)
        return score.item()

    def eval(self, dataframe, input_key):
        self.logger.info(f"Evaluating {self.score_name}...")
        scores = []
        for sample in tqdm(dataframe[input_key], desc="PairQualScorer Evaluating..."):
            score = self.inference(sample)
            scores.append(score)
        self.logger.info("Evaluation complete!")
        return np.array(scores)

    def run(self, storage: DataFlowStorage, input_key: str, output_key: str='PairQualScore'):
        dataframe = storage.read("dataframe")
        scores = self.eval(dataframe, input_key)
        dataframe[output_key] = scores
        storage.write(dataframe)


class BertForRegression_en(PreTrainedModel):
    config_class = BertConfig

    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.regression = nn.Sequential(
            nn.Linear(config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1)
        )

        self.post_init()

    def forward(self, inputs):
        encoded = self.bert(**inputs)
        score = self.regression(encoded['pooler_output'])
        return encoded, score

class BertForRegression_zh(PreTrainedModel):
    config_class = BertConfig

    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.regression = nn.Sequential(
            nn.Linear(config.hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        self.post_init()

    def forward(self, inputs):
        encoded = self.bert(**inputs)
        score = self.regression(encoded['pooler_output'])
        return encoded, score