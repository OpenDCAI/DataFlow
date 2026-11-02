import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dataflow.core import OperatorABC
from dataflow import get_logger
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage
from tqdm import tqdm
import numpy as np

@OPERATOR_REGISTRY.register()
class FineWebEduSampleEvaluator(OperatorABC):
    def __init__(self, model_cache_dir: str = './dataflow_cache', device: str = 'cuda'):
        self.logger = get_logger()
        self.logger.info(f'Initializing {self.__class__.__name__}...')
        self.model_name = 'HuggingFaceTB/fineweb-edu-classifier'
        self.model_cache_dir = model_cache_dir
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 1
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.model_cache_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, cache_dir=self.model_cache_dir).to(self.device)
        self.model.eval()
        self.score_name = 'FineWebEduScore'
        self.logger.info(f'{self.__class__.__name__} initialized.')
    
    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "基于Fineweb-Edu分类器(HuggingFaceTB/fineweb-edu-classifier)评估文本的教育价值。该分类器使用预训练的序列分类模型对文本进行评估，返回模型的原始回归输出（logit，代码中未做归一化或截断，因此不保证落在0-1之间），" 
                "分数越高表示文本的教育价值越高。适用于筛选具有教育意义的文本内容。\n" 
                "输入参数：\n" 
                "- text: 待评估的文本字符串\n" 
                "- input_key：run()读取的输入文本字段名，无默认值，必须指定\n" 
                "- output_key：写入分数的输出字段名，默认为'FinewebEduScore'\n" 
                "输出参数：\n" 
                "- float: 教育价值分数（模型原始输出），越高表示教育价值越大\n" 
                "- 不过滤任何行；每行新增output_key字段，值为该行的浮点分数，该字段保留在输出中；run()无返回值(None)"
            )
        else:
            return (
                "Evaluate the educational value of text using the Fineweb-Edu classifier (HuggingFaceTB/fineweb-edu-classifier). This classifier uses a pre-trained sequence classification model " 
                "to assess text and returns the model's raw regression output (logit; the code applies no normalization or clipping, so the value is not guaranteed to lie in 0-1), where higher scores indicate greater educational value. Suitable for filtering educational content.\n" 
                "Input parameters:\n" 
                "- text: Text string to be evaluated\n" 
                "- input_key: Name of the input text field read by run(); no default, must be given\n" 
                "- output_key: Name of the output field that receives the score, default is 'FinewebEduScore'\n" 
                "Output parameters:\n" 
                "- float: Educational value score (raw model output), higher values indicate greater educational value\n" 
                "- No rows are filtered; each row gets a new output_key field holding its float score, which is kept in the output; run() returns None"
            )

    def _score_func(self, sample):
        tokenized_inputs = self.tokenizer(sample, return_tensors="pt", padding="longest", truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**tokenized_inputs)
            logits = outputs.logits.squeeze(-1).float().detach().cpu().numpy() 
        
        return logits.tolist()[0]  # Return as list for individual sample

    def eval(self, dataframe, input_key):
        scores = []
        self.logger.info(f"Evaluating {self.score_name}...")
        for sample in tqdm(dataframe[input_key], desc="Fineweb-edu model evaluating..."):
            score = self._score_func(sample)
            scores.append(score)
        self.logger.info("Evaluation complete!")
        return np.array(scores)

    def run(self, storage: DataFlowStorage, input_key: str, output_key: str='FinewebEduScore'):
        self.input_key = input_key
        self.output_key = output_key
        dataframe = storage.read("dataframe")
        scores = self.eval(dataframe, input_key)
        dataframe[self.output_key] = scores
        storage.write(dataframe)
