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
                "基于Fineweb-Edu分类器评估文本的教育价值。该分类器使用预训练的序列分类模型对文本进行评估，返回0-1之间的分数，" 
                "分数越高表示文本的教育价值越高。适用于筛选具有教育意义的文本内容。\n" 
                "输入参数：\n" 
                "- text: 待评估的文本字符串\n" 
                "输出参数：\n" 
                "- float: 0-1之间的教育价值分数，越高表示教育价值越大"
            )
        else:
            return (
                "Evaluate the educational value of text using the Fineweb-Edu classifier. This classifier uses a pre-trained sequence classification model " 
                "to assess text and returns a score between 0 and 1, where higher scores indicate greater educational value. Suitable for filtering educational content.\n" 
                "Input parameters:\n" 
                "- text: Text string to be evaluated\n" 
                "Output parameters:\n" 
                "- float: Educational value score between 0 and 1, higher values indicate greater educational value"
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
