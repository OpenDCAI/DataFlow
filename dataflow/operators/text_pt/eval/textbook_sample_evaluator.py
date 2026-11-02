from typing import List
import re
from huggingface_hub import hf_hub_download
import fasttext
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage
from tqdm import tqdm
import numpy as np

@OPERATOR_REGISTRY.register()
class TextbookSampleEvaluator(OperatorABC):
    def __init__(self, model_cache_dir='./dataflow_cache'):
        self.logger = get_logger()
        self.logger.info(f'Initializing {self.__class__.__name__}...')
        model_path = hf_hub_download(
            repo_id='kenhktsui/llm-data-textbook-quality-fasttext-classifer-v2',
            filename='model.bin',
            cache_dir=model_cache_dir
        )
        low_score=1.0
        mid_score=3.0
        high_score=5.0
        self.model = fasttext.load_model(model_path)
        self.score_type = float
        self.data_type = 'text'
        self.score_name = 'TextbookScore'
        self.score_dict = {
            '__label__Low': low_score,
            '__label__Mid': mid_score,
            '__label__High': high_score
        }
        self.logger.info(f'{self.__class__.__name__} initialized.')
    
    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "基于FastText分类器(kenhktsui/llm-data-textbook-quality-fasttext-classifer-v2)评估文本的教育价值，将文本分为低(Low)、中(Mid)、高(High)三个等级，" 
                "并映射为1.0、3.0、5.0的分数，最终分数为各等级分数按分类器预测概率加权求和。适用于筛选适合作为教材的高质量文本内容。\n"
                "输入参数：\n"
                "- text: 待评估的文本字符串\n"
                "- input_key：run()读取的输入文本字段名，无默认值，必须指定\n"
                "- output_key：写入分数的输出字段名，默认为'TextbookScore'\n"
                "输出参数：\n"
                "- float: 教育价值分数，为1.0(低)、3.0(中)、5.0(高)按预测概率加权的连续值（约在1.0到5.0之间），不是离散的三个取值\n"
                "- 不过滤任何行；设计上将分数写入output_key字段并保留在输出中；run()无返回值(None)。"
                "注意：当前run()以三个参数调用eval(dataframe, input_key, output_key)，而eval只接受(dataframe, input_key)，因此run()会抛出TypeError，不会写入任何数据；"
                "即使修正该问题，run()在逐行循环中把每行的单元素分数数组赋给整列dataframe[output_key]，而不是赋给该行（多于一行时pandas会抛出ValueError）"
            )
        else:
            return (
                "Assess the educational value of text using a FastText classifier (kenhktsui/llm-data-textbook-quality-fasttext-classifer-v2), categorizing text into Low, Mid, and High levels, " 
                "mapped to scores of 1.0, 3.0, and 5.0 respectively; the final score is the sum of these level scores weighted by the classifier's predicted probabilities. Suitable for filtering high-quality text content suitable as teaching materials.\n"
                "Input parameters:\n"
                "- text: Text string to be evaluated\n"
                "- input_key: Name of the input text field read by run(); no default, must be given\n"
                "- output_key: Name of the output field that receives the score, default is 'TextbookScore'\n"
                "Output parameters:\n"
                "- float: Educational value score, a probability-weighted continuous value of 1.0 (Low), 3.0 (Mid), 5.0 (High) (roughly between 1.0 and 5.0), not one of three discrete values\n"
                "- No rows are filtered; by design the score is written to the output_key field and kept in the output; run() returns None. "
                "Note: run() currently calls eval(dataframe, input_key, output_key) with three arguments while eval accepts only (dataframe, input_key), so run() raises TypeError and writes nothing; "
                "even with that fixed, run() assigns each row's one-element score array to the entire dataframe[output_key] column inside a per-row loop instead of to that row (pandas raises ValueError for a DataFrame with more than one row)"
            )

    @staticmethod
    def replace_newlines(text: str) -> str:
        return re.sub("\n+", " ", text)

    def _score_func(self, text_list: List[str]) -> List[float]:
        text_list = [self.replace_newlines(text) for text in text_list]
        pred = self.model.predict(text_list, k=-1)
        
        score_list = []
        for labels, scores in zip(*pred):
            score = 0
            for label, score_value in zip(labels, scores):
                score += self.score_dict.get(label, 0) * score_value
            score_list.append(float(score))
        
        return score_list

    def eval(self, dataframe, input_key):
        scores = []
        text_list = dataframe[input_key]
        self.logger.info(f"Evaluating {self.score_name}...")
        for sample in tqdm(text_list, desc="TextbookScorer Evaluating..."):
            score = self._score_func([sample])
            scores.append(score)
        self.logger.info("Evaluation complete!")
        return np.array(scores)

    def run(self, storage: DataFlowStorage, input_key: str, output_key: str='TextbookScore'):
        dataframe = storage.read("dataframe")  
        scores = self.eval(dataframe, input_key, output_key) 
        for i, score_list in enumerate(scores):
            dataframe[output_key] = score_list 
        storage.write(dataframe)
