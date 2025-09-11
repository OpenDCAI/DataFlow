import os
import json
import pickle
from tqdm import tqdm
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.operators.general_text.eval.cider.cider import Cider

def load_idf(idf_path):
    with open(idf_path, 'rb') as f:
        idf = pickle.load(f, encoding='utf-8')  
    return idf

@OPERATOR_REGISTRY.register()
class CiderSampleEvaluator(OperatorABC):
    def __init__(self, n=4, sigma=6.0, df_mode="coco-val-df", idf_path="./dataflow/operators/general_pt/eval/cider/coco-val-df.p"):
        self.logger = get_logger()
        self.logger.info(f'Initializing {self.__class__.__name__}...')
        self.score_name = 'CiderScore'
        self.n = n  # Max n-gram length (default: 4)
        self.sigma = sigma  # Sigma for Gaussian penalty (default: 6.0)
        self.df_mode = df_mode
        if self.df_mode != "corpus":
            # The idf file can be downloaded at https://github.com/ramavedantam/coco-caption/blob/master/data/coco-val-df.p
            # Put the file in the correct idf_path
            self.idf = load_idf(idf_path)
        else:
            self.idf = None  # No need to load IDF for 'corpus' mode
        self.logger.info(f'{self.__class__.__name__} initialized.')
    
    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "使用CIDEr指标评估生成文本与参考文本的相似度，基于TF-IDF加权的n-gram重叠度。\n"
                "输入参数：\n"
                "- n：最大n-gram长度，默认为4\n"
                "- sigma：高斯惩罚参数，默认为6.0\n"
                "- df_mode：文档频率模式，默认为'coco-val-df'\n"
                "- idf_path：IDF文件路径，默认为预训练COCO数据集IDF\n"
                "- input_key：生成文本字段名\n"
                "- reference_key：参考文本字段名\n"
                "- output_key：输出得分字段名，默认为'CiderScore'\n"
                "输出参数：\n"
                "- 包含CIDEr得分的DataFrame"
            )
        elif lang == "en":
            return (
                "Evaluate text similarity using CIDEr metric with TF-IDF weighted n-gram overlap.\n"
                "Input Parameters:\n"
                "- n: Maximum n-gram length, default 4\n"
                "- sigma: Gaussian penalty parameter, default 6.0\n"
                "- df_mode: Document frequency mode, default 'coco-val-df'\n"
                "- idf_path: Path to IDF file, default pre-trained COCO dataset IDF\n"
                "- input_key: Field name for generated text\n"
                "- reference_key: Field name for reference text\n"
                "- output_key: Field name for output score, default 'CiderScore'\n"
                "Output Parameters:\n"
                "- DataFrame containing CIDEr scores"
            )
        else:
            return "Evaluate text similarity using CIDEr metric."
    
    def _score_func(self, eval_text, ref_text):
        cider_scorer = Cider(
            test=eval_text,
            refs=[ref_text],
            n=self.n,
            sigma=self.sigma,
            idf=self.idf  # Pass IDF (None if using 'corpus')
        )
        # Pass df_mode dynamically based on the argument
        cider_score, _ = cider_scorer.compute_score(df_mode='corpus' if self.idf is None else 'coco-val-df')  
        return cider_score

    def eval(self, dataframe, input_key, reference_key):
        eval_data = dataframe[input_key]
        ref_data = dataframe[reference_key]
        self.logger.info(f"Evaluating {self.score_name}...")
        scores = [self._score_func(eval_text, ref_text) for eval_text, ref_text in tqdm(zip(eval_data, ref_data), desc="CiderScorer Evaluating...")]
        self.logger.info("Evaluation complete!")
        return scores
    
    def run(self, storage: DataFlowStorage, input_key: str, reference_key: str, output_key: str='CiderScore'):
        self.input_key = input_key
        self.reference_key = reference_key
        self.output_key = output_key
        dataframe = storage.read("dataframe")
        scores = self.eval(dataframe, input_key, reference_key)
        dataframe[self.output_key] = scores
        storage.write(dataframe)
