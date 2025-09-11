from vendi_score import text_utils
from dataflow.utils.storage import DataFlowStorage
import pandas as pd
from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger

# VendiScore dataset diversity evaluation
# Cited from: The Vendi Score: A Diversity Evaluation Metric for Machine Learning
@OPERATOR_REGISTRY.register()
class VendiDatasetEvaluator(OperatorABC):
    def __init__(self, device='cuda'):
        self.logger = get_logger()
        self.logger.info(f'Initializing {self.__class__.__name__}...')
        self.bert_model_path = 'bert-base-uncased'
        self.simcse_model_path = 'princeton-nlp/unsup-simcse-bert-base-uncased'
        self.device = device
        self.score_name = 'VendiScore'
        self.logger.info(f'{self.__class__.__name__} initialized.')
    
    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "通过计算VendiScore来评估数据集的多样性，使用BERT和SimCSE模型生成嵌入并计算分数。\n"
                "输入参数：\n"
                "- device：计算设备，默认为'cuda'\n"
                "- input_key：输入文本字段名\n"
                "输出参数：\n"
                "- BERTVendiScore：基于BERT的多样性得分\n"
                "- SimCSEVendiScore：基于SimCSE的多样性得分"
            )
        elif lang == "en":
            return (
                "Assess dataset diversity using VendiScore with embeddings from BERT and SimCSE models.\n"
                "Input Parameters:\n"
                "- device: Computing device, default 'cuda'\n"
                "- input_key: Field name for input text\n"
                "Output Parameters:\n"
                "- BERTVendiScore: Diversity score based on BERT\n"
                "- SimCSEVendiScore: Diversity score based on SimCSE"
            )
        else:
            return "Assess dataset diversity using VendiScore."
    
    def get_score(self, sentences):
        result = {}
        bert_vs = text_utils.embedding_vendi_score(sentences, model_path=self.bert_model_path, device=self.device)
        result["BERTVendiScore"] = round(bert_vs, 2)
        simcse_vs = text_utils.embedding_vendi_score(sentences, model_path=self.simcse_model_path, device=self.device)
        result["SimCSEVendiScore"] = round(simcse_vs, 2)
        return result

    def run(self, storage: DataFlowStorage, input_key: str):
        dataframe = storage.read("dataframe")
        samples = dataframe[input_key].to_list()
        self.logger.info(f"Evaluating {self.score_name}...")
        vendiscore = self.get_score(samples)
        self.logger.info("Evaluation complete!")
        self.logger.info(f"VendiScore: {vendiscore}")
        return vendiscore