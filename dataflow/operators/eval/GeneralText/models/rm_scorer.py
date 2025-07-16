from transformers import AutoModelForSequenceClassification, AutoTokenizer
from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
import torch
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.utils import get_logger

# RMScorer for evaluating based on reward-model-deberta-v3-large-v2
@OPERATOR_REGISTRY.register()
class RMScorer(OperatorABC):
    def __init__(self, device='cuda', model_cache_dir='./dataflow_cache', ):
        self.logger = get_logger()
        self.logger.info(f'Initializing {self.__class__.__name__}...')
        self.model_name = 'OpenAssistant/reward-model-deberta-v3-large-v2'
        self.model_cache_dir = model_cache_dir
        self.score_name = 'RewardModelScore'
        self.device = device
        self.rank_model = AutoModelForSequenceClassification.from_pretrained(self.model_name, cache_dir=self.model_cache_dir).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.model_cache_dir)
        self.logger.info(f'{self.__class__.__name__} initialized.')
    
    @staticmethod
    def get_desc(lang: str = "zh"):
        return "基于人类价值判断的奖励模型打分，高分代表质量较高。" if lang == "zh" else "Quality scoring using reward model trained with human preference data."

    def eval(self, dataframe, input_instruction_key: str = 'instruction', input_output_key: str = 'output'):
        input_texts = dataframe.get(input_instruction_key, '').to_list()
        output_texts = dataframe.get(input_output_key, '').to_list()
        inputs = self.tokenizer(input_texts, output_texts, return_tensors='pt', padding=True, truncation=True).to(self.device)
        self.logger.info(f"Evaluating {self.score_name}...")
        with torch.no_grad():
            logits = self.rank_model(**inputs).logits.cpu().detach().numpy()
        scores = logits.squeeze() 
        if scores.ndim == 0:  
            scores = [float(scores)]
        self.logger.info("Evaluation complete!")
        return scores.tolist() 

    def run(self, storage: DataFlowStorage, input_instruction_key: str = 'instruction', input_output_key: str = 'output', output_key: str = 'RMScore'):
        dataframe = storage.read("dataframe")
        scores = self.eval(dataframe, input_instruction_key, input_output_key)
        dataframe[output_key] = scores        
        storage.write(dataframe)