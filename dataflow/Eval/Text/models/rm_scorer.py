from transformers import AutoModelForSequenceClassification, AutoTokenizer
from dataflow.core import TextScorer
from dataflow.utils.registry import MODEL_REGISTRY
import torch

# RMScorer for evaluating based on reward-model-deberta-v3-large-v2
@MODEL_REGISTRY.register()
class RMScorer(TextScorer):
    def __init__(self, args_dict):
        super().__init__(args_dict)
        self.model_name = args_dict.get('model_name')
        self.model_cache_dir = args_dict.get('model_cache_dir')  # 增加缓存目录
        self.batch_size = args_dict.get('batch_size')
        self.device = args_dict.get('device')
        self.score_type = float
        self.data_type = 'text'
        self.score_name = 'RewardModelScore'
        
        self.rank_model = AutoModelForSequenceClassification.from_pretrained(self.model_name, cache_dir=self.model_cache_dir).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.model_cache_dir)


        
    def evaluate_batch(self, batch):
        input_texts = batch.get('instruction', '')
        output_texts = batch.get('output', '')
        inputs = self.tokenizer(input_texts, output_texts, return_tensors='pt', padding=True, truncation=True).to(self.device)

        with torch.no_grad():
            logits = self.rank_model(**inputs).logits.cpu().detach().numpy()

        scores = logits.squeeze() 

        if scores.ndim == 0:  
            scores = [float(scores)]

        return scores.tolist() 