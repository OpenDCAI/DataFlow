import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dataflow.core import TextScorer
from dataflow.utils.registry import MODEL_REGISTRY
# FineWeb-Edu quality classifier (Huggingface)
# cited from: The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale
@MODEL_REGISTRY.register()
class FineWebEduScorer(TextScorer):
    def __init__(self, args_dict):
        super().__init__(args_dict)
        self.model_name = args_dict.get('model_name')
        self.model_cache_dir = args_dict.get('model_cache_dir') 
        self.device = args_dict.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 1
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.model_cache_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, cache_dir=self.model_cache_dir).to(self.device)
        self.model.eval()
        self.score_type = float
        self.data_type = 'text'
        self.score_name = 'FineWebEduScore'

    def evaluate_batch(self, batch) -> list:
        input_texts = next(iter(batch.values()))
        tokenized_inputs = self.tokenizer(input_texts, return_tensors="pt", padding="longest", truncation=True).to(self.device)

        with torch.no_grad():
            outputs = self.model(**tokenized_inputs)
            logits = outputs.logits.squeeze(-1).float().detach().cpu().numpy() 

        results = logits.tolist()

        return results

