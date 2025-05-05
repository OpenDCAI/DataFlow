from dataflow.core import TextScorer
from dataflow.utils.registry import MODEL_REGISTRY
import torch
from torch import nn
import transformers

class BertForRegression(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.bert = transformers.BertModel.from_pretrained(model_name)
        self.regression = nn.Sequential(
            nn.Linear(self.bert.pooler.dense.out_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1)
        )

    def forward(self, inputs):
        encoded = self.bert(**inputs)
        score = self.regression(encoded['pooler_output'])
        return encoded, score

@MODEL_REGISTRY.register()
class PairQualScorer(TextScorer):
    def __init__(self, args_dict: dict):
        super().__init__(args_dict)
        self.device = args_dict.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = args_dict.get('model_name', 'BAAI/bge-base-en-v1.5')
        self.model_state_dict = args_dict.get('model_state_dict', None)
        self.model_cache_dir = args_dict.get('model_cache_dir')
        self.max_length = args_dict.get('max_length', 512)
        self.score_type = float
        self.data_type = 'text'
        self.score_name = 'PairQualScorer'
        self.batch_size = 1
        
        self.model = BertForRegression(self.model_name)
        self.tokenizer = transformers.BertTokenizerFast.from_pretrained(self.model_name, cache_dir=self.model_cache_dir)
        if self.model_state_dict:
            self.model.load_state_dict(torch.load(self.model_state_dict, map_location='cpu'))
        self.model.to(self.device).eval()
    
    def evaluate_batch(self, batch):
        input_texts = next(iter(batch.values()))
        inputs = self.tokenizer(input_texts, return_tensors='pt', padding=True, truncation=True, max_length=self.max_length)
        inputs.to(self.device)
        _, score = self.model(inputs)
        return [score.item()]