import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
from dataflow.core import TextScorer
from dataflow.utils.registry import MODEL_REGISTRY
from huggingface_hub import PyTorchModelHubMixin
from torch.utils.data import DataLoader
from tqdm import tqdm

# Nvidia quality-classifier-deberta (Huggingface)
@MODEL_REGISTRY.register()
class DebertaV3Scorer(TextScorer):
    def __init__(self, args_dict):
        super().__init__(args_dict)
        self.model_name = args_dict.get('model_name')
        self.model_cache_dir = args_dict.get('model_cache_dir') 
        self.device = args_dict.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = args_dict.get('batch_size')
        self.score_type = str
        self.data_type = 'text'
        self.score_name = 'DebertaV3Score'
        self.config = AutoConfig.from_pretrained(self.model_name, cache_dir=self.model_cache_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.model_cache_dir)
        self.model = QualityModel.from_pretrained(self.model_name, cache_dir=self.model_cache_dir).to(self.device)
        self.model.eval()



    def evaluate_batch(self, batch):
        input_texts = next(iter(batch.values()))
        inputs = self.tokenizer(
            input_texts, return_tensors="pt", padding="longest", truncation=True
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(inputs["input_ids"], inputs["attention_mask"])
        predicted_classes = torch.argmax(outputs, dim=1)
        predicted_domains = [
            self.config.id2label[class_idx.item()] for class_idx in predicted_classes.cpu().numpy()
        ]
        return predicted_domains


class QualityModel(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config):
        super(QualityModel, self).__init__()
        self.model = AutoModel.from_pretrained(config["base_model"])
        self.dropout = nn.Dropout(config["fc_dropout"])
        self.fc = nn.Linear(self.model.config.hidden_size, len(config["id2label"]))

    def forward(self, input_ids, attention_mask):
        features = self.model(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        dropped = self.dropout(features)
        outputs = self.fc(dropped)
        return torch.softmax(outputs[:, 0, :], dim=1)
