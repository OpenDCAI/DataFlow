import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
from huggingface_hub import PyTorchModelHubMixin
from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage
from tqdm import tqdm
from dataflow import get_logger

@OPERATOR_REGISTRY.register()
class DebertaV3SampleEvaluator(OperatorABC):
    def __init__(self, model_name, model_cache_dir='./dataflow_cache', device='cuda'):
        self.logger = get_logger()
        self.logger.info(f'Initializing {self.__class__.__name__}...')
        self.model_name = model_name
        self.model_cache_dir = model_cache_dir
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.score_name = 'DebertaV3Score'
        self.config = AutoConfig.from_pretrained(self.model_name, cache_dir=self.model_cache_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.model_cache_dir)
        self.model = QualityModel.from_pretrained(self.model_name, cache_dir=self.model_cache_dir).to(self.device)
        self.model.eval()
        self.logger.info(f'{self.__class__.__name__} initialized.')
    
    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "基于Nvidia Deberta V3模型的质量分类器，用于评估文本质量并返回分类结果。\n"
                "输入参数：\n"
                "- model_name：预训练模型名称\n"
                "- model_cache_dir：模型缓存目录，默认为'./dataflow_cache'\n"
                "- device：计算设备，默认为'cuda'\n"
                "- input_key：输入文本字段名\n"
                "- output_key：输出分类结果字段名，默认为'Debertav3Score'\n"
                "输出参数：\n"
                "- 包含文本质量分类结果的DataFrame"
            )
        elif lang == "en":
            return (
                "Text quality classifier based on Nvidia Deberta V3 model for quality assessment and classification.\n"
                "Input Parameters:\n"
                "- model_name: Pretrained model name\n"
                "- model_cache_dir: Model cache directory, default './dataflow_cache'\n"
                "- device: Computing device, default 'cuda'\n"
                "- input_key: Field name for input text\n"
                "- output_key: Field name for output classification, default 'Debertav3Score'\n"
                "Output Parameters:\n"
                "- DataFrame containing text quality classification results"
            )
        else:
            return "Text quality classifier based on Nvidia Deberta V3."
    
    def _score_func(self, sample):
        inputs = self.tokenizer(
            sample, return_tensors="pt", padding="longest", truncation=True
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(inputs["input_ids"], inputs["attention_mask"])
        predicted_classes = torch.argmax(outputs, dim=1)
        predicted_domains = [
            self.config.id2label[class_idx.item()] for class_idx in predicted_classes.cpu().numpy()
        ]
        return predicted_domains[0]  # Assuming one sample per batch

    def eval(self, dataframe, input_key):
        scores = []
        self.logger.info(f"Evaluating {self.score_name}...")
        for sample in tqdm(dataframe[input_key], desc="DebertaV3 modle evaluating..."):
            score = self._score_func(sample)
            scores.append(score)
        self.logger.info("Evaluation complete!")
        return scores

    def run(self, storage: DataFlowStorage, input_key: str, output_key: str='Debertav3Score'):
        self.input_key = input_key
        self.output_key = output_key
        dataframe = storage.read("dataframe")
        scores = self.eval(dataframe, input_key)
        dataframe[self.output_key] = scores
        storage.write(dataframe)


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
