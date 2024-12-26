from typing import List
import re
from huggingface_hub import hf_hub_download
import fasttext
from dataflow.core import TextScorer
from dataflow.utils.registry import MODEL_REGISTRY

# Textbook quality classifier (Huggingface)
# cited from: Textbooks Are All You Need
@MODEL_REGISTRY.register()
class TextbookScorer(TextScorer):
    def __init__(self, args_dict):
        super().__init__(args_dict)
        model_path = hf_hub_download(
            repo_id=args_dict.get('model_repo'),
            filename=args_dict.get('model_file'),
            cache_dir=args_dict.get('model_cache_dir')  # 设置自定义缓存路径
        )

        self.model = fasttext.load_model(model_path)
        self.batch_size = args_dict.get('batch_size') 
        self.score_type = float
        self.data_type = 'text'  
        self.score_name = 'TextbookScore'  

        self.score_dict = {
            '__label__': 0,
            '__label__Low': args_dict.get('low_score', 1.0),
            '__label__Mid': args_dict.get('mid_score', 3.0),
            '__label__High': args_dict.get('high_score', 5.0)
        }

    @staticmethod
    def replace_newlines(text: str) -> str:
        return re.sub("\n+", " ", text)

    def evaluate_batch(self, batch) -> List[float]:
        text_list = next(iter(batch.values()))
        text_list = [self.replace_newlines(text) for text in text_list]
        pred = self.model.predict(text_list, k=-1)

        score_list = []
        for labels, scores in zip(*pred):
            score = 0
            for label, score_value in zip(labels, scores):
                score += self.score_dict.get(label, 0) * score_value
            score_list.append(float(score))
        return score_list
