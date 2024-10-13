from vendi_score import text_utils
from dataflow.core import TextScorer
from dataflow.utils.registry import MODEL_REGISTRY

# VendiScore dataset diversity evaluation
# cited from: The Vendi Score: A Diversity Evaluation Metric for Machine Learning
@MODEL_REGISTRY.register()
class VendiScorer(TextScorer):
    def __init__(self, args_dict):
        super().__init__(args_dict)
        self.bert_model_path = args_dict.get('bert_model_path') 
        self.simcse_model_path = args_dict.get('simcse_model_path')  
        self.metrics_to_keep = args_dict.get('metrics_to_keep')
        self.device = args_dict.get('device')
        self.batch_size = -1
        self.use_meta = True
        self.score_type = float 
        self.data_type = 'text'
        self.score_name = 'VendiScore'

    def evaluate_batch(self, batch):
        sentences = next(iter(batch.values()))

        result = {}

        if self.metrics_to_keep.get("ngram", False):
            ngram_vs = text_utils.ngram_vendi_score(sentences, ns=[1, 2, 3, 4])
            result["N-gramsVendiScore"] = round(ngram_vs, 2)

        if self.metrics_to_keep.get("bert", False):
            bert_vs = text_utils.embedding_vendi_score(sentences, model_path=self.bert_model_path, device=self.device)
            result["BERTVendiScore"] = round(bert_vs, 2)

        if self.metrics_to_keep.get("simcse", False):
            simcse_vs = text_utils.embedding_vendi_score(sentences, model_path=self.simcse_model_path, device=self.device)
            result["SimCSEVendiScore"] = round(simcse_vs, 2)

        return result
