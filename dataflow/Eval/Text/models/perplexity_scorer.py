from dataflow.core import TextScorer
from dataflow.utils.registry import MODEL_REGISTRY
from dataflow.Eval.Text.models.Kenlm.model import KenlmModel

# Kenlm models perplexity evaluation
@MODEL_REGISTRY.register()
class PerplexityScorer(TextScorer):
    def __init__(self, args_dict):
        super().__init__(args_dict)
        self.model_path = args_dict.get('model_path')
        self.language = args_dict.get('language')
        self.batch_size = 1
        self.score_type = float
        self.data_type = 'text'
        self.score_name = 'PerplexityScore'
        self.model = KenlmModel.from_pretrained(self.model_path, self.language)

    def evaluate_batch(self, batch):
        input_texts = next(iter(batch.values()))

        results = []
        for text in input_texts:
            perplexity = self.model.get_perplexity(text)
            results.append(perplexity)
        
        return results


