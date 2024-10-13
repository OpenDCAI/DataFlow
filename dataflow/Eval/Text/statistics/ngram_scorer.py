import re
from dataflow.core import TextScorer
from dataflow.utils.registry import MODEL_REGISTRY

# N-gram repetition evaluation
@MODEL_REGISTRY.register()
class NgramScorer(TextScorer):
    def __init__(self, args_dict: dict):
        super().__init__(args_dict)
        self.ngrams = args_dict.get('ngrams') 
        self.batch_size = 1
        self.data_type = 'text'
        self.score_name = 'NgramScore'
        self.score_type = float  
        
    def evaluate_batch(self, batch):
        input_data = next(iter(batch.values())) 
        scores = []
        for sample in input_data:
            content = sample 
            content = content.lower()
            content = re.sub(r'[^\w\s]', '', content)
            words = content.split()
            ngrams = [' '.join(words[i:i + self.ngrams]) for i in range(len(words) - (self.ngrams - 1))]
            unique_ngrams = set(ngrams)

            total_ngrams = len(ngrams)
            unique_ngrams_count = len(unique_ngrams)

            repetition_score = unique_ngrams_count / total_ngrams if total_ngrams > 0 else 0.0
            scores.append(repetition_score) 

        return scores 
