from dataflow.core import TextScorer
from datasets import Dataset
from dataflow.utils.registry import MODEL_REGISTRY
from .Qurating.qurater_annotate import ModelAnnotator
from .Qurating.qurater_annotate import TokenizeAndChunk
import torch

# Qurating text quality evaluation
# cited from: QuRating: Selecting High-Quality Data for Training Language Models
@MODEL_REGISTRY.register()
class QuratingScorer(TextScorer):
    def __init__(self, args_dict):
        super().__init__(args_dict)
        self.model = args_dict.get('model')
        self.tokens_field = args_dict.get('tokens_field')
        self.tokens = args_dict.get('tokens')
        self.map_batch_size = args_dict.get('map_batch_size')
        self.batch_size = -1 
        self.num_workers = args_dict.get('num_workers', 1)
        self.labels = args_dict.get('labels', [])
        self.device_batch_size = args_dict.get('device_batch_size')
        self.device = args_dict.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.score_type = float 
        self.data_type = 'text'  
        self.score_name = 'QuratingScore' 

    def evaluate_batch(self, batch):
        input_texts = next(iter(batch.values()))
        batch_dict = {'text': input_texts}
        dataset = Dataset.from_dict(batch_dict)
        dataset = dataset.map(
            TokenizeAndChunk(self.model, 'text', self.tokens_field, self.tokens),
            batched=True,
            batch_size=self.map_batch_size,
            num_proc=self.num_workers,
            remove_columns=dataset.column_names
        )
        dataset = dataset.map(
            ModelAnnotator(self.model, self.labels, self.device_batch_size, self.device),
            batched=True,
            with_indices=True,
            batch_size=self.map_batch_size,
            remove_columns=dataset.column_names
        )

        results_dict = dataset.to_dict()
        results = {}

        for i in range(len(dataset)):
            result = {key: results_dict[key][i] for key in results_dict}
            result_filtered = {}
            for label in self.labels:
                average_key = f"{label}_average"
                if average_key in result:
                    new_key = f"Qurating{''.join([word.capitalize() for word in label.split('_')])}Score"
                    result_filtered[new_key] = result[average_key]

            for key, value in result_filtered.items():
                if key not in results:
                    results[key] = []
                results[key].append(value)
        return results
