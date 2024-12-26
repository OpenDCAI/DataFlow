import evaluate
from dataflow.core import GenTextScorer
from dataflow.utils.registry import MODEL_REGISTRY

@MODEL_REGISTRY.register()
class TERScorer(GenTextScorer):
    def __init__(self, args_dict: dict):
        super().__init__(args_dict)
        self.batch_size = 1
        self.data_type = "text"
        self.scorer_name = "TERScorer"
        self.score_type = float

        # Optional, you can add parameters for TER scoring if needed
        self.normalized = args_dict.get("normalized", True)
        self.case_sensitive = args_dict.get("case_sensitive", True)

        # Load the TER metric
        self.ter = evaluate.load("ter")

    def evaluate_batch(self, eval_batch, ref_batch=None):
        eval_data = next(iter(eval_batch.values()))  # Evaluate texts
        ref_data = next(iter(ref_batch.values())) if ref_batch else None  # Reference texts

        if ref_data is None:
            raise ValueError("Reference data must be provided for TER Scorer.")

        # Compute TER score for each pair of evaluation and reference texts
        results = []
        for eval_text, ref_text in zip(eval_data, ref_data):
            # TER expects references as a list of lists, so we wrap the reference in a list
            reference_list = [ref_text] if isinstance(ref_text, str) else ref_text
            result = self.ter.compute(predictions=[eval_text], references=[reference_list], 
                                      normalized=self.normalized, case_sensitive=self.case_sensitive)

            # Extract the TER score from the result dictionary
            ter_score = result['score']  
            results.append(ter_score) 

        return results
