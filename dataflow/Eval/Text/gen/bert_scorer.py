import evaluate
from dataflow.core import GenTextScorer
from dataflow.utils.registry import MODEL_REGISTRY

@MODEL_REGISTRY.register()
class BERTScoreScorer(GenTextScorer):
    def __init__(self, args_dict: dict):
        super().__init__(args_dict)
        self.batch_size =  1  # Default batch size
        self.data_type = "text"
        self.scorer_name = "BERTScoreScorer"
        self.score_type = float

        # Additional parameters for BERTScore
        self.lang = args_dict.get("lang", "en")  # Language (default: English)
        self.model_type = args_dict.get("model_type", "distilbert-base-uncased")  # Pretrained model for BERTScore
        self.idf = args_dict.get("idf", False)  # Whether to use IDF weighting
        self.rescale_with_baseline = args_dict.get("rescale_with_baseline", False)  # Rescale scores with baseline
        
        # Load the BERTScore metric
        self.bertscore = evaluate.load("bertscore")

    def evaluate_batch(self, eval_batch, ref_batch=None):
        eval_data = next(iter(eval_batch.values()))  # Extract predictions
        ref_data = next(iter(ref_batch.values())) if ref_batch else None  # Extract references

        if ref_data is None:
            raise ValueError("Reference data must be provided for BERTScore Scorer.")

        # Compute BERTScore for predictions and references
        results = self.bertscore.compute(
            predictions=eval_data,
            references=ref_data,
            lang=self.lang,
            model_type=self.model_type,
            idf=self.idf,
            rescale_with_baseline=self.rescale_with_baseline
        )
        
        # Extract F1 scores for batch and return
        scores = results["f1"]
        return scores
