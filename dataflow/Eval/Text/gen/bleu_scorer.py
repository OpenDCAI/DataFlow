from dataflow.core import GenTextScorer
from dataflow.utils.registry import MODEL_REGISTRY
from dataflow.Eval.Text.gen.bleuscorer.bleu import Bleu  

@MODEL_REGISTRY.register()
class BleuScorer(GenTextScorer):
    def __init__(self, args_dict: dict):
        super().__init__(args_dict)
        self.batch_size = 1  # Default batch size to 1
        self.data_type = "text"
        self.scorer_name = "BleuScorer"
        self.score_type = float
        
        self.n = args_dict.get("n", 4)  # Max n-gram length (default: 4)
        self.eff = args_dict.get("eff", "average")  # Effective reference length calculation method
        self.special_reflen = args_dict.get("special_reflen", None)  # Special reference length if specified

    def evaluate_batch(self, eval_batch, ref_batch=None):
        eval_data = next(iter(eval_batch.values()))  # Extract generated text
        ref_data = next(iter(ref_batch.values())) if ref_batch else None  # Extract reference text

        if ref_data is None:
            raise ValueError("Reference data must be provided for BLEU Scorer.")

        scores = []
        for eval_text, ref_text in zip(eval_data, ref_data):
            bleu_scorer = Bleu(
                test=eval_text,
                refs=[ref_text],
                n=self.n,
                special_reflen=self.special_reflen,
            )
            
            bleu_score, _ = bleu_scorer.compute_score(option=self.eff)
            scores.append(bleu_score[0]) 

        return scores
