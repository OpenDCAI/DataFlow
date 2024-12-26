import sacrebleu
from dataflow.core import GenTextScorer
from dataflow.utils.registry import MODEL_REGISTRY

@MODEL_REGISTRY.register()
class CHRFScorer(GenTextScorer):
    def __init__(self, args_dict: dict):
        super().__init__(args_dict)
        self.batch_size = 1
        self.data_type = "text"
        self.scorer_name = "CHRFScorer"
        self.score_type = float

        # Optional, you can add parameters for CHRF scoring if needed
        self.char_order = args_dict.get("char_order", 6)
        self.word_order = args_dict.get("word_order", 0)
        self.beta = args_dict.get("beta", 3)

    def evaluate_batch(self, eval_batch, ref_batch=None):
        eval_data = next(iter(eval_batch.values()))  # Evaluate texts
        ref_data = next(iter(ref_batch.values())) if ref_batch else None  # Reference texts

        if ref_data is None:
            raise ValueError("Reference data must be provided for CHRF Scorer.")

        # Compute CHRF score for each pair of evaluation and reference texts
        results = []
        for eval_text, ref_text in zip(eval_data, ref_data):
            # Compute the CHRF score for hypotheses and references
            result = sacrebleu.sentence_chrf(
                hypothesis=eval_text,
                references=[ref_text],
                char_order=self.char_order,
                word_order=self.word_order,
                beta=self.beta
            ).score
            results.append(result)

        return results
