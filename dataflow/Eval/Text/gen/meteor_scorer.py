from dataflow.core import GenTextScorer
from dataflow.utils.registry import MODEL_REGISTRY
from dataflow.Eval.Text.gen.meteorscorer.meteor import Meteor  

@MODEL_REGISTRY.register()
class MeteorScorer(GenTextScorer):
    def __init__(self, args_dict: dict):
        super().__init__(args_dict)
        self.batch_size = 1  
        self.data_type = "text"
        self.scorer_name = "MeteorScorer"
        self.score_type = float
        
        self.language = args_dict.get("language", "en") 
    def evaluate_batch(self, eval_batch, ref_batch=None):
        eval_data = next(iter(eval_batch.values()))  
        ref_data = next(iter(ref_batch.values())) if ref_batch else None  

        if ref_data is None:
            raise ValueError("Reference data must be provided for Meteor Scorer.")

        scores = []
        for eval_text, ref_text in zip(eval_data, ref_data):

            meteor_scorer = Meteor(language=self.language)  
            score = meteor_scorer.compute_score(eval_text, [ref_text])  

            scores.append(score)

        return scores
