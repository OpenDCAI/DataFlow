from dataflow.core import GenTextScorer
from dataflow.utils.registry import MODEL_REGISTRY
import evaluate

@MODEL_REGISTRY.register()
class BleurtScorer(GenTextScorer):
    def __init__(self, args_dict: dict):
        super().__init__(args_dict)
        self.batch_size = 1 
        self.data_type = "text"
        self.scorer_name = "BleurtScorer"
        self.score_type = float
        

        self.bleurt = evaluate.load("bleurt", module_type="metric", checkpoint="bleurt-base-128")

    def evaluate_batch(self, eval_batch, ref_batch=None):
        eval_data = next(iter(eval_batch.values()))  
        ref_data = next(iter(ref_batch.values())) if ref_batch else None  

        if ref_data is None:
            raise ValueError("Reference data must be provided for BLEURT Scorer.")

        scores = []
        for eval_text, ref_text in zip(eval_data, ref_data):
        
            results = self.bleurt.compute(predictions=[eval_text], references=[ref_text])
            scores.append(results['scores'][0]) 

        return scores
