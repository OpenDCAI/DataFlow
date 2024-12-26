from dataflow.core import GenTextScorer
from dataflow.utils.registry import MODEL_REGISTRY
from hlepor import hlepor_score
    
def str_to_callable(func_str):

    parts = func_str.split('.')
    if len(parts) == 2 and parts[0] == 'str':
        return getattr(str, parts[1])  
    raise ValueError(f"Unsupported function: {func_str}")

@MODEL_REGISTRY.register()
class HLEPORScorer(GenTextScorer):

    def __init__(self, args_dict: dict):
        super().__init__(args_dict)
        self.batch_size = 1 
        self.data_type = "text"
        self.scorer_name = "HLEPORScorer"
        self.score_type = float

        self.alpha = args_dict.get("alpha", 1.0)  
        self.beta = args_dict.get("beta", 1.0)    
        self.n = args_dict.get("n", 1)             
        self.weight_elp = args_dict.get("weight_elp", 1.0)   
        self.weight_pos = args_dict.get("weight_pos", 1.0)   
        self.weight_pr = args_dict.get("weight_pr", 1.0)   

        self.preprocess = str_to_callable(args_dict.get("preprocess", str.lower) )
        self.separate_punctuation = args_dict.get("separate_punctuation", False)

        # Ensure preprocess is callable
        if not callable(self.preprocess):
            raise ValueError(f"Preprocess must be callable, but got {type(self.preprocess)}")

    def evaluate_batch(self, eval_batch, ref_batch=None):

        eval_data = next(iter(eval_batch.values()))  
        ref_data = next(iter(ref_batch.values())) if ref_batch else None  

        if ref_data is None:
            raise ValueError("Reference data must be provided for HLEPOR Scorer.")

        scores = []
        for eval_text, ref_text in zip(eval_data, ref_data):
            hlepor_value = hlepor_score(
                [ref_text], 
                [eval_text], 
                alpha=self.alpha, 
                beta=self.beta, 
                n=self.n, 
                weight_elp=self.weight_elp, 
                weight_pos=self.weight_pos, 
                weight_pr=self.weight_pr,
                preprocess=self.preprocess,
                separate_punctuation=self.separate_punctuation
            )
            scores.append(hlepor_value)

        return scores
    
