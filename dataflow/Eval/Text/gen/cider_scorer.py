import os
import json
from dataflow.core import GenTextScorer
from dataflow.utils.registry import MODEL_REGISTRY
from dataflow.Eval.Text.gen.ciderscorer.cider import Cider
import pickle


def load_idf(idf_path):
    with open(idf_path, 'rb') as f:
        idf = pickle.load(f, encoding='utf-8')  
    return idf

@MODEL_REGISTRY.register()
class CiderScorer(GenTextScorer):
    def __init__(self, args_dict: dict):
        super().__init__(args_dict)
        self.batch_size = 1
        self.data_type = "text"
        self.scorer_name = "CiderScorer"
        self.score_type = float
        self.n = args_dict.get("n", 4)  # Max n-gram length (default: 4)
        self.sigma = args_dict.get("sigma", 6.0)  # Sigma for Gaussian penalty (default: 6.0)
        
        # Decide which IDF file to load based on 'df_mode'
        df_mode = args_dict.get("df_mode", "coco-val-df")  # Default to 'coco-val-df'
        if df_mode != "corpus":
            idf_path = args_dict.get("idf_path", "dataflow/Eval/Text/gen/ciderscorer/coco-val-df.p")
            self.idf = load_idf(idf_path)
        else:
            self.idf = None  # No need to load IDF for 'corpus' mode

    def evaluate_batch(self, eval_batch, ref_batch=None):
        eval_data = next(iter(eval_batch.values()))  
        ref_data = next(iter(ref_batch.values())) if ref_batch else None 

        if ref_data is None:
            raise ValueError("Reference data must be provided for CIDEr Scorer.")

        scores = []
        for eval_text, ref_text in zip(eval_data, ref_data):
            cider_scorer = Cider(
                test=eval_text,
                refs=[ref_text],
                n=self.n,
                sigma=self.sigma,
                idf=self.idf  # Pass IDF (None if using 'corpus')
            )

            # Pass df_mode dynamically based on the argument
            cider_score, _ = cider_scorer.compute_score(df_mode='corpus' if self.idf is None else 'coco-val-df')  
            scores.append(cider_score)  

        return scores
