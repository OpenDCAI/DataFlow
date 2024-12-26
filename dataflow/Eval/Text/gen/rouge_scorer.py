from dataflow.core import GenTextScorer
from dataflow.utils.registry import MODEL_REGISTRY
from dataflow.Eval.Text.gen.rouge_scorer.rouge import Rouge  

@MODEL_REGISTRY.register()
class RougeScorer(GenTextScorer):
    def __init__(self, args_dict: dict):
        """
        Initializes the RougeScorer with the provided configuration.
        :param args_dict: dict : Configuration dictionary for the scorer
        """
        super().__init__(args_dict)
        
        self.batch_size = 1  # Default batch size to 1
        self.data_type = "text"
        self.scorer_name = "RougeScorer"
        self.score_type = float
        
        # Get beta for ROUGE-L score computation, default is 1.2
        self.beta = args_dict.get("beta", 1.2)

    def evaluate_batch(self, eval_batch, ref_batch=None):
        """
        Evaluate the batch of generated text against reference text using ROUGE-L.
        :param eval_batch: dict : Batch of generated text, indexed by image or example ID
        :param ref_batch: dict : Batch of reference text, indexed by image or example ID
        :returns: list : List of ROUGE-L scores for each evaluation example
        """
        # Extract generated and reference text from the batches
        eval_data = next(iter(eval_batch.values()))  # Generated text
        ref_data = next(iter(ref_batch.values())) if ref_batch else None  # Reference text

        if ref_data is None:
            raise ValueError("Reference data must be provided for ROUGE Scorer.")
        
        scores = []
        
        # Loop through each pair of generated text and reference text
        for eval_text, ref_text in zip(eval_data, ref_data):
            # Instantiate the Rouge scorer
            rouge_scorer = Rouge(beta=self.beta)
            
            # Compute the ROUGE-L score using the calc_score method from Rouge
            rouge_score = rouge_scorer.calc_score([eval_text], [ref_text])
            scores.append(rouge_score)

        return scores
