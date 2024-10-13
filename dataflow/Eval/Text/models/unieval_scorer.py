from dataflow.core import TextScorer
from .UniEval.utils import convert_to_json
from .UniEval.metric.evaluator import get_evaluator
from dataflow.utils.registry import MODEL_REGISTRY
import torch
# Unieval multi-dimension quality evaluation
# cited from: Towards a Unified Multi-Dimensional Evaluator for Text Generation
@MODEL_REGISTRY.register()
class UnievalScorer(TextScorer):
    def __init__(self, args_dict):
        super().__init__(args_dict)
        self.metrics_to_keep = args_dict.get('metrics_to_keep')
        self.device = args_dict.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = -1
        self.score_type = float 
        self.data_type = 'text'  
        self.score_name = 'UnievalScore' 

    def evaluate_batch(self, batch):
        output_list = next(iter(batch.values()))
        results = {}
        if self.metrics_to_keep.get('fluency'):
            sum_task = 'summarization'
            sum_data = convert_to_json(output_list=output_list, src_list=[''] * len(output_list), ref_list=[''] * len(output_list))
            sum_evaluator = get_evaluator(sum_task, device=self.device)
            fluency_scores = sum_evaluator.evaluate(sum_data, dims=['fluency'], print_result=False)
            results['UniEvalFluencyScore'] = [score.get('fluency', None) for score in fluency_scores]

        if self.metrics_to_keep.get('naturalness') or self.metrics_to_keep.get('understandability'):
            dialogue_task = 'dialogue'
            dialogue_data = convert_to_json(output_list=output_list, src_list=[''] * len(output_list), context_list=[''] * len(output_list))
            dialogue_evaluator = get_evaluator(dialogue_task, device=self.device)
            dialogue_scores = dialogue_evaluator.evaluate(dialogue_data, dims=['naturalness', 'understandability'], print_result=False)

            if self.metrics_to_keep.get('naturalness'):
                results['UniEvalNaturalnessScore'] = [score.get('naturalness', None) for score in dialogue_scores]

            if self.metrics_to_keep.get('understandability'):
                results['UniEvalUnderstandabilityScore'] = [score.get('understandability', None) for score in dialogue_scores]

        return results
