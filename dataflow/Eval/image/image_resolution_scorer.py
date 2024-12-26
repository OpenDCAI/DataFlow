from dataflow.core.scorer import ImageScorer
from dataflow.utils.registry import MODEL_REGISTRY
from dataflow.utils.image_utils import image_collate_fn

@MODEL_REGISTRY.register()
class ImageResolutionScorer(ImageScorer):
    def __init__(self, args_dict: dict):
        super().__init__(args_dict)
        self.data_type = "image"
        self.scorer_name = "ImageResolutionScorer"
        self.collate_fn = image_collate_fn
        self.score_type_list = ['width', 'height']
    
    def evaluate_batch(self, sample):
        # format of return scores:
        # [
        #  {'width': ndarray, 'height': ndarray},
        #  {'width': ndarray, 'height': ndarray},
        #  ...
        # ]
        scores = []
        for img in sample:
            scores.append({'width': img.size[0], 'height': img.size[1]})

        return scores
    