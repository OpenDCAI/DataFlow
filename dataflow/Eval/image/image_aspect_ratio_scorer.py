import numpy as np

from dataflow.core.scorer import ImageScorer
from dataflow.utils.registry import MODEL_REGISTRY
from dataflow.utils.image_utils import image_collate_fn

@MODEL_REGISTRY.register()
class ImageAspectRatioScorer(ImageScorer):
    def __init__(self, args_dict: dict):
        super().__init__(args_dict)
        self.data_type = "image"
        self.scorer_name = "ImageAspectRatioScorer"
        self.collate_fn = image_collate_fn
    
    def evaluate_batch(self, sample):
        scores = []
        for img in sample:
            scores.append(img.size[0] / img.size[1])
        scores = np.array(scores)

        return scores
    