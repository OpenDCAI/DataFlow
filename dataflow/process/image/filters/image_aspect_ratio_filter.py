import numpy as np
from dataflow.core import ImageFilter
from dataflow.Eval.image import ImageAspectRatioScorer
from dataflow.utils.registry import PROCESSOR_REGISTRY

@PROCESSOR_REGISTRY.register()
class ImageAspectRatioFilter(ImageFilter):
    def __init__(self, args_dict: dict):
        super().__init__()
        self.min_ratio = args_dict["min_ratio"] if "min_ratio" in args_dict else -np.inf
        self.max_ratio = args_dict["max_ratio"] if "max_ratio" in args_dict else np.inf

        self.scorer = ImageAspectRatioScorer(args_dict=args_dict)

    def filter_func(self, sample):
        _, score = self.scorer(sample)

        result = np.array(((self.min_ratio <= score['Default']) & (score['Default'] <= self.max_ratio)).astype(int))

        return result
