import numpy as np
from dataflow.core import ImageFilter
from dataflow.Eval.image import ImageResolutionScorer
from dataflow.utils.registry import PROCESSOR_REGISTRY

@PROCESSOR_REGISTRY.register()
class ImageResolutionFilter(ImageFilter):
    def __init__(self, args_dict: dict):
        super().__init__()
        self.min_width = args_dict["min_width"]
        self.max_width = args_dict["max_width"]
        self.min_height = args_dict["min_height"]
        self.max_height = args_dict["max_height"]
        self.scorer = ImageResolutionScorer(args_dict=args_dict)

    def filter_func(self, sample):
        _, score = self.scorer(sample)
        width_condition = (self.min_width <= score['width']) & (score['width'] <= self.max_width)
        height_condition = (self.min_height <= score['height']) & (score['height'] <= self.max_height)
        result = np.array((width_condition & height_condition).astype(int))
        return result
