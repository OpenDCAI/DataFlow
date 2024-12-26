import sys
from jsonargparse.typing import PositiveInt
from functools import partial
import numpy as np
from dataflow.core import VideoFilter
from dataflow.data import DataFlowDataset
from dataflow.Eval.video import VideoResolutionScorer
from dataflow.utils.registry import PROCESSOR_REGISTRY

@PROCESSOR_REGISTRY.register()
class VideoResolutionFilter(VideoFilter):

    def __init__(self, args_dict: dict):

        super().__init__(args_dict)
        self.min_width = args_dict['min_width']
        self.max_width = args_dict['max_width']
        self.min_height = args_dict['min_height']
        self.max_height = args_dict['max_height']
        self.scorer = VideoResolutionScorer(args_dict['scorer_args'])

    def filter_func(self, dataset):
        _, scores = self.scorer(dataset)
        print(scores)
        return np.array([
            self.min_width <= width <= self.max_width and
            self.min_height <= height <= self.max_height
                for width, height in zip(scores['width'], scores['height'])
        ]).astype(int)
