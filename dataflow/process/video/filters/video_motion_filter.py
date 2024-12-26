from dataflow.Eval.video import VideoMotionScorer
from dataflow.data import DataFlowDataset
from dataflow.core import VideoFilter
import numpy as np
from dataflow.utils.registry import PROCESSOR_REGISTRY

@PROCESSOR_REGISTRY.register()
class VideoMotionFilter(VideoFilter):

    def __init__(self, args_dict: dict):
        super().__init__(args_dict)
        self.min_score = args_dict['min_score']
        self.max_score = args_dict['max_score']
        self.scorer = VideoMotionScorer(args_dict['scorer_args'])

    def filter_func(self, dataset):
        _, scores = self.scorer(dataset)
        # print(sample['video'], scores)
        return np.array([self.min_score <= score <= self.max_score for score in scores['Default']]).astype(int)
