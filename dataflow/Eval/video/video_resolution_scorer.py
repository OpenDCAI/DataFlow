import sys
import av

import numpy as np
from jsonargparse.typing import PositiveInt
from dataflow.core import VideoScorer
from dataflow.utils.registry import MODEL_REGISTRY

@MODEL_REGISTRY.register()
class VideoResolutionScorer(VideoScorer):

    def __init__(self, args_dict):
        super().__init__(args_dict)

    def init_score(self, len_dataset):
        '''
        return empty score dict for this scorer
        eg: {'Default': np.array([-1] * len_dataset)}
        '''
        return {'width': np.array([np.nan] * len_dataset), 'height': np.array([np.nan] * len_dataset)}


    def evaluate_batch(self, sample, key=None, rank=None):
        video_data = av.open(sample['video'][0])
        video_stream = video_data.streams.video[0]
        video_width, video_height = video_stream.codec_context.width, video_stream.codec_context.height
        for video_stream in video_data.streams.video:
            video_stream.close(strict=False)

        video_data.close()
        return {'width': video_width, 'height': video_height}

