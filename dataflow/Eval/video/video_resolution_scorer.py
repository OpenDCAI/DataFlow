import sys
import av

import numpy as np
from jsonargparse.typing import PositiveInt
from dataflow.core import Scorer
from dataflow.utils.registry import MODEL_REGISTRY

@MODEL_REGISTRY.register()
class VideoResolutionScorer(Scorer):

    def __init__(self,
                 min_width: PositiveInt = 1,
                 max_width: PositiveInt = sys.maxsize,
                 min_height: PositiveInt = 1,
                 max_height: PositiveInt = sys.maxsize,
                 any_or_all: str = 'any',
                 *args,
                 **kwargs):
        self.min_width = min_width
        self.max_width = max_width
        self.min_height = min_height
        self.max_height = max_height 
        if any_or_all not in ['any', 'all']:
            raise ValueError(f'Keep strategy [{any_or_all}] is not supported. '
                             f'Can only be one of ["any", "all"].')
        self.any = (any_or_all == 'any')
       
    def evaluate(self, sample, key=None, rank=None):
        video_data = av.open(sample['video'])
        video_stream = video_data.streams.video[0]
        video_width, video_height = video_stream.codec_context.width, video_stream.codec_context.height
        for video_stream in video_data.streams.video:
            video_stream.close(strict=False)

        video_data.close()
        return {'width': video_width, 'height': video_height}
    
    def __call__(self, sample, key=None, rank=None):
        return self.evaluate(sample, key)

