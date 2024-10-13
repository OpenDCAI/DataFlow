import av
import numpy as np
from jsonargparse.typing import ClosedUnitInterval, PositiveInt
# import aesthetics_predictor
import torch
import transformers
from dataflow.utils import close_video, extract_key_frames, get_key_frame_seconds, extract_video_frames_uniformly, prepare_huggingface_model
from dataflow.utils.registry import Registry, MODEL_REGISTRY
from dataflow.core import Scorer

@MODEL_REGISTRY.register()
class VideoAestheticScorer(Scorer):

    _accelerator = 'cuda'

    def __init__(self,
                 hf_scorer_model='shunk031/aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE',
                 frame_sampling_method: str = 'uniform',
                 frame_num: PositiveInt = 3,
                 any_or_all: str = 'any',
                 reduce_mode: str = 'avg',
                 *args,
                 **kwargs):
        
        self.hf_model_id = hf_scorer_model
        self.reduce_mode = reduce_mode

        self.need_normalized_by_ten = ('shunk031/aesthetics-predictor'
                                in hf_scorer_model)
        self.frame_sampling_method = frame_sampling_method
        self.frame_num = frame_num

        self.sampled_frames_key_suffix = f'-{frame_sampling_method}' + \
            ('' if frame_sampling_method == 'all_keyframes'
             else f'-{frame_num}')
        
    def evaluate(self, sample, key=None, rank=None):
        
        aesthetics_scores = []

        video = av.open(sample['video'])
        if self.frame_sampling_method == 'all_keyframes':
            frames = extract_key_frames(video)
        elif self.frame_sampling_method == 'uniform':
            frames = extract_video_frames_uniformly(
                video, self.frame_num)
        frame_images = [frame.to_image() for frame in frames]
        if len(frame_images) > 0:
            # compute aesthetics_scores
            model, processor = prepare_huggingface_model(self.hf_model_id, trust_remote_code=True)
            model.to(f'cuda:{rank}')
            # processor.to(f'cuda:{rank}')
            
            inputs = processor(images=frame_images,
                                return_tensors='pt').to(model.device)
            # print(inputs.shape)
            with torch.no_grad():
                outputs = model(**inputs)
                print(outputs)
            if self.need_normalized_by_ten:
                aesthetics_score = outputs.logits / 10.0
            else:
                aesthetics_score = outputs.logits

            if self.reduce_mode == 'avg':
                aesthetics_score = float(aesthetics_score.mean())
            elif self.reduce_mode == 'max':
                aesthetics_score = float(aesthetics_score.max())
            else:
                aesthetics_score = float(aesthetics_score.min())
        else:
            aesthetics_score = 0.0

        aesthetics_scores.append(aesthetics_score)
        print(aesthetics_scores)
        # logger.debug(f'aesthetics_score: {aesthetics_scores}')

        # sample[Fields.stats][StatsKeys.video_frames_aesthetics_score] = (
        #     aesthetics_scores)

        close_video(video)

        return aesthetics_scores

        
    def __call__(self, sample, key=None):
        return self.evaluate(sample, key)
