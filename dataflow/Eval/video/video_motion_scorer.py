import os
import cv2
import torch
import numpy as np
from decord import VideoReader

from dataflow.data import PureVideoDataset
from dataflow.core import VideoScorer
from dataflow.utils.registry import MODEL_REGISTRY
from jsonargparse.typing import PositiveFloat, PositiveInt
@MODEL_REGISTRY.register()
class VideoMotionScorer(VideoScorer):

    def __init__(self, args_dict: dict, cfg=None):
        
        super().__init__(args_dict)
        self.cfg = cfg
        self.scorer_name = 'VideoMotionScorer'
        self.flow_kwargs = {
            'pyr_scale': 0.5,
            'levels': 3,
            'winsize': 15,
            'iterations': 3,
            'poly_n': 5,
            'poly_sigma': 1.2,
            'flags': 0
        }
        if 'sampling_fps' in args_dict.keys():
            self.sampling_fps = args_dict['sampling_fps']
        if 'relative' in args_dict.keys():
            self.relative = args_dict['relative']

    def evaluate_batch(self, sample, key=None, rank=None):
        video_motion_scores = []
        cap = cv2.VideoCapture(sample['video'][0])
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            sampling_fps = min(self.sampling_fps, fps)
            sampling_step = round(fps / sampling_fps)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            sampling_step = max(min(sampling_step, total_frames - 1),
                                1)
        prev_frame = None
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_frame is None:
                prev_frame = gray_frame
                continue
            
            # Calculate the magnitude of pixel movement between frames.
            flow = cv2.calcOpticalFlowFarneback(
                prev_frame, gray_frame, None, **self.flow_kwargs)
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            frame_motion_score = np.mean(mag)
            if self.relative:
                frame_motion_score /= np.hypot(*flow.shape[:2])
            video_motion_scores.append(frame_motion_score)
            prev_frame = gray_frame

            # quickly skip frames
            frame_count += sampling_step
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        cap.release()
        return torch.tensor([video_motion_scores]).mean()