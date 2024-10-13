import json
import os
import cv2
import torch
import numpy as np
from decord import VideoReader
from .dataflow_dataset import DataFlowDataset, DataFlowSubset

class PureVideoDataset(DataFlowDataset):

    def __init__(self, meta_data, video_folder):
        super().__init__()
        self.meta_data = meta_data
        self.video_folder = video_folder

    def __getitem__(self, index):
        sample_metadata = self.meta_data[index]
        if 'flickr_id' in sample_metadata.keys():
            sample_metadata['video'] = os.path.join(self.video_folder, str(sample_metadata['flickr_id'])) + '.mp4'
        elif 'videoID' in sample_metadata.keys():
            sample_metadata['video'] = os.path.join(self.video_folder, str(sample_metadata['videoID'])) + '.mp4'
        else:
            sample_metadata['video'] = os.path.join(self.video_folder, str(sample_metadata['video']))
        for func in self.map_func:
            sample_metadata = func(sample_metadata)
        return {'video': sample_metadata['video']}

    def __len__(self):
        return len(self.meta_data)