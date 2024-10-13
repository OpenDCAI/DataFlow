import os
from .dataflow_dataset import DataFlowDataset, DataFlowSubset

import numpy as np
import torch
from typing import List, Dict, Any

class VideoCaptionDataset(DataFlowDataset):

    def __init__(self, meta_data, video_folder):
        
        super().__init__()
        self.meta_data = meta_data
        self.video_folder = video_folder

    def __getitem__(self, index) :
        
        sample_meta_data = self.meta_data[index]

        return {
            'captions': sample_meta_data['enCap'].tolist(),
            'video': os.path.join(self.video_folder, sample_meta_data['videoID'] + '.mp4') if 'videoID' in sample_meta_data.keys() else os.path.join(self.video_folder, sample_meta_data['video'])
        }
    
    def __len__(self):
        return len(self.meta_data)