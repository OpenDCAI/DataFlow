import os
import tempfile
import numpy as np
from .dataflow_dataset import DataFlowDataset

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
    
    def get_dump_data(self):
        return self.meta_data    
    
    def dump(self, save_path):
        import json
        import uuid
        if os.path.exists(save_path):
            save_file = save_path if os.path.isfile(save_path) else save_path + uuid.uuid4().hex + '.json'
            with open(save_file, 'w+') as f:
                json.dump(self.meta_data)
                