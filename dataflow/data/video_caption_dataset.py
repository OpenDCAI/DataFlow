import os
from .dataflow_dataset import DataFlowDataset
class VideoCaptionDataset(DataFlowDataset):

    def __init__(self, meta_data, video_folder):
        
        super().__init__()
        self.meta_data = meta_data
        self.video_folder = video_folder

    def __getitem__(self, index) :
        
        sample_meta_data = self.meta_data[index]

        return {
            'captions': sample_meta_data['enCap'].tolist() if type(sample_meta_data['enCap']) is not list else sample_meta_data['enCap'],
            'video': os.path.join(self.video_folder, sample_meta_data['videoID'] + '.mp4') if 'videoID' in sample_meta_data.keys() else os.path.join(self.video_folder, sample_meta_data['video'])
        }
    
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