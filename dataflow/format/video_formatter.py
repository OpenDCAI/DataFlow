import pandas as pd
from dataflow.data import PureVideoDataset, VideoCaptionDataset
from torch.utils.data import Dataset
from dataflow.utils.registry import FORMATTER_REGISTRY

@FORMATTER_REGISTRY.register()
class PureVideoFormatter():

    def __init__(self, cfg):
        self.meta_path = cfg['meta_data_path']
        self.video_path = cfg['data_path']

    def load_dataset(self) -> Dataset:
        if self.meta_path.endswith('.csv'):
            df = pd.read_csv(self.meta_path)
        elif self.meta_path.endswith('.tsv'):
            df = pd.read_csv(self.meta_path, sep="\t")
        elif self.meta_path.endswith('.json'):
            df = pd.read_json(self.meta_path)
        elif self.meta_path.endswith('.parquet'):
            df = pd.read_parquet(self.meta_path)
        else:
            return ValueError(f"Unsupported file type: {self.meta_path}")
        meta_data = df.to_dict(orient='records')
        return PureVideoDataset(meta_data, self.video_path)

@FORMATTER_REGISTRY.register()
class VideoCaptionFormatter():
    
    def __init__(self, cfg):
        self.meta_path = cfg['meta_data_path']
        self.video_path = cfg['data_path']

    def load_dataset(self) -> Dataset:
        if self.meta_path.endswith('.csv'):
            df = pd.read_csv(self.meta_path)
        elif self.meta_path.endswith('.tsv'):
            df = pd.read_csv(self.meta_path, sep="\t")
        elif self.meta_path.endswith('.json'):
            df = pd.read_json(self.meta_path)
        elif self.meta_path.endswith('.parquet'):
            df = pd.read_parquet(self.meta_path)
        else:
            return ValueError(f"Unsupported file type: {self.meta_path}")
        meta_data = df.to_dict(orient='records')
        return VideoCaptionDataset(meta_data, self.video_path)