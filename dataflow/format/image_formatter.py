import pandas as pd
import json
from dataflow.data import ImageDataset, ImageCaptionDataset
from dataflow.utils.registry import FORMATTER_REGISTRY

@FORMATTER_REGISTRY.register()
class PureImageFormatter():
    def __init__(self, cfg):
        self.image_key = cfg["image_key"]
        self.image_folder_path = cfg["data_path"]
        if hasattr(cfg, 'id_key'):
            self.id_key = cfg["id_key"]
        else:
            self.id_key = None
        self.meta_path = cfg["meta_data_path"]

    def load_dataset(self):
        if self.meta_path.endswith('.csv'):
            dataset = pd.read_csv(self.meta_path)
        elif self.meta_path.endswith('.tsv'):
            dataset = pd.read_csv(self.meta_path, sep="\t")
        elif self.meta_path.endswith('.json'):
            # df = pd.read_json(self.meta_path)
            with open(self.meta_path, 'r', encoding='utf-8') as file:
                dataset = json.load(file)
        elif self.meta_path.endswith('.parquet'):
            dataset = pd.read_parquet(self.meta_path)
        else:
            raise ValueError(f"Unsupported file type: {self.meta_path}")
        return ImageDataset(dataset=dataset, image_key=self.image_key, image_folder_path=self.image_folder_path, id_key=self.id_key)

@FORMATTER_REGISTRY.register()
class ImageCaptionFormatter():
    def __init__(self, cfg):
        print(f"cfg: {cfg}")
        self.image_key = cfg["image_key"]
        self.caption_key = cfg["image_caption_key"]
        self.image_folder_path = cfg["data_path"]
        if hasattr(cfg, 'id_key'):
            self.id_key = cfg["id_key"]
        else:
            self.id_key = None
        self.meta_path = cfg["meta_data_path"]

    def load_dataset(self):
        if self.meta_path.endswith('.csv'):
            dataset = pd.read_csv(self.meta_path)
        elif self.meta_path.endswith('.tsv'):
            dataset = pd.read_csv(self.meta_path, sep="\t")
        elif self.meta_path.endswith('.json'):
            with open(self.meta_path, 'r', encoding='utf-8') as file:
                dataset = json.load(file)
        elif self.meta_path.endswith('.parquet'):
            dataset = pd.read_parquet(self.meta_path)
        else:
            raise ValueError(f"Unsupported file type: {self.meta_path}")
        return ImageCaptionDataset(dataset=dataset, image_key=self.image_key, text_key=self.caption_key, image_folder_path=self.image_folder_path, id_key=self.id_key)
  
    
@FORMATTER_REGISTRY.register()
class GenImageFormatter():
    def __init__(self, cfg):
        self.image_key = cfg['image_key']
        self.meta_data_path = cfg['meta_data_path']
        self.data_path = cfg['data_path']
        if hasattr(cfg, 'id_key'):
            self.id_key = cfg['id_key']
        else:
            self.id_key = None
        if 'ref_meta_data_path' in cfg:
            self.ref_meta_data_path = cfg['ref_meta_data_path']
            self.ref_data_path = cfg['ref_data_path']
        else:
            self.ref_meta_data_path = None
            self.ref_data_path = None

    def load(self, meta_data_path, data_path):
        if meta_data_path.endswith('.csv'):
            dataset = pd.read_csv(meta_data_path)
        elif meta_data_path.endswith('.tsv'):
            dataset = pd.read_csv(meta_data_path, sep="\t")
        elif meta_data_path.endswith('.json'):
            with open(meta_data_path, 'r', encoding='utf-8') as file:
                dataset = json.load(file)
        elif meta_data_path.endswith('.jsonl'):
            with open(meta_data_path, 'r', encoding='utf-8') as file:
                dataset = []
                for line in file:
                    dataset.append(json.loads(line))
        elif meta_data_path.endswith('.parquet'):
            dataset = pd.read_parquet(meta_data_path)
        else:
            raise ValueError(f"Unsupported file type: {meta_data_path}")

        return ImageDataset(dataset=dataset, image_key=self.image_key, image_folder_path=data_path, id_key=self.id_key)
    
    def load_dataset(self):
        if self.ref_data_path is None:
            return self.load(self.meta_data_path, self.data_path), None
        else:
            return self.load(self.meta_data_path, self.data_path), self.load(self.ref_meta_data_path, self.ref_data_path)
