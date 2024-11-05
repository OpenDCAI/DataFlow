import datasets
import json
import pyarrow.parquet as pq
from dataflow.utils.registry import FORMATTER_REGISTRY
from dataflow.data.text_dataset import TextDataset
import os

@FORMATTER_REGISTRY.register()
class TextFormatter:
    def __init__(self, cfg):
        self.dataset_name = cfg.get('dataset_name', None) 
        self.dataset_split = cfg.get('dataset_split', None) 
        self.name = cfg.get('name', None) 
        self.data_dir = cfg.get('data_path', None) 
        self.keys = cfg.get('keys', None)  
        self.use_hf = cfg.get('use_hf')

    def load_dataset(self) -> TextDataset:
        if self.use_hf:
            return self.load_hf_dataset(
                dataset_name=self.dataset_name,
                dataset_split=self.dataset_split,
                name=self.name,
                keys=self.keys
            )
        elif self.data_dir:
            return self.load_local_dataset(
                file_path=self.data_dir,
                keys=self.keys            
            )
        else:
            raise RuntimeError("No valid dataset configuration found. Please provide either 'dataset_name' or 'data_dir'.")

    def load_hf_dataset(self, dataset_name, dataset_split=None, name=None, keys=None) -> TextDataset:
        load_kwargs = {
            "path": dataset_name,        
            "split": dataset_split,    
            "name": name                  
        }
        
        dataset = datasets.load_dataset(**{k: v for k, v in load_kwargs.items() if v is not None})

        metadata = {
            "description": dataset.info.description if hasattr(dataset, "info") else None,
            "features": dataset.info.features if hasattr(dataset, "info") else None,
            "version": dataset.info.version if hasattr(dataset, "info") else None
        }

        return TextDataset(
            dataset=dataset,
            keys=keys,
            metadata=metadata 
        )

    def load_local_dataset(self, file_path: str, keys=None) -> TextDataset:
        file_extension = os.path.splitext(file_path)[1].lower()
        metadata = None
        dataset = None

        if file_extension == '.json':
            with open(file_path, 'r') as f:
                json_data = json.load(f)

            if "metadata" in json_data:
                metadata = json_data.pop("metadata")
            
            dataset = json_data["data"] if "data" in json_data else json_data
        
        elif file_extension == '.jsonl':
            dataset = []
            metadata = None 

            with open(file_path, 'r') as f:
                for line in f:
                    dataset.append(json.loads(line.strip()))
        
        elif file_extension == '.parquet':
            table = pq.read_table(file_path)
            dataset = table.to_pydict()
            dataset = [{k: v[i] for k, v in dataset.items()} for i in range(len(next(iter(dataset.values()))))]
            metadata = table.schema.metadata  
        
        else:
            raise RuntimeError(f"Unsupported file format: {file_extension}. Only .json, .jsonl and .parquet are supported.")

        return TextDataset(
            dataset=dataset,
            keys=keys,
            metadata=metadata 
        )
