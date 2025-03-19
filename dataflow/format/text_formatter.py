import json
import pyarrow.parquet as pq
import datasets
from dataflow.utils.registry import FORMATTER_REGISTRY
from dataflow.data.text_dataset import TextDataset
import os

@FORMATTER_REGISTRY.register()
class TextFormatter:
    def __init__(self, data_dir, key, sft_single_round, sft_multi_round, RLHF):
        self.use_hf = False
        self.data_dir = data_dir
        self.keys = key
        # 以下留空
        self.sft_single_round = sft_single_round
        self.sft_multi_round = sft_multi_round
        self.RLHF = RLHF

    def load_dataset(self) -> TextDataset:
        if self.use_hf:
            return self.load_hf_dataset(
                dataset_name=self.dataset_name,
                dataset_split=self.dataset_split,
                name=self.name,
                revision=self.revision,
                keys=self.keys
            )
        elif self.data_dir:
            return self.load_local_dataset(
                file_path=self.data_dir,
                keys=self.keys            
            )
        else:
            raise RuntimeError("No valid dataset configuration found. Please provide either 'dataset_name' or 'data_dir'.")

    def load_hf_dataset(self, dataset_name, dataset_split=None, revision=None, name=None, keys=None) -> TextDataset:
        load_kwargs = {
            "path": dataset_name,        
            "split": dataset_split,
            "revision": revision,    
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
        file_extension = file_path.split('.')[-1].lower()
        supported_formats = ['json', 'jsonl', 'parquet']
        
        if file_extension not in supported_formats:
            raise RuntimeError(f"Unsupported file format: .{file_extension}. Supported formats: {supported_formats}.")
        
        dataset_format = "json" if file_extension in ["json", "jsonl"] else "parquet"
        
        dataset = datasets.load_dataset(dataset_format, data_files=file_path, split="train")


        return TextDataset(
            dataset=dataset,
            keys=keys,
            metadata=None 
        )

@FORMATTER_REGISTRY.register()
class GenTextFormatter:
    def __init__(self, cfg):
        self.eval_data_path = cfg.get('eval_data_path')  
        self.ref_data_path = cfg.get('ref_data_path')   
        self.eval_key = cfg.get('eval_key', None)
        self.ref_key = cfg.get('ref_key', None)               

    def load_local_dataset(self, file_path: str):
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
            with open(file_path, 'r') as f:
                for line in f:
                    dataset.append(json.loads(line.strip()))

        elif file_extension == '.parquet':
            table = pq.read_table(file_path)
            dataset = table.to_pydict()
            dataset = [{k: v[i] for k, v in dataset.items()} for i in range(len(next(iter(dataset.values()))))]
            metadata = table.schema.metadata

        else:
            raise RuntimeError(f"Unsupported file format: {file_extension}. Only .json, .jsonl, and .parquet are supported.")

        return TextDataset(dataset=dataset, keys=None, metadata=metadata)

    def load_dataset(self):
        if not self.eval_data_path or not self.ref_data_path:
            raise ValueError("Both 'eval_data_path' and 'ref_data_path' must be provided.")
        
        eval_dataset = self.load_local_dataset(self.eval_data_path)
        ref_dataset = self.load_local_dataset(self.ref_data_path)
        eval_dataset.keys = self.eval_key
        ref_dataset.keys = self.ref_key
        
        return eval_dataset, ref_dataset