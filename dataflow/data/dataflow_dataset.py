import os
import json
import typing
import numpy as np
import torch

class DataFlowDataset(torch.utils.data.Dataset):

    def __init__(self, args=None):
        self.map_func = []
        self.cache = {}
        self.score_record = None
        pass

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass
    
    def set_score_record(self, score_record):
        self.score_record = score_record

    def apply(self, function: typing.Callable):
        print(len(self))
        return np.array([function(sample) for sample in self])

    def map(self, function: typing.Callable, is_lazy=True, is_copy=False):
        self.map_func.append(function)
        self.cache.clear()
        return self
    
    def filter(self, function: typing.Callable):
        labels = self.apply(function)
        indices = np.where(labels == 1)[0]
        return DataFlowSubset(self, list(indices))        
        
    
class DataFlowSubset(DataFlowDataset):

    def __init__(self, dataset: DataFlowDataset, indices: list[int]) -> None:
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx: int):
        if isinstance(idx, list):
            return self.dataset[[self.indices[i] for i in idx]]
        return self.dataset[int(self.indices[idx])]

    def __getitems__(self, indices: list[int]) -> list:
        if callable(getattr(self.dataset, "__getitems__", None)):
            return self.dataset.__getitems__([self.indices[idx] for idx in indices])
        else:
            return [self.dataset[self.indices[idx]] for idx in indices]

    def __len__(self):
        return len(self.indices)
            
