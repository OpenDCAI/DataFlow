import os
import json
import typing
import numpy as np
import torch
from dataflow.utils import check_serializable_fields

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
    
    def filter(self, labels):
        if not isinstance(labels, np.ndarray):
            labels = np.array(labels)  
        indices = np.where(labels == 1)[0]
        return DataFlowSubset(self, indices.tolist())

    
    def dump(self, save_path):
        import json
        import uuid
        if os.path.isdir(save_path):
            save_file = os.path.join(save_path, uuid.uuid4().hex + '.jsonl')  
        else:
            if not save_path.endswith('.jsonl'):
                save_file = save_path + '.jsonl' 
            else:
                save_file = save_path

        os.makedirs(os.path.dirname(save_file), exist_ok=True)

        first_item = self.get_dump_data()[0]
        serializable_fields = check_serializable_fields(first_item)

        with open(save_file, 'w') as f:
            for item in self.get_dump_data():
                item_copy = {key: value for key, value in item.items() if key in serializable_fields}
                json.dump(item_copy, f)
                f.write('\n')

        print(f'Data saved to {save_file}')

    def set_image_preprocess(self, preprocess):
        self.image_preprocess = preprocess

    def set_text_preprocess(self, preprocess):
        self.text_preprocess = preprocess
        
    
class DataFlowSubset(DataFlowDataset):

    def __init__(self, dataset: DataFlowDataset, indices: list[int]) -> None:
        self.dataset = dataset
        self.indices = indices
        if hasattr(dataset, 'keys'):
            self.keys = dataset.keys
        # self.score_record = dataset.score_record

    def get_dataset(self):
        return self.dataset
    
    def get_indices(self):
        return self.indices

    def __getitem__(self, idx: int):
        if isinstance(idx, list):
            return self.dataset[[self.indices[i] for i in idx]]
        return self.dataset[int(self.indices[idx])]

    def __getitems__(self, indices: list[int]) -> list:
        if callable(getattr(self.dataset, "__getitems__", None)):
            return self.dataset.__getitems__([self.indices[idx] for idx in indices])
        else:
            return [self.dataset[self.indices[idx]] for idx in indices]

    def dump(self, save_path):
        import json
        import uuid
        if os.path.isdir(save_path):  
            save_file = os.path.join(save_path, uuid.uuid4().hex + '.jsonl') 
        else:
            if not save_path.endswith('.jsonl'):
                save_file = save_path + '.jsonl' 
            else:
                save_file = save_path 
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        first_item = self.dataset.get_dump_data()[0]
        serializable_fields = check_serializable_fields(first_item)
        original_dataset = self.dataset.get_dump_data()
        with open(save_file, 'w') as f:
            for index in self.indices:
                item = original_dataset[index]
                item_copy = {key: value for key, value in item.items() if key in serializable_fields}

                json.dump(item_copy, f)
                f.write('\n')
        print(f'Data saved to {save_file}')


    
    def __len__(self):
        return len(self.indices)
    
    def set_image_preprocess(self, preprocess):
        self.dataset.set_image_preprocess(preprocess)

    def set_text_preprocess(self, preprocess):
        self.dataset.set_text_preprocess(preprocess)
            
class DataFlowDSDict():
    
    def __init__(self):
        self.ds_dict = {}
        self.indices = None
    
    def __setitem__(self, name: str, value: DataFlowDataset) -> None:
        if name not in self.ds_dict.keys():
            if isinstance(value, DataFlowSubset):
                self.ds_dict[name] = value.get_dataset()
            else:
                self.ds_dict[name] = value
        else:
            if isinstance(value, DataFlowSubset):
                if self.indices is None:
                    self.indices = value.get_indices()
                else:
                    import numpy as np
                    self.indices = np.array(self.indices)[value.get_indices()].tolist()
                
    def __getitem__(self, key):
        if self.indices is None:
            return self.ds_dict[key]
        else:
            return DataFlowSubset(self.ds_dict[key], self.indices)

    def __delitem__(self, key: str):
        del self.ds_dict[key]
    
    def keys(self):
        return self.ds_dict.keys()
    
    def values(self):
        return [self[key] for key in self.keys()]
        
    def items(self):
        return [(key, self[key]) for key in self.keys()]
