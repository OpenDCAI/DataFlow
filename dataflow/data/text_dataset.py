from typing import Optional, Union, Generator, List, Dict
import pandas as pd
from dataflow.data.dataflow_dataset import DataFlowDataset, DataFlowSubset
import pandas as pd
import hashlib
import json
import os
import numpy as np

_MAX_ROWS_FOR_DIGEST_COMPUTATION = 10000

class TextDataset(DataFlowDataset):
    def __init__(self, dataset, keys: Optional[str] = None, metadata: Optional[dict] = None):
        super().__init__()
        self.dataset = dataset  
        self.keys = keys  
        self.metadata = metadata if metadata is not None else {}

    def __getitem__(self, index):
        if isinstance(index, slice):
            sliced_dataset = self.dataset[index]  
            return TextDataset(sliced_dataset, keys=self.keys)
        else:
            sample = self.dataset[index]
            return sample


    def __len__(self):
        return len(self.dataset)
    
    # def filter(self, labels):
    #     indices = np.where(labels == 1)[0]
    #     return TextSubset(self, list(indices))    
    
    def compute_pandas_digest(self, df: pd.DataFrame) -> str:
        df_str = df.to_string(index=False, header=False)
        digest = hashlib.sha256(df_str.encode('utf-8')).hexdigest()
        return digest

    def _compute_digest(self) -> str:
        if hasattr(self.dataset, "to_pandas"):
            df = next(self.dataset.to_pandas(batch_size=_MAX_ROWS_FOR_DIGEST_COMPUTATION, batched=True))
        else:
            df = pd.DataFrame(self.dataset[:_MAX_ROWS_FOR_DIGEST_COMPUTATION])
        return self.compute_pandas_digest(df)

    def get_data(self) -> Generator[Union[dict, str], None, None]:
        for data_raw in self.dataset:
            yield data_raw

    @property
    def profile(self) -> Optional[dict]:
        num_rows = len(self.dataset)
        if hasattr(self.dataset, "info"):
            dataset_size = self.dataset.info.size_in_bytes / (1024 * 1024) 
        else:
            dataset_size = len(pd.DataFrame(self.dataset).to_json()) / (1024 * 1024) 
        return {
            "metadata": self.get_metadata(),
            "num_rows": num_rows,
            "dataset_size": f"{dataset_size:.2f} MB", 
            "digest": self._compute_digest(),
            "keys": self.keys
        }

    def to_list(self) -> List[Dict]:
        """
        Convert the TextDataset to a list of dictionaries.
        """
        return [self[i] for i in range(len(self))]
    
    def to_dict(self) -> List[Dict]:
        data_as_dicts = []
        for i in range(len(self)):
            sample = self[i]
            if isinstance(sample, dict):
                data_as_dicts.append(sample)
            else:
                data_as_dicts.append({"data": sample})
        return data_as_dicts

    
    def update_metadata(self, new_metadata: dict):
        self.metadata.update(new_metadata)

    def get_metadata(self):
        return self.metadata
    
    def get_dump_data(self):
        return self.dataset


# class TextSubset(TextDataset):
#     def __init__(self, dataset: TextDataset, indices: list[int]) -> None:
#         super().__init__(dataset=dataset.dataset, keys=dataset.keys, metadata=dataset.metadata)
#         self.indices = indices 

#     def __getitem__(self, idx: Union[int, list[int]]):
#         if isinstance(idx, slice):
#             start, stop, step = idx.start, idx.stop, idx.step
#             sliced_indices = self.indices[start:stop:step]
#             return TextSubset(self, sliced_indices)
#         elif isinstance(idx, list):
#             subset_indices = [self.indices[i] for i in idx]
#             return TextSubset(self, subset_indices)
#         else:
#             return self.dataset[int(self.indices[idx])]

#     def __len__(self):
#         return len(self.indices)
    
#     def get_indices(self):
#         return self.indices
