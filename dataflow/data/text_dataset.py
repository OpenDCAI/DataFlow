from typing import Optional, Union, Generator, List, Dict
import pandas as pd
from dataflow.data.dataflow_dataset import DataFlowDataset
import pandas as pd
import hashlib

_MAX_ROWS_FOR_DIGEST_COMPUTATION = 10000

class TextDataset(DataFlowDataset):
    def __init__(self, dataset, keys: Optional[str] = None, metadata: Optional[dict] = None):
        super().__init__()
        self.dataset = dataset  
        self.keys = keys  
        self.metadata = metadata if metadata is not None else {}

    def __getitem__(self, index):
        if isinstance(index, slice):
            start = index.start if index.start is not None else 0
            stop = index.stop if index.stop is not None else len(self.dataset)
            step = index.step if index.step is not None else 1
            indices = list(range(start, stop, step))
            sliced_dataset = self.dataset.select(indices)
            
            return TextDataset(sliced_dataset, keys=self.keys) 
        else:
            sample = self.dataset[index]
            return sample


    def __len__(self):
        return len(self.dataset)
    
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
    
    def update_metadata(self, new_metadata: dict):
        self.metadata.update(new_metadata)

    def get_metadata(self):
        return self.metadata
