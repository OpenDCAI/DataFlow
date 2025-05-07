import json
import pandas as pd
from abc import ABC, abstractmethod
from typing import Any, Literal

class DataFlowStorage(ABC):
    
    @abstractmethod
    def read(self, key: Any) -> Any:
        pass
    
    @abstractmethod
    def write(self, key: Any, data: Any) -> Any:
        pass
    
class DataFlowFileStorage(DataFlowStorage):

    def __init__():
        pass
        
    def read(self, key: str, type: Literal['dataframe', 'dict']):
        if type == "dict":
            with open(key, 'r') as f:
                return [json.loads(_) for _ in f]
        elif type == "dataframe":
          return pd.read_json(key)  
        
    def write(self, key: str, data: list):
        with open(key, 'w') as f:
            for item in data:
                for k, v in item.items():
                    if pd.isna(v):
                        item[k] = None
                json.dump(item, f)
                f.write('\n')
                
    def write(self, key: str, data: pd.DataFrame):
        data.to_json(key, orient='records', lines=True)
