from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
import pandas as pd


@OPERATOR_REGISTRY.register()
class DifficultyInfo(OperatorABC):
    def __init__(self):
        self.logger = get_logger()
        self.logger.info(f'Initializing {self.__class__.__name__}...')
        self.logger.info(f'{self.__class__.__name__} initialized.')
        self.information_name = "Difficulty Information"

    def get_category_info(self, samples, input_diffulty_key="difficulty_score"):
        diffultys = [sample.get(input_diffulty_key, 'null') for sample in samples]
        diffultys_count = pd.Series(diffultys).value_counts().to_dict()
        self.logger.info(f"Difficulty information: {diffultys_count}")
        return diffultys_count



        
        
    
    def run(self,storage: DataFlowStorage, input_diffulty_key: str = "difficulty_score"):
        self.input_diffulty_key = input_diffulty_key
        dataframe = storage.read("dataframe")
        if self.input_diffulty_key not in dataframe.columns:
            self.logger.error(f"Input key {self.input_diffulty_key} not found in dataframe columns.")
            return {}
        samples = dataframe.to_dict(orient='records')
        category_info = self.get_category_info(samples, self.input_diffulty_key)
        return category_info
        