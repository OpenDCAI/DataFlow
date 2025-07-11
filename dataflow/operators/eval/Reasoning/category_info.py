from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.reasoning.CategoryFuzz import CategoryUtils
import pandas as pd


@OPERATOR_REGISTRY.register()
class CategoryInfo(OperatorABC):
    def __init__(self):
        self.logger = get_logger()
        self.logger.info(f'Initializing {self.__class__.__name__}...')
        self.logger.info(f'{self.__class__.__name__} initialized.')
        self.information_name = "Category Information"
        self.category_list = CategoryUtils().secondary_categories

    def get_category_info(self, samples, input_primary_category_key = "primary_category", input_secondary_category_key = "secondary_category"):
        primary_categories = [sample.get(input_primary_category_key, '') for sample in samples]
        secondary_categories = [sample.get(input_secondary_category_key, '') for sample in samples]
        primary_categories_count = pd.Series(primary_categories).value_counts().to_dict()
        secondary_categories_count = pd.Series(secondary_categories).value_counts().to_dict()

        output = []
        for primary in self.category_list:
            js = {}
            if primary not in primary_categories_count:
                continue
            js["primary_num"] = primary_categories_count[primary]
            for secondary in self.category_list[primary]:
                if secondary not in secondary_categories_count:
                    continue
                js[secondary] = secondary_categories_count[secondary]
            output[primary] = js
        self.logger.info(f"Category information: {output}")
        return output


        
        
    
    def run(self,storage: DataFlowStorage, input_primary_category_key: str = "primary_category", input_secondary_category_key: str = "secondary_category"):
        self.input_primary_category_key = input_primary_category_key
        self.input_secondary_category_key = input_secondary_category_key
        dataframe = storage.read("dataframe")
        if self.input_primary_category_key not in dataframe.columns or self.input_secondary_category_key not in dataframe.columns:
            self.logger.error(f"Input keys {self.input_primary_category_key} or {self.input_secondary_category_key} not found in dataframe columns.")
            return {}
        samples = dataframe.to_dict(orient='records')
        category_info = self.get_category_info(samples, self.input_primary_category_key, self.input_secondary_category_key)
        return category_info
        