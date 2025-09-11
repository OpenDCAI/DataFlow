from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.reasoning.CategoryFuzz import CategoryUtils
import pandas as pd


@OPERATOR_REGISTRY.register()
class ReasoningCategoryDatasetEvaluator(OperatorABC):
    def __init__(self):
        self.logger = get_logger()
        self.logger.info(f'Initializing {self.__class__.__name__}...')
        self.logger.info(f'{self.__class__.__name__} initialized.')
        self.information_name = "Category Information"
        self.category_list = CategoryUtils().secondary_categories

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "该算子用于统计数据集中的类别信息，包括主类别和次类别的分布情况。"
                "它计算每个类别的样本数量，并返回类别分布的统计结果。\n"
                "输入参数：\n"
                "- input_primary_category_key：主类别字段名，默认为'primary_category'\n"
                "- input_secondary_category_key：次类别字段名，默认为'secondary_category'\n"
                "输出参数：\n"
                "- 返回包含类别统计信息的字典，主类别作为键，值为包含该类别样本数量和次类别分布的字典"
            )
        elif lang == "en":
            return (
                "This operator analyzes category distribution in the dataset, including primary and secondary categories. "
                "It counts the number of samples in each category and returns statistical results of category distribution.\n"
                "Input Parameters:\n"
                "- input_primary_category_key: Field name for primary category, default is 'primary_category'\n"
                "- input_secondary_category_key: Field name for secondary category, default is 'secondary_category'\n\n"
                "Output Parameters:\n"
                "- Returns a dictionary containing category statistics, with primary categories as keys and values as dictionaries "
                "containing sample counts and secondary category distribution"
            )
        else:
            return (
                "CategoryInfo analyzes and reports the distribution of primary and secondary categories in the dataset."
            )
    
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
        