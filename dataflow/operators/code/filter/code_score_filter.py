import pandas as pd
from typing import List, Literal

# Assuming these are the correct import paths for your framework
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC

# TODO: remove and add to general filter
@OPERATOR_REGISTRY.register()
class CodeGenericScoreFilter(OperatorABC):
    """
    CodeGenericScoreFilter is a generic score-based filtering operator that filters
    datasets based on numerical score columns. It provides flexible comparison
    methods to remove samples that don't meet specified threshold criteria.
    
    This filter directly applies score-based filtering without using evaluator scores:
    - Removes samples with scores below minimum threshold
    - Removes samples with scores above maximum threshold
    - Removes samples that don't meet specific score criteria
    - Keeps samples that meet the specified threshold criteria
    """

    def __init__(self, score_threshold: int = 8, filter_method: Literal["greater", "greater_equal", "less", "less_equal", "equal"] = "greater_equal"):
        """
        Initializes the operator.
        """
        self.logger = get_logger()
        self.score_threshold = score_threshold
        self.filter_method = filter_method
    
    @staticmethod
    def get_desc(lang: str = "en"):
        """
        Provides a description of the operator's function and parameters.
        """
        if lang == "zh":
            return (
                "基于数值分数列直接过滤数据集，提供灵活的阈值比较方法。\n\n"
                "比较方法：\n"
                "- greater_equal: 分数 >= 阈值\n"
                "- greater: 分数 > 阈值\n"
                "- less_equal: 分数 <= 阈值\n"
                "- less: 分数 < 阈值\n"
                "- equal: 分数 = 阈值\n\n"
                "输入参数：\n"
                "- input_key: 包含分数的字段名\n"
                "- output_key: 输出标签字段名 (默认: 'generic_score_filter_label')\n"
                "- score_threshold: 分数阈值 (默认: 8)\n"
                "- filter_method: 比较方法 (默认: 'greater_equal')\n\n"
                "输出参数：\n"
                "- 过滤后的DataFrame，仅保留符合分数条件的样本\n"
                "- 返回包含输出标签字段名的列表"
            )
        else: # Default to English
            return (
                "Filter datasets based on numerical score columns with flexible threshold comparison methods.\n\n"
                "Comparison Methods:\n"
                "- greater_equal: score >= threshold\n"
                "- greater: score > threshold\n"
                "- less_equal: score <= threshold\n"
                "- less: score < threshold\n"
                "- equal: score = threshold\n\n"
                "Input Parameters:\n"
                "- input_key: Field name containing the score\n"
                "- output_key: Output label field name (default: 'generic_score_filter_label')\n"
                "- score_threshold: Numerical threshold for filtering (default: 8)\n"
                "- filter_method: Comparison method to use (default: 'greater_equal')\n\n"
                "Output Parameters:\n"
                "- Filtered DataFrame containing only samples meeting score criteria\n"
                "- List containing output label field name"
            )

    def _validate_dataframe(self, dataframe: pd.DataFrame):
        """
        Validates the DataFrame to ensure the required score column exists.
        """
        required_keys = [self.input_score_key]

        missing = [k for k in required_keys if k not in dataframe.columns]

        if missing:
            raise ValueError(f"Missing required column(s) for ScoreFilter: {missing}")
        
        # Also check if the column is numeric
        if not pd.api.types.is_numeric_dtype(dataframe[self.input_score_key]):
            raise TypeError(f"Column '{self.input_score_key}' for ScoreFilter must be of a numeric type.")

    def run(
        self, 
        storage: DataFlowStorage, 
        input_key: str,
        output_key: str = "generic_score_filter_label"
    ) -> List[str]:
        """
        Execute the filtering process.
        
        Reads data from storage, applies the filter based on the score,
        and writes the filtered data back to storage.
        
        Args:
            storage: Data storage object
            input_key: Field name containing the score
            output_key: Key name for output label
            score_threshold: Numerical threshold for filtering
            filter_method: Comparison method to use
            
        Returns:
            List[str]: List containing output key name
        """
        self.input_key = input_key
        self.output_key = output_key
        self.logger.info(f"Running {self.__class__.__name__} with input_key: {self.input_key} and output_key: {self.output_key}...")
        
        # Store key for use in helper methods
        self.input_score_key = input_key

        # 1. Read data from the current step
        dataframe = storage.read("dataframe")
        if dataframe.empty:
            self.logger.warning("Input dataframe is empty. Skipping.")
            storage.write(dataframe)
            return [self.output_key]

        original_count = len(dataframe)
        
        # 2. Validate the data
        self._validate_dataframe(dataframe)
        
        # 3. Apply the filter logic and add label
        if self.filter_method == "greater_equal":
            filter_mask = dataframe[self.input_score_key] >= self.score_threshold
        elif self.filter_method == "greater":
            filter_mask = dataframe[self.input_score_key] > self.score_threshold
        elif self.filter_method == "less_equal":
            filter_mask = dataframe[self.input_score_key] <= self.score_threshold
        elif self.filter_method == "less":
            filter_mask = dataframe[self.input_score_key] < self.score_threshold
        elif self.filter_method == "equal":
            filter_mask = dataframe[self.input_score_key] == self.score_threshold
        else:
            # This case should ideally not be hit due to Literal type hint, but is good for robustness
            raise ValueError(f"Unsupported filter_method: '{filter_method}'")
        
        dataframe[self.output_key] = filter_mask.astype(int)
        filtered_df = dataframe[filter_mask]
        
        filtered_count = len(filtered_df)
        self.logger.info(f"Filtering completed. Total records passing filter: {filtered_count}.")

        # 4. Write the results back to storage
        storage.write(filtered_df)

        # 5. Return output key
        return [self.output_key]