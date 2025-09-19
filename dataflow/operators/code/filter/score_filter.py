import pandas as pd
from typing import List, Literal

# Assuming these are the correct import paths for your framework
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC

@OPERATOR_REGISTRY.register()
class ScoreFilter(OperatorABC):
    """
    ScoreFilter is a non-LLM operator that filters a dataset based on a numerical
    score column. It's used to remove low-quality samples before they proceed
    to more computationally expensive stages like sandbox validation.
    """

    def __init__(self):
        """
        Initializes the operator.
        """
        self.logger = get_logger()
    
    @staticmethod
    def get_desc(lang: str = "en"):
        """
        Provides a description of the operator's function and parameters.
        """
        if lang == "zh":
            return (
                "该算子根据一个分数栏位对数据集进行过滤，移除不符合阈值条件的样本。\n\n"
                "输入参数：\n"
                "- input_score_key: 包含分数的字段名 (默认: 'quality_score')\n"
                "- score_threshold: 用于过滤的分数阈值 (默认: 8)\n"
                "- filter_method: 过滤方法 ('greater_equal', 'less_than', etc.) (默认: 'greater_equal')\n"
            )
        else: # Default to English
            return (
                "This operator filters the dataset based on a score column, removing samples that do not meet the threshold criteria.\n\n"
                "Input Parameters:\n"
                "- input_score_key: Field name containing the score (default: 'quality_score')\n"
                "- score_threshold: The numerical threshold for filtering (default: 8)\n"
                "- filter_method: The comparison method to use ('greater_equal', 'less_than', etc.) (default: 'greater_equal')\n"
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
        input_score_key: str = "quality_score",
        score_threshold: int = 8,
        filter_method: Literal["greater", "greater_equal", "less", "less_equal", "equal"] = "greater_equal"
    ) -> List[str]:
        """
        Execute the filtering process.
        
        Reads data from storage, applies the filter based on the score,
        and writes the filtered data back to storage.
        
        Args:
            storage: Data storage object
            input_score_key: Field name containing the score
            score_threshold: Numerical threshold for filtering
            filter_method: Comparison method to use
            
        Returns:
            An empty list, as no new columns are created.
        """
        self.logger.info(f"Running ScoreFilter operator with method '{filter_method}' and threshold '{score_threshold}'...")
        
        # Store key for use in helper methods
        self.input_score_key = input_score_key

        # 1. Read data from the current step
        dataframe = storage.read("dataframe")
        if dataframe.empty:
            self.logger.warning("Input dataframe for ScoreFilter is empty. Skipping.")
            storage.write(dataframe)
            return []

        original_count = len(dataframe)
        
        # 2. Validate the data
        self._validate_dataframe(dataframe)
        
        # 3. Apply the filter logic
        if filter_method == "greater_equal":
            filtered_df = dataframe[dataframe[self.input_score_key] >= score_threshold]
        elif filter_method == "greater":
            filtered_df = dataframe[dataframe[self.input_score_key] > score_threshold]
        elif filter_method == "less_equal":
            filtered_df = dataframe[dataframe[self.input_score_key] <= score_threshold]
        elif filter_method == "less":
            filtered_df = dataframe[dataframe[self.input_score_key] < score_threshold]
        elif filter_method == "equal":
            filtered_df = dataframe[dataframe[self.input_score_key] == score_threshold]
        else:
            # This case should ideally not be hit due to Literal type hint, but is good for robustness
            raise ValueError(f"Unsupported filter_method: '{filter_method}'")
        
        filtered_count = len(filtered_df)
        self.logger.info(f"Filtering complete. Kept {filtered_count} / {original_count} rows.")

        # 4. Write the results back to storage
        output_file = storage.write(filtered_df)
        self.logger.success(f"ScoreFilter finished. Filtered data saved to {output_file}")

        # 5. Return an empty list as no new columns were added
        return []