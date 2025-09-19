import pandas as pd
from typing import List, Set

from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC

@OPERATOR_REGISTRY.register()
class FileTypeFilter(OperatorABC):
    """
    FileTypeFilter is an operator that filters data based on file types and content characteristics.
    It implements specific filtering rules for different file types, such as large file filtering,
    HTML visible text filtering, etc.
    """

    # File types that require size checking
    SIZE_CHECK_TYPES: Set[str] = {
        "text", "json", "yaml", "web ontology language", 
        "graphviz", "dot"
    }

    # Valid filename set for Text files
    VALID_TEXT_NAMES: Set[str] = {
        "readme", "notes", "todo", "description", "cmakelists"
    }

    def __init__(self):
        """
        Initialize the operator and set up the logger.
        """
        self.logger = get_logger()

    @staticmethod
    def get_desc(lang: str = "en"):
        """
        Provide operator functionality description and parameter documentation.
        """
        if lang == "zh":
            return (
                "该算子根据文件类型和内容特征进行过滤，支持多种过滤规则：\n\n"
                "过滤规则：\n"
                "1. 对特定类型文件（如Text/JSON等）进行大小过滤（>512行）\n"
                "2. 对HTML文件进行可见文本质量过滤\n"
                "3. 对Text文件进行文件名规则过滤\n\n"
                "输入参数：\n"
                "- input_dataframe_key: 输入数据的键名 (默认: 'dataframe')\n"
                "- output_dataframe_key: 输出数据的键名 (默认: 'filtered_dataframe')\n"
            )
        return (
            "This operator filters data based on file types and content characteristics:\n\n"
            "Filtering Rules:\n"
            "1. Size filtering for specific file types (Text/JSON etc.)\n"
            "2. Visible text quality filtering for HTML files\n"
            "3. Filename rule filtering for Text files\n\n"
            "Input Parameters:\n"
            "- input_dataframe_key: Key for input data (default: 'dataframe')\n"
            "- output_dataframe_key: Key for output data (default: 'filtered_dataframe')\n"
        )

    def _validate_dataframe(self, dataframe: pd.DataFrame):
        """
        Validate DataFrame to ensure required columns exist.
        """
        required_keys = ["filetype", "filename", "line_count"]
        missing = [k for k in required_keys if k not in dataframe.columns]

        if missing:
            raise ValueError(f"FileTypeFilter missing required columns: {missing}")

    def _is_large_file(self, row: pd.Series) -> bool:
        """
        Check if the file is large (line count > 512).
        """
        return row.get("line_count", 0) > 512

    def _is_html_valid(self, row: pd.Series) -> bool:
        """
        Check if HTML file meets visible text requirements.
        """
        visible_text_len = row.get("visible_text_length", 0)
        total_code_len = row.get("total_code_length", 1)
        ratio = visible_text_len / max(total_code_len, 1)
        return visible_text_len >= 100 and ratio >= 0.2

    def _is_text_filename_valid(self, filename: str) -> bool:
        """
        Check if Text filename meets requirements.
        """
        filename_lower = filename.lower()
        name_without_ext = filename_lower.rsplit('.', 1)[0]
        return (
            "requirement" in filename_lower
            or name_without_ext in self.VALID_TEXT_NAMES
        )

    def run(
        self,
        storage: DataFlowStorage,
        input_dataframe_key: str = "dataframe",
        output_dataframe_key: str = "filtered_dataframe"
    ) -> List[str]:
        """
        Execute file type filtering operation.

        Args:
            storage: Data storage object
            input_dataframe_key: Key name for input data
            output_dataframe_key: Key name for output data

        Returns:
            List[str]: List containing newly generated output key names
        """
        self.logger.info("Running FileTypeFilter operator...")

        # 1. Read data
        dataframe = storage.read(input_dataframe_key)
        if dataframe.empty:
            self.logger.warning("Input data is empty, skipping processing.")
            storage.write(dataframe)
            return [output_dataframe_key]

        original_count = len(dataframe)

        # 2. Validate data
        self._validate_dataframe(dataframe)

        # 3. Define filtering logic
        def filter_row(row: pd.Series) -> bool:
            filetype = row.get("filetype", "").lower()
            filename = row.get("filename", "")

            if filetype in self.SIZE_CHECK_TYPES:
                return not self._is_large_file(row)
            elif filetype == "html":
                return self._is_html_valid(row)
            elif filetype == "text":
                return self._is_text_filename_valid(filename)
            return True

        # 4. Apply filtering
        filtered_df = dataframe[dataframe.apply(filter_row, axis=1)].reset_index(drop=True)

        # 5. Count results
        filtered_count = len(filtered_df)
        self.logger.info(
            f"Filtering completed. Kept {filtered_count}/{original_count} samples "
            f"(removed {original_count - filtered_count})."
        )

        # 6. Write back results
        output_file = storage.write(filtered_df)
        self.logger.success(f"FileTypeFilter completed. Results saved to {output_file}")

        return [output_dataframe_key]