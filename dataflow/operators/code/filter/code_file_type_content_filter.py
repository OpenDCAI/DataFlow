import pandas as pd
from typing import List, Set

from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC

@OPERATOR_REGISTRY.register()
class CodeFileTypeContentFilter(OperatorABC):
    """
    CodeFileTypeContentFilter filters code samples based on file types and content characteristics,
    applying different rules for different file formats to ensure quality and relevance.
    
    This filter directly applies filtering rules without using evaluator scores:
    - Removes oversized text files (>512 lines for Text/JSON/YAML files)
    - Removes HTML files with insufficient visible text content
    - Removes text files with inappropriate filenames (not documentation-related)
    - Keeps files that meet format-specific quality criteria
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
                "基于文件类型和内容特征直接过滤代码样本，针对不同文件格式应用特定规则。\n\n"
                "过滤规则：\n"
                "- Text/JSON/YAML/Graphviz文件：行数 > 512 行\n"
                "- HTML文件：可见文本长度 < 100字符 或 可见文本比例 < 20%\n"
                "- Text文件：文件名不符合文档规范（非readme/notes/todo等）\n\n"
                "输入参数：\n"
                "- input_key: 输入字段名（需要包含'filetype'、'filename'、'line_count'等列）\n"
                "- output_key: 输出标签字段名 (默认: 'file_type_content_filter_label')\n\n"
                "输出参数：\n"
                "- 过滤后的DataFrame，仅保留符合文件类型规则的样本\n"
                "- 返回包含输出标签字段名的列表"
            )
        else:
            return (
                "Filter code samples based on file types and content characteristics, applying specific rules for different file formats.\n\n"
                "Filtering Rules:\n"
                "- Text/JSON/YAML/Graphviz files: line count > 512\n"
                "- HTML files: visible text length < 100 chars OR visible text ratio < 20%\n"
                "- Text files: filename doesn't follow documentation conventions (not readme/notes/todo etc.)\n\n"
                "Input Parameters:\n"
                "- input_key: Input field name (requires 'filetype', 'filename', 'line_count' columns)\n"
                "- output_key: Output label field name (default: 'file_type_content_filter_label')\n\n"
                "Output Parameters:\n"
                "- Filtered DataFrame containing only samples meeting file type rules\n"
                "- List containing output label field name"
            )

    def _validate_dataframe(self, dataframe: pd.DataFrame):
        """
        Validate DataFrame to ensure required columns exist.
        """
        required_keys = ["filetype", "filename", "line_count"]
        missing = [k for k in required_keys if k not in dataframe.columns]

        if missing:
            raise ValueError(f"{self.__class__.__name__} missing required columns: {missing}")

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
        input_key: str,
        output_key: str = "file_type_content_filter_label"
    ) -> List[str]:
        """
        Execute file type filtering operation.

        Args:
            storage: Data storage object
            input_key: Key name for input data
            output_key: Key name for output label

        Returns:
            List[str]: List containing output key name
        """
        self.input_key = input_key
        self.output_key = output_key
        self.logger.info(f"Running {self.__class__.__name__} with input_key: {self.input_key} and output_key: {self.output_key}...")

        # 1. Read data
        dataframe = storage.read("dataframe")
        if dataframe.empty:
            self.logger.warning("Input data is empty, skipping processing.")
            storage.write(dataframe)
            return [self.output_key]

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

        # 4. Apply filtering and add label
        filter_mask = dataframe.apply(filter_row, axis=1)
        dataframe[self.output_key] = filter_mask.astype(int)
        filtered_df = dataframe[filter_mask].reset_index(drop=True)

        # 5. Count results
        filtered_count = len(filtered_df)
        self.logger.info(f"Filtering completed. Total records passing filter: {filtered_count}.")

        # 6. Write back results
        storage.write(filtered_df)

        return [self.output_key]