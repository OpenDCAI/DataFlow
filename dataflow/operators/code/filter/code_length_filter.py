import pandas as pd
from typing import List

from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC

@OPERATOR_REGISTRY.register()
class CodeLengthFilter(OperatorABC):
    """
    CodeLengthFilter is an operator for filtering code files that are too long.
    It filters inappropriate samples by analyzing the number of lines, average line length,
    and maximum line length of the code.
    Mainly used to remove low-quality samples such as oversized files and auto-generated
    long-line code.
    """

    # List of languages that require special handling (these languages allow longer line lengths)
    EXCLUDED_LANGS = {
        "HTML", "JSON", "Markdown", "Roff", "Roff Manpage", 
        "SMT", "TeX", "Text", "XML"
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
                "该算子基于代码长度特征进行过滤，去除不合适的样本。\n\n"
                "过滤规则：\n"
                "- 所有语言：文件超过10万行直接过滤\n"
                "- 普通语言：平均行长>100或最大行长>1000过滤\n"
                "- 特殊语言：最大行长>10万过滤\n\n"
                "输入参数：\n"
                "- input_lines_key: 代码行列表的字段名 (默认: 'lines')\n"
                "- input_language_key: 编程语言的字段名 (默认: 'language')\n"
                "输出参数：\n"
                "- output_pass_key: 过滤结果的字段名 (默认: 'length_filter_pass')\n"
            )
        return (
            "This operator filters code samples based on length characteristics.\n\n"
            "Filtering Rules:\n"
            "- All languages: filter if total lines > 100k\n"
            "- Normal languages: filter if avg length > 100 or max length > 1000\n"
            "- Special languages: filter if max length > 100k\n\n"
            "Input Parameters:\n"
            "- input_lines_key: Field name containing code lines (default: 'lines')\n"
            "- input_language_key: Field name for programming language (default: 'language')\n"
            "Output Parameters:\n"
            "- output_pass_key: Field name for filter results (default: 'length_filter_pass')\n"
        )

    def _validate_dataframe(self, dataframe: pd.DataFrame):
        """
        Validate DataFrame to ensure required columns exist.
        """
        required_keys = [self.input_lines_key, self.input_language_key]
        missing = [k for k in required_keys if k not in dataframe.columns]

        if missing:
            raise ValueError(f"CodeLengthFilter missing required columns: {missing}")

    def run(
        self,
        storage: DataFlowStorage,
        input_lines_key: str = "lines",
        input_language_key: str = "language",
        output_pass_key: str = "length_filter_pass"
    ) -> List[str]:
        """
        Execute code length filtering.

        Args:
            storage: Data storage object
            input_lines_key: Field name containing code lines
            input_language_key: Field name containing programming language
            output_pass_key: Field name for filter results

        Returns:
            List[str]: List containing new output column names
        """
        self.logger.info("Running CodeLengthFilter operator...")

        # Store key names for use by helper methods
        self.input_lines_key = input_lines_key
        self.input_language_key = input_language_key

        # 1. Read data
        dataframe = storage.read("dataframe")
        if dataframe.empty:
            self.logger.warning("Input data is empty, skipping processing.")
            storage.write(dataframe)
            return [output_pass_key]

        original_count = len(dataframe)

        # 2. Validate data
        self._validate_dataframe(dataframe)

        # 3. Define length checking function
        def check_length(lines: List[str], language: str) -> bool:
            """Check if code length meets requirements"""
            # Check total number of lines
            n_lines = len(lines)
            if n_lines > 100_000:
                return False

            # Calculate line length statistics
            avg_len = sum(len(l) for l in lines) / max(1, n_lines)
            max_len = max((len(l) for l in lines), default=0)

            # Apply different rules based on language type
            if language not in self.EXCLUDED_LANGS:
                if avg_len > 100 or max_len > 1000:
                    return False
            else:
                if max_len > 100_000:
                    return False
            return True

        # 4. Apply length checking
        self.logger.info("Checking code length characteristics...")
        dataframe[output_pass_key] = dataframe.apply(
            lambda row: check_length(row[input_lines_key], row[input_language_key]),
            axis=1
        )

        # 5. Count results
        passed_count = dataframe[output_pass_key].sum()
        self.logger.info(f"Filtering completed. {passed_count}/{original_count} samples passed length check.")

        # 6. Write back results
        output_file = storage.write(dataframe)
        self.logger.success(f"CodeLengthFilter completed. Results saved to {output_file}")

        return [output_pass_key]