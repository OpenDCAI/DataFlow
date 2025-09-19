import re
import pandas as pd
from typing import List

from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC

@OPERATOR_REGISTRY.register()
class DataPatternFilter(OperatorABC):
    """
    DataPatternFilter is an operator for detecting and filtering code files containing
    large amounts of encoded data patterns.
    Mainly used to identify and remove files containing large amounts of Base64 encoded
    data, hexadecimal data, or Unicode escape sequences, which are typically compiled
    resources, binary data, or auto-generated code.
    """

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
                "该算子用于检测和过滤包含大量特殊数据模式的代码文件：\n\n"
                "检测模式：\n"
                "- Base64编码字符串\n"
                "- 十六进制数据序列\n"
                "- Unicode转义序列\n\n"
                "过滤规则：\n"
                "- 匹配长度 > 1024字符 或\n"
                "- 匹配内容占总文本比例 > 50%\n\n"
                "输入参数：\n"
                "- input_text_key: 代码文本的字段名 (默认: 'text')\n"
                "输出参数：\n"
                "- output_pass_key: 过滤结果的字段名 (默认: 'pattern_filter_pass')\n"
            )
        return (
            "This operator detects and filters code files containing large amounts of encoded data patterns:\n\n"
            "Detection Patterns:\n"
            "- Base64 encoded strings\n"
            "- Hexadecimal data sequences\n"
            "- Unicode escape sequences\n\n"
            "Filtering Rules:\n"
            "- Match length > 1024 characters or\n"
            "- Match ratio > 50% of total text\n\n"
            "Input Parameters:\n"
            "- input_text_key: Field name containing code text (default: 'text')\n"
            "Output Parameters:\n"
            "- output_pass_key: Field name for filter results (default: 'pattern_filter_pass')\n"
        )

    def _validate_dataframe(self, dataframe: pd.DataFrame):
        """
        Validate DataFrame to ensure required columns exist.
        """
        required_keys = [self.input_text_key]
        missing = [k for k in required_keys if k not in dataframe.columns]

        if missing:
            raise ValueError(f"DataPatternFilter missing required columns: {missing}")

    def run(
        self,
        storage: DataFlowStorage,
        input_text_key: str = "text",
        output_pass_key: str = "pattern_filter_pass"
    ) -> List[str]:
        """
        Execute data pattern detection and filtering.

        Args:
            storage: Data storage object
            input_text_key: Field name containing code text
            output_pass_key: Field name for filter results

        Returns:
            List[str]: List containing new output column names
        """
        self.logger.info("Running DataPatternFilter operator...")

        # Store key name for use by helper methods
        self.input_text_key = input_text_key

        # 1. Read data
        dataframe = storage.read("dataframe")
        if dataframe.empty:
            self.logger.warning("Input data is empty, skipping processing.")
            storage.write(dataframe)
            return [output_pass_key]

        original_count = len(dataframe)

        # 2. Validate data
        self._validate_dataframe(dataframe)

        # 3. Define pattern detection function
        def check_patterns(text: str) -> bool:
            """Check if text contains problematic data patterns"""
            # Compile regular expressions for different data patterns
            patterns = [
                re.compile(r"[a-zA-Z0-9+/=\n]{64,}"),  # Base64
                re.compile(r"(?:\b(?:0x|\\x)?[0-9a-fA-F]{2}(?:,|\b\s*)){8,}"),  # Hex
                re.compile(r"(?:\\u[0-9a-fA-F]{4}){8,}")  # Unicode
            ]

            # Check each pattern
            for pattern in patterns:
                for match in pattern.finditer(text):
                    match_len = len(match.group())
                    # Filter if match is too long or represents too much of the text
                    if match_len > 1024 or match_len / max(1, len(text)) > 0.5:
                        return False
            return True

        # 4. Apply filtering
        self.logger.info("Checking data patterns...")
        dataframe[output_pass_key] = dataframe[input_text_key].apply(check_patterns)

        # 5. Count results
        passed_count = dataframe[output_pass_key].sum()
        self.logger.info(f"Filtering completed. {passed_count}/{original_count} samples passed the check.")

        # 6. Write back results
        output_file = storage.write(dataframe)
        self.logger.success(f"DataPatternFilter completed. Results saved to {output_file}")

        return [output_pass_key]