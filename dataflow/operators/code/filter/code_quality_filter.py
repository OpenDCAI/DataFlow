import pandas as pd
from typing import List

from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC

@OPERATOR_REGISTRY.register()
class CodeTextQualityFilter(OperatorABC):
    """
    CodeTextQualityFilter is a filtering operator for checking code text quality.
    It determines code readability and quality by analyzing the ratio of alphabetic
    and numeric characters in the code.
    Mainly used to filter binary files, encrypted files, or other non-normal code text.
    """

    # List of languages that require special handling
    SPECIAL_LANGS = {"Motorola 68K Assembly", "WebAssembly"}

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
                "该算子通过分析代码文本的字符组成来过滤低质量样本。\n\n"
                "过滤规则：\n"
                "- 普通语言：要求字母字符比例 >= 25%\n"
                "- 汇编语言：要求字母数字字符比例 >= 25%\n\n"
                "输入参数：\n"
                "- input_text_key: 代码文本的字段名 (默认: 'text')\n"
                "- input_language_key: 编程语言的字段名 (默认: 'language')\n"
                "- output_pass_key: 输出过滤结果的字段名 (默认: 'text_quality_pass')\n"
            )
        return (
            "This operator filters low-quality samples by analyzing the character composition of code text.\n\n"
            "Filtering Rules:\n"
            "- Normal languages: requires alphabetic character ratio >= 25%\n"
            "- Assembly languages: requires alphanumeric character ratio >= 25%\n\n"
            "Input Parameters:\n"
            "- input_text_key: Field name containing the code text (default: 'text')\n"
            "- input_language_key: Field name for programming language (default: 'language')\n"
            "- output_pass_key: Field name for filter results (default: 'text_quality_pass')\n"
        )

    def _validate_dataframe(self, dataframe: pd.DataFrame):
        """
        Validate DataFrame to ensure required columns exist.
        """
        required_keys = [self.input_text_key, self.input_language_key]
        missing = [k for k in required_keys if k not in dataframe.columns]

        if missing:
            raise ValueError(f"CodeTextQualityFilter missing required columns: {missing}")

    def run(
        self,
        storage: DataFlowStorage,
        input_text_key: str = "text",
        input_language_key: str = "language",
        output_pass_key: str = "text_quality_pass"
    ) -> List[str]:
        """
        Execute code text quality check filtering.

        Args:
            storage: Data storage object
            input_text_key: Field name containing code text
            input_language_key: Field name containing programming language
            output_pass_key: Field name for filter results

        Returns:
            List[str]: List containing new output column names
        """
        self.logger.info("Running CodeTextQualityFilter operator...")

        # Store key names for use by helper methods
        self.input_text_key = input_text_key
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

        # 3. Define quality check function
        def check_text_quality(text: str, language: str) -> bool:
            """Check code text quality based on character composition"""
            if language in self.SPECIAL_LANGS:
                # For assembly languages, check alphanumeric character ratio
                alnum_ratio = sum(c.isalnum() for c in text) / max(1, len(text))
                return alnum_ratio >= 0.25
            else:
                # For normal languages, check alphabetic character ratio
                alpha_ratio = sum(c.isalpha() for c in text) / max(1, len(text))
                return alpha_ratio >= 0.25

        # 4. Apply quality check
        self.logger.info("Checking code text quality...")
        dataframe[output_pass_key] = dataframe.apply(
            lambda row: check_text_quality(row[input_text_key], row[input_language_key]),
            axis=1
        )

        # 5. Count results
        passed_count = dataframe[output_pass_key].sum()
        self.logger.info(f"Filtering completed. {passed_count}/{original_count} samples passed quality check.")

        # 6. Write back results
        output_file = storage.write(dataframe)
        self.logger.success(f"CodeTextQualityFilter completed. Results saved to {output_file}")

        return [output_pass_key]