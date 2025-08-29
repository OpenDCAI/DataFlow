import re
import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC

@OPERATOR_REGISTRY.register()
class CodeMathFormatter(OperatorABC):
    """Identify and format embedded code snippets (`code`, ```code``` blocks) and LaTeX math expressions."""

    def __init__(self):
        self.logger = get_logger()

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "该算子在文本中识别嵌入的代码片段与数学表达式并进行标准化格式化。\n\n"
                "输入参数：\n"
                "- input_key：包含原始文本的字段名\n"
                "- output_key：格式化后文本字段名\n\n"
                "输出参数：\n"
                "- output_key：格式化后的文本"
            )
        else:
            return (
                "This operator detects embedded code snippets and LaTeX math expressions in text and formats them.\n\n"
                "Input Parameters:\n"
                "- input_key: Field containing raw text\n"
                "- output_key: Field to store formatted text\n\n"
                "Output Parameters:\n"
                "- output_key: Formatted text"
            )

    @staticmethod
    def _format_inline_code(text: str) -> str:
        return re.sub(r"`([^`]+)`", r"<code>\\1</code>", text)

    @staticmethod
    def _format_code_block(text: str) -> str:
        pattern = re.compile(r"```(.*?\n)([\s\S]*?)```", re.MULTILINE)
        def repl(match):
            first_line, code = match.group(1), match.group(2)
            lang = first_line.strip()
            lang_cls = f" class=\"language-{lang}\"" if lang else ""
            return f"<pre><code{lang_cls}>{code.rstrip()}</code></pre>"
        return pattern.sub(repl, text)

    @staticmethod
    def _format_math(text: str) -> str:
        # display math $$...$$ => \[ ... \]
        text = re.sub(r"\$\$(.*?)\$\$", r"\\[\\1\\]", text, flags=re.S)
        # inline math $...$ => \( ... \) (ignore already processed $$ blocks)
        text = re.sub(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)", r"\\(\\1\\)", text, flags=re.S)
        return text

    @classmethod
    def format_text(cls, text: str) -> str:
        text = cls._format_code_block(text)
        text = cls._format_inline_code(text)
        text = cls._format_math(text)
        return text

    def _validate_dataframe(self, df: pd.DataFrame, input_key: str):
        if input_key not in df.columns:
            self.logger.error(f"Missing required column: {input_key}")
            raise ValueError(f"Input key '{input_key}' not found in dataframe.")

    def run(self, storage: DataFlowStorage, input_key: str = "instruction", output_key: str = "formatted_text") -> list:
        df = storage.read("dataframe")
        self._validate_dataframe(df, input_key)

        self.logger.info(f"Formatting {len(df)} rows ...")
        df[output_key] = df[input_key].apply(self.format_text)

        output_file = storage.write(df)
        self.logger.info(f"Formatted text saved to {output_file}")
        return [output_key]


# ======== Auto-generated runner ========
from dataflow.utils.storage import FileStorage
from dataflow.serving import APILLMServing_request, LocalModelLLMServing_vllm, LocalModelLLMServing_sglang
from dataflow.core import LLMServingABC

if __name__ == "__main__":
    # 1. FileStorage
    storage = FileStorage(
        first_entry_file_name="../example_data/DataflowAgent/agent_test_data.json",
        cache_path="./cache_local",
        file_name_prefix="dataflow_cache_step",
        cache_type="jsonl",
    )

    # 2. LLM-Serving
 
# 3. Instantiate operator
operator = CodeMathFormatter()

# 4. Run
operator.run(storage=storage.step())
