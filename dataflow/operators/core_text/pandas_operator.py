from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.registry import OPERATOR_REGISTRY
import pandas as pd

@OPERATOR_REGISTRY.register()
class PandasOperator(OperatorABC):

    def __init__(self, process_fn: list):
        self.logger = get_logger()
        self.process_fn = process_fn
        self.logger.info(f"Initializing {self.__class__.__name__} with transform functions: {self.process_fn}")

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "该算子支持通过多个自定义函数对 DataFrame 进行任意操作（如添加列、重命名、排序等）。\n\n"
                "每个函数（通常为 lambda 表达式）接受一个 DataFrame 并返回一个修改后的 DataFrame。\n\n"
                "输入参数：\n"
                "- process_fn：一个函数列表，每个函数形式为 lambda df: ...，"
                "必须返回一个 DataFrame。\n\n"
                "示例：\n"
                "  - lambda df: df.assign(score2=df['score'] * 2)\n"
                "  - lambda df: df.sort_values('score', ascending=False)"
            )
        elif lang == "en":
            return (
                "This operator applies a list of transformation functions to a DataFrame.\n\n"
                "Each function (typically a lambda) takes a DataFrame and returns a modified DataFrame.\n\n"
                "Input Parameters:\n"
                "- process_fn: A list of functions, each in the form of lambda df: ..., "
                "and must return a DataFrame.\n\n"
                "Examples:\n"
                "  - lambda df: df.assign(score2=df['score'] * 2)\n"
                "  - lambda df: df.sort_values('score', ascending=False)"
            )
        else:
            return "Applies a sequence of transformation functions to a DataFrame."

    def run(self, storage: DataFlowStorage):
        df = storage.read("dataframe")
        for fn in self.process_fn:
            if not callable(fn):
                raise ValueError("Each transform function must be callable (e.g., lambda df: ...)")
            df = fn(df)
            if not isinstance(df, pd.DataFrame):
                raise ValueError("Each transform function must return a DataFrame")
        self.logger.info(f"Transformation complete. Final shape: {df.shape}")
        storage.write(df)
        return ""
