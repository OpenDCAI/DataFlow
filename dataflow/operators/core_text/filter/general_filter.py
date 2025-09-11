from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.registry import OPERATOR_REGISTRY
import pandas as pd

@OPERATOR_REGISTRY.register()
class GeneralFilter(OperatorABC):

    def __init__(self, filter_rules: list):
        self.logger = get_logger()
        self.filter_rules = filter_rules
        self.logger.info(f"Initializing {self.__class__.__name__} with rules: {self.filter_rules}")  

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "该算子支持通过多个自定义函数对 DataFrame 进行灵活过滤。\n\n"
                "每条过滤规则是一个函数（例如 lambda 表达式），接受一个 DataFrame 并返回一个布尔类型的 Series，"
                "用于指定保留哪些行。\n\n"
                "输入参数：\n"
                "- filter_rules：一个函数列表，每个函数形式为 lambda df: ...，"
                "需返回一个与 df 长度一致的布尔 Series。所有规则之间采用与（AND）关系组合。\n\n"
                "示例：\n"
                "  - lambda df: df['score'] > 0.5\n"
                "  - lambda df: df['label'].isin(['A', 'B'])"
            )
        elif lang == "en":
            return (
                "This operator applies custom filtering functions to a DataFrame.\n\n"
                "Each filter rule is a function (e.g., lambda expression) that takes a DataFrame "
                "and returns a boolean Series indicating which rows to retain.\n\n"
                "Input Parameters:\n"
                "- filter_rules: A list of functions, each in the form of lambda df: ..., "
                "returning a boolean Series of the same length as the DataFrame. "
                "All rules are combined using logical AND.\n\n"
                "Examples:\n"
                "  - lambda df: df['score'] > 0.5\n"
                "  - lambda df: df['label'].isin(['A', 'B'])"
            )
        else:
            return "GeneralFilter filters DataFrame rows using a list of functions returning boolean Series."

        
    def _validate_dataframe(self, dataframe: pd.DataFrame):
        required_keys = [self.input_key]
        forbidden_keys = []

        missing = [k for k in required_keys if k not in dataframe.columns]
        conflict = [k for k in forbidden_keys if k in dataframe.columns]

        if missing:
            raise ValueError(f"Missing required column(s): {missing}")
        if conflict:
            raise ValueError(f"The following column(s) already exist and would be overwritten: {conflict}")

    def run(self,
            storage: DataFlowStorage,
            ):
        df = storage.read("dataframe")
        mask = pd.Series(True, index=df.index)

        for rule_fn in self.filter_rules:
            if not callable(rule_fn):
                raise ValueError("Each filter rule must be a callable(e.g., lambda df: ...)")
            cond = rule_fn(df)
            if not isinstance(cond, pd.Series) or cond.dtype != bool:
                raise ValueError("Each filter function must return a boolean Series")
            mask &= cond
            
        filtered_df = df[mask]
        self.logger.info(f"Filtering complete. Remaining rows: {len(filtered_df)}")
        storage.write(filtered_df)
        self.logger.info(f"Filtering completed. Total records passing filter: {len(filtered_df)}.")
        return ""



