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
                "该算子支持多种过滤条件，可对 DataFrame 中的字段进行灵活筛选，"
                "包括数值区间（range）、相等、不等、集合包含、字符串匹配、缺失值判断等，"
                "还支持通过自定义函数（custom）实现更复杂的过滤逻辑。\n\n"
                "输入参数：\n"
                "- filter_rules：一个包含多个过滤规则的列表。每条规则需包含以下字段：\n"
                "  - key：要过滤的字段名（custom 操作时可不指定）\n"
                "  - op：过滤操作符，如 '==', '>=', 'in', 'range', 'custom' 等\n"
                "  - value：用于比较的值或范围（如 [0.8, 1.0]），"
                "custom 操作时为一个接受 DataFrame 并返回布尔 Series 的函数\n\n"
                "支持的操作符包括：==, !=, >, >=, <, <=, in, not in, contains, startswith, endswith, isna, notna, range, custom"
            )
        elif lang == "en":
            return (
                "This operator applies flexible filtering rules to DataFrame fields, "
                "supporting a wide range of conditions including numeric ranges, comparisons, set inclusion, "
                "string matching, missing value checks, and also supports custom functions (custom) "
                "for more complex filtering logic.\n\n"
                "Input Parameters:\n"
                "- filter_rules: A list of filter rules. Each rule must include:\n"
                "  - key: The field name to filter on (optional for 'custom' op)\n"
                "  - op: The operator, such as '==', '>=', 'in', 'range', 'custom', etc.\n"
                "  - value: The comparison value or range (e.g., [0.8, 1.0]). "
                "For 'custom' op, a function that takes a DataFrame and returns a boolean Series\n\n"
                "Supported operators: ==, !=, >, >=, <, <=, in, not in, contains, startswith, endswith, isna, notna, range, custom"
            )
        else:
            return "FlexibleFilter applies advanced filtering rules on DataFrame columns using a variety of operations, including custom functions."
        
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

        for rule in self.filter_rules:
            key = rule.get("key")
            op = rule.get("op")
            val = rule.get("value")
            if op == "custom":
                if not callable(val):
                    raise ValueError("For 'custom' op, 'value' must be callable")
                cond = val(df)
                if not isinstance(cond, pd.Series) or cond.dtype != bool:
                    raise ValueError("Custom function must return a boolean pandas Series")
            else:
                if key not in df.columns:
                    raise ValueError(f"Column '{key}' not found in DataFrame.")

                if op == "==":
                    cond = df[key] == val
                elif op == "!=":
                    cond = df[key] != val
                elif op == ">":
                    cond = df[key] > val
                elif op == ">=":
                    cond = df[key] >= val
                elif op == "<":
                    cond = df[key] < val
                elif op == "<=":
                    cond = df[key] <= val
                elif op == "in":
                    cond = df[key].isin(val)
                elif op == "not in":
                    cond = ~df[key].isin(val)
                elif op == "contains":
                    cond = df[key].astype(str).str.contains(val)
                elif op == "startswith":
                    cond = df[key].astype(str).str.startswith(val)
                elif op == "endswith":
                    cond = df[key].astype(str).str.endswith(val)
                elif op == "isna":
                    cond = df[key].isna()
                elif op == "notna":
                    cond = df[key].notna()
                elif op == "range":
                    if not (isinstance(val, (list, tuple)) and len(val) == 2):
                        raise ValueError(f"Invalid value for 'range': {val}")
                    cond = (df[key] >= val[0]) & (df[key] <= val[1])
                else:
                    raise ValueError(f"Unsupported operator: {op}")

            mask &= cond
        filtered_df = df[mask]
        self.logger.info(f"Filtering complete. Remaining rows: {len(filtered_df)}")
        storage.write(filtered_df)
        self.logger.info(f"Filtering completed. Total records passing filter: {len(filtered_df)}.")
        return ""