"""Filter explicitly evaluated records without modifying their contents."""
from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.training_operator_batch import filter_validation_rows


@OPERATOR_REGISTRY.register()
class PreferenceStructureFilter(OperatorABC):
    def __init__(self):
        super().__init__()

    def run(self, storage, input_key="preference_valid"):
        return filter_validation_rows(storage, input_key)

    @staticmethod
    def get_desc(lang="zh"):
        if lang == "zh":
            return ("按 preference 验证列过滤记录，不调用模型、无初始化参数。run 参数 input_key='preference_valid'；"
                    "None 不适用于该必需输入列，不能用来关闭过滤条件。该列必须存在且每个值都必须是 True 或 False。"
                    "保留值为 True 的记录，删除值为 False 的记录；保持保留记录的字段、内容和顺序，不新增列。"
                    "缺列、非布尔值或存储错误直接抛出异常，返回空输出字段列表。")
        return ("Filter preference records by a boolean validation column without model calls or constructor options. "
                "run parameter: input_key='preference_valid'. None is not a valid value for this required input and cannot disable filtering. "
                "The column must exist and every value must be True or False. Keeps True rows and removes False rows, preserving retained fields, content and order. "
                "Missing columns, non-boolean values and storage errors raise. Returns an empty output field list.")
