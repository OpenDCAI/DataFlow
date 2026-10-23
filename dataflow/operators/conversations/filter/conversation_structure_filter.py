"""Filter explicitly evaluated records without modifying their contents."""
from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.training_data import check_keys
from dataflow.utils.training_operator_batch import filter_validation_rows


@OPERATOR_REGISTRY.register()
class ConversationStructureFilter(OperatorABC):
    def __init__(self):
        super().__init__()

    def run(self, storage, input_key="conversation_valid"):
        check_keys(input_key)
        return filter_validation_rows(storage, input_key)

    @staticmethod
    def get_desc(lang="zh"):
        if lang == "zh":
            return ("显式过滤评估结果为 False 的记录；无初始化参数、不调用模型。run: input_key='conversation_valid'。"
                    "该列必须全为布尔值；缺列或非法值报错。保持剩余记录内容和顺序，不添加列，返回空输出字段列表。")
        return ("Filter False validation flags without a model or constructor options. run: input_key='conversation_valid'. "
                "Requires boolean flags; missing/invalid flags raise. Preserves retained row contents and order, "
                "adds no columns and returns an empty output field list.")
