"""Rule-based evaluation of string-valued preference pairs."""
from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.training_data import validate_preference, check_keys
from dataflow.utils.training_operator_batch import evaluate_rows


@OPERATOR_REGISTRY.register()
class PreferenceStructureSampleEvaluator(OperatorABC):
    def __init__(self):
        super().__init__()

    def run(self, storage, input_prompt_key="prompt", input_chosen_key="chosen", input_rejected_key="rejected",
            output_valid_key="preference_valid", output_error_key="preference_error"):
        inputs = [input_prompt_key, input_chosen_key, input_rejected_key]
        check_keys(*inputs)
        if len(set(inputs)) != 3:
            raise ValueError("Preference input fields must be distinct")
        return evaluate_rows(storage, lambda row: validate_preference(row, *inputs),
                             output_valid_key, output_error_key, inputs)

    @staticmethod
    def get_desc(lang="zh"):
        if lang == "zh":
            return ("评估字符串型 DPO 结构，不调用模型、无初始化参数。run: input_prompt_key='prompt'、"
                    "input_chosen_key='chosen'、input_rejected_key='rejected'，三列须存在且不同；"
                    "output_valid_key='preference_valid'、output_error_key='preference_error'，输出列须不存在。"
                    "保留所有记录、内容和顺序，写出布尔合法性及 field/code/message/row 错误；合法时错误为 None。"
                    "缺列和存储错误直接失败，不支持多轮/多模态 DPO，返回两个输出列名。")
        return ("Evaluate string-valued DPO without model calls or constructor options. run: input_prompt_key='prompt', "
                "input_chosen_key='chosen', input_rejected_key='rejected' (distinct existing columns); "
                "output_valid_key='preference_valid', output_error_key='preference_error' (new columns). "
                "Preserves all rows/order/content; emits boolean validity and field/code/message/row errors, None when valid. "
                "Missing columns/storage errors raise. Conversational/multimodal DPO unsupported. Returns both output names.")
