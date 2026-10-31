"""Rule-based evaluation of string-valued preference pairs."""
from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.training_data import require_text
from dataflow.utils.training_operator_batch import evaluate_rows


@OPERATOR_REGISTRY.register()
class PreferenceStructureSampleEvaluator(OperatorABC):
    def __init__(self):
        super().__init__()

    def run(self, storage, input_prompt_key="prompt", input_chosen_key="chosen", input_rejected_key="rejected",
            output_valid_key="preference_valid", output_error_key="preference_error"):
        inputs = [input_prompt_key, input_chosen_key, input_rejected_key]
        if any(not isinstance(key, str) or not key.strip() for key in inputs):
            raise ValueError("Preference input fields must be nonblank strings")
        if len(set(inputs)) != 3:
            raise ValueError("Preference input fields must be distinct")
        def validate(row):
            for key in inputs:
                require_text(row, key)
        return evaluate_rows(storage, validate, output_valid_key=output_valid_key,
                             output_error_key=output_error_key, input_columns=inputs)

    @staticmethod
    def get_desc(lang="zh"):
        if lang == "zh":
            return ("检查字符串型 preference/DPO 样本的三列结构，不调用模型、不删行。"
                    "run 参数：input_prompt_key='prompt'、input_chosen_key='chosen'、input_rejected_key='rejected'；"
                    "三列都必须存在、名称非空且互不相同。output_valid_key='preference_valid'、"
                    "output_error_key='preference_error' 为新增输出列，不能覆盖已有列。"
                    "每个字段必须是非空字符串。合法记录写入 True 和 None；非法记录写入 False 及 field/code/message/row 错误对象。"
                    "保留所有行、原字段和原始内容；不支持多轮或多模态 preference。缺列、输出列冲突或存储错误直接抛出异常。返回两个输出列名。")
        return ("Validate the three string fields of a preference/DPO record without model calls or filtering. "
                "run parameters: input_prompt_key='prompt', input_chosen_key='chosen', and input_rejected_key='rejected'; "
                "all three columns must exist, have nonblank names, and be distinct. output_valid_key='preference_valid' and "
                "output_error_key='preference_error' are new output columns and must not overwrite existing columns. "
                "Each field must be a nonblank string. Valid rows receive True and None; invalid rows receive False and an error object with field/code/message/row. "
                "Preserves all rows, source columns and original content. Conversational and multimodal preference records are unsupported. "
                "Missing columns, output conflicts and storage errors raise. Returns both output column names.")
