"""Source evaluation shares the converter's rules but never removes rows."""
from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.training_data_formats import convert_sharegpt, optional_text, add_system
from dataflow.utils.training_operator_batch import evaluate_rows


@OPERATOR_REGISTRY.register()
class ShareGPTStructureSampleEvaluator(OperatorABC):
    def __init__(self):
        super().__init__()

    def run(self, storage, input_key="conversations", input_system_key=None, output_valid_key="source_valid", output_error_key="source_error"):
        def validate(row):
            system = optional_text(row, input_system_key) if input_system_key is not None else None
            add_system(convert_sharegpt(row, input_key), None, system, input_system_key or input_key, input_key)
        return evaluate_rows(
            storage,
            validate,
            output_valid_key=output_valid_key,
            output_error_key=output_error_key,
            input_columns=[input_key] + ([] if input_system_key is None else [input_system_key]),
        )

    @staticmethod
    def get_desc(lang="zh"):
        if lang == "zh":
            return ("检查文本 ShareGPT(from/value) 源结构，不调用模型、不转换字段、不删行。"
                    "run 参数：input_key='conversations'（ShareGPT 列）、input_system_key=None（可选的顶层 system 列）、"
                    "output_valid_key='source_valid'、output_error_key='source_error'。"
                    "input_system_key=None 表示不读取顶层 system 列；传入列名后该列必须存在。"
                    "只支持文本 from/value 消息，不支持权重字段或多模态字段；顶层 system 与对话内 system 冲突时记录为非法。"
                    "合法记录写入 True 和 None，非法记录写入 False 及 field/code/message/row 错误对象。"
                    "保留所有行、原字段和原始内容；缺列、输出列冲突或存储错误直接抛出异常。返回两个输出列名。")
        return ("Validate text ShareGPT from/value source structure without model calls, field conversion or filtering. "
                "run parameters: input_key='conversations', input_system_key=None (optional top-level system column), "
                "output_valid_key='source_valid', and output_error_key='source_error'. "
                "input_system_key=None disables reading the top-level system column; when a column name is provided, it must exist. "
                "Only text from/value messages are supported; weights and multimodal fields are unsupported. A top-level system conflicting with an in-conversation system is invalid. "
                "Valid rows receive True and None; invalid rows receive False and an error object with field/code/message/row. "
                "Preserves all rows, source columns and original content. Missing columns, output conflicts and storage errors raise. Returns both output column names.")
