"""Source evaluation shares the converter's rules but never removes rows."""
from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.training_data_formats import convert_alpaca, optional_text
from dataflow.utils.training_operator_batch import evaluate_rows


@OPERATOR_REGISTRY.register()
class AlpacaStructureSampleEvaluator(OperatorABC):
    def __init__(self):
        super().__init__()

    def run(self, storage, input_key="instruction", input_answer_key="output", input_context_key=None, input_history_key=None, input_system_key=None, output_valid_key="source_valid", output_error_key="source_error"):
        def validate(row):
            convert_alpaca(row, input_key, input_answer_key, input_context_key, input_history_key)
            if input_system_key is not None:
                optional_text(row, input_system_key)
        return evaluate_rows(
            storage,
            validate,
            output_valid_key=output_valid_key,
            output_error_key=output_error_key,
            input_columns=[input_key, input_answer_key] + [
                key for key in (input_context_key, input_history_key, input_system_key)
                if key is not None
            ],
        )

    @staticmethod
    def get_desc(lang="zh"):
        if lang == "zh":
            return ("检查 Alpaca 源字段的结构，不调用模型、不转换字段、不删行。"
                    "run 参数：input_key='instruction'（指令列）、input_answer_key='output'（答案列）、"
                    "input_context_key=None、input_history_key=None、input_system_key=None（可选输入列）、"
                    "output_valid_key='source_valid'、output_error_key='source_error'。"
                    "可选输入参数为 None 时表示不读取对应列；传入列名后该列必须存在。"
                    "instruction 和 output 必须是非空字符串；context、history、system 按 Alpaca 文本格式校验。"
                    "合法记录写入 True 和 None，非法记录写入 False 及 field/code/message/row 错误对象。"
                    "保留所有行、原字段和原始内容；缺列、输出列冲突或存储错误直接抛出异常。返回两个输出列名。")
        return ("Validate Alpaca source fields without model calls, field conversion or filtering. "
                "run parameters: input_key='instruction', input_answer_key='output', "
                "input_context_key=None, input_history_key=None, input_system_key=None, "
                "output_valid_key='source_valid', and output_error_key='source_error'. "
                "An optional input set to None is disabled and its column is not read; when a column name is provided, that column must exist. "
                "instruction and output must be nonblank strings; context, history and system are validated according to the Alpaca text format. "
                "Valid rows receive True and None; invalid rows receive False and an error object with field/code/message/row. "
                "Preserves all rows, source columns and original content. Missing columns, output conflicts and storage errors raise. Returns both output column names.")
