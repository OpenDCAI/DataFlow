"""Source evaluation shares the converter's rules but never removes rows."""
from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.training_data import require_text
from dataflow.utils.training_operator_batch import evaluate_rows


@OPERATOR_REGISTRY.register()
class QAStructureSampleEvaluator(OperatorABC):
    def __init__(self):
        super().__init__()

    def run(self, storage, input_key="prompt", input_answer_key="answer", output_valid_key="source_valid", output_error_key="source_error"):
        def validate(row):
            require_text(row, input_key)
            require_text(row, input_answer_key)
        return evaluate_rows(storage, validate, output_valid_key=output_valid_key,
                             output_error_key=output_error_key,
                             input_columns=[input_key, input_answer_key])

    @staticmethod
    def get_desc(lang="zh"):
        if lang == "zh":
            return ("检查 QA 源字段的结构，不调用模型、不转换字段、不删行。"
                    "run 参数：input_key='prompt'（问题列）、input_answer_key='answer'（答案列）、"
                    "output_valid_key='source_valid'（布尔结果列）、output_error_key='source_error'（错误详情列）。"
                    "输入列必须存在，输出列必须是新列；本算子没有可选列，因此不存在用 None 禁用输入列的情况。"
                    "每条记录的问题和答案都必须是非空字符串。合法记录写入 True 和 None，非法记录写入 False 及包含 field/code/message/row 的错误对象。"
                    "保留所有行、原字段和原始内容；缺列、输出列冲突或存储错误直接抛出异常。返回两个输出列名。")
        return ("Validate QA source fields without model calls, field conversion or filtering. "
                "run parameters: input_key='prompt' (question column), input_answer_key='answer' (answer column), "
                "output_valid_key='source_valid' (boolean result column), and output_error_key='source_error' (error detail column). "
                "Input columns must exist and output columns must be new. This operator has no optional input columns, so None does not disable a run input. "
                "Each question and answer must be a nonblank string. Valid rows receive True and None; invalid rows receive False and an error object with field/code/message/row. "
                "Preserves all rows, source columns and original content. Missing columns, output conflicts and storage errors raise. Returns both output column names.")
