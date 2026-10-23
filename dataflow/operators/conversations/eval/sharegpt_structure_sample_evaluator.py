"""Source evaluation shares the converter's rules but never removes rows."""
from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.training_data import require_text
from dataflow.utils.training_data_formats import convert_alpaca, convert_sharegpt, optional_text, add_system
from dataflow.utils.training_operator_batch import evaluate_rows


@OPERATOR_REGISTRY.register()
class ShareGPTStructureSampleEvaluator(OperatorABC):
    def __init__(self, system_prompt=None):
        super().__init__()
        if system_prompt is not None and (not isinstance(system_prompt, str) or not system_prompt.strip()):
            raise ValueError("system_prompt must be nonblank text or None")
        self.system_prompt = system_prompt

    def run(self, storage, input_key="conversations", input_system_key=None, output_valid_key="source_valid", output_error_key="source_error"):
        def validate(row):
            system = optional_text(row, input_system_key) if input_system_key is not None else None
            add_system(convert_sharegpt(row, input_key), self.system_prompt, system, input_system_key or input_key, input_key)
        return evaluate_rows(storage, validate, output_valid_key, output_error_key, [input_key] + ([] if input_system_key is None else [input_system_key]))

    @staticmethod
    def get_desc(lang="zh"):
        if lang == "zh":
            return ("源格式检查，不转换、不删行、不调用模型。初始化 system_prompt=None，可指定非空系统提示词。"
                    "run 输入：input_key='conversations', input_system_key=None。None 禁用可选列；启用后必须存在。"
                    "output_valid_key='source_valid'、output_error_key='source_error'；输出列必须尚不存在。"
                    "合法记录为 True/None，非法内容为 False 和 field/code/message/row，可含当前输入物理位置。"
                    "缺列或存储错误直接失败。保留行数和内容，返回两个输出列名。")
        return ("Evaluate source structure without conversion, filtering or model calls. Constructor: system_prompt=None, optional nonblank text. "
                "run inputs: input_key='conversations', input_system_key=None. None disables optional fields; enabled columns must exist. "
                "output_valid_key='source_valid', output_error_key='source_error', both new columns. Valid records get True/None; "
                "invalid content gets False and field/code/message/row, optionally current input physical locations. "
                "Missing columns/storage errors raise. Preserves rows and returns both output field names.")
