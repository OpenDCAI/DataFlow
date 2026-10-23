"""Convert the text from/value ShareGPT dialect, preserving every row."""
from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.training_data import check_keys
from dataflow.utils.training_data_formats import convert_sharegpt, optional_text, add_system
from dataflow.utils.training_operator_batch import transform_rows


@OPERATOR_REGISTRY.register()
class ShareGPTMessagesGenerator(OperatorABC):
    def __init__(self, system_prompt=None):
        super().__init__()
        if system_prompt is not None and (not isinstance(system_prompt, str) or not system_prompt.strip()):
            raise ValueError("system_prompt must be nonblank text or None")
        self.system_prompt = system_prompt

    def run(self, storage, input_key="conversations", output_key="messages", input_system_key=None):
        check_keys(input_key, output_key)
        if input_system_key is not None:
            check_keys(input_system_key)
        def convert(row):
            if input_system_key is not None and input_system_key not in row:
                raise ValueError(f"Missing explicitly enabled field: {input_system_key}")
            system = optional_text(row, input_system_key) if input_system_key is not None else None
            return {output_key: add_system(convert_sharegpt(row, input_key), self.system_prompt, system,
                                          input_system_key or input_key, input_key)}
        return transform_rows(storage, convert, [output_key], [input_key] + ([] if input_system_key is None else [input_system_key]))

    @staticmethod
    def get_desc(lang="zh"):
        if lang == "zh":
            return ("文本 ShareGPT(from/value) 转 messages；初始化 system_prompt=None，可指定非空系统提示词；不调用模型。"
                    "run: input_key='conversations'、output_key='messages'、input_system_key=None(不读取)。"
                    "保持行数、顺序和源字段；输出列须不存在；不支持权重或多模态方言；非法内容报错，整批不写入。返回输出字段列表。")
        return ("Convert text ShareGPT from/value messages, without a model; constructor system_prompt=None accepts optional nonblank text. "
                "run: input_key='conversations', output_key='messages', input_system_key=None (disabled). "
                "Preserves rows, order and source fields; requires a new output column. "
                "Weights and multimodal dialects are unsupported. Invalid content raises before writing. Returns output field names.")
