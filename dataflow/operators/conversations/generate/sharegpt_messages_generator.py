"""Convert the text from/value ShareGPT dialect, preserving every row."""
from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
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
            return ("将文本 ShareGPT(from/value) 转换为 role/content messages，不调用模型、不删行。"
                    "构造参数 system_prompt=None 表示不添加 system 消息；传入非空字符串时，将其作为第一条 system 消息。"
                    "run 参数：input_key='conversations'、output_key='messages'、input_system_key=None。"
                    "input_system_key=None 表示不读取顶层 system 列；传入列名后该列必须存在。"
                    "只支持文本 from/value 消息，不支持权重字段或多模态字段；output_key 必须是新列。"
                    "保留行数、顺序、源字段和原始内容；非法内容或存储错误直接抛出异常，整批成功后才写入。返回输出字段列表。")
        return ("Convert text ShareGPT from/value messages into role/content messages without model calls or filtering. "
                "Constructor system_prompt=None means no system message is added; a nonblank string is added as the first system message. "
                "run parameters: input_key='conversations', output_key='messages', and input_system_key=None. "
                "input_system_key=None disables reading a top-level system column; when a name is provided, that column must exist. "
                "Only text from/value messages are supported; weights and multimodal fields are unsupported, and output_key must be new. "
                "Preserves row count, order, source columns and original content. Invalid content or storage errors raise, and the batch is written only after success. "
                "Returns the output column list.")
