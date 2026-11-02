"""Prepare Alpaca messages using explicit optional field dependencies."""
from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.training_data_formats import convert_alpaca, optional_text, add_system
from dataflow.utils.training_operator_batch import transform_rows


@OPERATOR_REGISTRY.register()
class AlpacaMessagesGenerator(OperatorABC):
    def __init__(self, system_prompt=None):
        super().__init__()
        if system_prompt is not None and (not isinstance(system_prompt, str) or not system_prompt.strip()):
            raise ValueError("system_prompt must be nonblank text or None")
        self.system_prompt = system_prompt

    def run(self, storage, input_key="instruction", input_answer_key="output", output_key="messages",
            input_context_key=None, input_history_key=None, input_system_key=None):
        def convert(row):
            # None means disabled, never an implicit default column read.
            selected = {input_key: row.get(input_key), input_answer_key: row.get(input_answer_key)}
            for key in (input_context_key, input_history_key, input_system_key):
                if key is not None:
                    if key not in row:
                        raise ValueError(f"Missing explicitly enabled field: {key}")
                    selected[key] = row[key]
            messages = convert_alpaca(selected, input_key, input_answer_key, input_context_key, input_history_key)
            system = optional_text(selected, input_system_key) if input_system_key is not None else None
            return {output_key: add_system(messages, self.system_prompt, system, input_system_key or input_key, input_key)}
        return transform_rows(storage, convert, [output_key], [input_key, input_answer_key] + [key for key in (input_context_key, input_history_key, input_system_key) if key is not None])

    @staticmethod
    def get_desc(lang="zh"):
        if lang == "zh":
            return ("将 Alpaca 字段转换为 messages，不调用模型、不删行。构造参数 system_prompt=None 表示不添加 system 消息；"
                    "传入非空字符串时，将其作为第一条 system 消息。run 参数：input_key='instruction'、input_answer_key='output'、"
                    "output_key='messages'、input_context_key=None、input_history_key=None、input_system_key=None。"
                    "后三个 None 分别表示不读取 context、history、system 列；传入列名后对应列必须存在。"
                    "instruction 和 output 必须是非空字符串；output_key 必须是新列。保留行数、顺序、源字段和原始内容，"
                    "非法内容或存储错误直接抛出异常，整批成功后才写入。返回输出字段列表。")
        return ("Convert Alpaca fields into a messages column without model calls or filtering. "
                "Constructor system_prompt=None means no system message is added; a nonblank string is added as the first system message. "
                "run parameters: input_key='instruction', input_answer_key='output', output_key='messages', "
                "input_context_key=None, input_history_key=None, and input_system_key=None. "
                "Each of the last three None values disables reading that optional column; when a name is provided, the column must exist. "
                "instruction and output must be nonblank strings, and output_key must be a new column. "
                "Preserves row count, order, source columns and original content. Invalid content or storage errors raise, and the batch is written only after success. "
                "Returns the output column list.")
