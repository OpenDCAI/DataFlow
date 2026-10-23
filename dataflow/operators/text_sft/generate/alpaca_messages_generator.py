"""Prepare Alpaca messages using explicit optional field dependencies."""
from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.training_data import check_keys
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
        keys = [input_key, input_answer_key, output_key]
        keys.extend(k for k in (input_context_key, input_history_key, input_system_key) if k is not None)
        check_keys(*keys)
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
            return ("Alpaca 转 messages；初始化 system_prompt=None，可指定非空系统提示词；不调用模型。run: input_key='instruction'、"
                    "input_answer_key='output'、output_key='messages'；input_context_key/input_history_key/input_system_key"
                    "默认 None 表示不读取，启用时显式传列名。保持行数、顺序和源字段；禁止覆盖输出列；"
                    "非法内容严格报错，整批不写入。返回输出字段列表。")
        return ("Convert Alpaca to messages without a model; constructor system_prompt=None accepts optional nonblank text. run: input_key='instruction', "
                "input_answer_key='output', output_key='messages'. input_context_key/input_history_key/input_system_key "
                "default to None (disabled); pass explicit column names to enable. Preserves rows, order and source fields. "
                "Output must be new; invalid content raises before writing. Returns output field names.")
