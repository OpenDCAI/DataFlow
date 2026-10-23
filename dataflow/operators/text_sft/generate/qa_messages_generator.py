"""Prepare messages from explicit QA fields without filtering records."""
from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.training_data import check_keys, require_text
from dataflow.utils.training_data_formats import add_system
from dataflow.utils.training_operator_batch import transform_rows


@OPERATOR_REGISTRY.register()
class QAMessagesGenerator(OperatorABC):
    def __init__(self, system_prompt=None):
        super().__init__()
        if system_prompt is not None and (not isinstance(system_prompt, str) or not system_prompt.strip()):
            raise ValueError("system_prompt must be nonblank text or None")
        self.system_prompt = system_prompt

    def run(self, storage, input_key="prompt", input_answer_key="answer", output_key="messages"):
        check_keys(input_key, input_answer_key, output_key)
        def convert(row):
            messages = [{"role": "user", "content": require_text(row, input_key)},
                        {"role": "assistant", "content": require_text(row, input_answer_key)}]
            return {output_key: add_system(messages, self.system_prompt, None, input_key, input_key)}
        return transform_rows(storage, convert, [output_key], [input_key, input_answer_key])

    @staticmethod
    def get_desc(lang="zh"):
        if lang == "zh":
            return ("将 QA 字段转换成 messages；纯规则处理，不调用模型。初始化 system_prompt=None，可提供非空系统提示词。"
                    "run: input_key='prompt'、input_answer_key='answer'、output_key='messages'。"
                    "保持行数、顺序和源字段；输出列必须尚不存在；内容非法时严格报错，整批不写入。返回输出字段列表。")
        return ("Convert QA into messages without a model. Constructor: system_prompt=None (optional nonblank text). "
                "run: input_key='prompt', input_answer_key='answer', output_key='messages'. "
                "Preserves row count, order and source columns; output must be a new column. "
                "Invalid content raises before the batch is written. Returns output field names.")
