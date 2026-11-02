"""Prepare messages from explicit QA fields without filtering records."""
from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.training_data import require_text
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
        def convert(row):
            messages = [{"role": "user", "content": require_text(row, input_key)},
                        {"role": "assistant", "content": require_text(row, input_answer_key)}]
            return {output_key: add_system(messages, self.system_prompt, None, input_key, input_key)}
        return transform_rows(storage, convert, [output_key], [input_key, input_answer_key])

    @staticmethod
    def get_desc(lang="zh"):
        if lang == "zh":
            return ("将 QA 字段转换为两轮 messages，不调用模型、不删行。构造参数 system_prompt=None 表示不添加 system 消息；"
                    "传入非空字符串时，将其作为第一条 system 消息。run 参数：input_key='prompt'、"
                    "input_answer_key='answer'、output_key='messages'。prompt 和 answer 必须是非空字符串；"
                    "output_key 必须是尚不存在的新列。保留行数、顺序、源字段和原始内容；非法内容或存储错误直接抛出异常，"
                    "整批成功后才写入。返回输出字段列表。")
        return ("Convert QA fields into a two-turn messages column without model calls or filtering. "
                "Constructor system_prompt=None means no system message is added; a nonblank string is added as the first system message. "
                "run parameters: input_key='prompt', input_answer_key='answer', and output_key='messages'. "
                "prompt and answer must be nonblank strings, and output_key must name a new column. "
                "Preserves row count, order, source columns and original content. Invalid content or storage errors raise, and the batch is written only after success. "
                "Returns the output column list.")
