"""Mark conversation structure errors without filtering records."""
from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.training_data import validate_messages, validate_conversation
from dataflow.utils.training_operator_batch import evaluate_rows


@OPERATOR_REGISTRY.register()
class ConversationStructureSampleEvaluator(OperatorABC):
    def __init__(self, profile="conversation"):
        super().__init__()
        if profile not in {"single_turn", "conversation"}:
            raise ValueError("profile must be single_turn or conversation")
        self.profile = profile

    def run(self, storage, input_key="messages", output_valid_key="conversation_valid",
            output_error_key="conversation_error"):
        validator = validate_messages if self.profile == "single_turn" else validate_conversation
        return evaluate_rows(
            storage,
            lambda row: validator(row, input_key),
            output_valid_key=output_valid_key,
            output_error_key=output_error_key,
            input_columns=[input_key],
        )

    @staticmethod
    def get_desc(lang="zh"):
        if lang == "zh":
            return ("按规则检查 messages 对话结构，不调用模型、不转换字段、不删行。"
                    "构造参数 profile='conversation'；profile='single_turn' 时只接受 system/user/assistant 的单轮结构。"
                    "run 参数：input_key='messages'、output_valid_key='conversation_valid'、output_error_key='conversation_error'。"
                    "本算子没有可选输入列，因此 run 参数中的 None 不表示禁用字段。"
                    "合法记录写入 True 和 None；非法记录写入 False 及 field/code/message/row 错误对象。"
                    "conversation 模式还检查工具调用、工具响应、顺序和最终 assistant 答案。保留所有行和原字段；"
                    "输入列缺失、输出列冲突或存储错误直接抛出异常。返回两个输出列名。")
        return ("Validate messages conversation structure without model calls, field conversion or filtering. "
                "Constructor profile='conversation'; profile='single_turn' accepts only a single system/user/assistant turn. "
                "run parameters: input_key='messages', output_valid_key='conversation_valid', and output_error_key='conversation_error'. "
                "There are no optional run input columns, so None does not disable a field. "
                "Valid rows receive True and None; invalid rows receive False and an error object with field/code/message/row. "
                "The conversation profile also checks tool calls, tool responses, ordering and the final assistant answer. "
                "Preserves all rows and source columns. Missing inputs, output conflicts and storage errors raise. Returns both output column names.")
