"""Mark conversation structure errors without filtering records."""
from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.training_data import check_keys, validate_messages, validate_conversation
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
        check_keys(input_key, output_valid_key, output_error_key)
        validator = validate_messages if self.profile == "single_turn" else validate_conversation
        return evaluate_rows(storage, lambda row: validator(row, input_key),
                             output_valid_key, output_error_key, [input_key])

    @staticmethod
    def get_desc(lang="zh"):
        if lang == "zh":
            return ("按规则评估对话结构，不调用模型、不删行。初始化 profile='conversation'，也可为 'single_turn'。"
                    "run: input_key='messages'、output_valid_key='conversation_valid'、output_error_key='conversation_error'。"
                    "非法样本标记 False，错误包含 field/code/message；合法样本错误为 None。输出列须不存在；"
                    "字段或存储错误直接失败。保留顺序和原字段，返回两个输出字段名。")
        return ("Evaluate conversation structure without a model or filtering. Constructor: profile='conversation' "
                "or 'single_turn'. run: input_key='messages', output_valid_key='conversation_valid', "
                "output_error_key='conversation_error'. Invalid samples get False and field/code/message; "
                "valid samples get True and None. Requires new output columns. Field/storage errors raise. "
                "Preserves row count, order and source fields; returns both output names.")
