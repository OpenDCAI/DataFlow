"""Prepare text messages from QA, Alpaca, ShareGPT or complete conversations."""

from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.training_data import check_keys, check_policy, process_batch, require_text
from dataflow.utils.training_data_formats import (
    add_system, convert_alpaca, convert_messages, convert_sharegpt, optional_text,
)


@OPERATOR_REGISTRY.register()
class TrainingMessagesConverter(OperatorABC):
    def __init__(self, system_prompt=None, on_invalid="raise", errors_path=None,
                 source_format="qa"):
        super().__init__()
        check_policy(on_invalid, errors_path)
        if system_prompt is not None and (not isinstance(system_prompt, str) or not system_prompt.strip()):
            raise ValueError("system_prompt must be nonblank text or None")
        if source_format not in {"qa", "alpaca", "sharegpt", "messages"}:
            raise ValueError("source_format must be 'qa', 'alpaca', 'sharegpt' or 'messages'")
        self.source_format = source_format
        self.system_prompt = system_prompt
        self.on_invalid = on_invalid
        self.errors_path = errors_path
        self.last_report = {}

    def run(self, storage: DataFlowStorage, input_key=None,
            input_answer_key=None, output_key: str = "messages", *,
            input_context_key=None, input_history_key=None, input_system_key=None):
        """Add messages; retain source fields and text. Defaults depend on format.

        QA defaults to prompt/answer, Alpaca to instruction/input/output/history/
        system, ShareGPT to conversations (from/value), messages to messages.
        Pass consumed input keys explicitly in Pipeline for dependency checking;
        optional Alpaca fields may be absent or null. No format is auto-detected.
        """
        defaults = {"qa": "prompt", "alpaca": "instruction", "sharegpt": "conversations", "messages": "messages"}
        source_key = defaults[self.source_format] if input_key is None else input_key
        answer_key = input_answer_key
        if self.source_format in {"qa", "alpaca"}:
            if answer_key is None:
                answer_key = "answer" if self.source_format == "qa" else "output"
        elif answer_key is not None:
            raise ValueError("input_answer_key is only used by qa and alpaca")
        if self.source_format != "alpaca" and any(key is not None for key in (input_context_key, input_history_key)):
            raise ValueError("input_context_key and input_history_key are only used by alpaca")
        context_key = "input" if input_context_key is None else input_context_key
        history_key = "history" if input_history_key is None else input_history_key
        system_key = input_system_key
        if system_key is None and self.source_format in {"alpaca", "sharegpt"}:
            system_key = "system"
        keys = [source_key, output_key]
        if answer_key is not None:
            keys.append(answer_key)
        if self.source_format == "alpaca":
            keys.extend([context_key, history_key])
        if system_key is not None:
            keys.append(system_key)
        check_keys(*keys)

        def convert(row):
            if self.source_format == "qa":
                messages = [
                    {"role": "user", "content": require_text(row, source_key)},
                    {"role": "assistant", "content": require_text(row, answer_key)},
                ]
            elif self.source_format == "alpaca":
                messages = convert_alpaca(row, source_key, answer_key, context_key, history_key)
            elif self.source_format == "sharegpt":
                messages = convert_sharegpt(row, source_key)
            else:
                messages = convert_messages(row, source_key)
            system = optional_text(row, system_key) if system_key is not None else None
            return {output_key: add_system(messages, self.system_prompt, system, system_key or source_key, source_key)}

        process_batch(self, storage, convert, [output_key])
        return [output_key]

    @staticmethod
    def get_desc(lang="zh"):
        if lang == "zh":
            return "将问答、Alpaca、ShareGPT 或已有多轮/工具对话统一为 messages，保留源字段，支持错误隔离。"
        return "Prepare messages from QA, Alpaca, ShareGPT or complete tool conversations; retain source fields."
