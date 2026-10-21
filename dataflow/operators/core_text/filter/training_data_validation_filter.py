"""Deterministic validation of text conversations and string-valued DPO."""

from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.training_data import (
    check_keys, check_policy, process_batch, validate_conversation,
    validate_messages, validate_preference,
)


@OPERATOR_REGISTRY.register()
class TrainingDataValidationFilter(OperatorABC):
    def __init__(self, schema="messages", on_invalid="raise", errors_path=None,
                 message_profile="single_turn"):
        super().__init__()
        if schema not in {"messages", "dpo"}:
            raise ValueError("schema must be 'messages' or 'dpo'")
        if message_profile not in {"single_turn", "conversation"}:
            raise ValueError("message_profile must be 'single_turn' or 'conversation'")
        if schema == "dpo" and message_profile != "single_turn":
            raise ValueError("message_profile='conversation' is only used by the messages schema")
        check_policy(on_invalid, errors_path)
        self.schema = schema
        self.message_profile = message_profile
        self.on_invalid = on_invalid
        self.errors_path = errors_path
        self.last_report = {}

    def run(self, storage: DataFlowStorage, input_key=None, input_prompt_key=None,
            input_chosen_key=None, input_rejected_key=None,
            output_key: str = "training_data_valid"):
        """Write valid records with a success flag; report invalid records.

        Pass active input keys explicitly in a Pipeline for compile-time field
        validation. Direct calls default to messages or prompt/chosen/rejected.
        """
        if self.schema == "messages" and any(
            key is not None for key in (input_prompt_key, input_chosen_key, input_rejected_key)
        ):
            raise ValueError("DPO input keys are not used by the messages schema")
        if self.schema == "dpo" and input_key is not None:
            raise ValueError("input_key is not used by the dpo schema")
        messages_key = input_key if input_key is not None else "messages"
        preference_keys = [
            input_prompt_key if input_prompt_key is not None else "prompt",
            input_chosen_key if input_chosen_key is not None else "chosen",
            input_rejected_key if input_rejected_key is not None else "rejected",
        ]
        inputs = [messages_key] if self.schema == "messages" else preference_keys
        check_keys(*inputs, output_key)
        if output_key in inputs:
            raise ValueError("Validation output_key must not overwrite a checked field")
        if len(set(inputs)) != len(inputs):
            raise ValueError("DPO input keys must be distinct")

        def validate(row):
            if self.schema == "messages":
                validator = validate_messages if self.message_profile == "single_turn" else validate_conversation
                validator(row, messages_key)
            else:
                validate_preference(row, *preference_keys)
            return {output_key: True}

        process_batch(self, storage, validate, [output_key])
        return [output_key]

    @staticmethod
    def get_desc(lang="zh"):
        if lang == "zh":
            return "检查单轮、多轮或工具调用对话，以及字符串型 DPO，定位字段错误，支持严格检查和隔离。"
        return "Validate text messages, complete tool conversations or string-valued DPO, with field-level errors."
