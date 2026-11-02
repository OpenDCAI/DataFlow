"""Explicit text training formats; no format detection or invented tool IDs."""

from copy import deepcopy
import math

from dataflow.utils.training_data import (
    TrainingDataError, _text_at, require_text, validate_conversation,
)


def optional_value(row: dict, key: str):
    """Treat missing/null optional columns, including pandas NaN, as absent."""
    value = row.get(key)
    if value is None or isinstance(value, float) and math.isnan(value):
        return None
    return value


def optional_text(row: dict, key: str):
    value = optional_value(row, key)
    if value is None:
        return None
    text = _text_at(value, key, allow_empty=True)
    return text if text.strip() else None


def add_system(messages: list, system_prompt, source_system=None,
               source_key="system", messages_key="messages") -> list:
    if system_prompt is not None and source_system is not None:
        raise TrainingDataError(source_key, "system_conflict", "Source system and system_prompt cannot both be set")
    system = source_system if source_system is not None else system_prompt
    if system is not None:
        if messages and messages[0].get("role") == "system":
            field = source_key if source_system is not None else f"{messages_key}[0]"
            raise TrainingDataError(field, "system_conflict", "Conversation already contains a system message")
        return [{"role": "system", "content": system}, *messages]
    return messages


def convert_alpaca(row: dict, instruction_key: str, answer_key: str,
                   context_key: str, history_key: str) -> list:
    """instruction + optional input separated by two newlines; history first."""
    instruction = require_text(row, instruction_key)
    answer = require_text(row, answer_key)
    context = optional_text(row, context_key)
    messages = []
    history = optional_value(row, history_key)
    if history is not None:
        if not isinstance(history, list):
            raise TrainingDataError(history_key, "invalid_type", "Expected a list of question/answer pairs")
        for index, pair in enumerate(history):
            path = f"{history_key}[{index}]"
            if not isinstance(pair, list) or len(pair) != 2:
                raise TrainingDataError(path, "invalid_structure", "Expected a two-element question/answer pair")
            for position, role in enumerate(("user", "assistant")):
                messages.append({"role": role, "content": _text_at(pair[position], f"{path}[{position}]")})
    messages.extend([
        {"role": "user", "content": instruction if context is None else instruction + "\n\n" + context},
        {"role": "assistant", "content": answer},
    ])
    return messages


def convert_sharegpt(row: dict, key: str) -> list:
    """Map text-only from/value messages; never drop unknown nested fields."""
    if key not in row:
        raise TrainingDataError(key, "missing_field", "Required field is missing")
    source = row[key]
    if not isinstance(source, list):
        raise TrainingDataError(key, "invalid_type", "Expected a list of conversations")
    roles = {"human": "user", "gpt": "assistant", "system": "system"}
    messages = []
    for index, message in enumerate(source):
        path = f"{key}[{index}]"
        if not isinstance(message, dict):
            raise TrainingDataError(path, "invalid_type", "Expected a conversation object")
        if set(message) - {"from", "value"}:
            raise TrainingDataError(path, "unsupported_fields", "Only from and value are supported in text ShareGPT")
        role = message.get("from")
        if not isinstance(role, str) or role not in roles:
            raise TrainingDataError(f"{path}.from", "invalid_role", "Expected human, gpt or system")
        messages.append({"role": roles[role], "content": _text_at(message.get("value"), f"{path}.value")})
    try:
        validate_conversation({key: messages}, key)
    except TrainingDataError as exc:
        field = exc.field
        if field.startswith(f"{key}[") and field.endswith(".role"):
            field = field[:-5] + ".from"
        raise TrainingDataError(field, exc.code, str(exc)) from exc
    return messages


def convert_messages(row: dict, key: str) -> list:
    validate_conversation(row, key)
    # Keep tool arguments/IDs/content and all messages intact; detach output
    # objects from the source column even when both columns are kept in memory.
    return deepcopy(row[key])


def map_preference_fields(row, input_prompt_key="prompt", input_chosen_key="chosen", input_rejected_key="rejected",
                          output_prompt_key="prompt", output_chosen_key="chosen", output_rejected_key="rejected"):
    """Return a copied preference record with explicit, lossless field mapping.

    This is a Python utility, not a registered conversion/filter operator.
    Target fields may equal their corresponding input, but may not overwrite
    unrelated source fields or another preference field.
    """
    from dataflow.utils.training_data import check_keys
    inputs = [input_prompt_key, input_chosen_key, input_rejected_key]
    outputs = [output_prompt_key, output_chosen_key, output_rejected_key]
    check_keys(*inputs, *outputs)
    if len(set(inputs)) != 3 or len(set(outputs)) != 3:
        raise ValueError("Preference fields must be distinct")
    values = [require_text(row, key) for key in inputs]
    for source, target in zip(inputs, outputs):
        if target in row and target != source:
            raise ValueError("Preference outputs must not overwrite unrelated source fields")
    return {**row, **dict(zip(outputs, values))}
