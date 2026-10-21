"""Shared, deterministic validation and batch handling for training data operators."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import pandas as pd


class TrainingDataError(ValueError):
    """A record error with a stable code and a field path."""

    def __init__(self, field: str, code: str, message: str):
        self.field = field
        self.code = code
        super().__init__(message)


def require_text(row: dict, key: str) -> str:
    """Return nonblank text without trimming or otherwise changing it."""
    if key not in row:
        raise TrainingDataError(key, "missing_field", "Required field is missing")
    value = row[key]
    if not isinstance(value, str):
        raise TrainingDataError(key, "invalid_type", "Expected a string")
    if not value.strip():
        raise TrainingDataError(key, "empty_content", "Expected nonblank text")
    return value


def validate_messages(row: dict, key: str) -> None:
    """Validate text-only single-turn messages, with an optional system message.

    Kept as the default profile for compatibility. Multi-turn/tool records use
    validate_conversation explicitly; multimodal content is not supported here.
    """
    if key not in row:
        raise TrainingDataError(key, "missing_field", "Required field is missing")
    messages = row[key]
    if not isinstance(messages, list):
        raise TrainingDataError(key, "invalid_type", "Expected a list of messages")
    if len(messages) not in (2, 3):
        raise TrainingDataError(key, "invalid_structure", "Expected user/assistant, optionally preceded by system")
    roles = ["user", "assistant"] if len(messages) == 2 else ["system", "user", "assistant"]
    for index, (message, role) in enumerate(zip(messages, roles)):
        path = f"{key}[{index}]"
        if not isinstance(message, dict):
            raise TrainingDataError(path, "invalid_type", "Expected a message object")
        if message.get("role") != role:
            raise TrainingDataError(f"{path}.role", "invalid_role", f"Expected role {role!r}")
        # Do not silently accept structures this text-only profile cannot validate.
        if set(message) - {"role", "content"}:
            raise TrainingDataError(path, "unsupported_fields", "Only role and content are supported in this profile")
        try:
            require_text(message, "content")
        except TrainingDataError as exc:
            raise TrainingDataError(f"{path}.{exc.field}", exc.code, str(exc)) from exc


def _text_at(value, path: str, *, allow_empty=False) -> str:
    if not isinstance(value, str):
        raise TrainingDataError(path, "invalid_type", "Expected a string")
    if not allow_empty and not value.strip():
        raise TrainingDataError(path, "empty_content", "Expected nonblank text")
    return value


def _only_fields(value: dict, allowed: set, path: str) -> None:
    if set(value) - allowed:
        raise TrainingDataError(path, "unsupported_fields", "Unsupported fields in this message profile")


def _reject_json_constant(value):
    raise ValueError("Non-finite JSON constant")


def _validate_tool_calls(calls, path: str, seen: set) -> set:
    if not isinstance(calls, list):
        raise TrainingDataError(path, "invalid_type", "Expected a list of tool calls")
    if not calls:
        raise TrainingDataError(path, "invalid_structure", "Expected at least one tool call")
    pending = set()
    for index, call in enumerate(calls):
        call_path = f"{path}[{index}]"
        if not isinstance(call, dict):
            raise TrainingDataError(call_path, "invalid_type", "Expected a tool call object")
        _only_fields(call, {"id", "type", "function"}, call_path)
        call_id = _text_at(call.get("id"), f"{call_path}.id")
        if call_id in seen:
            raise TrainingDataError(f"{call_path}.id", "duplicate_tool_call", "Tool call IDs must be unique within a record")
        if call.get("type") != "function":
            raise TrainingDataError(f"{call_path}.type", "invalid_type", "Expected a function tool call")
        function = call.get("function")
        if not isinstance(function, dict):
            raise TrainingDataError(f"{call_path}.function", "invalid_type", "Expected a function object")
        _only_fields(function, {"name", "arguments"}, f"{call_path}.function")
        _text_at(function.get("name"), f"{call_path}.function.name")
        arguments_path = f"{call_path}.function.arguments"
        arguments = _text_at(function.get("arguments"), arguments_path)
        try:
            decoded = json.loads(arguments, parse_constant=_reject_json_constant)
        except ValueError as exc:
            raise TrainingDataError(arguments_path, "invalid_json", "Arguments must be valid JSON") from exc
        if not isinstance(decoded, dict):
            raise TrainingDataError(arguments_path, "invalid_type", "Arguments must decode to a JSON object")
        seen.add(call_id)
        pending.add(call_id)
    return pending


def validate_conversation(row: dict, key: str) -> None:
    """Validate a complete text conversation, including ordered tool exchanges.

    A turn starts with user and ends with a nonblank assistant answer. Within a
    turn, assistant function calls must each receive exactly one tool response
    before assistant continues. Parallel responses may arrive in any order.
    This checks structure, not tool execution, tool schemas or answer quality.
    """
    if key not in row:
        raise TrainingDataError(key, "missing_field", "Required field is missing")
    messages = row[key]
    if not isinstance(messages, list):
        raise TrainingDataError(key, "invalid_type", "Expected a list of messages")
    if not messages:
        raise TrainingDataError(key, "invalid_structure", "Expected a complete conversation")
    expected = "user"
    seen, returned, pending = set(), set(), set()
    call_path = None
    for index, message in enumerate(messages):
        path = f"{key}[{index}]"
        if not isinstance(message, dict):
            raise TrainingDataError(path, "invalid_type", "Expected a message object")
        role = message.get("role")
        if not isinstance(role, str) or role not in {"system", "user", "assistant", "tool"}:
            raise TrainingDataError(f"{path}.role", "invalid_role", "Unsupported message role")
        if role == "system":
            if index != 0:
                raise TrainingDataError(f"{path}.role", "invalid_structure", "System message is only allowed first")
            _only_fields(message, {"role", "content"}, path)
            _text_at(message.get("content"), f"{path}.content")
            continue
        if role == "tool":
            _only_fields(message, {"role", "content", "tool_call_id"}, path)
            response_id = _text_at(message.get("tool_call_id"), f"{path}.tool_call_id")
            if response_id in returned:
                raise TrainingDataError(f"{path}.tool_call_id", "duplicate_tool_response", "A call must have exactly one tool response")
            if response_id not in pending:
                raise TrainingDataError(f"{path}.tool_call_id", "unexpected_tool_response", "Tool response has no pending call")
            # An empty tool result is legal, unlike an empty training answer.
            _text_at(message.get("content"), f"{path}.content", allow_empty=True)
            pending.remove(response_id)
            returned.add(response_id)
            expected = "tool" if pending else "assistant"
            continue
        if pending:
            raise TrainingDataError(f"{path}.role", "missing_tool_response", "All pending calls need responses before the conversation continues")
        if role != expected:
            raise TrainingDataError(f"{path}.role", "invalid_structure", f"Expected role {expected!r}")
        if role == "assistant" and "tool_calls" in message:
            _only_fields(message, {"role", "content", "tool_calls"}, path)
            if message.get("content") is not None:
                _text_at(message["content"], f"{path}.content", allow_empty=True)
            call_path = f"{path}.tool_calls"
            pending = _validate_tool_calls(message["tool_calls"], call_path, seen)
            expected = "tool"
        else:
            _only_fields(message, {"role", "content"}, path)
            _text_at(message.get("content"), f"{path}.content")
            expected = "assistant" if role == "user" else "user"
    if pending:
        raise TrainingDataError(call_path, "missing_tool_response", "Tool calls are missing responses")
    if expected != "user" or messages[-1].get("role") != "assistant":
        raise TrainingDataError(key, "invalid_structure", "Conversation must end with a nonblank assistant answer")


def validate_preference(row: dict, prompt: str, chosen: str, rejected: str) -> None:
    """Validate a string-valued DPO record; preserve both responses verbatim."""
    for key in (prompt, chosen, rejected):
        require_text(row, key)


def check_policy(on_invalid: str, errors_path) -> None:
    if on_invalid not in {"raise", "quarantine"}:
        raise ValueError("on_invalid must be 'raise' or 'quarantine'")
    if on_invalid == "quarantine" and errors_path is None:
        raise ValueError("quarantine requires errors_path so rejected records are auditable")


def check_keys(*keys: str) -> None:
    if any(not isinstance(key, str) or not key.strip() for key in keys):
        raise ValueError("Field keys must be nonblank strings")


def process_batch(operator, storage, transform: Callable, output_columns: list[str]) -> None:
    """Write a batch only after validation; keep source columns and row order.

    Error locations are positions in this step's input, not original-source IDs
    after upstream filtering. Error ledgers append across calls; last_report is
    per call. There is deliberately no resume/replay contract in this helper.
    """
    frame = storage.read("dataframe")
    if operator.errors_path is not None:
        error_path = Path(operator.errors_path).resolve()
        source = getattr(storage, "first_entry_file_name", None)
        if source and error_path == Path(source).resolve():
            raise ValueError("errors_path must not be the source file")
        cache_file = getattr(storage, "_get_cache_file_path", None)
        step = getattr(storage, "operator_step", None)
        if cache_file is not None and step is not None:
            for position in (step, step + 1):
                if error_path == Path(cache_file(position)).resolve():
                    raise ValueError("errors_path must not be a Storage input or output file")

    batch = getattr(storage, "batch_step", 0)
    size = getattr(storage, "batch_size", None)
    offset = batch * size if size is not None else 0
    operator.last_report = {"input": len(frame), "accepted": 0, "rejected": 0}
    rows, errors = [], []
    locations = frame.attrs.get("dataflow_source_locations")
    # pandas drops records entirely for a frame with rows but zero columns.
    # Keep those records so required-field validation can reject each one.
    records = frame.to_dict(orient="records") if len(frame.columns) else [{} for _ in range(len(frame))]
    for position, row in enumerate(records):
        try:
            updates = transform(row)
        except TrainingDataError as exc:
            error = {
                "schema_version": 1,
                "operator": type(operator).__name__,
                "step": getattr(storage, "operator_step", None),
                "row": offset + position + 1,
                "field": exc.field,
                "code": exc.code,
                "message": str(exc),
            }
            if locations is not None:
                error["source_location"] = locations[position]
            operator.last_report["rejected"] += 1
            if operator.on_invalid == "raise":
                _append_errors(operator.errors_path, [error])
                raise TrainingDataError(
                    exc.field, exc.code,
                    f"row {error['row']}, {exc.field}: {exc}",
                ) from exc
            errors.append(error)
            continue
        rows.append({**row, **updates})
        operator.last_report["accepted"] += 1

    _append_errors(operator.errors_path, errors)
    columns = list(dict.fromkeys([*frame.columns, *output_columns]))
    # A DataFrame preserves the schema even when every record is quarantined.
    storage.write(pd.DataFrame(rows, columns=columns))


def _append_errors(path, errors: list[dict]) -> None:
    if path is None or not errors:
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as stream:
        for error in errors:
            stream.write(json.dumps(error, ensure_ascii=False) + "\n")
