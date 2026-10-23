"""Frozen content oracles and error cases retained from pre-standardization tests."""
def user(text):
    return {"role": "user", "content": text}

def assistant(text):
    return {"role": "assistant", "content": text}

def call(call_id="c1", name="add", arguments='{"a": 1, "b": 2}'):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}

def result(call_id="c1", content="3"):
    return {"role": "tool", "tool_call_id": call_id, "content": content}

TOOL_MESSAGES = [
    user("先查两次，再回答"),
    {"role": "assistant", "content": None, "tool_calls": [call(), call("c2", "lookup", '{ "字": "词" }')]},
    result("c2", "词典"), result(),
    {"role": "assistant", "tool_calls": [call("c3", "finish", "{}")], "content": "核对结果"},
    result("c3", ""), assistant("结果是 3"),
    user("再说一遍"), assistant(" 3\n"),
]

ALPACA = {
    "id": "alpaca", "system": "你是助教", "history": [["1+1?", "2"], ["2+2?", "4"]],
    "instruction": "翻译", "input": " 你好\n", "output": " Hello\n", "metadata": {"split": "train"},
}

ALPACA_EXPECTED = [
    {"role": "system", "content": "你是助教"},
    {"role": "user", "content": "1+1?"}, {"role": "assistant", "content": "2"},
    {"role": "user", "content": "2+2?"}, {"role": "assistant", "content": "4"},
    {"role": "user", "content": "翻译\n\n 你好\n"}, {"role": "assistant", "content": " Hello\n"},
]

SHAREGPT = {
    "id": "sharegpt", "conversations": [
        {"from": "system", "value": "帮助用户"}, {"from": "human", "value": " Hi\n"},
        {"from": "gpt", "value": "你好"}, {"from": "human", "value": "再见"},
        {"from": "gpt", "value": " Bye "},
    ],
}

SHAREGPT_EXPECTED = [
    {"role": "system", "content": "帮助用户"}, {"role": "user", "content": " Hi\n"},
    {"role": "assistant", "content": "你好"}, {"role": "user", "content": "再见"},
    {"role": "assistant", "content": " Bye "},
]

CONVERSION_ERRORS = [
    ("alpaca", {"output": "A"}, {}, "instruction", "missing_field"),
    ("alpaca", {"instruction": "Q", "output": " "}, {}, "output", "empty_content"),
    ("alpaca", {"instruction": "Q", "output": "A", "input": []}, {}, "input", "invalid_type"),
    ("alpaca", {"instruction": "Q", "output": "A", "history": {}}, {}, "history", "invalid_type"),
    ("alpaca", {"instruction": "Q", "output": "A", "history": [["P"]]}, {}, "history[0]", "invalid_structure"),
    ("alpaca", {"instruction": "Q", "output": "A", "history": [["P", " "]]}, {}, "history[0][1]", "empty_content"),
    ("alpaca", {"instruction": "Q", "output": "A", "system": 2}, {}, "system", "invalid_type"),
    ("alpaca", ALPACA, {"system_prompt": "override"}, "system", "system_conflict"),
    ("sharegpt", SHAREGPT, {"system_prompt": "override"}, "conversations[0]", "system_conflict"),
    ("sharegpt", {"system": "S", **SHAREGPT}, {}, "system", "system_conflict"),
    ("sharegpt", {}, {}, "conversations", "missing_field"),
    ("sharegpt", {"conversations": "Q"}, {}, "conversations", "invalid_type"),
    ("sharegpt", {"conversations": []}, {}, "conversations", "invalid_structure"),
    ("sharegpt", {"conversations": ["Q"]}, {}, "conversations[0]", "invalid_type"),
    ("sharegpt", {"conversations": [{"from": "function_call", "value": "{}"}]}, {}, "conversations[0].from", "invalid_role"),
    ("sharegpt", {"conversations": [{"from": [], "value": "Q"}]}, {}, "conversations[0].from", "invalid_role"),
    ("sharegpt", {"conversations": [{"from": "human", "value": " "}]}, {}, "conversations[0].value", "empty_content"),
    ("sharegpt", {"conversations": [{"from": "gpt", "value": "A"}]}, {}, "conversations[0].from", "invalid_structure"),
    ("sharegpt", {"conversations": [{"from": "human", "value": "Q", "tool_calls": []}]}, {}, "conversations[0]", "unsupported_fields"),
    ("messages", {"messages": [user("Q")]}, {}, "messages", "invalid_structure"),
]

CONVERSATION_ERRORS = [
    (None, "messages", "invalid_type"),
    ([], "messages", "invalid_structure"),
    ([user("Q"), "A"], "messages[1]", "invalid_type"),
    ([{"role": [], "content": "Q"}], "messages[0].role", "invalid_role"),
    ([{"role": "human", "content": "Q"}], "messages[0].role", "invalid_role"),
    ([assistant("A")], "messages[0].role", "invalid_structure"),
    ([user("Q"), user("Q2")], "messages[1].role", "invalid_structure"),
    ([user("Q"), assistant("A"), assistant("A2")], "messages[2].role", "invalid_structure"),
    ([user("Q"), {"role": "system", "content": "S"}], "messages[1].role", "invalid_structure"),
    ([{"role": "system", "content": "S"}], "messages", "invalid_structure"),
    ([user("Q"), assistant("A"), user("Q2")], "messages", "invalid_structure"),
    ([user(" "), assistant("A")], "messages[0].content", "empty_content"),
    ([user("Q"), assistant("")], "messages[1].content", "empty_content"),
    ([user([{"type": "image_url"}]), assistant("A")], "messages[0].content", "invalid_type"),
    ([{**user("Q"), "tool_calls": [call()]}, assistant("A")], "messages[0]", "unsupported_fields"),
    ([user("Q"), {**assistant("A"), "reasoning_content": "R"}], "messages[1]", "unsupported_fields"),
    ([user("Q"), result()], "messages[1].tool_call_id", "unexpected_tool_response"),
    ([result(), user("Q"), assistant("A")], "messages[0].tool_call_id", "unexpected_tool_response"),
    ([user("Q"), {"role": "assistant", "tool_calls": []}], "messages[1].tool_calls", "invalid_structure"),
    ([user("Q"), {"role": "assistant", "tool_calls": {}}], "messages[1].tool_calls", "invalid_type"),
    ([user("Q"), {"role": "assistant", "tool_calls": [call()], "content": 3}], "messages[1].content", "invalid_type"),
    ([user("Q"), {"role": "assistant", "tool_calls": [call()]}], "messages[1].tool_calls", "missing_tool_response"),
    ([user("Q"), {"role": "assistant", "tool_calls": [call()]}, assistant("A")], "messages[2].role", "missing_tool_response"),
    ([user("Q"), {"role": "assistant", "tool_calls": [call()]}, result("unknown"), assistant("A")], "messages[2].tool_call_id", "unexpected_tool_response"),
    ([user("Q"), {"role": "assistant", "tool_calls": [call()]}, result(), result(), assistant("A")], "messages[3].tool_call_id", "duplicate_tool_response"),
    ([user("Q"), {"role": "assistant", "tool_calls": [call(), call("c2")]}, result(), assistant("A")], "messages[3].role", "missing_tool_response"),
    ([user("Q"), {"role": "assistant", "tool_calls": [call(), call()]}, result(), assistant("A")], "messages[1].tool_calls[1].id", "duplicate_tool_call"),
    ([user("Q"), {"role": "assistant", "tool_calls": [call()]}, result()], "messages", "invalid_structure"),
    ([user("Q"), {"role": "assistant", "tool_calls": [call()]}, result(content=None), assistant("A")], "messages[2].content", "invalid_type"),
    ([user("Q"), {"role": "assistant", "tool_calls": [call()]}, {"role": "tool", "content": "3"}, assistant("A")], "messages[2].tool_call_id", "invalid_type"),
    ([user("Q"), {"role": "assistant", "tool_calls": [call()]}, result(), {"role": "assistant", "tool_calls": [call()]}], "messages[3].tool_calls[0].id", "duplicate_tool_call"),
]

CALL_ERRORS = [
    ("call", "", "invalid_type"),
    ({**call(), "id": " "}, ".id", "empty_content"),
    ({**call(), "type": "other"}, ".type", "invalid_type"),
    ({**call(), "extra": 1}, "", "unsupported_fields"),
    ({**call(), "function": "add"}, ".function", "invalid_type"),
    ({**call(), "function": {"name": "add", "arguments": "{}", "extra": 1}}, ".function", "unsupported_fields"),
    (call(name=""), ".function.name", "empty_content"),
    (call(arguments={}), ".function.arguments", "invalid_type"),
    (call(arguments=""), ".function.arguments", "empty_content"),
    (call(arguments="{"), ".function.arguments", "invalid_json"),
    (call(arguments='{"x":NaN}'), ".function.arguments", "invalid_json"),
    (call(arguments='{"x":Infinity}'), ".function.arguments", "invalid_json"),
    (call(arguments="[]"), ".function.arguments", "invalid_type"),
    (call(arguments="null"), ".function.arguments", "invalid_type"),
]

QA_ERRORS = [
    ({"prompt": "q"}, "answer", "missing_field"),
    ({"prompt": "q", "answer": None}, "answer", "invalid_type"),
    ({"prompt": "q", "answer": 42}, "answer", "invalid_type"),
    ({"prompt": "q", "answer": " \n"}, "answer", "empty_content"),
    ({"prompt": ["q"], "answer": "a"}, "prompt", "invalid_type"),
]

SINGLE_TURN_ERRORS = [
    (None, "messages", "invalid_type"),
    ([], "messages", "invalid_structure"),
    ([{"role": "user", "content": "q"}, {"role": "tool", "content": "a"}], "messages[1].role", "invalid_role"),
    ([{"role": "user", "content": "q"}, {"role": "assistant", "content": " "}], "messages[1].content", "empty_content"),
    ([{"role": "user", "content": "q"}, {"role": "assistant", "content": None, "tool_calls": []}], "messages[1]", "unsupported_fields"),
    ([{"role": "user", "content": "q"}, "answer"], "messages[1]", "invalid_type"),
]
