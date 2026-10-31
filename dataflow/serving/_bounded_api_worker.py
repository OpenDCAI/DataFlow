"""Private, killable HTTP worker. Executed as a file, never imports Serving."""

import json
import re
import sys
import time

import requests


def format_api_response(response, is_embedding=False):
    """Shared response formatting without importing the Serving package.

    Keep the legacy formatter's return values; the recorded worker separately
    decides whether an empty result should be retried within its total budget.
    """
    if is_embedding:
        return response.get("data", [{}])[0].get("embedding", [])
    message = response.get("choices", [{}])[0].get("message") or {}
    content = message.get("content")
    reasoning_content = message.get("reasoning_content")
    if content is None:
        if reasoning_content:
            content = ""
        else:
            return None
    if re.search(r"<think>.*?</think>.*?<answer>.*?</answer>", content, re.DOTALL):
        return content
    if reasoning_content:
        return f"<think>{reasoning_content}</think>\n<answer>{content}</answer>"
    return content


def main():
    spec = json.load(sys.stdin)
    code = "empty_response"
    attempts = 0
    with requests.Session() as session:
        for attempt in range(spec["max_retries"]):
            attempts = attempt + 1
            # Only attempt counts go to stderr. Never print payloads or credentials.
            print(json.dumps({"attempt": attempts}), file=sys.stderr, flush=True)
            try:
                response = session.post(spec["url"], headers=spec["headers"],
                                        json=spec["payload"], timeout=tuple(spec["timeout"]))
                if response.status_code == 200:
                    try:
                        data = response.json()
                        formatted = format_api_response(data, spec.get("is_embedding", False))
                    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
                        code = "invalid_response_json"
                    else:
                        if formatted is not None and not (isinstance(formatted, str) and not formatted.strip()):
                            print(json.dumps({"data": data, "attempts": attempts, "error_code": None}))
                            return
                        code = "empty_response"
                else:
                    code = "http_error"
            except requests.exceptions.ConnectTimeout:
                code = "connect_timeout"
            except requests.exceptions.Timeout:
                code = "read_timeout"
            except requests.exceptions.ConnectionError:
                code = "connection_error"
            if attempt + 1 < spec["max_retries"]:
                time.sleep(2 ** attempt)
    print(json.dumps({"data": None, "attempts": attempts, "error_code": code}))


if __name__ == "__main__":
    main()
