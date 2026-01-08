import requests
import pytest


@pytest.mark.api
def test_health(dummy_server_base_url):
    r = requests.get(f"{dummy_server_base_url}/health", timeout=(2, 2))
    assert r.status_code == 200
    assert r.json()["ok"] is True


@pytest.mark.api
def test_chat_ok(dummy_server_base_url):
    r = requests.post(
        f"{dummy_server_base_url}/v1/chat/completions?body=hello&think=reason&stream=0",
        json={"model": "dummy", "messages": []},
        timeout=(2, 2),
    )
    assert r.status_code == 200
    data = r.json()
    assert data["choices"][0]["message"]["content"] == "hello"
    assert data["choices"][0]["message"]["reasoning_content"] == "reason"


@pytest.mark.api
def test_chat_500(dummy_server_base_url):
    r = requests.post(
        f"{dummy_server_base_url}/v1/chat/completions?status=500",
        json={"model": "dummy", "messages": []},
        timeout=(2, 2),
    )
    assert r.status_code == 500
    # 也可以顺便验证 error body
    assert "error" in r.json()


@pytest.mark.api
def test_chat_queue_keepalive_non_stream_then_json(dummy_server_base_url):
    """
    验证：非流式排队阶段持续返回空行，最终仍能 json() 成功解析。
    """
    url = (
        f"{dummy_server_base_url}/v1/chat/completions"
        f"?queue=0.3&ka_interval=0.05&stream=0&body=hello"
    )
    r = requests.post(
        url,
        json={"model": "dummy", "messages": []},
        timeout=(2, 2),
    )
    assert r.status_code == 200
    data = r.json()
    assert data["choices"][0]["message"]["content"] == "hello"


@pytest.mark.api
def test_chat_queue_keepalive_stream_sse(dummy_server_base_url):
    """
    验证：流式模式下排队阶段会发 ': keep-alive'，随后发 data chunk + [DONE]
    注意：这里用 stream=True 来读取行。
    """
    url = (
        f"{dummy_server_base_url}/v1/chat/completions"
        f"?queue=0.3&ka_interval=0.05&stream=1&body=hello"
    )
    with requests.post(
        url,
        json={"model": "dummy", "messages": [], "stream": True},
        timeout=(2, 2),
        stream=True,
    ) as r:
        assert r.status_code == 200
        assert "text/event-stream" in (r.headers.get("Content-Type") or "")

        lines = []
        # iter_lines 会按 \n 切分；keep-alive 行是 ": keep-alive"
        for line in r.iter_lines(decode_unicode=True):
            if line is None:
                continue
            # requests 可能给空行；我们都保留
            lines.append(line)
            # 读到 DONE 就可以结束
            if line.strip() == "data: [DONE]":
                break

    # 至少应出现过 keep-alive 注释（排队阶段）
    assert any(l.strip() == ": keep-alive" for l in lines)
    # 应出现 data chunk 和 DONE
    assert any(l.startswith("data: ") and "chat.completion.chunk" in l for l in lines)
    assert any(l.strip() == "data: [DONE]" for l in lines)


@pytest.mark.api
def test_chat_disconnect_when_max_wait_exceeded(dummy_server_base_url):
    url = (
        f"{dummy_server_base_url}/v1/chat/completions"
        f"?queue=999&max_wait=0.2&ka_interval=999&stream=0"
    )

    with pytest.raises((requests.exceptions.RequestException, requests.exceptions.ConnectionError)):
        requests.post(
            url,
            json={"model": "dummy", "messages": []},
            timeout=(1, 5),  # 不靠 read timeout
        )
