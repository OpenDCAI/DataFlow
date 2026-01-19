# tests/test_api_llm_serving_request.py
import pytest
from dataflow.serving import APILLMServing_request


@pytest.mark.api
def test_non_stream_keepalive_then_json_parse_ok(dummy_server_base_url, monkeypatch):
    monkeypatch.setenv("DF_API_KEY", "dummy-key")

    api_url = (
        f"{dummy_server_base_url}/v1/chat/completions"
        f"?queue=0.3&ka_interval=0.05&stream=0"
        f"&body=hello&think=reason"
    )

    cli = APILLMServing_request(
        api_url=api_url,
        key_name_of_api_key="DF_API_KEY",
        model_name="dummy-model",
        # timeout=(1.0, 3.0),
        connect_timeout=1.0,
        read_timeout=3.0,
        max_retries=1,
        max_workers=1,
    )

    _id, resp = cli._api_chat_with_id(
        id=0,
        payload=[{"role": "user", "content": "hi"}],
        model="dummy-model",
        is_embedding=False,
    )

    assert _id == 0
    assert resp is not None
    assert "<think>reason</think>" in resp
    assert "<answer>hello</answer>" in resp

    cli.cleanup()


@pytest.mark.api
def test_timeout_should_warn(dummy_server_base_url, monkeypatch):
    """
    不要用 monkeypatch time.sleep，加速会破坏 server 的排队逻辑导致忙循环。
    用 max_wait 让服务端在排队阶段主动断开，保证测试确定性 & 不会卡。
    """
    monkeypatch.setenv("DF_API_KEY", "dummy-key")

    api_url = (
        f"{dummy_server_base_url}/v1/chat/completions"
        f"?queue=999&max_wait=0.2&ka_interval=999&stream=0"
    )

    cli = APILLMServing_request(
        api_url=api_url,
        key_name_of_api_key="DF_API_KEY",
        model_name="dummy-model",
        timeout=(1.0, 5.0),   # 不依赖 read timeout
        max_retries=1,
        max_workers=1,
    )

    with pytest.warns(RuntimeWarning):
        _id, resp = cli._api_chat_with_id(
            id=0,
            payload=[{"role": "user", "content": "hi"}],
            model="dummy-model",
            is_embedding=False,
        )

    assert resp is None
    cli.cleanup()


@pytest.mark.api
def test_connection_error_should_raise(monkeypatch):
    monkeypatch.setenv("DF_API_KEY", "dummy-key")

    api_url = "http://127.0.0.1:1/v1/chat/completions"

    cli = APILLMServing_request(
        api_url=api_url,
        key_name_of_api_key="DF_API_KEY",
        model_name="dummy-model",
        timeout=(0.2, 0.2),
        max_retries=1,
        max_workers=1,
    )

    with pytest.raises(RuntimeError):
        cli._api_chat_with_id(
            id=0,
            payload=[{"role": "user", "content": "hi"}],
            model="dummy-model",
            is_embedding=False,
        )

    cli.cleanup()
