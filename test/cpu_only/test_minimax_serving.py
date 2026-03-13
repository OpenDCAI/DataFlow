"""Tests for APIMinimaxServing."""
import pytest
from dataflow.serving import APIMinimaxServing


class TestMinimaxServingInit:
    """Unit tests for APIMinimaxServing initialisation and constraints."""

    def test_default_values(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
        serving = APIMinimaxServing()
        assert serving.api_url == "https://api.minimax.io/v1/chat/completions"
        assert serving.model_name == "MiniMax-M2.5"
        assert serving.configs.get("temperature") == 1.0
        serving.cleanup()

    def test_custom_model(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
        serving = APIMinimaxServing(model_name="MiniMax-M2.5-highspeed")
        assert serving.model_name == "MiniMax-M2.5-highspeed"
        serving.cleanup()

    def test_china_endpoint(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
        serving = APIMinimaxServing(
            api_url="https://api.minimaxi.com/v1/chat/completions"
        )
        assert "minimaxi.com" in serving.api_url
        serving.cleanup()

    def test_temperature_zero_rejected(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
        with pytest.raises(ValueError, match="temperature"):
            APIMinimaxServing(temperature=0.0)

    def test_temperature_negative_rejected(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
        with pytest.raises(ValueError, match="temperature"):
            APIMinimaxServing(temperature=-0.1)

    def test_temperature_above_one_rejected(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
        with pytest.raises(ValueError, match="temperature"):
            APIMinimaxServing(temperature=1.5)

    def test_valid_temperature(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
        serving = APIMinimaxServing(temperature=0.7)
        assert serving.configs.get("temperature") == 0.7
        serving.cleanup()

    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
        with pytest.raises(ValueError, match="MINIMAX_API_KEY"):
            APIMinimaxServing()

    def test_json_schema_stripped(self, monkeypatch, dummy_server_base_url):
        """json_schema should be ignored (set to None) for MiniMax."""
        monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
        api_url = (
            f"{dummy_server_base_url}/v1/chat/completions"
            f"?queue=0.3&ka_interval=0.05&stream=0&body=ok"
        )
        serving = APIMinimaxServing(
            api_url=api_url,
            connect_timeout=1.0,
            read_timeout=3.0,
            max_retries=1,
            max_workers=1,
        )
        _id, resp = serving._api_chat_with_id(
            id=0,
            payload=[{"role": "user", "content": "hi"}],
            model="MiniMax-M2.5",
            json_schema={"type": "object"},
        )
        assert _id == 0
        assert resp is not None
        serving.cleanup()


@pytest.mark.api
class TestMinimaxServingWithDummyServer:
    """Tests that use the dummy OpenAI-compatible server."""

    def test_generate_response(self, monkeypatch, dummy_server_base_url):
        monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
        api_url = (
            f"{dummy_server_base_url}/v1/chat/completions"
            f"?queue=0.3&ka_interval=0.05&stream=0&body=hello"
        )
        serving = APIMinimaxServing(
            api_url=api_url,
            connect_timeout=1.0,
            read_timeout=3.0,
            max_retries=1,
            max_workers=1,
        )
        _id, resp = serving._api_chat_with_id(
            id=0,
            payload=[{"role": "user", "content": "hi"}],
            model="MiniMax-M2.5",
        )
        assert _id == 0
        assert resp is not None
        assert "hello" in resp
        serving.cleanup()
