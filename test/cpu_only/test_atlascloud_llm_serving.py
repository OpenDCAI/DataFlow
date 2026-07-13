import pytest

from dataflow.serving import AtlasCloudLLMServing


def test_atlascloud_defaults(monkeypatch):
    monkeypatch.setenv("ATLASCLOUD_API_KEY", "dummy-key")

    serving = AtlasCloudLLMServing(max_workers=1, max_retries=1)

    assert serving.api_url == "https://api.atlascloud.ai/v1/chat/completions"
    assert serving.model_name == "qwen/qwen3.5-flash"
    assert serving.api_key == "dummy-key"
    serving.cleanup()


def test_atlascloud_uses_custom_model(monkeypatch):
    monkeypatch.setenv("ATLASCLOUD_API_KEY", "dummy-key")

    serving = AtlasCloudLLMServing(model_name="deepseek-ai/deepseek-v4-pro")

    assert serving.model_name == "deepseek-ai/deepseek-v4-pro"
    serving.cleanup()


def test_atlascloud_requires_api_key(monkeypatch):
    monkeypatch.delenv("ATLASCLOUD_API_KEY", raising=False)

    with pytest.raises(ValueError, match="ATLASCLOUD_API_KEY"):
        AtlasCloudLLMServing()
