from __future__ import annotations

import inspect
import sys
import types

import pytest

from dataflow.serving import LiteLLMServing


pytestmark = pytest.mark.cpu


class RecordingLiteLLM(types.ModuleType):
    def __init__(self):
        super().__init__("litellm")
        self.completion_calls = []

    def completion(
        self,
        messages,
        max_tokens,
        model,
        timeout,
        api_key=None,
        api_base=None,
        api_version=None,
        custom_llm_provider=None,
    ):
        self.completion_calls.append(
            {
                "messages": messages,
                "max_tokens": max_tokens,
                "model": model,
                "timeout": timeout,
                "api_key": api_key,
                "api_base": api_base,
                "api_version": api_version,
                "custom_llm_provider": custom_llm_provider,
            }
        )
        return {"choices": [{"message": {"content": "ok"}}]}


def test_litellm_docs_match_mutable_instance_contract():
    class_docs = inspect.getdoc(LiteLLMServing) or ""
    init_docs = inspect.getdoc(LiteLLMServing.__init__) or ""
    combined = f"{class_docs}\n{init_docs}".lower()

    assert "immutable after initialization" not in combined
    assert "switch_model()" in class_docs
    assert "switch_model()" in init_docs
    assert "instance" in combined


def test_switch_model_mutates_only_target_instance_and_validation_uses_new_state(monkeypatch):
    fake_litellm = RecordingLiteLLM()
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)
    monkeypatch.setenv("DF_API_KEY", "old-key")
    monkeypatch.setenv("NEW_API_KEY", "new-key")

    target = LiteLLMServing(
        validate_on_init=False,
        model_name="old-model",
        api_url="https://old.example/v1",
        api_version="v1",
        temperature=0.7,
    )
    control = LiteLLMServing(
        validate_on_init=False,
        model_name="control-model",
        api_url="https://control.example/v1",
        api_version="control-v1",
        temperature=0.6,
    )

    target.switch_model(
        model_name="new-model",
        key_name_of_api_key="NEW_API_KEY",
        api_url="https://new.example/v1",
        api_version="v2",
        temperature=0.2,
        response_format="json",
    )

    assert target.model_name == "new-model"
    assert target.api_key == "new-key"
    assert target.key_name_of_api_key == "NEW_API_KEY"
    assert target.api_url == "https://new.example/v1"
    assert target.api_version == "v2"
    assert target.temperature == 0.2
    assert target.kwargs["response_format"] == "json"

    assert control.model_name == "control-model"
    assert control.api_key == "old-key"
    assert control.api_url == "https://control.example/v1"
    assert control.api_version == "control-v1"
    assert control.temperature == 0.6

    assert fake_litellm.completion_calls == [
        {
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 1,
            "model": "new-model",
            "timeout": 60,
            "api_key": "new-key",
            "api_base": "https://new.example/v1",
            "api_version": "v2",
            "custom_llm_provider": None,
        }
    ]
