from __future__ import annotations

import sys
import types
from types import SimpleNamespace

from PIL import Image
import pytest

from dataflow.serving import LocalVLMServing_vllm


pytestmark = pytest.mark.cpu


class RecordingTokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        self.calls.append(
            {
                "messages": messages,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
            }
        )
        return "rendered prompt"


class RecordingLLM:
    def __init__(self):
        self.calls = []

    def generate(self, inputs, sampling_params):
        self.calls.append({"inputs": inputs, "sampling_params": sampling_params})
        return [SimpleNamespace(outputs=[SimpleNamespace(text="ok")]) for _ in inputs]


class FakeGuidedDecodingParams:
    def __init__(self, json=None):
        self.json = json


class FakeSamplingParams:
    def __init__(
        self,
        temperature=1.0,
        top_p=1.0,
        max_tokens=16,
        top_k=-1,
        repetition_penalty=1.0,
        seed=None,
        guided_decoding=None,
    ):
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.seed = seed
        self.guided_decoding = guided_decoding


def _install_fake_vllm(monkeypatch):
    vllm_module = types.ModuleType("vllm")
    vllm_module.SamplingParams = FakeSamplingParams
    sampling_module = types.ModuleType("vllm.sampling_params")
    sampling_module.GuidedDecodingParams = FakeGuidedDecodingParams
    monkeypatch.setitem(sys.modules, "vllm", vllm_module)
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", sampling_module)


def _build_serving(tmp_path):
    image_path = tmp_path / "image.png"
    Image.new("RGB", (2, 2), "white").save(image_path)

    serving = LocalVLMServing_vllm(
        hf_model_name_or_path="Qwen2.5-VL-test",
        vllm_temperature=0.31,
        vllm_top_p=0.83,
        vllm_max_tokens=77,
        vllm_top_k=19,
        vllm_repetition_penalty=1.07,
        vllm_seed=123,
    )
    serving.backend_initialized = True
    serving.tokenizer = RecordingTokenizer()
    serving.llm = RecordingLLM()
    serving.sampling_params = object()
    return serving, image_path


def test_public_local_vlm_json_schema_reaches_single_image_guided_decoding(monkeypatch, tmp_path):
    _install_fake_vllm(monkeypatch)
    serving, image_path = _build_serving(tmp_path)
    default_sampling_params = serving.sampling_params
    schema = {
        "type": "object",
        "properties": {"caption": {"type": "string"}},
        "required": ["caption"],
    }

    result = serving.generate_from_input(
        user_inputs=[str(image_path)],
        system_prompt="Return structured output.",
        json_schema=schema,
    )

    assert result == ["ok"]
    assert len(serving.llm.calls) == 1
    runtime_sampling_params = serving.llm.calls[0]["sampling_params"]
    assert runtime_sampling_params is not default_sampling_params
    assert isinstance(runtime_sampling_params, FakeSamplingParams)
    assert runtime_sampling_params.guided_decoding.json == schema
    assert runtime_sampling_params.temperature == 0.31
    assert runtime_sampling_params.top_p == 0.83
    assert runtime_sampling_params.max_tokens == 77
    assert runtime_sampling_params.top_k == 19
    assert runtime_sampling_params.repetition_penalty == 1.07
    assert runtime_sampling_params.seed == 123
    assert serving.tokenizer.calls[0]["messages"][0] == {
        "role": "system",
        "content": "Return structured output.",
    }


def test_public_local_vlm_without_schema_reuses_default_sampling_params(tmp_path):
    serving, image_path = _build_serving(tmp_path)
    default_sampling_params = serving.sampling_params

    result = serving.generate_from_input(
        user_inputs=[str(image_path)],
        system_prompt="Describe the image.",
        json_schema=None,
    )

    assert result == ["ok"]
    assert len(serving.llm.calls) == 1
    assert serving.llm.calls[0]["sampling_params"] is default_sampling_params
