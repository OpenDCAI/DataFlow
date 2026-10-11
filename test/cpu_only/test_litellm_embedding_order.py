import threading

import pytest

from dataflow.serving.lite_llm_serving import LiteLLMServing


class _SilentLogger:
    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass


class _ControlledLiteLLM:
    def __init__(self, failures=None):
        self.failures = set(failures or [])
        self.vectors = {
            "slow": [0.0],
            "fast": [1.0],
            "mid": [2.0],
        }
        self._lock = threading.Lock()
        self._non_slow_completed = 0
        self._release_slow = threading.Event()

    def embedding(self, input, **kwargs):
        text = input[0]

        if text == "slow":
            if not self._release_slow.wait(timeout=2):
                raise RuntimeError("test synchronization timed out")
            return {"data": [{"embedding": self.vectors[text]}]}

        try:
            if text in self.failures:
                raise RuntimeError(f"forced failure for {text}")
            return {"data": [{"embedding": self.vectors[text]}]}
        finally:
            with self._lock:
                self._non_slow_completed += 1
                if self._non_slow_completed == 2:
                    self._release_slow.set()


def _make_serving(fake_litellm):
    serving = LiteLLMServing.__new__(LiteLLMServing)
    serving.model_name = "fake-model"
    serving.timeout = 60
    serving.api_key = None
    serving.api_url = None
    serving.api_version = None
    serving.max_retries = 1
    serving.max_workers = 3
    serving._litellm = fake_litellm
    serving.logger = _SilentLogger()
    return serving


@pytest.mark.cpu
def test_generate_embedding_preserves_input_order_when_futures_complete_out_of_order():
    serving = _make_serving(_ControlledLiteLLM())

    embeddings = serving.generate_embedding_from_input(["slow", "fast", "mid"])

    assert embeddings == [[0.0], [1.0], [2.0]]


@pytest.mark.cpu
def test_generate_embedding_keeps_failure_placeholder_in_original_input_slot():
    serving = _make_serving(_ControlledLiteLLM(failures={"fast"}))

    embeddings = serving.generate_embedding_from_input(["slow", "fast", "mid"])

    assert embeddings == [[0.0], [], [2.0]]
