"""Explicit capability contracts; no new methods are imposed on all servings."""
from __future__ import annotations

from typing import Any, Protocol, Sequence

CASCADE_MODEL_IDS = ("Qwen3.5-4B", "Qwen3.5-9B", "Qwen3.5-27B", "Qwen3.5-122B-A10B")


class VisualCoTGenerationService(Protocol):
    source_model: str

    def complete(self, *, prompt: str, images: Sequence[str], phase: str) -> dict[str, Any]: ...


class VisualCoTLikelihoodService(VisualCoTGenerationService, Protocol):
    def score_answer(self, *, prompt: str, images: Sequence[str], response: str) -> dict[str, Any]: ...


class VisualCoTImageFilter(Protocol):
    def reset(self) -> None: ...
    def manifest(self) -> dict[str, Any]: ...
    def evaluate(self, path: str, question: str, answer: str) -> dict[str, Any]: ...


def source_model(service: VisualCoTGenerationService) -> str:
    return str(getattr(service, "source_model", "")).split("/")[-1]
