"""Model roles and an adapter for DataFlow serving resources."""
from __future__ import annotations

from typing import Any, Callable, Sequence

from dataflow.core import LLMServingABC
from .api_serving import VisualCoTAPIServing
from .clip_serving import LocalCLIPEncoder
from .contracts import VisualCoTGenerationService, VisualCoTLikelihoodService, CASCADE_MODEL_IDS, source_model

__all__ = ["VisualCoTServingBundle", "VisualCoTServingAdapter", "VisualCoTAPIServing", "LocalCLIPEncoder"]


class VisualCoTServingBundle(LLMServingABC):
    """One framework-managed serving resource for the visual CoT models.

    Construction never starts model servers. Lifecycle calls are forwarded once
    per unique supplied object. Models are validated in cascade order.
    """
    def __init__(self, small_model: VisualCoTGenerationService,
                 teacher_servings: Sequence[VisualCoTLikelihoodService],
                 verifier_serving: VisualCoTGenerationService,
                 cleaner_serving: VisualCoTGenerationService,
                 auxiliary_resources: Sequence[Any] = ()) -> None:
        self.small_model = small_model
        self.teacher_servings = list(teacher_servings)
        self.verifier_serving = verifier_serving
        self.cleaner_serving = cleaner_serving
        self.auxiliary_resources = tuple(auxiliary_resources)
        self._cleaned = False
        if tuple(source_model(s) for s in [small_model, *teacher_servings]) != CASCADE_MODEL_IDS:
            raise ValueError("Required source-model order: Qwen3.5-4B -> 9B -> 27B -> 122B-A10B")
        if source_model(verifier_serving) != CASCADE_MODEL_IDS[2] or source_model(cleaner_serving) != CASCADE_MODEL_IDS[3]:
            raise ValueError("Verifier must be Qwen3.5-27B; cleaner must be Qwen3.5-122B-A10B")
        for provider in self.providers():
            if not callable(getattr(provider, "complete", None)):
                raise TypeError("Visual CoT providers must implement complete")
        for teacher in teacher_servings:
            if not callable(getattr(teacher, "score_answer", None)) or getattr(teacher, "supports_answer_likelihood", True) is False:
                raise TypeError("Every teacher must support actual answer-token likelihood")

    def providers(self) -> list[VisualCoTGenerationService]:
        return [self.small_model, *self.teacher_servings, self.verifier_serving, self.cleaner_serving]

    def _lifecycle(self, method: str) -> None:
        seen = set()
        for resource in [*self.providers(), *self.auxiliary_resources]:
            if id(resource) in seen:
                continue
            seen.add(id(resource))
            callback = getattr(resource, method, None)
            if callable(callback):
                callback()

    def start_serving(self) -> None:
        self._cleaned = False
        self._lifecycle("start_serving")

    def cleanup(self) -> None:
        if not self._cleaned:
            self._lifecycle("cleanup")
            self._cleaned = True

    def generate_from_input(self, user_inputs: list[str], system_prompt: str = "") -> list[str]:
        return [self.small_model.complete(prompt=system_prompt + "\n" + text, images=[],
                                          phase="bundle_text_0")["text"] for text in user_inputs]


class VisualCoTServingAdapter:
    """Adapt standard DataFlow generation; require an explicit likelihood scorer.

    Existing LLMServingABC implementations need not implement complete or score.
    Their sampling/template settings remain caller-owned and are not verified here.
    """
    def __init__(self, llm_serving: LLMServingABC, source_model: str,
                 purpose: str = "generation", revision: str = "unverified",
                 likelihood_scorer: Callable[..., dict[str, Any]] | None = None) -> None:
        if purpose not in {"generation", "cleaner", "verifier"}:
            raise ValueError("Unknown visual CoT serving purpose")
        self.llm_serving = llm_serving
        self.source_model, self.purpose, self.revision = source_model, purpose, revision
        self._scorer = likelihood_scorer or getattr(llm_serving, "score_answer", None)
        self.supports_answer_likelihood = callable(self._scorer)
        self.mocked = bool(getattr(llm_serving, "mocked", False))
        self.runtime_manifest = {"adapter": "DataFlow standard interface", "purpose": purpose,
                                 "sampling_template_verified": False}

    def complete(self, *, prompt: str, images: Sequence[str], phase: str) -> dict[str, Any]:
        from dataflow.prompts.chinese_visual_cot import ChineseVisualCoTPrompt
        system = ChineseVisualCoTPrompt().build_system_prompt("verification") if self.purpose == "verifier" else ""
        if self.purpose == "verifier" and images:
            raise ValueError("The verifier takes text only")
        if images:
            generate = getattr(self.llm_serving, "generate_from_input_multi_images", None)
            if not callable(generate):
                raise TypeError("The supplied DataFlow Serving must support actual image inputs")
            outputs = generate(list_of_image_paths=[list(images)],
                               list_of_image_labels=[[f"image{i + 1}" for i in range(len(images))]],
                               system_prompt=system, user_prompts=[prompt])
        else:
            outputs = self.llm_serving.generate_from_input([prompt], system_prompt=system)
        if not isinstance(outputs, list) or len(outputs) != 1 or not isinstance(outputs[0], str):
            raise RuntimeError("DataFlow Serving returned a misaligned result")
        return {"text": outputs[0], "phase": phase, "finish_reason": "unavailable_from_standard_interface"}

    def score_answer(self, *, prompt: str, images: Sequence[str], response: str) -> dict[str, Any]:
        if not self.supports_answer_likelihood:
            raise TypeError("Supply an actual token-likelihood scorer; no surrogate scoring is allowed")
        return self._scorer(prompt=prompt, images=list(images), response=response)

    def start_serving(self) -> None:
        self.llm_serving.start_serving()

    def cleanup(self) -> None:
        self.llm_serving.cleanup()
