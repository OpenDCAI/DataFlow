"""Image transport and completion parsing for the visual CoT services."""
from __future__ import annotations

from base64 import b64encode
from io import BytesIO
import json
import math
import re

from PIL import Image, ImageOps


def encode_image(path: str) -> str:
    with Image.open(path) as original:
        image = ImageOps.exif_transpose(original).convert("RGB")
    short, long = sorted(image.size)
    scale = min(2048 / long, max(1.0, 32 / short))
    size = tuple(max(1, round(axis * scale)) for axis in image.size)
    if size != image.size:
        filter_type = Image.Resampling.LANCZOS if scale < 1 else Image.Resampling.BICUBIC
        image = image.resize(size, filter_type)
    encoded = BytesIO()
    image.save(encoded, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + b64encode(encoded.getvalue()).decode("ascii")


def normalize_completion(text: str) -> str:
    """Accept the backend's alternate reasoning tag casing and spelling."""
    return re.sub(
        r"<(/?)(think|reasoning|analysis|answer)>",
        lambda match: f"<{match[1]}{'answer' if match[2].lower() == 'answer' else 'think'}>",
        text.strip(),
        flags=re.IGNORECASE,
    )


def parse_verdict(text: str) -> bool | None:
    """Distinguish a negative verdict from an unusable service reply."""
    text = text.strip()
    if text.startswith("{"):
        payload = text
    else:
        lines = [line.strip() for line in text.splitlines() if line.strip().startswith("{")]
        if len(lines) != 1:
            return None
        payload = lines[0]
    try:
        verdict = json.loads(payload)
    except ValueError:
        return None
    if not isinstance(verdict, dict):
        return None
    answer, consistent = verdict.get("verdict"), verdict.get("reasoning_consistent")
    if answer not in ("correct", "incorrect") or type(consistent) is not bool:
        return None
    return answer == "correct" and consistent


def answer_span(response: str) -> tuple[str, str]:
    """Return text through the opening answer tag and the answer itself."""
    match = re.search(r"<answer>\s*(?P<answer>.*?)\s*</answer>\.?\s*$", response, re.IGNORECASE | re.DOTALL)
    if match is None or not match["answer"]:
        raise ValueError("Expected a nonempty final answer")
    return response[:match.start("answer")], match["answer"]


def prompt_field(reply: dict, *names: str):
    choices = reply.get("choices")
    containers = [reply]
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        containers.append(choices[0])
    return next(
        (container[name] for container in containers for name in names if container.get(name) is not None),
        None,
    )


def prompt_ids(reply: dict) -> list[int]:
    values = prompt_field(reply, "prompt_token_ids", "prompt_tokens")
    if not isinstance(values, list):
        raise RuntimeError("Likelihood endpoint must return actual prompt token IDs")
    if any(type(token) is not int and not (isinstance(token, str) and token.isascii() and token.isdecimal())
           for token in values):
        raise RuntimeError("Likelihood endpoint must return actual prompt token IDs")
    return [int(token) for token in values]


def answer_token_logprobs(prefix: list[int], full: list[int], reply: dict) -> tuple[int, list[float]]:
    start = next((index for index, (a, b) in enumerate(zip(prefix, full)) if a != b), min(len(prefix), len(full)))
    if not prefix or not full or start == 0 or start < len(prefix) - 1:
        raise RuntimeError("Scoring prompts diverged before the answer-token boundary")
    scores = prompt_field(reply, "prompt_logprobs")
    if not isinstance(scores, list) or len(scores) != len(full):
        raise RuntimeError("Expected aligned prompt_logprobs for all prompt token IDs")
    if start == len(full):
        raise RuntimeError("No actual answer-token logprobs; refusing surrogate score")
    selected = []
    for token, choices in zip(full[start:], scores[start:]):
        value = None
        if isinstance(choices, dict):
            if "logprob" in choices and str(choices.get("token_id", token)) == str(token):
                value = choices["logprob"]
            else:
                entry = choices.get(str(token), choices.get(token))
                if isinstance(entry, dict):
                    value = entry.get("logprob")
        if type(value) not in (int, float) or not math.isfinite(value):
            raise RuntimeError("Incomplete or nonfinite answer-token logprobs")
        if value > 0:
            raise RuntimeError("Invalid positive answer-token logprob")
        selected.append(float(value))
    return start, selected
