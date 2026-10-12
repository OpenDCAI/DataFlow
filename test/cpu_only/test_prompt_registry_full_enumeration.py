import subprocess
import sys
import textwrap

import pytest


pytestmark = pytest.mark.cpu


def _run_fresh_python(script: str):
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_prompt_registry_full_enumeration_uses_real_prompt_tree():
    result = _run_fresh_python(
        """
        from dataflow.core.prompt import PromptABC
        from dataflow.utils.registry import PROMPT_REGISTRY

        # Normal lookup for an already-registered prompt must stay lazy.
        assert None in PROMPT_REGISTRY.loader_map.values()
        general_prompt = PROMPT_REGISTRY.get("GeneralAnswerGeneratorPrompt")
        assert general_prompt.__name__ == "GeneralAnswerGeneratorPrompt"
        assert None in PROMPT_REGISTRY.loader_map.values()

        before = dict(PROMPT_REGISTRY.get_obj_map())
        assert "GeneralAnswerGeneratorPrompt" in before
        assert "FormatStrPrompt" not in before
        assert "AnswerJudgePrompt" not in before

        # Exercise the real full-enumeration boundary and real prompt module tree.
        PROMPT_REGISTRY._get_all()

        prompts = PROMPT_REGISTRY.get_obj_map()
        assert None not in PROMPT_REGISTRY.loader_map.values()
        assert len(prompts) > len(before)
        assert "GeneralAnswerGeneratorPrompt" in prompts
        assert "FormatStrPrompt" in prompts
        assert "ExtractSmilesFromTextPrompt" in prompts
        # model_evaluation is a real nested prompt package without its own __init__.py.
        assert "AnswerJudgePrompt" in prompts
        assert all(
            isinstance(prompt, type) and issubclass(prompt, PromptABC)
            for prompt in prompts.values()
        )

        # Single-item lookup must still resolve the same registered class.
        assert PROMPT_REGISTRY.get("FormatStrPrompt") is prompts["FormatStrPrompt"]
        assert PROMPT_REGISTRY.get("GeneralAnswerGeneratorPrompt") is general_prompt
        """
    )

    assert result.returncode == 0, (
        "fresh-process PROMPT_REGISTRY enumeration failed\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
