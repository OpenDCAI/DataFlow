"""Regression tests for four crashes caused by names that were never bound.

Every test here fails on the parent commit and passes with the accompanying fix.
"""
import importlib

import pandas as pd
import pytest


class InMemoryStorage:
    """Minimal DataFlowStorage stand-in so operators can run without disk I/O."""

    file_name_prefix = "regression_test"

    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe

    def read(self, output_type):
        return self.dataframe

    def write(self, dataframe):
        self.dataframe = dataframe
        return "in_memory.json"


def test_text2sql_prompts_import():
    """`from re import template` broke every text2sql operator on Python 3.13+.

    `re.template` was deprecated in 3.11 and removed in 3.13, so importing this
    module raised ImportError before any operator could be constructed.
    """
    module = importlib.import_module("dataflow.prompts.text2sql")

    assert hasattr(module, "SelectSQLGeneratorPrompt")
    assert not hasattr(module, "template")


@pytest.mark.parametrize(
    "operator_name",
    ["SQLGenerator", "SQLVariationGenerator", "Text2SQLPromptGenerator"],
)
def test_text2sql_operators_import(operator_name):
    """The operators that transitively import the prompt module above."""
    operators = importlib.import_module("dataflow.operators.text2sql")

    assert getattr(operators, operator_name) is not None


def test_lalm_serving_binds_base64():
    """`_read_audio_base64` called base64.b64decode without importing base64."""
    module = importlib.import_module("dataflow.serving.localmodel_lalm_serving")

    assert hasattr(module, "base64")


def test_format_response_handles_null_content(monkeypatch):
    """Providers send `"content": null` when a reply carries only tool calls.

    `.get('content', '')` returns None in that case, and the regex below it
    raised TypeError, which the caller swallowed into a silently dropped row.
    """
    monkeypatch.setenv("DF_API_KEY", "dummy-key")
    from dataflow.serving import APILLMServing_request

    serving = APILLMServing_request(key_name_of_api_key="DF_API_KEY")
    response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"id": "call_1", "type": "function"}],
                }
            }
        ]
    }

    assert serving.format_response(response) == ""
    serving.cleanup()


def test_format_response_wraps_reasoning_when_content_is_null(monkeypatch):
    """Reasoning models may send reasoning_content alongside a null content."""
    monkeypatch.setenv("DF_API_KEY", "dummy-key")
    from dataflow.serving import APILLMServing_request

    serving = APILLMServing_request(key_name_of_api_key="DF_API_KEY")
    response = {
        "choices": [{"message": {"content": None, "reasoning_content": "step one"}}]
    }

    assert serving.format_response(response) == "<think>step one</think>\n<answer></answer>"
    serving.cleanup()


def test_code_score_filter_rejects_unknown_filter_method():
    """The guard raised NameError instead of the ValueError it was written to raise."""
    from dataflow.operators.code import CodeGenericScoreFilter

    score_filter = CodeGenericScoreFilter(
        score_threshold=0.5,
        filter_method="not_a_real_method",
    )
    storage = InMemoryStorage(pd.DataFrame({"score": [0.1, 0.9]}))

    with pytest.raises(ValueError, match="not_a_real_method"):
        score_filter.run(storage=storage, input_key="score", output_key="kept")


@pytest.mark.parametrize("keep_all_samples, expected_rows", [(False, 0), (True, 1)])
def test_bench_evaluator_handles_missing_reference_answers(keep_all_samples, expected_rows):
    """`return required_columns` referenced a name the method never bound.

    Reached on an ordinary data condition -- every reference answer empty --
    and `self.keep_all_samples` was read one line earlier without ever being
    set in __init__, so the branch raised AttributeError before that.
    """
    from dataflow.prompts.model_evaluation.general import AnswerJudgePromptQuestion
    from dataflow.operators.core_text import BenchDatasetEvaluatorQuestion

    evaluator = BenchDatasetEvaluatorQuestion(
        compare_method="semantic",
        llm_serving=object(),
        prompt_template=AnswerJudgePromptQuestion(),
        keep_all_samples=keep_all_samples,
    )
    storage = InMemoryStorage(
        pd.DataFrame(
            {"question": ["a"], "generated_cot": ["x"], "golden_answer": [""]}
        )
    )

    returned_keys = evaluator.run(storage=storage, input_question_key="question")

    assert "answer_match_result" in returned_keys
    assert len(storage.dataframe) == expected_rows
