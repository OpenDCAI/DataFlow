"""Regression tests for crash fixes in the undefined-name branch.

Each test fails if its corresponding production fix is reverted.
"""

import base64
import importlib

import pandas as pd
import pytest

pytestmark = pytest.mark.cpu


class InMemoryStorage:
    """Minimal DataFlowStorage stand-in so operators can run without disk I/O."""

    file_name_prefix = "regression_test"

    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe

    def read(self, _output_type: str) -> pd.DataFrame:
        return self.dataframe

    def write(self, dataframe: pd.DataFrame) -> str:
        self.dataframe = dataframe
        return "in_memory.json"


@pytest.fixture
def api_serving(monkeypatch):
    monkeypatch.setenv("DF_API_KEY", "dummy-key")
    from dataflow.serving import APILLMServing_request

    serving = APILLMServing_request(key_name_of_api_key="DF_API_KEY")
    yield serving
    serving.cleanup()


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


def test_lalm_serving_decodes_base64_audio(monkeypatch):
    """`_read_audio_base64` called base64.b64decode without importing base64."""
    module = importlib.import_module("dataflow.serving.localmodel_lalm_serving")
    audio_bytes = b"regression-test-audio"
    encoded_audio = base64.b64encode(audio_bytes).decode("ascii")
    expected_waveform = object()

    def read_audio_bytes(data: bytes, sr: int):
        assert data == audio_bytes
        assert sr == 16_000
        return expected_waveform, sr

    monkeypatch.setattr(module, "_read_audio_bytes", read_audio_bytes)

    assert module._read_audio_base64(
        f"data:audio/wav;base64,{encoded_audio}",
        sr=16_000,
    ) == (expected_waveform, 16_000)


def test_format_response_handles_null_content(api_serving):
    """Providers send `"content": null` when a reply carries only tool calls.

    `.get('content', '')` returns None in that case, and the regex below it
    raised TypeError, which the caller swallowed into a silently dropped row.
    """
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

    assert api_serving.format_response(response) == ""


def test_format_response_wraps_reasoning_when_content_is_null(api_serving):
    """Reasoning models may send reasoning_content alongside a null content."""
    response = {
        "choices": [{"message": {"content": None, "reasoning_content": "step one"}}]
    }

    assert (
        api_serving.format_response(response)
        == "<think>step one</think>\n<answer></answer>"
    )


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
def test_bench_evaluator_handles_missing_reference_answers(
    keep_all_samples, expected_rows
):
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
        pd.DataFrame({"question": ["a"], "generated_cot": ["x"], "golden_answer": [""]})
    )

    returned_keys = evaluator.run(storage=storage, input_question_key="question")

    assert returned_keys == [
        "generated_cot",
        "golden_answer",
        "question",
        "answer_match_result",
    ]
    assert len(storage.dataframe) == expected_rows


def test_bench_evaluator_validates_columns_before_reading_them():
    """Missing input columns should be reported instead of raising KeyError."""
    from dataflow.prompts.model_evaluation.general import AnswerJudgePromptQuestion
    from dataflow.operators.core_text import BenchDatasetEvaluatorQuestion

    evaluator = BenchDatasetEvaluatorQuestion(
        compare_method="semantic",
        llm_serving=object(),
        prompt_template=AnswerJudgePromptQuestion(),
    )
    storage = InMemoryStorage(pd.DataFrame({"question": ["a"]}))

    returned_keys = evaluator.run(storage=storage, input_question_key="question")

    assert returned_keys == ["generated_cot", "golden_answer", "question"]
    assert "answer_match_result" not in storage.dataframe.columns


def test_bench_match_mode_initializes_subquestion_setting(tmp_path, monkeypatch):
    """Match-mode statistics must not read an attribute only semantic mode defines."""
    from dataflow.prompts.model_evaluation.general import AnswerJudgePromptQuestion
    from dataflow.operators.core_text import BenchDatasetEvaluatorQuestion

    evaluator = BenchDatasetEvaluatorQuestion(
        compare_method="match",
        eval_result_path=str(tmp_path / "statistics.json"),
        prompt_template=AnswerJudgePromptQuestion(),
    )
    monkeypatch.setattr(
        evaluator.answer_extractor,
        "extract_answer",
        lambda answer, _: answer,
    )
    monkeypatch.setattr(
        evaluator, "compare", lambda answer, expected: answer == expected
    )
    storage = InMemoryStorage(
        pd.DataFrame({"generated_cot": ["42"], "golden_answer": ["42"]})
    )

    returned_keys = evaluator.run(storage=storage)

    assert returned_keys == [
        "generated_cot",
        "golden_answer",
        "answer_match_result",
    ]
    assert storage.dataframe["answer_match_result"].tolist() == [True]


@pytest.mark.parametrize("compare_method", ["match", "semantic"])
def test_bench_evaluator_handles_non_default_index(compare_method, tmp_path, monkeypatch):
    """Upstream filters hand downstream a sliced frame whose index is not 0..n-1.

    Row lookups used to index the Series by position, which is a label lookup
    on a pandas Series, so a gapped index raised KeyError.
    """
    from dataflow.prompts.model_evaluation.general import AnswerJudgePromptQuestion
    from dataflow.operators.core_text import BenchDatasetEvaluatorQuestion

    class StubLLMServing:
        def generate_from_input(self, user_inputs, system_prompt=None):
            return ['{"judgement_result": true}', '{"judgement_result": false}']

    evaluator = BenchDatasetEvaluatorQuestion(
        compare_method=compare_method,
        eval_result_path=str(tmp_path / "statistics.json"),
        llm_serving=StubLLMServing(),
        prompt_template=AnswerJudgePromptQuestion(),
    )
    if compare_method == "match":
        monkeypatch.setattr(
            evaluator.answer_extractor, "extract_answer", lambda answer, _: answer
        )
        monkeypatch.setattr(
            evaluator, "compare", lambda answer, expected: answer == expected
        )
    sliced_frame = pd.DataFrame(
        {
            "question": ["q1", "q2"],
            "generated_cot": ["42", "7"],
            "golden_answer": ["42", "9"],
        },
        index=[3, 7],
    )
    storage = InMemoryStorage(sliced_frame)

    evaluator.run(storage=storage, input_question_key="question")

    assert storage.dataframe["answer_match_result"].tolist() == [True, False]
