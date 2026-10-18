"""Regression tests for ``ReferenceRemoverRefiner``.

The operator must actually write back the cleaned text (it previously appended
the original value, silently discarding the refinement), and must count each
modified row exactly once (see issue #554).
"""
import pandas as pd
import pytest

from dataflow.operators.general_text import ReferenceRemoverRefiner
from dataflow.utils.storage import DataFlowStorage


class MemoryStorage(DataFlowStorage):
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.result = None

    def get_keys_from_dataframe(self):
        return self.dataframe.columns.tolist()

    def read(self, output_type="dataframe"):
        assert output_type == "dataframe"
        return self.dataframe.copy()

    def write(self, dataframe):
        self.result = dataframe.copy()
        return "memory://reference-remover"


def _run(rows):
    storage = MemoryStorage(pd.DataFrame({"text": rows}))
    ReferenceRemoverRefiner().run(storage, input_key="text")
    return storage.result["text"].tolist()


@pytest.mark.cpu
def test_removes_closed_ref_tags_and_cite_templates():
    assert _run([
        "Paris is the capital of France.<ref>Smith 2020</ref>",
        "Water boils at 100 C.{{cite web|url=http://example.com}}",
    ]) == [
        "Paris is the capital of France.",
        "Water boils at 100 C.",
    ]


@pytest.mark.cpu
def test_removes_unclosed_ref_and_cite_forms():
    assert _run([
        "abc<ref name=x>unclosed at end",
        "open {{cite book|title=abc no close",
    ]) == ["abc", "open "]


@pytest.mark.cpu
def test_rows_without_markers_are_unchanged():
    rows = ["A plain row.", "Another plain row."]
    assert _run(rows) == rows


@pytest.mark.cpu
def test_each_modified_row_is_counted_once(caplog):
    with caplog.at_level("INFO", logger="DataFlow"):
        _run([
            "Paris is the capital of France.<ref>Smith 2020</ref>",
            "Water boils at 100 C.{{cite web|url=http://example.com}}",
            "A plain row with no markers.",
        ])

    summary = [r.message for r in caplog.records if "Total modified items" in r.message]
    assert summary, "expected a 'Total modified items' summary log"
    # Two rows were modified, and each must be counted exactly once (not 4).
    assert "Total modified items: 2" in summary[-1]
