from pathlib import Path

import pandas as pd
import pytest
import tiktoken

from dataflow.utils.storage import DataFlowStorage


class MemoryStorage(DataFlowStorage):
    def __init__(self, dataframe):
        self.dataframe = dataframe.copy()
        self.result = None

    def get_keys_from_dataframe(self):
        return self.dataframe.columns.tolist()

    def read(self, output_type="dataframe"):
        return self.dataframe.copy()

    def write(self, dataframe):
        self.result = dataframe.copy()
        return "memory://chunked-results"


class CharacterEncoder:
    def encode(self, text):
        return list(text)


class FakeServing:
    def generate_from_input(self, user_inputs):
        return [f"Generated result {i}" for i in range(len(user_inputs))]


@pytest.fixture
def generator(monkeypatch):
    # The constructor evaluates the default tokenizer at import time. Avoid a
    # network download; the operator already supports an injected encoder.
    with monkeypatch.context() as patch:
        patch.setattr(tiktoken, "get_encoding", lambda name: CharacterEncoder())
        from dataflow.operators.core_text import ChunkedPromptedGenerator

    return ChunkedPromptedGenerator(FakeServing(), enc=CharacterEncoder())


@pytest.mark.cpu
@pytest.mark.parametrize(
    "directory, filenames, output_names",
    [
        ("reports", ["report.v1.txt", "report.v2.txt"],
         ["report.v1_llm_output.txt", "report.v2_llm_output.txt"]),
        ("reports.v1", ["first.txt", "second.txt"],
         ["first_llm_output.txt", "second_llm_output.txt"]),
        ("reports", ["first.txt", "second.txt"],
         ["first_llm_output.txt", "second_llm_output.txt"]),
        ("reports", ["README", ".hidden", "plain.txt"],
         ["README_llm_output.txt", ".hidden_llm_output.txt", "plain_llm_output.txt"]),
    ],
)
def test_output_paths_preserve_directory_and_filename(
    tmp_path, generator, directory, filenames, output_names
):
    parent = tmp_path / directory
    parent.mkdir()
    paths = [parent / filename for filename in filenames]
    for i, path in enumerate(paths):
        path.write_text(f"Source {i}", encoding="utf-8")
    storage = MemoryStorage(pd.DataFrame(
        {"input_path": [str(path) for path in paths]},
        index=[10 + i * 3 for i in range(len(paths))],
    ))

    result_key = generator.run(storage, "input_path", "output_path")

    assert result_key == "output_path"
    outputs = storage.result["output_path"].tolist()
    assert outputs == [str(parent / name) for name in output_names]
    assert len(set(outputs)) == len(paths)
    for i, output in enumerate(outputs):
        assert Path(output).read_text(encoding="utf-8") == f"Generated result {i}"
        assert paths[i].read_text(encoding="utf-8") == f"Source {i}"


@pytest.mark.cpu
def test_relative_output_path_remains_relative(tmp_path, monkeypatch, generator):
    monkeypatch.chdir(tmp_path)
    source = Path("reports.v1") / "report.v2.txt"
    source.parent.mkdir()
    source.write_text("Source", encoding="utf-8")
    storage = MemoryStorage(pd.DataFrame({"input_path": [str(source)]}))

    generator.run(storage, "input_path", "output_path")

    output = Path(storage.result.loc[0, "output_path"])
    assert output == source.parent / "report.v2_llm_output.txt"
    assert not output.is_absolute()
    assert output.read_text(encoding="utf-8") == "Generated result 0"
