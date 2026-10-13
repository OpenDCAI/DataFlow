import pandas as pd
import pytest

from dataflow.operators.reasoning.generate.reasoning_pretrain_format_convert_generator import (
    ReasoningPretrainFormatConvertGenerator,
)
from dataflow.utils.storage import BatchedFileStorage


def _zero_row_source(tmp_path):
    source = tmp_path / "zero_rows.csv"
    pd.DataFrame(columns=["question", "answer"]).to_csv(source, index=False)
    return source


@pytest.mark.cpu
@pytest.mark.parametrize("cache_type", ["jsonl", "csv"])
def test_zero_row_list_round_trips_through_real_batched_operator(tmp_path, cache_type):
    """A valid zero-row dataset must survive a real operator write and next-step read."""
    storage = BatchedFileStorage(
        first_entry_file_name=str(_zero_row_source(tmp_path)),
        cache_path=str(tmp_path / "cache"),
        file_name_prefix=f"p07_04_{cache_type}",
        cache_type=cache_type,
    ).step()
    storage.batch_size = 8

    # Zero rows are already a supported BatchedFileStorage input state.
    assert storage.read("dict") == []
    assert storage.record_count == 0

    # This public operator naturally converts a zero-row dataframe to [] and writes it.
    ReasoningPretrainFormatConvertGenerator().run(
        storage=storage,
        input_read_key_question="question",
        input_read_key_answer="answer",
        output_key="text",
    )

    # The persisted zero-row artifact must remain readable by the next real storage step.
    next_storage = storage.step()
    next_storage.batch_size = 8
    next_storage.batch_step = 0
    assert next_storage.read("dict") == []
    assert next_storage.record_count == 0


@pytest.mark.cpu
@pytest.mark.parametrize("cache_type", ["jsonl", "csv"])
def test_batched_write_preserves_non_empty_list_and_dataframe(tmp_path, cache_type):
    source = tmp_path / "source.csv"
    pd.DataFrame({"seed": ["row"]}).to_csv(source, index=False)

    list_storage = BatchedFileStorage(
        first_entry_file_name=str(source),
        cache_path=str(tmp_path / "list-cache"),
        file_name_prefix="list_contract",
        cache_type=cache_type,
    ).step()
    list_storage.write([{"value": "alpha"}])
    assert list_storage.step().read("dict") == [{"value": "alpha"}]

    dataframe_storage = BatchedFileStorage(
        first_entry_file_name=str(source),
        cache_path=str(tmp_path / "dataframe-cache"),
        file_name_prefix="dataframe_contract",
        cache_type=cache_type,
    ).step()
    dataframe_storage.write(pd.DataFrame({"value": ["beta"]}))
    assert dataframe_storage.step().read("dict") == [{"value": "beta"}]


@pytest.mark.cpu
@pytest.mark.parametrize("unsupported", [[1], ["not-a-dict"], [None]])
def test_batched_write_still_rejects_unsupported_list_elements(tmp_path, unsupported):
    source = tmp_path / "source.csv"
    pd.DataFrame({"seed": ["row"]}).to_csv(source, index=False)
    storage = BatchedFileStorage(
        first_entry_file_name=str(source),
        cache_path=str(tmp_path / "cache"),
        cache_type="jsonl",
    ).step()

    with pytest.raises(ValueError, match="Unsupported data type"):
        storage.write(unsupported)
