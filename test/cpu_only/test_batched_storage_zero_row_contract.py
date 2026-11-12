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


def _source_with_one_row(tmp_path, name="source.csv"):
    source = tmp_path / name
    pd.DataFrame({"seed": ["row"]}).to_csv(source, index=False)
    return source


def _batched_storage(tmp_path, cache_type, cache_name):
    return BatchedFileStorage(
        first_entry_file_name=str(_source_with_one_row(tmp_path, f"{cache_name}-source.csv")),
        cache_path=str(tmp_path / cache_name),
        file_name_prefix=cache_name,
        cache_type=cache_type,
    ).step()


def _write_batches(storage, batches):
    for batch_step, batch in enumerate(batches):
        storage.batch_step = batch_step
        storage.write(batch)


def _read_next_dataframe(storage):
    next_storage = storage.step()
    next_storage.batch_size = None
    next_storage.batch_step = 0
    return next_storage.read("dataframe")


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
def test_csv_all_zero_batches_round_trip_as_empty(tmp_path):
    storage = _batched_storage(tmp_path, "csv", "csv-all-zero")
    _write_batches(storage, [[], [], []])

    dataframe = _read_next_dataframe(storage)
    assert dataframe.to_dict("records") == []
    assert dataframe.columns.tolist() == []


@pytest.mark.cpu
def test_csv_empty_first_batch_then_non_empty_batches_preserve_header_and_records(tmp_path):
    storage = _batched_storage(tmp_path, "csv", "csv-empty-first")
    expected = [
        {"value": "alpha", "rank": 1},
        {"value": "beta", "rank": 2},
        {"value": "gamma", "rank": 3},
    ]
    _write_batches(
        storage,
        [
            [],
            expected[:2],
            expected[2:],
        ],
    )

    dataframe = _read_next_dataframe(storage)
    assert dataframe.columns.tolist() == ["value", "rank"]
    assert dataframe.to_dict("records") == expected


@pytest.mark.cpu
def test_csv_empty_middle_and_trailing_batches_do_not_corrupt_prior_data(tmp_path):
    storage = _batched_storage(tmp_path, "csv", "csv-empty-middle-trailing")
    expected = [
        {"value": "alpha", "rank": 1},
        {"value": "beta", "rank": 2},
    ]
    _write_batches(storage, [expected[:1], [], expected[1:], []])

    dataframe = _read_next_dataframe(storage)
    assert dataframe.columns.tolist() == ["value", "rank"]
    assert dataframe.to_dict("records") == expected


@pytest.mark.cpu
def test_csv_non_empty_first_batch_keeps_existing_append_behavior(tmp_path):
    storage = _batched_storage(tmp_path, "csv", "csv-normal-first")
    expected = [
        {"value": "alpha", "rank": 1},
        {"value": "beta", "rank": 2},
    ]
    _write_batches(storage, [expected[:1], expected[1:]])

    dataframe = _read_next_dataframe(storage)
    assert dataframe.columns.tolist() == ["value", "rank"]
    assert dataframe.to_dict("records") == expected


@pytest.mark.cpu
def test_jsonl_multi_batch_empty_positions_preserve_existing_behavior(tmp_path):
    storage = _batched_storage(tmp_path, "jsonl", "jsonl-empty-positions")
    expected = [{"value": "alpha"}, {"value": "beta"}]
    _write_batches(storage, [[], expected[:1], [], expected[1:], []])

    dataframe = _read_next_dataframe(storage)
    assert dataframe.columns.tolist() == ["value"]
    assert dataframe.to_dict("records") == expected


@pytest.mark.cpu
@pytest.mark.parametrize("cache_type", ["jsonl", "csv"])
def test_batched_write_preserves_non_empty_list_and_dataframe(tmp_path, cache_type):
    source = _source_with_one_row(tmp_path)

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
    source = _source_with_one_row(tmp_path)
    storage = BatchedFileStorage(
        first_entry_file_name=str(source),
        cache_path=str(tmp_path / "cache"),
        cache_type="jsonl",
    ).step()

    with pytest.raises(ValueError, match="Unsupported data type"):
        storage.write(unsupported)


@pytest.mark.cpu
def test_csv_missing_file_preserves_existing_error(tmp_path):
    storage = BatchedFileStorage(
        first_entry_file_name=str(tmp_path / "source.csv"),
        cache_path=str(tmp_path / "cache"),
        cache_type="csv",
    )
    missing = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError) as exc_info:
        storage._load_local_file(str(missing), "csv")

    assert str(exc_info.value) == f"File {missing} does not exist. Please check the path."


@pytest.mark.cpu
def test_csv_malformed_file_preserves_existing_wrapped_error(tmp_path):
    storage = BatchedFileStorage(
        first_entry_file_name=str(tmp_path / "source.csv"),
        cache_path=str(tmp_path / "cache"),
        cache_type="csv",
    )
    invalid = tmp_path / "invalid.csv"
    invalid.write_text('a,b\n1,"unterminated\n', encoding="utf-8")

    with pytest.raises(ValueError, match=r"^Failed to load csv file:"):
        storage._load_local_file(str(invalid), "csv")
