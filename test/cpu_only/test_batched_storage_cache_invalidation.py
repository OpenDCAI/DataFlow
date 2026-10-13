import pandas as pd
import pytest

from dataflow.core import OperatorABC
from dataflow.pipeline import BatchedPipelineABC
from dataflow.utils.storage import BatchedFileStorage


def _write_source(tmp_path, cache_type, rows=None):
    rows = rows or [{"seed": "row"}]
    source = tmp_path / f"input.{cache_type}"
    dataframe = pd.DataFrame(rows)
    if cache_type == "jsonl":
        dataframe.to_json(source, orient="records", lines=True, force_ascii=False)
    else:
        dataframe.to_csv(source, index=False)
    return source


def _read_persisted(path, cache_type):
    if cache_type == "jsonl":
        return pd.read_json(path, lines=True)
    return pd.read_csv(path)


class _VersionedProducer(OperatorABC):
    def __init__(self, version: str):
        super().__init__()
        self.version = version
        self.calls = 0

    def run(self, storage):
        self.calls += 1
        dataframe = storage.read("dataframe").copy()
        dataframe["version"] = self.version
        storage.write(dataframe)


class _ObservingConsumer(OperatorABC):
    def __init__(self, fail_once=False):
        super().__init__()
        self.seen_versions = []
        self.fail_once = fail_once

    def run(self, storage):
        dataframe = storage.read("dataframe").copy()
        self.seen_versions.append(dataframe["version"].tolist())
        if self.fail_once:
            self.fail_once = False
            raise RuntimeError("intentional consumer failure")
        storage.write(dataframe)


class _RerunnableBatchedPipeline(BatchedPipelineABC):
    def __init__(self, source, cache_path, cache_type, producer, consumer):
        super().__init__()
        self.storage = BatchedFileStorage(
            first_entry_file_name=str(source),
            cache_path=str(cache_path),
            file_name_prefix="p07_02",
            cache_type=cache_type,
        )
        self.producer = producer
        self.consumer = consumer

    def forward(self):
        self.producer.run(self.storage.step())
        self.consumer.run(self.storage.step())


@pytest.mark.cpu
@pytest.mark.parametrize("cache_type", ["jsonl", "csv"])
def test_public_step_rewrite_is_immediately_visible(tmp_path, cache_type):
    source = _write_source(tmp_path, cache_type)
    root = BatchedFileStorage(
        first_entry_file_name=str(source),
        cache_path=str(tmp_path / "cache-public"),
        file_name_prefix="public_transition",
        cache_type=cache_type,
    )

    producer_storage = root.step()
    consumer_storage = root.step()

    producer_storage.write([{"value": "old"}])
    assert consumer_storage.read("dict") == [{"value": "old"}]

    producer_storage.write([{"value": "new"}])
    persisted_path = (
        tmp_path / "cache-public" / f"public_transition_step1.{cache_type}"
    )
    assert _read_persisted(persisted_path, cache_type).to_dict("records") == [
        {"value": "new"}
    ]

    assert consumer_storage.read("dict") == [{"value": "new"}]


@pytest.mark.cpu
@pytest.mark.parametrize("cache_type", ["jsonl", "csv"])
def test_batched_pipeline_rerun_without_resume_consumes_current_output(
    tmp_path, cache_type
):
    source = _write_source(tmp_path, cache_type)
    cache_path = tmp_path / "cache-pipeline"
    producer = _VersionedProducer("first")
    consumer = _ObservingConsumer()
    pipeline = _RerunnableBatchedPipeline(
        source, cache_path, cache_type, producer, consumer
    )

    pipeline.compile()
    pipeline.forward(batch_size=1, resume_from_last=False)
    assert consumer.seen_versions[-1] == ["first"]

    producer.version = "second"
    pipeline.forward(batch_size=1, resume_from_last=False)

    persisted_step1 = cache_path / f"p07_02_step1.{cache_type}"
    assert _read_persisted(persisted_step1, cache_type)["version"].tolist() == [
        "second"
    ]
    assert consumer.seen_versions[-1] == ["second"]


@pytest.mark.cpu
@pytest.mark.parametrize("cache_type", ["jsonl", "csv"])
def test_successful_write_invalidates_only_next_step_cache(tmp_path, cache_type):
    source = _write_source(
        tmp_path,
        cache_type,
        [{"seed": "first"}, {"seed": "second"}, {"seed": "third"}],
    )
    storage = BatchedFileStorage(
        first_entry_file_name=str(source),
        cache_path=str(tmp_path / "cache-slicing"),
        file_name_prefix="slicing",
        cache_type=cache_type,
    ).step()
    storage.batch_size = 2
    storage.batch_step = 0

    assert storage.read("dict") == [{"seed": "first"}, {"seed": "second"}]
    storage.write([{"derived": "current"}])

    # A write to step 1 must not globally discard the still-current step 0 cache.
    source.unlink()
    storage.batch_step = 1
    assert storage.read("dict") == [{"seed": "third"}]
    assert storage.record_count == 3


@pytest.mark.cpu
@pytest.mark.parametrize("cache_type", ["jsonl", "csv"])
def test_failed_write_does_not_invalidate_cached_generation(tmp_path, cache_type):
    source = _write_source(tmp_path, cache_type)
    root = BatchedFileStorage(
        first_entry_file_name=str(source),
        cache_path=str(tmp_path / "cache-failed-write"),
        file_name_prefix="failed_write",
        cache_type=cache_type,
    )
    producer_storage = root.step()
    consumer_storage = root.step()

    producer_storage.write([{"value": "old"}])
    assert consumer_storage.read("dict") == [{"value": "old"}]

    with pytest.raises(ValueError, match="Unsupported data type"):
        producer_storage.write([1])

    assert consumer_storage.read("dict") == [{"value": "old"}]


@pytest.mark.cpu
@pytest.mark.parametrize("cache_type", ["jsonl", "csv"])
def test_resume_from_last_keeps_last_successful_generation(tmp_path, cache_type):
    source = _write_source(tmp_path, cache_type)
    cache_path = tmp_path / "cache-resume"
    producer = _VersionedProducer("first")
    consumer = _ObservingConsumer(fail_once=True)
    pipeline = _RerunnableBatchedPipeline(
        source, cache_path, cache_type, producer, consumer
    )

    pipeline.compile()
    with pytest.raises(RuntimeError, match="intentional consumer failure"):
        pipeline.forward(batch_size=1, resume_from_last=True)

    assert producer.calls == 1
    assert consumer.seen_versions[-1] == ["first"]

    # Resume must skip the already-successful producer. Changing its configuration
    # cannot affect the persisted generation that the resumed consumer should read.
    producer.version = "second"
    pipeline.forward(batch_size=1, resume_from_last=True)

    assert producer.calls == 1
    assert consumer.seen_versions[-1] == ["first"]
    persisted_step1 = cache_path / f"p07_02_step1.{cache_type}"
    assert _read_persisted(persisted_step1, cache_type)["version"].tolist() == [
        "first"
    ]
