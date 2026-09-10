"""Regression tests for flushing lazy output buffers without rewriting inputs."""

from pathlib import Path

import pandas as pd
import pytest

from dataflow.utils import storage as storage_module
from dataflow.utils.storage import LazyFileStorage

pytestmark = pytest.mark.cpu


@pytest.fixture
def source_file(tmp_path):
    source = tmp_path / "source.csv"
    source.write_bytes(b'id,text\r\n1,"original, text"\r\n')
    return source


def make_storage(source_file, flush_all_steps=False):
    return LazyFileStorage(
        str(source_file),
        cache_path=str(source_file.parent / "cache"),
        cache_type="jsonl",
        save_on_exit=False,
        flush_all_steps=flush_all_steps,
    )


def assert_no_replace(*args, **kwargs):
    pytest.fail("Flushing unchanged buffers must not replace any files")


@pytest.mark.parametrize("flush_all_steps", [False, True])
def test_flush_all_preserves_source_and_persists_outputs(
    source_file, flush_all_steps, monkeypatch
):
    original = source_file.read_bytes()
    storage = make_storage(source_file, flush_all_steps)
    first_step = storage.step()
    first_step.read()
    first_output = Path(first_step.write([{"text": "first output"}]))
    second_step = storage.step()
    assert second_step.read("dict") == [{"text": "first output"}]
    latest_output = Path(second_step.write([{"text": "latest output"}]))

    assert not first_output.exists()
    assert not latest_output.exists()
    storage.flush_all()

    assert source_file.read_bytes() == original
    assert first_output.exists() is flush_all_steps
    if flush_all_steps:
        assert pd.read_json(first_output, lines=True).to_dict("records") == [
            {"text": "first output"}
        ]
    assert pd.read_json(latest_output, lines=True).to_dict("records") == [
        {"text": "latest output"}
    ]

    # In the default mode, repeated flushes must not walk back through older
    # dirty steps after the latest output has already been persisted.
    monkeypatch.setattr(storage_module.os, "replace", assert_no_replace)
    storage.flush_all()
    assert first_output.exists() is flush_all_steps


@pytest.mark.parametrize("flush_all_steps", [False, True])
def test_flush_all_leaves_read_only_source_untouched(
    source_file, flush_all_steps, monkeypatch
):
    original = source_file.read_bytes()
    storage = make_storage(source_file, flush_all_steps)
    assert storage.step().read("dict") == [{"id": 1, "text": "original, text"}]

    monkeypatch.setattr(storage_module.os, "replace", assert_no_replace)
    storage.flush_all()
    storage.flush_all()

    assert source_file.read_bytes() == original
    assert not Path(storage.cache_path).exists()


@pytest.mark.parametrize("flush_all_steps", [False, True])
def test_flush_all_without_buffers_is_a_noop(source_file, flush_all_steps):
    original = source_file.read_bytes()
    storage = make_storage(source_file, flush_all_steps)

    storage.flush_all()

    assert source_file.read_bytes() == original
    assert not Path(storage.cache_path).exists()


@pytest.mark.parametrize("flush_all_steps", [False, True])
def test_flush_all_does_not_rewrite_outputs_loaded_from_disk(
    source_file, flush_all_steps
):
    original = source_file.read_bytes()
    storage = make_storage(source_file, flush_all_steps)
    storage.step().read()
    cached_output = Path(storage.cache_path) / "dataflow_cache_step_step1.jsonl"
    cached_output.parent.mkdir()
    cached_bytes = b'{ "text": "cached output" }\n'
    cached_output.write_bytes(cached_bytes)
    next_step = storage.step()
    assert next_step.read("dict") == [{"text": "cached output"}]
    new_output = Path(next_step.write([{"text": "new output"}]))

    storage.flush_all()

    assert source_file.read_bytes() == original
    assert cached_output.read_bytes() == cached_bytes
    assert pd.read_json(new_output, lines=True).to_dict("records") == [
        {"text": "new output"}
    ]


def test_explicit_flush_step_can_persist_an_older_output(source_file):
    storage = make_storage(source_file)
    first_output = Path(storage.step().write([{"text": "first output"}]))
    latest_output = Path(storage.step().write([{"text": "latest output"}]))
    storage.flush_all()
    assert not first_output.exists()
    assert latest_output.exists()

    storage.flush_step(1)

    assert pd.read_json(first_output, lines=True).to_dict("records") == [
        {"text": "first output"}
    ]


@pytest.mark.parametrize("flush_all_steps", [False, True])
def test_latest_output_can_be_updated_after_flushing(source_file, flush_all_steps):
    original = source_file.read_bytes()
    storage = make_storage(source_file, flush_all_steps)
    step = storage.step()
    step.read()
    output = Path(step.write([{"text": "first version"}]))
    storage.flush_all()
    assert pd.read_json(output, lines=True).to_dict("records") == [
        {"text": "first version"}
    ]

    step.write([{"text": "updated version"}])
    storage.flush_all()

    assert source_file.read_bytes() == original
    assert pd.read_json(output, lines=True).to_dict("records") == [
        {"text": "updated version"}
    ]
