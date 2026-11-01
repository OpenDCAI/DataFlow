"""Verify concurrent flushes cannot overwrite a newer persisted generation."""

from pathlib import Path
from threading import Event, Thread, current_thread

import pandas as pd
import pytest

from dataflow.utils import storage as storage_module
from dataflow.utils.storage import LazyFileStorage

pytestmark = pytest.mark.cpu


@pytest.mark.parametrize("cache_type", ["jsonl", "csv"])
def test_two_overlapping_flushes_persist_latest_generation(tmp_path, monkeypatch, cache_type):
    source = tmp_path / "source.csv"
    source.write_text("text\\nsource\\n", encoding="utf-8")
    storage = LazyFileStorage(
        str(source),
        cache_path=str(tmp_path / "cache"),
        cache_type=cache_type,
        save_on_exit=False,
    )
    step = storage.step()
    output_path = Path(step.write([{"text": "first"}]))

    old_replace_entered = Event()
    release_old = Event()
    new_replace_completed = Event()
    errors = []
    real_replace = storage_module.os.replace

    def ordered_replace(src, dst):
        if current_thread().name == "old-flush":
            old_replace_entered.set()
            if not release_old.wait(timeout=15):
                raise AssertionError("old flush was not released")
        result = real_replace(src, dst)
        if current_thread().name == "new-flush":
            new_replace_completed.set()
        return result

    def flush_job():
        try:
            storage.flush_step(1)
        except BaseException as exc:
            errors.append(exc)

    monkeypatch.setattr(storage_module.os, "replace", ordered_replace)
    old_thread = Thread(target=flush_job, name="old-flush")
    new_thread = Thread(target=flush_job, name="new-flush")
    old_thread.start()
    assert old_replace_entered.wait(timeout=10), "old flush did not reach replace"

    # Newer generation arrives while first flush is paused just before replace.
    step.write([{"text": "second"}])
    try:
        new_thread.start()
        # On the buggy implementation, the new flush commits first and the old
        # one subsequently overwrites it. A serialized implementation can keep
        # the new flush waiting until the first one completes.
        new_replace_completed.wait(timeout=3)
    finally:
        release_old.set()
        old_thread.join(timeout=15)
        if new_thread.ident is not None:
            new_thread.join(timeout=15)

    assert not old_thread.is_alive() and not new_thread.is_alive()
    assert not errors, repr(errors)
    if cache_type == "jsonl":
        rows = pd.read_json(output_path, lines=True).to_dict("records")
    else:
        rows = pd.read_csv(output_path).to_dict("records")
    assert rows == [{"text": "second"}], "late old flush overwrote the newer data"
    assert 1 not in storage._dirty_steps
