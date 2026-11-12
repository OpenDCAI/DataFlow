"""Concurrency regression tests for LazyFileStorage flush generation handling."""

from pathlib import Path
from threading import Event, Thread

import pandas as pd
import pytest

from dataflow.utils import storage as storage_module
from dataflow.utils.storage import LazyFileStorage

pytestmark = pytest.mark.cpu


def _make_storage(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("text\nsource\n", encoding="utf-8")
    return LazyFileStorage(
        str(source),
        cache_path=str(tmp_path / "cache"),
        cache_type="jsonl",
        save_on_exit=False,
    )


def test_write_during_older_flush_remains_dirty_and_persistable(tmp_path, monkeypatch):
    storage = _make_storage(tmp_path)
    step = storage.step()
    output = Path(step.write([{"text": "old generation"}]))

    replace_entered = Event()
    release_replace = Event()
    original_replace = storage_module.os.replace

    def blocking_replace(src, dst):
        # Synchronize below LazyFileStorage.flush_step(): the real flush has
        # already selected and serialized its old dataframe snapshot here.
        replace_entered.set()
        assert release_replace.wait(timeout=10), "test failed to release blocked persistence"
        return original_replace(src, dst)

    monkeypatch.setattr(storage_module.os, "replace", blocking_replace)

    flush_thread = Thread(target=storage.flush_step, args=(1,))
    flush_thread.start()
    assert replace_entered.wait(timeout=10), "flush never reached persistence boundary"

    # Real concurrent write to the same output step while the older flush is paused.
    step.write([{"text": "new generation"}])

    release_replace.set()
    flush_thread.join(timeout=10)
    assert not flush_thread.is_alive()

    # The completed old flush may persist only its selected snapshot.
    assert pd.read_json(output, lines=True).to_dict("records") == [
        {"text": "old generation"}
    ]

    # Restore real persistence and use the normal public flush path. The newer
    # write must still be considered dirty so flush_all() persists it.
    monkeypatch.setattr(storage_module.os, "replace", original_replace)
    storage.flush_all()

    assert pd.read_json(output, lines=True).to_dict("records") == [
        {"text": "new generation"}
    ]
    assert 1 not in storage._dirty_steps
