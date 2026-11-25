"""Preserve QA strings at file boundaries and publish journal records atomically."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

import pandas as pd

from dataflow.utils.storage import FileStorage, LazyFileStorage


def input_path(value):
    path = Path(value).resolve()
    if path.suffix.lower() not in {".jsonl", ".json"}:
        raise ValueError("Expected a local .jsonl or .json QA file")
    return path


class QAReader:
    """Preserve JSON string values, including leading zeroes and decimal precision."""
    def read(self, file_path):
        path = input_path(file_path)
        with path.open(encoding="utf-8-sig") as stream:
            if path.suffix.lower() == ".jsonl":
                rows = [json.loads(line) for line in stream if line.strip()]
            else:
                rows = json.load(stream)
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise ValueError("Expected one JSON object per QA record")
        return pd.DataFrame(rows, dtype=object)


def read_qa_frame(storage):
    if (type(storage) in {FileStorage, LazyFileStorage}
            and storage.operator_step > 0 and storage.cache_type in {"json", "jsonl"}):
        # Compilation may have inferred numeric IDs while loading a saved stage.
        # A pending LazyFileStorage write is newer than its on-disk version.
        path = Path(storage._get_cache_file_path(storage.operator_step))
        if type(storage) is FileStorage or (storage.operator_step not in storage._dirty_steps and path.is_file()):
            return QAReader().read(path)
    if type(storage) in {FileStorage, LazyFileStorage} and storage.operator_step == 0:
        path = Path(storage.first_entry_file_name)
        if path.suffix.lower() == ".json":
            return QAReader().read(path)
        if path.suffix.lower() == ".jsonl" and getattr(storage, "jsonl_read_policy", None) is None:
            # Pipeline compilation may already have buffered an inferred frame.
            return QAReader().read(path)
        if path.suffix.lower() in {".pickle", ".pkl"}:
            raise ValueError("Expected a local .jsonl or .json QA file")
    return storage.read("dataframe").copy()


def publish_text(path, content):
    """Publish a completed journal file without overwriting an existing entry."""
    data = content.encode("utf-8")
    stream = tempfile.NamedTemporaryFile(dir=path.parent, delete=False)
    temporary = Path(stream.name)
    try:
        with stream:
            stream.write(data)
        try:
            if os.name == "nt":
                os.rename(temporary, path)  # On Windows rename refuses existing targets.
            else:
                os.link(temporary, path)
        except FileExistsError:
            raise FileExistsError(f"Refusing to overwrite existing journal entry: {path}") from None
    finally:
        temporary.unlink(missing_ok=True)
