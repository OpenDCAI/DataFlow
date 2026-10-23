"""Opt-in, auditable local JSONL reading without changing pandas defaults."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pandas as pd


class JSONLReadPolicy:
    """Quarantine malformed records; publish one error ledger per source scan.

    The supplied error path belongs to one input file. Repeated header/count/data
    reads replace the ledger rather than append duplicate errors. No raw input
    is recorded. Filesystem failures propagate. Only local initial JSONL inputs
    use this policy; intermediate Storage files retain their existing readers.
    """

    def __init__(self, errors_path):
        self.errors_path = Path(errors_path).resolve()
        self.last_report = {}
        self._inspected = None
        self._source_path = None

    def bind(self, storage):
        source = Path(storage.first_entry_file_name).resolve()
        if self._source_path is not None and self._source_path != source:
            raise ValueError("Use a separate JSONL read policy and ledger per source")
        if self.errors_path == source:
            raise ValueError("JSONL error ledger must differ from the source")
        if self.errors_path.parent == Path(storage.cache_path).resolve() and self.errors_path.name.startswith(storage.file_name_prefix + "_step"):
            raise ValueError("JSONL error ledger must not use a Storage cache filename")

    @staticmethod
    def _identity(path):
        stat = path.stat()
        return (str(path), stat.st_ino, stat.st_size, stat.st_mtime_ns)

    @staticmethod
    def _records(path):
        with path.open("rb") as source:
            line, offset = 0, 0
            while True:
                raw = source.readline()
                if not raw:
                    break
                line += 1
                location = {"source": str(path), "line": line, "byte_offset": offset}
                offset += len(raw)
                try:
                    text = raw.decode("utf-8")
                    if not text.strip():
                        yield None, location, "blank_line"
                        continue
                    def reject_constant(value):
                        raise ValueError("Nonstandard JSON constant")
                    row = json.loads(text, parse_constant=reject_constant)
                    if not isinstance(row, dict):
                        yield None, location, "not_object"
                        continue
                    yield row, location, None
                except UnicodeDecodeError:
                    yield None, location, "invalid_utf8"
                except (json.JSONDecodeError, ValueError):
                    yield None, location, "invalid_json"

    def inspect(self, file_path):
        path = Path(file_path).resolve()
        if self._source_path is not None and self._source_path != path:
            raise ValueError("Use a separate JSONL read policy and ledger per source")
        self._source_path = path
        if path == self.errors_path:
            raise ValueError("JSONL error ledger must differ from the source")
        identity = self._identity(path)
        if self._inspected == identity and self.errors_path.exists():
            return {**self.last_report, "keys": list(self.last_report["keys"])}
        self.errors_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(prefix=".jsonl-errors-", dir=self.errors_path.parent)
        scanned = accepted = 0
        keys = {}
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as errors:
                for row, location, code in self._records(path):
                    scanned += 1
                    if code is not None:
                        errors.write(json.dumps({"schema_version": 1, **location, "code": code}) + "\n")
                    else:
                        accepted += 1
                        keys.update(dict.fromkeys(row))
                errors.flush()
                os.fsync(errors.fileno())
            if self._identity(path) != identity:
                raise ValueError("JSONL source changed during inspection")
            os.replace(name, self.errors_path)
        finally:
            if os.path.exists(name):
                os.unlink(name)
        self.last_report = {
            "source": str(path), "scanned": scanned,
            "accepted": accepted, "rejected": scanned - accepted,
            "keys": list(keys),
        }
        self._inspected = identity
        return {**self.last_report, "keys": list(self.last_report["keys"])}

    def iter_chunks(self, file_path, batch_size):
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError("JSONL batch size must be a positive integer")
        path = Path(file_path).resolve()
        report = self.inspect(path)
        identity = self._inspected
        rows, locations = [], []
        for row, location, code in self._records(path):
            if code is not None:
                continue
            rows.append(row)
            locations.append(location)
            if len(rows) == batch_size:
                frame = pd.DataFrame(rows, columns=report["keys"])
                frame.attrs["dataflow_source_locations"] = locations
                yield frame
                rows, locations = [], []
        if rows:
            frame = pd.DataFrame(rows, columns=report["keys"])
            frame.attrs["dataflow_source_locations"] = locations
            yield frame
        if self._identity(path) != identity:
            raise ValueError("JSONL source changed during reading")

    def read(self, file_path):
        # Non-streaming Storage already materializes the input by contract.
        report = self.inspect(file_path)
        chunks = list(self.iter_chunks(file_path, 10000))
        if not chunks:
            return pd.DataFrame(columns=report["keys"])
        locations = [item for frame in chunks for item in frame.attrs["dataflow_source_locations"]]
        for frame in chunks:
            frame.attrs = {}
        result = pd.concat(chunks, ignore_index=True)
        result.attrs["dataflow_source_locations"] = locations
        return result
