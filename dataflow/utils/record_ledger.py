"""Disk-backed outcomes and idempotent replay for independent JSONL records."""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path

from dataflow.utils.jsonl_reader import JSONLReadPolicy
from dataflow.utils.pipeline_checkpoint import atomic_json, canonical_digest, file_digest, state_lock


class RecordProcessingError(ValueError):
    """An expected sample failure. Do not wrap arbitrary programming errors."""

    def __init__(self, code, stage="processing", attempts=1):
        self.code, self.stage, self.attempts = code, stage, attempts
        super().__init__(code)


@dataclass(frozen=True)
class RecordResult:
    updates: dict
    attempts: int = 1


class RecordLedger:
    """One ledger per stage/input/configuration, used by a single writer.

    Only source references and generated updates are stored; inputs are reread
    from the unchanged source. A successful result can rebuild output after a
    crash without calling the model again. A crash during a model request can
    still repeat that request. JSONL parsing errors belong to the read policy.
    """

    def __init__(self, directory, source, config, read_policy=None, _source_inspection=None):
        if not isinstance(config, dict):
            raise ValueError("A JSON-serializable configuration declaration is required")
        self.directory = Path(directory).resolve()
        self.source = Path(source).resolve()
        if self.source.suffix.lower() != ".jsonl":
            raise ValueError("Record ledgers require a local JSONL input")
        self.config_digest = canonical_digest(config)
        self.source_stat = self._stat()
        if _source_inspection is not None and _source_inspection["source"] == str(self.source) and _source_inspection["stat"] == self.source_stat:
            digest, total = _source_inspection["digest"], _source_inspection["total"]
        else:
            digest = file_digest(self.source)
            if read_policy is not None:
                total = read_policy.inspect(self.source)["accepted"]
            else:
                total = 0
                blank_seen = False
                for row, location, code in JSONLReadPolicy._records(self.source):
                    if code == "blank_line":
                        blank_seen = True
                        continue
                    if code is not None:
                        raise ValueError(f"Invalid source at line {location['line']}; use a read policy first")
                    total += 1
                if total and blank_seen:
                    raise ValueError("Source contains blank lines; use a read policy first")
        if self._stat() != self.source_stat:
            raise ValueError("Ledger source changed during inspection")
        self.source_inspection = {"source": str(self.source), "stat": self.source_stat, "digest": digest, "total": total}
        self.identity = canonical_digest({"source": str(self.source), "sha256": digest, "config": self.config_digest})
        self.source_digest = digest
        self.directory.mkdir(parents=True, exist_ok=True)
        manifest = self.directory / "identity.json"
        if manifest.exists():
            try:
                existing = json.loads(manifest.read_text(encoding="utf-8"))
            except (ValueError, OSError) as exc:
                raise ValueError("Invalid record ledger identity") from exc
            if existing.get("version") != 1 or existing.get("identity") != self.identity:
                raise ValueError("Ledger input/configuration changed; use a new directory")
        else:
            if (self.directory / "outcomes.sqlite").exists():
                raise ValueError("Ledger database exists without an identity")
            atomic_json(manifest, {"version": 1, "identity": self.identity, "source": str(self.source), "sha256": digest, "config": self.config_digest})
        self.total = total
        self.read_policy = read_policy
        self.database = sqlite3.connect(self.directory / "outcomes.sqlite")
        self.database.execute("PRAGMA synchronous=FULL")
        self.database.execute("CREATE TABLE IF NOT EXISTS outcomes (line INTEGER PRIMARY KEY, status TEXT NOT NULL, stage TEXT, code TEXT, attempts INTEGER NOT NULL, result TEXT)")
        self.database.commit()

    def _stat(self):
        stat = self.source.stat()
        return stat.st_size, stat.st_mtime_ns, stat.st_ino

    def verify_source(self):
        if self._stat() != self.source_stat:
            raise ValueError("Ledger source changed while processing")

    def get(self, line):
        row = self.database.execute("SELECT status,stage,code,attempts,result FROM outcomes WHERE line=?", (line,)).fetchone()
        if row is None:
            return None
        return {"status": row[0], "stage": row[1], "code": row[2], "attempts": row[3], "result": json.loads(row[4]) if row[4] is not None else None}

    def record(self, line, status, result=None, stage=None, code=None, attempts=1):
        self.verify_source()
        if type(line) is not int or line < 1 or status not in {"success", "failed", "skipped"}:
            raise ValueError("Invalid record identity or terminal status")
        if type(attempts) is not int or attempts < 0:
            raise ValueError("attempts must be a nonnegative integer")
        if status == "success" and not isinstance(result, dict):
            raise ValueError("Successful results must be a dictionary of output updates")
        if status == "failed" and (not isinstance(code, str) or not code):
            raise ValueError("Failed records require an error code")
        encoded = json.dumps(result, ensure_ascii=False, allow_nan=False) if result is not None else None
        with self.database:
            self.database.execute(
                "INSERT INTO outcomes VALUES (?,?,?,?,?,?) ON CONFLICT(line) DO UPDATE SET status=excluded.status,stage=excluded.stage,code=excluded.code,attempts=outcomes.attempts+excluded.attempts,result=excluded.result",
                (line, status, stage, code, attempts, encoded),
            )

    def report(self):
        counts = {"success": 0, "failed": 0, "skipped": 0}
        counts.update(dict(self.database.execute("SELECT status,COUNT(*) FROM outcomes GROUP BY status")))
        processed = sum(counts.values())
        if processed > self.total:
            raise ValueError("Ledger contains more outcomes than source records")
        attempts = self.database.execute("SELECT COALESCE(SUM(attempts),0) FROM outcomes").fetchone()[0]
        return {"schema_version": 1, "identity": self.identity, "input": self.total, **counts,
                "pending": self.total - processed, "attempts": attempts, "complete": processed == self.total}

    def publish(self):
        atomic_json(self.directory / "report.json", self.report())
        self._write_jsonl(self.directory / "failures.jsonl", (
            {"schema_version": 1, "identity": self.identity, "source": str(self.source),
             "source_sha256": self.source_digest, "line": row[0], "stage": row[1], "code": row[2], "attempts": row[3],
             "resource_base": str(self.source.parent)}
            for row in self.database.execute("SELECT line,stage,code,attempts FROM outcomes WHERE status='failed' ORDER BY line")
        ))

    def _write_jsonl(self, destination, rows):
        destination = Path(destination).resolve()
        if destination in {self.source, self.directory / "identity.json", self.directory / "outcomes.sqlite", self.directory / ".lock"}:
            raise ValueError("Output must not overwrite source or ledger state")
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(prefix=".records-", dir=destination.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            self.verify_source()
            os.replace(name, destination)
        finally:
            if os.path.exists(name):
                os.unlink(name)

    def export_results(self, output_path):
        """Rebuild ordered successful output; repeated exports never append."""
        if Path(output_path).resolve().is_relative_to(self.directory):
            raise ValueError("Export results outside the ledger state directory")
        def rows():
            for row, location, code in JSONLReadPolicy._records(self.source):
                if code is not None:
                    if self.read_policy is None and not (code == "blank_line" and self.total == 0):
                        raise ValueError("Source is not valid JSONL")
                    continue
                outcome = self.get(location["line"])
                if outcome is not None and outcome["status"] == "success":
                    yield {**row, **outcome["result"]}
        self._write_jsonl(output_path, rows())

    def export_outcomes(self, output_path, output_key="generated_content", status_key="generation_outcome"):
        """Export every parsed source record, including failures, in source order.

        Status line numbers refer to this ledger's immutable source snapshot,
        never positions inferred from a filtered upstream dataset.
        """
        if Path(output_path).resolve().is_relative_to(self.directory):
            raise ValueError("Export outcomes outside the ledger state directory")
        if output_key == status_key:
            raise ValueError("Output and status fields must be distinct")
        def rows():
            for row, location, code in JSONLReadPolicy._records(self.source):
                if code is not None:
                    if self.read_policy is None and not (code == "blank_line" and self.total == 0):
                        raise ValueError("Source is not valid JSONL")
                    continue
                if output_key in row or status_key in row:
                    raise ValueError("Outcome fields must not overwrite source columns")
                outcome = self.get(location["line"])
                status = outcome["status"] if outcome else "pending"
                updates = outcome["result"] if status == "success" else {output_key: None}
                yield {**row, **updates, status_key: {
                    "status": status, "line": location["line"], "source": str(self.source),
                    "attempts": outcome["attempts"] if outcome else 0,
                    "stage": outcome["stage"] if outcome else None,
                    "code": outcome["code"] if outcome else None}}
        self._write_jsonl(output_path, rows())

    def process(self, processor, only_failed=False):
        """Process missing records or replay only recorded failures.

        processor(row) returns output updates. Relative assets remain based at
        source.parent; replay never relocates source records into another folder.
        Only RecordProcessingError is treated as an expected sample failure.
        """
        with state_lock(self.directory):
            for row, location, code in JSONLReadPolicy._records(self.source):
                if code is not None:
                    if self.read_policy is None and not (code == "blank_line" and self.total == 0):
                        raise ValueError("Source is not valid JSONL")
                    continue
                line = location["line"]
                old = self.get(line)
                if (only_failed and (old is None or old["status"] != "failed")) or (not only_failed and old is not None):
                    continue
                try:
                    result = processor(row)
                except RecordProcessingError as exc:
                    self.record(line, "failed", stage=exc.stage, code=exc.code, attempts=exc.attempts)
                else:
                    if isinstance(result, RecordResult):
                        self.record(line, "success", result=result.updates, attempts=result.attempts)
                    else:
                        self.record(line, "success", result=result)
            self.publish()

    def close(self):
        self.database.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
