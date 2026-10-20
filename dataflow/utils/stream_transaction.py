"""Crash-consistent, opt-in transactions for local streaming pipeline outputs.

The durable manifest is the sole authority for committed data and cursor state.
Legacy BatchedFileStorage paths do not use this module.
"""
from __future__ import annotations

import csv
import hashlib
import inspect
import json
import os
from pathlib import Path
import tempfile
import uuid

import pandas as pd


class TransactionError(RuntimeError):
    """Transactional streaming cannot safely continue."""

# Lifecycle failures after durable stage commits require a process restart.
# This guard is process-local; a new process may recover from the manifest.
_UNSAFE_LIFECYCLE_WORKSPACES = set()


def _digest(value):
    return hashlib.sha256(value).hexdigest()


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _hash_file(path, limit=None):
    sha = hashlib.sha256()
    with open(path, "rb") as handle:
        remaining = limit
        while True:
            n = 1024 * 1024 if remaining is None else min(1024 * 1024, remaining)
            if n <= 0:
                break
            chunk = handle.read(n)
            if not chunk:
                break
            sha.update(chunk)
            if remaining is not None:
                remaining -= len(chunk)
        if remaining is not None and remaining != 0:
            raise TransactionError("Committed output is shorter than manifest offset")
    return sha.hexdigest()


def _flush_directory(path):
    # Directory fsync is not portable to Windows; process-crash consistency is
    # supported there, but power-loss durability is not promised.
    if os.name != "posix":
        return
    fd = os.open(str(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _safe_value(value):
    if isinstance(value, str):
        return value.encode("utf-8", "replace").decode("utf-8")
    if isinstance(value, dict):
        return {k: _safe_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_safe_value(v) for v in value]
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return _safe_value(str(value))


class StreamTransactionCoordinator:
    """One coordinator, lock and manifest for all compiled streaming stages."""

    def __init__(self, nodes, batch_size):
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise TransactionError("Transactional streaming requires batch_size > 0")
        if not nodes:
            raise TransactionError("No streaming operators")
        self.nodes = nodes
        self.batch_size = batch_size
        first = nodes[0].storage
        self.root = Path(first.cache_path).expanduser().resolve()
        self.prefix = first.file_name_prefix
        self.manifest_path = self.root / (self.prefix + "_stream_transaction.json")
        self.legacy_path = self.root / (self.prefix + "_last_success_step.txt")
        self.lock_path = self.root / (self.prefix + "_stream_transaction.lock")
        self.registered = {}
        self.paths = []
        self._lock_handle = None
        self._lock_acquired = False
        self._hashers = {}
        self.active = None
        self.failed = False
        self.manifest = None

    def _preflight(self):
        from dataflow.utils.storage import StreamBatchedFileStorage
        if self.root in _UNSAFE_LIFECYCLE_WORKSPACES:
            raise TransactionError(
                "Previous stage lifecycle cleanup failed; restart the process before resuming"
            )
        # Async shutdown/cleanup needs an event-loop contract which the sync
        # compiled streaming pipeline cannot safely supply. Reject BEFORE work.
        seen_servings = set()
        previous_serving = None
        for node in self.nodes:
            shutdown = getattr(node.op_obj, "shutdown", None)
            if shutdown is not None and (
                inspect.iscoroutinefunction(shutdown)
                or inspect.isasyncgenfunction(shutdown)
            ):
                raise TransactionError("Async operator shutdown is unsupported in transactional mode")
            serving = node.llm_serving
            if serving is not None:
                cleanup = getattr(serving, "cleanup", None)
                if not callable(cleanup):
                    raise TransactionError("Serving requires callable synchronous cleanup()")
                if inspect.iscoroutinefunction(cleanup) or inspect.isasyncgenfunction(cleanup):
                    raise TransactionError(
                        "Async serving cleanup is unsupported in synchronous transactional mode"
                    )
            # A -> B -> A would reuse A after it has been cleaned up. Do not
            # assume cleaned serving instances are restartable.
            identity = id(serving) if serving is not None else None
            if identity != previous_serving and identity in seen_servings:
                raise TransactionError(
                    "Non-contiguous serving reuse (A -> B -> A) is unsupported"
                )
            if identity is not None:
                seen_servings.add(identity)
            previous_serving = identity
        first = self.nodes[0].storage
        source = Path(first.first_entry_file_name).expanduser().resolve()
        if not source.is_file() or source.suffix.lower() not in (".jsonl", ".csv"):
            raise TransactionError("Transactional mode requires a local CSV or JSONL input file")
        self.source = source
        self.kind = first.cache_type
        if self.kind not in ("jsonl", "csv"):
            raise TransactionError("Transactional mode supports only JSONL/CSV output")
        descriptors = []
        existing_inodes = set()
        if source.exists():
            st = source.stat()
            existing_inodes.add((st.st_dev, st.st_ino))
        for index, node in enumerate(self.nodes):
            storage = node.storage
            operator = node.op_obj
            if type(storage) is not StreamBatchedFileStorage:
                raise TransactionError("Transactional mode requires unmodified StreamBatchedFileStorage")
            if (storage.operator_step != index or storage.cache_type != self.kind
                    or Path(storage.cache_path).expanduser().resolve() != self.root
                    or Path(storage.first_entry_file_name).expanduser().resolve() != source
                    or storage.file_name_prefix != self.prefix):
                raise TransactionError("Inconsistent streaming stage storage")
            if not getattr(operator, "transaction_safe", False):
                raise TransactionError("Operator must explicitly declare transaction_safe")
            signature = getattr(operator, "transaction_signature", None)
            if not callable(signature):
                raise TransactionError("Operator must provide transaction_signature()")
            try:
                config = _json({"operator": type(operator).__module__ + "." + type(operator).__qualname__,
                                "signature": signature(), "arguments": node.kwargs})
            except (ValueError, TypeError, OverflowError) as exc:
                raise TransactionError("Unverifiable operator configuration") from exc
            dest = (self.root / (self.prefix + "_step" + str(index + 1) + "." + self.kind))
            # Prevent aliasing and symlink substitution of registered outputs.
            if dest.is_symlink() or dest.resolve() == source or dest.parent.resolve() != self.root:
                raise TransactionError("Unsafe or aliased output destination")
            if dest.exists():
                st = dest.stat()
                inode = (st.st_dev, st.st_ino)
                if inode in existing_inodes:
                    raise TransactionError("Output hardlink aliases input or another output")
                existing_inodes.add(inode)
            if dest in self.paths:
                raise TransactionError("Duplicate output destination")
            self.paths.append(dest)
            self.registered[id(storage)] = index
            descriptors.append(config)
        self.identity = _digest(_json({
            "source": str(source), "source_digest": _hash_file(source),
            "batch_size": self.batch_size, "format": self.kind,
            "outputs": list(map(str, self.paths)), "operators": descriptors,
        }).encode("utf-8"))

    def __enter__(self):
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock_handle = open(self.lock_path, "a+b")
        try:
            self._lock_handle.seek(0)
            self._lock_handle.write(b"1")
            self._lock_handle.flush()
            self._lock_handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self._lock_handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._lock_acquired = True
            # Validate output ownership after acquiring the single-writer lock.
            self._preflight()
            self._open_manifest()
            for node in self.nodes:
                node.storage._stream_transaction = self
            return self
        except BaseException:
            self._close_lock()
            raise

    def _close_lock(self):
        if self._lock_handle is not None:
            try:
                if self._lock_acquired:
                    self._unlock_handle()
            finally:
                self._lock_handle.close()
                self._lock_handle = None
                self._lock_acquired = False

    def _unlock_handle(self):
        self._lock_handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(self._lock_handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_UN)

    def __exit__(self, exc_type, exc, tb):
        for node in self.nodes:
            if getattr(node.storage, "_stream_transaction", None) is self:
                del node.storage._stream_transaction
        self._close_lock()

    def mark_lifecycle_unsafe(self):
        """Block further transactions on this workspace within this process."""
        _UNSAFE_LIFECYCLE_WORKSPACES.add(self.root)

    def _initial(self):
        return {
            "version": 1, "run_id": uuid.uuid4().hex, "identity": self.identity,
            "generation": 0, "stage": 0, "batch": 0,
            "outputs": [{"offset": 0, "rows": 0, "header": False,
                         "columns": None, "sha256": _digest(b"")}
                        for _ in self.paths],
        }

    def _sealed(self, state):
        payload = dict(state)
        payload["seal"] = _digest(_json(state).encode("utf-8"))
        return payload

    def _read_manifest(self):
        try:
            with open(self.manifest_path, "r", encoding="utf-8") as handle:
                envelope = json.load(handle)
            seal = envelope.pop("seal")
            if seal != _digest(_json(envelope).encode("utf-8")):
                raise TransactionError("Manifest integrity check failed")
            if (envelope["version"] != 1 or envelope["identity"] != self.identity
                    or not isinstance(envelope["generation"], int)
                    or envelope["generation"] < 0
                    or not 0 <= envelope["stage"] <= len(self.nodes)
                    or not isinstance(envelope["batch"], int) or envelope["batch"] < 0
                    or len(envelope["outputs"]) != len(self.paths)):
                raise TransactionError("Invalid manifest identity or cursor")
            if envelope["stage"] == len(self.nodes) and envelope["batch"] != 0:
                raise TransactionError("Invalid completed cursor")
            for o in envelope["outputs"]:
                if (type(o["offset"]) is not int or o["offset"] < 0
                        or type(o["rows"]) is not int or o["rows"] < 0
                        or not isinstance(o["header"], bool)
                        or not isinstance(o["sha256"], str)):
                    raise TransactionError("Invalid output metadata")
            return envelope
        except TransactionError:
            raise
        except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
            raise TransactionError("Missing or corrupted transactional manifest") from exc

    def _atomic_manifest(self, state):
        fd, name = tempfile.mkstemp(prefix=".stream_txn_", dir=self.root)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(_json(self._sealed(state)) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(name, self.manifest_path)
            _flush_directory(self.root)
        finally:
            if os.path.exists(name):
                os.unlink(name)

    def _open_manifest(self):
        if self.manifest_path.exists():
            self.manifest = self._read_manifest()
        else:
            if self.legacy_path.exists() or any(p.exists() for p in self.paths):
                raise TransactionError("Legacy checkpoint or unowned output; refuse transactional takeover")
            self.manifest = self._initial()
            self._atomic_manifest(self.manifest)
        self.reconcile()

    def _invalidate(self):
        for node in self.nodes:
            storage = node.storage
            storage._dataframe_buffer.clear()
            if hasattr(storage, "_cached_record_count"):
                del storage._cached_record_count
            storage._current_streaming_chunk = None
            if hasattr(storage, "record_count"):
                del storage.record_count

    def reconcile(self):
        state = self._read_manifest()
        recovered_hashers = {}
        for index, item in enumerate(state["outputs"]):
            path = self.paths[index]
            offset = item["offset"]
            if not path.exists():
                if offset != 0:
                    raise TransactionError("Committed output file is missing")
                if item["sha256"] != _digest(b"") or item["rows"] != 0 or item["header"]:
                    raise TransactionError("Missing output conflicts with manifest")
                recovered_hashers[index] = hashlib.sha256()
                continue
            if path.is_symlink():
                raise TransactionError("Output replaced by symlink")
            size = path.stat().st_size
            if size < offset:
                raise TransactionError("Committed output is shorter than manifest")
            sha = hashlib.sha256()
            with open(path, "rb") as output:
                remaining = offset
                while remaining:
                    chunk = output.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise TransactionError("Committed output prefix missing")
                    sha.update(chunk)
                    remaining -= len(chunk)
            if sha.hexdigest() != item["sha256"]:
                raise TransactionError("Committed output prefix has changed")
            recovered_hashers[index] = sha
            if size > offset:
                with open(path, "r+b") as output:
                    output.truncate(offset)
                    output.flush()
                    os.fsync(output.fileno())
        self.manifest = state
        self._hashers = recovered_hashers
        self.active = None
        self.failed = False
        self._invalidate()

    def _check(self, stage, batch):
        if self.failed or self.active is not None:
            raise TransactionError("Coordinator not ready for a new transaction")
        if self.manifest["stage"] != stage or self.manifest["batch"] != batch:
            raise TransactionError("Cursor mismatch")

    def begin(self, stage, batch):
        self._check(stage, batch)
        self.active = {"stage": stage, "batch": batch, "writes": 0,
                       "error": False, "pending": None, "hasher": None}

    def write(self, storage, data):
        tx = self.active
        if tx is None or self.failed:
            raise TransactionError("Write outside active transaction")
        try:
            if self.registered.get(id(storage)) != tx["stage"] or tx["writes"] != 0:
                raise TransactionError("Wrong storage stage or multiple storage.write calls")
            tx["writes"] = 1  # Count attempted writes so caught failures still abort.
            if isinstance(data, list):
                if not all(isinstance(item, dict) for item in data):
                    raise TransactionError("List output must contain dictionaries")
                frame = pd.DataFrame(_safe_value(data))
            elif isinstance(data, pd.DataFrame):
                frame = data.map(_safe_value)
            else:
                raise TransactionError("Unsupported transactional output type")
            index = tx["stage"]
            meta = self.manifest["outputs"][index]
            columns = list(frame.columns)
            if meta["columns"] is not None and columns and columns != meta["columns"]:
                raise TransactionError("CSV/JSONL schema changed during transactional run")
            if not columns:
                columns = meta["columns"]
            header = meta["header"]
            if len(frame) == 0:
                raw = b""
            elif self.kind == "jsonl":
                raw = frame.to_json(orient="records", lines=True, force_ascii=False).encode("utf-8")
            else:
                raw = frame.to_csv(index=False, header=not header).encode("utf-8")
                header = True
            path = self.paths[index]
            current_size = path.stat().st_size if path.exists() else 0
            if current_size != meta["offset"]:
                raise TransactionError("Output size diverged from committed offset")
            if raw:
                with open(path, "ab") as handle:
                    handle.write(raw)
                    handle.flush()
            pending_hasher = self._hashers[index].copy()
            pending_hasher.update(raw)
            tx["hasher"] = pending_hasher
            tx["pending"] = {"rows": meta["rows"] + len(frame), "header": header,
                             "columns": columns, "offset": meta["offset"] + len(raw),
                             "sha256": None}
            return str(path)
        except BaseException:
            tx["error"] = True
            raise

    def _commit(self, state):
        # Any error here, including a post-replace directory-sync error, can
        # have published either generation. Reload and reconcile, never roll
        # back using stale in-memory metadata.
        self.failed = True
        self.active = None
        try:
            self._atomic_manifest(state)
        except BaseException:
            self.reconcile()
            raise TransactionError("Manifest commit uncertain; state reconciled, retry from durable cursor")
        self.manifest = state
        self.failed = False
        # Invalidate stage-local and shared shallow-copy caches after commit.
        self._invalidate()

    def commit(self, stage, batch):
        tx = self.active
        if (tx is None or tx["stage"] != stage or tx["batch"] != batch
                or tx["writes"] != 1 or tx["error"] or tx["pending"] is None):
            raise TransactionError("Executed batch requires exactly one successful write")
        path = self.paths[stage]
        pending = dict(tx["pending"])
        if pending["offset"]:
            with open(path, "r+b") as handle:
                handle.flush()
                os.fsync(handle.fileno())
        if (path.stat().st_size if path.exists() else 0) != pending["offset"]:
            raise TransactionError("Output changed before commit")
        pending["sha256"] = tx["hasher"].hexdigest()
        nxt = json.loads(_json(self.manifest))
        nxt["generation"] += 1
        nxt["outputs"][stage] = pending
        nxt["batch"] = batch + 1
        self._commit(nxt)
        # The manifest now commits exactly the bytes represented by this hasher.
        # An indeterminate commit takes the exception/reconcile path instead.
        self._hashers[stage] = tx["hasher"]

    def transition(self, stage):
        self._check(stage, self.manifest["batch"])
        nxt = json.loads(_json(self.manifest))
        nxt["generation"] += 1
        nxt["stage"] = stage + 1
        nxt["batch"] = 0
        self._commit(nxt)

    def abort(self):
        if self.active is not None or self.failed:
            self.failed = True
            self.active = None
            self.reconcile()

    def input_rows(self, stage):
        if stage:
            return self.manifest["outputs"][stage - 1]["rows"]
        if self.source.stat().st_size == 0:
            return 0
        if self.source.suffix.lower() == ".jsonl":
            with open(self.source, "r", encoding="utf-8") as f:
                return sum(bool(line.strip()) for line in f)
        with open(self.source, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            next(reader, None)
            return sum(1 for _ in reader)
