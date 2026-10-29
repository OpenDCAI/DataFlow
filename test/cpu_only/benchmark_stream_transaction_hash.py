"""P07.01 hashing microbenchmark.

Compare the legacy full-prefix checksum approach (O(batches * output size))
with the production incremental SHA-256 algorithm. This is a hash benchmark,
not an end-to-end pipeline throughput measurement.
Run: python test/cpu_only/benchmark_stream_transaction_hash.py
"""
import hashlib
from pathlib import Path
import tempfile
import time

from dataflow.utils.stream_transaction import _hash_file


def main():
    batch_count = 32
    batch_bytes = 512 * 1024
    chunk = (b'{"value":"x"}\n' * (batch_bytes // len(b'{"value":"x"}\n')))
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "sample.jsonl"
        old_hashes = []
        with path.open("wb") as out:
            before = time.perf_counter()
            for _ in range(batch_count):
                out.write(chunk)
                out.flush()
                old_hashes.append(_hash_file(path, out.tell()))
            elapsed_full = time.perf_counter() - before

        with path.open("wb") as out:
            sha = hashlib.sha256()
            before = time.perf_counter()
            for i in range(batch_count):
                out.write(chunk)
                out.flush()
                sha.update(chunk)
                assert sha.hexdigest() == old_hashes[i], "Incremental hash must match full prefix"
            elapsed_incremental = time.perf_counter() - before

    total_mb = len(chunk) * batch_count / (1024 * 1024)
    print(f"hash benchmark: {batch_count} batches, {total_mb:.1f} MiB output")
    print(f"full prefix: {elapsed_full:.4f}s")
    print(f"incremental: {elapsed_incremental:.4f}s")
    print(f"speedup: {elapsed_full / max(elapsed_incremental, 1e-10):.2f}x")


if __name__ == "__main__":
    main()
