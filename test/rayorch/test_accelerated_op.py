"""
Serial vs Parallel timing comparison for RayAcceleratedOperator.

A dummy operator that sleeps per row simulates per-sample model inference.
Pipeline-style test following DataFlow conventions.

Usage:
    python test/rayorch/test_accelerated_op.py
"""
from __future__ import annotations

import time

import pandas as pd

from dataflow.core.operator import OperatorABC
from dataflow.rayorch import RayAcceleratedOperator
from dataflow.rayorch.memory_storage import InMemoryStorage
from dataflow.utils.storage import DataFlowStorage


# ---------------------------------------------------------------------------
# Dummy operator
# ---------------------------------------------------------------------------
class DummySleepScorer(OperatorABC):
    def __init__(self, sleep_per_row: float = 0.1):
        super().__init__()
        self.sleep_per_row = sleep_per_row

    def run(
        self,
        storage: DataFlowStorage,
        input_key: str = "text",
        output_key: str = "score",
    ) -> None:
        df = storage.read("dataframe")
        scores = []
        for text in df[input_key]:
            time.sleep(self.sleep_per_row)
            scores.append(len(str(text)))
        df[output_key] = scores
        storage.write(df)


# ---------------------------------------------------------------------------
# Pipeline: serial baseline
# ---------------------------------------------------------------------------
class DummySleepSerialPipeline:
    def __init__(self, n_rows: int = 40, sleep_per_row: float = 0.1):
        self.storage = InMemoryStorage(
            pd.DataFrame({"text": [f"sample text {i}" for i in range(n_rows)]})
        )
        self.scorer = DummySleepScorer(sleep_per_row=sleep_per_row)

    def forward(self):
        self.scorer.run(
            storage=self.storage.step(),
            input_key="text",
            output_key="score",
        )


# ---------------------------------------------------------------------------
# Pipeline: Ray-accelerated parallel
# ---------------------------------------------------------------------------
class DummySleepRayPipeline:
    def __init__(self, n_rows: int = 40, sleep_per_row: float = 0.1, replicas: int = 4):
        self.storage = InMemoryStorage(
            pd.DataFrame({"text": [f"sample text {i}" for i in range(n_rows)]})
        )
        self.scorer = RayAcceleratedOperator(
            DummySleepScorer,
            replicas=replicas,
        ).op_cls_init(sleep_per_row=sleep_per_row)

    def forward(self):
        self.scorer.run(
            storage=self.storage.step(),
            input_key="text",
            output_key="score",
        )

    def shutdown(self):
        self.scorer.shutdown()


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def verify_result(storage: InMemoryStorage, n_rows: int, label: str) -> None:
    result = storage.result
    assert len(result) == n_rows, f"[{label}] expected {n_rows} rows, got {len(result)}"
    assert "score" in result.columns, f"[{label}] missing 'score' column"
    expected = [len(f"sample text {i}") for i in range(n_rows)]
    actual = result["score"].tolist()
    assert actual == expected, (
        f"[{label}] score mismatch (row order may be wrong)\n"
        f"  expected[:5] = {expected[:5]}\n"
        f"  actual[:5]   = {actual[:5]}"
    )
    print(f"  ✓ correctness check passed ({n_rows} rows, scores match)")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import ray

    N_ROWS = 40
    SLEEP_PER_ROW = 0.1
    REPLICAS = 4

    ray.init(ignore_reinit_error=True)

    # --- Serial ---
    print(f"\n{'=' * 60}")
    print(f"[Serial] {N_ROWS} rows, sleep={SLEEP_PER_ROW}s/row")
    print(f"{'=' * 60}")
    serial_pipe = DummySleepSerialPipeline(N_ROWS, SLEEP_PER_ROW)
    t0 = time.perf_counter()
    serial_pipe.forward()
    serial_time = time.perf_counter() - t0
    verify_result(serial_pipe.storage, N_ROWS, "Serial")
    print(f"  Time: {serial_time:.2f}s  (expected ≈{N_ROWS * SLEEP_PER_ROW:.2f}s)")

    # --- Parallel (cold) ---
    print(f"\n{'=' * 60}")
    print(f"[Parallel] {N_ROWS} rows, sleep={SLEEP_PER_ROW}s/row, replicas={REPLICAS}")
    print(f"{'=' * 60}")
    ray_pipe = DummySleepRayPipeline(N_ROWS, SLEEP_PER_ROW, REPLICAS)
    t_cold = time.perf_counter()
    ray_pipe.forward()
    cold_time = time.perf_counter() - t_cold
    print(f"  Cold run (includes actor init): {cold_time:.2f}s")

    # --- Parallel (warm) ---
    ray_pipe.storage = InMemoryStorage(
        pd.DataFrame({"text": [f"sample text {i}" for i in range(N_ROWS)]})
    )
    t0 = time.perf_counter()
    ray_pipe.forward()
    parallel_time = time.perf_counter() - t0
    verify_result(ray_pipe.storage, N_ROWS, "Parallel-warm")
    ideal = N_ROWS * SLEEP_PER_ROW / REPLICAS
    print(f"  Warm compute: {parallel_time:.2f}s  (ideal ≈{ideal:.2f}s)")

    ray_pipe.shutdown()

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print("[Summary]  (parallel = warm compute only)")
    print(f"  Serial:   {serial_time:.2f}s")
    print(f"  Parallel: {parallel_time:.2f}s")
    speedup = serial_time / parallel_time if parallel_time > 0 else float("inf")
    print(f"  Speedup:  {speedup:.1f}x  (ideal {REPLICAS}x)")
    print(f"{'=' * 60}")

    ray.shutdown()
