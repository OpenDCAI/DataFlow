"""
Test RayAcceleratedOperator under PipelineABC.compile() workflow.

Verifies that:
  1. compile() correctly discovers and wraps the operator via AutoOP
  2. Key validation passes (input_* kwargs match dataset columns)
  3. _compiled_forward() triggers lazy actor init and produces correct results
  4. Results match a serial (non-Ray) compiled pipeline

Uses SuperfilteringSampleEvaluator + Alpaca dataset + FileStorage.

Usage:
    python test/rayorch/test_pipeline_compile.py
    python test/rayorch/test_pipeline_compile.py --rows 256 --replicas 4
"""
from __future__ import annotations

import argparse
import os
import shutil
import tempfile
import time

import numpy as np
import pandas as pd

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from dataflow.pipeline.Pipeline import PipelineABC
from dataflow.rayorch import RayAcceleratedOperator
from dataflow.utils.storage import FileStorage


# ===================================================================
# Pipelines (extend PipelineABC — participate in compile())
# ===================================================================
class SerialCompiledPipeline(PipelineABC):
    """Standard single-GPU pipeline that goes through compile()."""

    def __init__(self, input_file: str, cache_path: str, op_cls, init_kwargs: dict):
        super().__init__()
        self.storage = FileStorage(
            first_entry_file_name=input_file,
            cache_path=cache_path,
            file_name_prefix="bench_step",
            cache_type="jsonl",
        )
        self.scorer = op_cls(**init_kwargs)

    def forward(self, **run_kwargs):
        self.scorer.run(
            storage=self.storage.step(),
            input_instruction_key="instruction",
            input_output_key="output",
        )


class RayCompiledPipeline(PipelineABC):
    """Ray-accelerated pipeline that goes through compile()."""

    def __init__(
        self,
        input_file: str,
        cache_path: str,
        op_cls,
        init_kwargs: dict,
        replicas: int = 4,
        num_gpus_per_replica: float = 1.0,
    ):
        super().__init__()
        self.storage = FileStorage(
            first_entry_file_name=input_file,
            cache_path=cache_path,
            file_name_prefix="bench_step",
            cache_type="jsonl",
        )
        self.scorer = RayAcceleratedOperator(
            op_cls,
            replicas=replicas,
            num_gpus_per_replica=num_gpus_per_replica,
        ).op_cls_init(**init_kwargs)

    def forward(self, **run_kwargs):
        self.scorer.run(
            storage=self.storage.step(),
            input_instruction_key="instruction",
            input_output_key="output",
        )

    def shutdown_actors(self):
        """Shut down Ray actors after compiled pipeline is done."""
        for op_node in self.op_nodes_list:
            if hasattr(op_node.op_obj, "shutdown"):
                op_node.op_obj.shutdown()


# ===================================================================
# Helpers
# ===================================================================
def load_alpaca_subset(n: int) -> pd.DataFrame:
    from datasets import load_dataset

    ds = load_dataset("tatsu-lab/alpaca", split=f"train[:{n}]")
    df = ds.to_pandas()
    df["input"] = df["input"].fillna("")
    return df


def save_to_jsonl(df: pd.DataFrame, path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_json(path, orient="records", lines=True, force_ascii=False)
    return path


def read_result(cache_path: str, output_key: str) -> list:
    result_file = os.path.join(cache_path, "bench_step_step1.jsonl")
    result_df = pd.read_json(result_file, lines=True)
    return result_df[output_key].tolist()


# ===================================================================
# Main
# ===================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Test RayAcceleratedOperator with pipeline.compile()")
    p.add_argument("--rows", type=int, default=128, help="Alpaca rows (default: 128)")
    p.add_argument("--replicas", type=int, default=4, help="Ray replicas (default: 4)")
    return p.parse_args()


if __name__ == "__main__":
    import ray
    from dataflow.operators.text_sft.eval.superfiltering_sample_evaluator import (
        SuperfilteringSampleEvaluator,
    )

    args = parse_args()
    N_ROWS = args.rows
    REPLICAS = args.replicas
    OUTPUT_KEY = "SuperfilteringScore"
    INIT_KWARGS = {"device": "cuda", "max_length": 512}

    ray.init(ignore_reinit_error=True)

    df = load_alpaca_subset(N_ROWS)
    print(f"Loaded {len(df)} rows from tatsu-lab/alpaca")

    tmp_root = tempfile.mkdtemp(prefix="rayorch_compile_test_")
    input_file = save_to_jsonl(df, os.path.join(tmp_root, "input.jsonl"))

    # ---------------------------------------------------------------
    # 1. Serial compiled pipeline
    # ---------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"[Serial] compile() + forward()  ({N_ROWS} rows)")
    print(f"{'=' * 60}")

    serial_cache = os.path.join(tmp_root, "serial")
    os.makedirs(serial_cache, exist_ok=True)

    serial_pipe = SerialCompiledPipeline(input_file, serial_cache, SuperfilteringSampleEvaluator, INIT_KWARGS)
    serial_pipe.compile()
    print(f"  compile() OK — {len(serial_pipe.op_runtimes)} op(s) registered")

    t0 = time.perf_counter()
    serial_pipe.forward()
    serial_time = time.perf_counter() - t0

    serial_scores = read_result(serial_cache, OUTPUT_KEY)
    print(f"  forward() OK — {serial_time:.2f}s")
    print(f"  Scores[:5]: {[round(s, 4) for s in serial_scores[:5]]}")

    # ---------------------------------------------------------------
    # 2. Ray compiled pipeline
    # ---------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"[Ray x{REPLICAS}] compile() + forward()  ({N_ROWS} rows)")
    print(f"{'=' * 60}")

    ray_cache = os.path.join(tmp_root, "ray")
    os.makedirs(ray_cache, exist_ok=True)

    ray_pipe = RayCompiledPipeline(
        input_file, ray_cache,
        SuperfilteringSampleEvaluator, INIT_KWARGS,
        replicas=REPLICAS, num_gpus_per_replica=1.0,
    )
    ray_pipe.compile()
    print(f"  compile() OK — {len(ray_pipe.op_runtimes)} op(s) registered")

    t0 = time.perf_counter()
    ray_pipe.forward()
    ray_time = time.perf_counter() - t0

    ray_scores = read_result(ray_cache, OUTPUT_KEY)
    print(f"  forward() OK — {ray_time:.2f}s (includes cold start)")
    print(f"  Scores[:5]: {[round(s, 4) for s in ray_scores[:5]]}")

    ray_pipe.shutdown_actors()

    # ---------------------------------------------------------------
    # 3. Correctness check
    # ---------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("[Correctness]")
    print(f"{'=' * 60}")

    assert len(ray_scores) == len(serial_scores), (
        f"Row count mismatch: serial={len(serial_scores)}, ray={len(ray_scores)}"
    )
    mismatches = sum(
        1 for s, p in zip(serial_scores, ray_scores)
        if not (s is None and p is None)
        and (s is None or p is None or not np.isclose(s, p, rtol=1e-3, equal_nan=True))
    )
    if mismatches:
        print(f"  ⚠ {mismatches}/{len(serial_scores)} mismatches")
    else:
        print(f"  ✓ All {len(serial_scores)} scores match (serial vs Ray x{REPLICAS})")

    print(f"\n  Serial:  {serial_time:.2f}s")
    print(f"  Ray x{REPLICAS}: {ray_time:.2f}s")
    if ray_time > 0:
        print(f"  Speedup: {serial_time / ray_time:.1f}x (includes cold start)")

    # ---------------------------------------------------------------
    # Cleanup
    # ---------------------------------------------------------------
    ray.shutdown()
    shutil.rmtree(tmp_root, ignore_errors=True)
    print(f"\nCleaned up {tmp_root}")
    print("PASSED" if mismatches == 0 else "FAILED")
