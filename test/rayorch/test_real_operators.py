"""
Real-operator benchmark for RayAcceleratedOperator on the Alpaca dataset.

Outer pipeline uses FileStorage (read file → process → write file),
matching real DataFlow deployment.  InMemoryStorage is only used
internally by _OpRunner inside Ray actors — fully transparent here.

Supports SuperfilteringSampleEvaluator and DeitaQualitySampleEvaluator.
Compares serial (1 GPU) vs Ray-parallel (multi-GPU) execution, verifying:
  1. Correctness — parallel results match serial baseline
  2. Speedup   — approaches linear scaling as data volume grows

Usage:
    python test/rayorch/test_real_operators.py --help
    python test/rayorch/test_real_operators.py --op superfiltering --rows 4096
    python test/rayorch/test_real_operators.py --op superfiltering --rows 4096 --replicas 2 4 8
    python test/rayorch/test_real_operators.py --op deita --rows 256 --replicas 2 4
    python test/rayorch/test_real_operators.py --op all --rows 1024
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from dataflow.rayorch import RayAcceleratedOperator
from dataflow.utils.storage import FileStorage


# ===================================================================
# Dataset helper
# ===================================================================
def load_alpaca_subset(n: int = 4096) -> pd.DataFrame:
    from datasets import load_dataset

    ds = load_dataset("tatsu-lab/alpaca", split=f"train[:{n}]")
    df = ds.to_pandas()
    df["input"] = df["input"].fillna("")
    return df


def save_to_jsonl(df: pd.DataFrame, path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_json(path, orient="records", lines=True, force_ascii=False)
    return path


# ===================================================================
# Operator registry
# ===================================================================
OPERATORS: dict[str, dict] = {}


def _register_superfiltering():
    from dataflow.operators.text_sft.eval.superfiltering_sample_evaluator import (
        SuperfilteringSampleEvaluator,
    )

    OPERATORS["superfiltering"] = {
        "cls": SuperfilteringSampleEvaluator,
        "init_kwargs": {"device": "cuda", "max_length": 512},
        "run_kwargs": {"input_instruction_key": "instruction", "input_output_key": "output"},
        "output_key": "SuperfilteringScore",
        "label": "SuperfilteringSampleEvaluator (gpt2, 124 M)",
    }


def _register_deita():
    from dataflow.operators.text_sft.eval.deita_quality_sample_evaluator import (
        DeitaQualitySampleEvaluator,
    )

    OPERATORS["deita"] = {
        "cls": DeitaQualitySampleEvaluator,
        "init_kwargs": {"device": "cuda", "max_length": 512},
        "run_kwargs": {"input_instruction_key": "instruction", "input_output_key": "output"},
        "output_key": "DeitaQualityScore",
        "label": "DeitaQualitySampleEvaluator (Llama-based, 7 B)",
    }


_REGISTER_FNS = {
    "superfiltering": _register_superfiltering,
    "deita": _register_deita,
}


# ===================================================================
# Pipeline wrappers (DataFlow convention: __init__ + forward)
#   Outer storage = FileStorage (real file I/O)
#   Inner storage = InMemoryStorage (inside Ray actors, transparent)
# ===================================================================
class SerialPipeline:
    """Single-GPU serial execution — standard DataFlow pattern."""

    def __init__(self, input_file: str, cache_path: str, op_cls, init_kwargs: dict):
        self.storage = FileStorage(
            first_entry_file_name=input_file,
            cache_path=cache_path,
            file_name_prefix="bench_step",
            cache_type="jsonl",
        )
        self.scorer = op_cls(**init_kwargs)

    def forward(self, **run_kwargs):
        self.scorer.run(storage=self.storage.step(), **run_kwargs)


class RayPipeline:
    """Multi-GPU Ray-accelerated — drop-in replacement of the operator."""

    def __init__(
        self,
        input_file: str,
        cache_path: str,
        op_cls,
        init_kwargs: dict,
        replicas: int = 8,
        num_gpus_per_replica: float = 1.0,
    ):
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
        self.scorer.run(storage=self.storage.step(), **run_kwargs)

    def shutdown(self):
        self.scorer.shutdown()


# ===================================================================
# Helpers
# ===================================================================
def _read_result(cache_path: str, output_key: str) -> list:
    """Read the operator output from FileStorage step-1 cache."""
    result_file = os.path.join(cache_path, "bench_step_step1.jsonl")
    result_df = pd.read_json(result_file, lines=True)
    return result_df[output_key].tolist()


# ===================================================================
# Benchmark harness
# ===================================================================
def bench(
    label: str,
    op_cls,
    init_kwargs: dict,
    run_kwargs: dict,
    input_file: str,
    warmup_file: str,
    output_key: str,
    n_rows: int,
    tmp_root: str,
    replicas_list: list[int],
    num_gpus_per_replica: float = 1.0,
) -> dict:
    """Run serial + parallel benchmarks and return structured results."""
    print(f"\n{'#' * 70}")
    print(f"# {label}  ({n_rows} rows, FileStorage)")
    print(f"{'#' * 70}")

    results: dict[int, dict] = {}

    for replicas in replicas_list:
        tag = "Serial" if replicas == 1 else f"Parallel x{replicas}"
        print(f"\n--- [{tag}] ---")

        if replicas == 1:
            cache_dir = os.path.join(tmp_root, "serial")
            os.makedirs(cache_dir, exist_ok=True)

            pipe = SerialPipeline(input_file, cache_dir, op_cls, init_kwargs)
            t0 = time.perf_counter()
            pipe.forward(**run_kwargs)
            elapsed = time.perf_counter() - t0
            scores = _read_result(cache_dir, output_key)
            results[replicas] = {"time": elapsed, "scores": scores, "cold": 0.0}
        else:
            cold_cache = os.path.join(tmp_root, f"ray_x{replicas}_cold")
            warm_cache = os.path.join(tmp_root, f"ray_x{replicas}_warm")
            os.makedirs(cold_cache, exist_ok=True)
            os.makedirs(warm_cache, exist_ok=True)

            # Cold run — creates actors + loads model
            pipe = RayPipeline(
                warmup_file, cold_cache, op_cls, init_kwargs,
                replicas=replicas, num_gpus_per_replica=num_gpus_per_replica,
            )
            t_cold = time.perf_counter()
            pipe.forward(**run_kwargs)
            cold_time = time.perf_counter() - t_cold
            print(f"  Cold start: {cold_time:.1f}s")

            # Warm run — reuse actors, fresh FileStorage with full data
            pipe.storage = FileStorage(
                first_entry_file_name=input_file,
                cache_path=warm_cache,
                file_name_prefix="bench_step",
                cache_type="jsonl",
            )
            t0 = time.perf_counter()
            pipe.forward(**run_kwargs)
            elapsed = time.perf_counter() - t0
            scores = _read_result(warm_cache, output_key)
            results[replicas] = {"time": elapsed, "scores": scores, "cold": cold_time}
            pipe.shutdown()

        print(f"  Time: {elapsed:.2f}s")
        print(f"  Scores[:5]: {[round(s, 4) if s is not None else None for s in scores[:5]]}")

    # --- correctness ---
    serial_time = results[replicas_list[0]]["time"]
    serial_scores = results[replicas_list[0]]["scores"]

    for r in replicas_list[1:]:
        par_scores = results[r]["scores"]
        assert len(par_scores) == len(serial_scores), "Row count mismatch"
        mismatches = sum(
            1 for s, p in zip(serial_scores, par_scores)
            if not (s is None and p is None)
            and (s is None or p is None or not np.isclose(s, p, rtol=1e-3, equal_nan=True))
        )
        tag = f"serial vs x{r}"
        print(f"  {'⚠ ' + str(mismatches) + ' mismatches' if mismatches else '✓ scores match'} ({tag})")

    # --- speedup summary ---
    print(f"\n  {'Speedup summary':=^40}")
    summary_rows = []
    for r in replicas_list:
        t = results[r]["time"]
        speedup = serial_time / t if t > 0 else 0
        tag = "serial" if r == 1 else f"x{r}"
        cold = results[r]["cold"]
        cold_str = f"  (cold {cold:.1f}s)" if cold > 0 else ""
        print(f"    {tag:>10s}: {t:7.2f}s  ({speedup:.1f}x){cold_str}")
        match_ok = True
        if r > 1:
            par_scores = results[r]["scores"]
            mismatch_cnt = sum(
                1 for s, p in zip(serial_scores, par_scores)
                if not (s is None and p is None)
                and (s is None or p is None or not np.isclose(s, p, rtol=1e-3, equal_nan=True))
            )
            match_ok = mismatch_cnt == 0
        summary_rows.append({
            "replicas": r,
            "time_s": round(t, 2),
            "speedup": round(speedup, 1),
            "correct": match_ok,
        })

    return {
        "label": label,
        "rows": n_rows,
        "serial_time_s": round(serial_time, 2),
        "details": summary_rows,
    }


# ===================================================================
# Argparse
# ===================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RayAcceleratedOperator benchmark on Alpaca dataset (FileStorage)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test/rayorch/test_real_operators.py --op superfiltering --rows 4096
  python test/rayorch/test_real_operators.py --op superfiltering --rows 4096 --replicas 2 4 8
  python test/rayorch/test_real_operators.py --op deita --rows 256 --replicas 2 4
  python test/rayorch/test_real_operators.py --op all --rows 1024
        """,
    )
    p.add_argument(
        "--op", type=str, default="superfiltering",
        choices=["superfiltering", "deita", "all"],
        help="Operator to benchmark (default: superfiltering)",
    )
    p.add_argument(
        "--rows", type=int, default=4096,
        help="Number of Alpaca rows to use (default: 4096)",
    )
    p.add_argument(
        "--replicas", type=int, nargs="+", default=[2, 4, 8],
        help="Parallel replica counts to test (default: 2 4 8)",
    )
    p.add_argument(
        "--gpus-per-replica", type=float, default=1.0,
        help="GPUs per replica (default: 1.0)",
    )
    p.add_argument(
        "--save-json", type=str, default=None,
        help="Path to save JSON results (default: test/rayorch/bench_results.json)",
    )
    return p.parse_args()


# ===================================================================
# main
# ===================================================================
if __name__ == "__main__":
    import ray

    args = parse_args()

    ops_to_run = list(_REGISTER_FNS.keys()) if args.op == "all" else [args.op]
    for name in ops_to_run:
        _REGISTER_FNS[name]()

    ray.init(ignore_reinit_error=True)

    # Load data and write to temp JSONL files (input for FileStorage)
    df = load_alpaca_subset(args.rows)
    print(f"Loaded {len(df)} rows from tatsu-lab/alpaca")
    print(f"Columns: {list(df.columns)}")

    tmp_root = tempfile.mkdtemp(prefix="rayorch_bench_")
    input_file = save_to_jsonl(df, os.path.join(tmp_root, "alpaca_input.jsonl"))
    warmup_file = save_to_jsonl(df.iloc[:2], os.path.join(tmp_root, "alpaca_warmup.jsonl"))
    print(f"Temp dir: {tmp_root}")

    replicas_list = sorted(set([1] + args.replicas))
    all_results = []

    for name in ops_to_run:
        cfg = OPERATORS[name]
        op_tmp = os.path.join(tmp_root, name)
        os.makedirs(op_tmp, exist_ok=True)

        result = bench(
            label=cfg["label"],
            op_cls=cfg["cls"],
            init_kwargs=cfg["init_kwargs"],
            run_kwargs=cfg["run_kwargs"],
            input_file=input_file,
            warmup_file=warmup_file,
            output_key=cfg["output_key"],
            n_rows=len(df),
            tmp_root=op_tmp,
            replicas_list=replicas_list,
            num_gpus_per_replica=args.gpus_per_replica,
        )
        all_results.append(result)

    ray.shutdown()

    # Clean up temp files
    shutil.rmtree(tmp_root, ignore_errors=True)
    print(f"Cleaned up {tmp_root}")

    default_json = str(Path(__file__).parent / "bench_results.json")
    out_path = args.save_json or default_json
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")
