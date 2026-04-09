# RayOrch Acceleration for DataFlow

`RayAcceleratedOperator` provides transparent data-parallel acceleration for
DataFlow's row-independent (map-style) operators.  Wrap any existing operator
to distribute inference across multiple GPUs — no changes to the operator
itself or the pipeline's `FileStorage` are required.

## Minimal Usage Example

```python
from dataflow.rayorch import RayAcceleratedOperator
from dataflow.operators.text_sft.eval.superfiltering_sample_evaluator import (
    SuperfilteringSampleEvaluator,
)
from dataflow.utils.storage import FileStorage

# 1. Wrap the operator — replicas × GPUs
scorer = RayAcceleratedOperator(
    SuperfilteringSampleEvaluator,   # any DataFlow OperatorABC
    replicas=4,                      # 4 parallel actors
    num_gpus_per_replica=1.0,        # 1 GPU each
).op_cls_init(device="cuda", max_length=512)   # original __init__ args

# 2. Use it exactly like the original operator
storage = FileStorage(
    first_entry_file_name="data/input.jsonl",
    cache_path="./cache",
    file_name_prefix="step",
    cache_type="jsonl",
)
scorer.run(
    storage=storage.step(),
    input_instruction_key="instruction",
    input_output_key="output",
)

# 3. Clean up when done (see shutdown notes below)
scorer.shutdown()
```

Actors are created **lazily** on the first `run()` call, so pipeline
compilation does not trigger model loading.

### About `shutdown()`

| Scenario | Must call? |
|----------|-----------|
| Single operator / script exits normally | **Optional**. Ray cleans up all actors on process exit. |
| Multiple `RayAcceleratedOperator` stages with GPUs | **Required**. Idle actors from earlier stages still hold GPU resource reservations — later stages will **hang indefinitely** waiting for resources. |
| `num_gpus_per_replica=0` (CPU-only) | **Optional**. CPU resources are abundant so no hang, but actors still consume memory (loaded model weights, etc.). |

> **Note**: When using `PipelineABC.compile()`, `_compiled_forward` **automatically** calls `shutdown()` after each stage completes — no manual cleanup is needed.

## Prerequisites

```bash
# Pick one:
# Re-install DataFlow (includes rayorch as a dependency)
pip install -e .

# Or install RayOrch separately
pip install rayorch==0.0.1
```

## Test Files

| File | Description |
|------|-------------|
| `test_compile_cpu.py` | **CI suite** (pytest, CPU): all 3 Pipeline types × compile × multi-op chains × ordering/content checks |
| `test_accelerated_op.py` | Dummy sleep operator — validates correctness & scheduling (CPU only) |
| `test_pipeline_compile.py` | Real operator (Superfiltering) compile-path integration test (GPU required) |
| `test_real_operators.py` | Real operator benchmark with argparse, using `FileStorage` externally |

## Quick Start

```bash
cd /path/to/DataFlow
export HF_ENDPOINT=https://hf-mirror.com

# Dummy operator test (CPU, ~30s)
python test/rayorch/test_accelerated_op.py

# Superfiltering benchmark — 4096 rows, 2/4/8 GPU parallel
python test/rayorch/test_real_operators.py --op superfiltering --rows 4096 --replicas 2 4 8

# Deita Quality — 256 rows, 2/4 GPU parallel
python test/rayorch/test_real_operators.py --op deita --rows 256 --replicas 2 4

# All operators
python test/rayorch/test_real_operators.py --op all --rows 1024

# Fractional GPU allocation (e.g. 0.5 GPU per replica)
python test/rayorch/test_real_operators.py --op superfiltering --rows 2048 --replicas 4 8 16 --gpus-per-replica 0.5

# Save results to a custom path
python test/rayorch/test_real_operators.py --op superfiltering --rows 4096 --save-json my_results.json
```

## CLI Arguments

```
--op {superfiltering,deita,all}   Operator to benchmark (default: superfiltering)
--rows N                          Number of Alpaca rows (default: 4096)
--replicas R [R ...]              Parallel replica counts (default: 2 4 8)
--gpus-per-replica G              GPUs per replica (default: 1.0)
--save-json PATH                  JSON output path (default: bench_results.json)
```

A serial baseline (1 GPU) is always included automatically.

## Output

For each operator the benchmark reports:
1. **Serial** — single-GPU wall time (baseline)
2. **Parallel xN** — N-replica wall time, split into cold start and warm compute
3. **Correctness** — row-by-row comparison against serial (rtol=1e-3)
4. **Speedup** — warm compute time relative to serial

Results are also saved as JSON (`bench_results.json`).

## Benchmark Results

> Measured on 8× NVIDIA A800-SXM4-80GB.

### SuperfilteringSampleEvaluator (gpt2, 124 M)

Dataset: `tatsu-lab/alpaca`, 4096 rows, outer storage = `FileStorage`

| Config | Warm Time | Speedup | Cold Start | Correct |
|--------|-----------|---------|------------|---------|
| Serial (1 GPU) | 66.41s | 1.0x | — | baseline |
| 2 GPUs | 36.48s | 1.8x | 12.1s | ✓ |
| 4 GPUs | 18.45s | 3.6x | 13.0s | ✓ |
| 8 GPUs | 10.80s | 6.1x | 18.4s | ✓ |

Superfiltering is based on GPT-2 (124 M) which runs at ~60 it/s per row.
Because per-row compute is light, Ray scheduling/serialization overhead is
relatively significant.  Heavier models (e.g. Deita 7 B) will show speedup
closer to linear.

### DeitaQualitySampleEvaluator (Llama-based, 7 B)

_Pending: requires `hkust-nlp/deita-quality-scorer` model (~14 GB) to be cached locally._

## Design Notes

- **Cold start**: The first `run()` call creates Ray actors and loads models (~12-20 s). Subsequent calls reuse warm actors.
- **Data volume vs speedup**: Larger datasets amortize Ray communication overhead, pushing speedup closer to linear. Recommend ≥1024 rows for benchmarking.
- **Storage separation**: The outer pipeline uses `FileStorage` (standard DataFlow). `InMemoryStorage` is only used internally by `_OpRunner` inside Ray actors — fully transparent to the caller.
- **Row-independent only**: Operators that require cross-row global state (e.g. semantic dedup with a full similarity matrix) should **not** use this wrapper.
