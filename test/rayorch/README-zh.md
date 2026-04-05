# RayOrch 加速 DataFlow 算子

`RayAcceleratedOperator` 为 DataFlow 的行独立（map-style）算子提供透明的数据并行加速。
只需包装已有算子即可将推理分发到多张 GPU——无需修改算子本身或 pipeline 的 `FileStorage`。

## 最简使用示例

```python
from dataflow.rayorch import RayAcceleratedOperator
from dataflow.operators.text_sft.eval.superfiltering_sample_evaluator import (
    SuperfilteringSampleEvaluator,
)
from dataflow.utils.storage import FileStorage

# 1. 包装算子 — replicas × GPU
scorer = RayAcceleratedOperator(
    SuperfilteringSampleEvaluator,   # 任意 DataFlow OperatorABC
    replicas=4,                      # 4 个并行 actor
    num_gpus_per_replica=1.0,        # 每个 actor 1 张 GPU
).op_cls_init(device="cuda", max_length=512)   # 原始 __init__ 参数

# 2. 和原始算子完全一样使用
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

# 3. 用完释放资源（见下方 shutdown 说明）
scorer.shutdown()
```

Actor 在首次 `run()` 时**懒加载**创建，pipeline compile 阶段不会触发模型加载。

### 关于 `shutdown()`

| 场景 | 是否必须调用 |
|------|-------------|
| 单算子 / 脚本结束退出 | **可选**。进程退出时 Ray 自动清理所有 actor。 |
| 多个 `RayAcceleratedOperator` 串联且占 GPU | **必须**。前一个 stage 的 actor 虽然空闲但仍持有 GPU 资源预留，后续 stage 创建新 actor 时会因资源不足而**永久阻塞**。 |
| `num_gpus_per_replica=0`（纯 CPU） | **可选**。CPU 资源充裕不会阻塞，但 actor 仍占用内存（已加载的模型权重等）。 |

> **Note**: 使用 `PipelineABC.compile()` 时，`_compiled_forward` 会在每个 stage 结束后**自动调用** `shutdown()`，无需手动释放。

## 环境要求

```bash
# 以下二选一：
# 在 DataFlow 目录下重新安装（已包含 rayorch 依赖）
pip install -e .

# 或仅额外安装 RayOrch
pip install rayorch==0.0.1
```

## 测试文件

| 文件 | 说明 |
|------|------|
| `test_compile_cpu.py` | **CI 用例**（pytest, CPU）：三种 Pipeline 类型 × compile × 多算子链 × 顺序/内容校验 |
| `test_accelerated_op.py` | Dummy sleep 算子——验证正确性和调度逻辑（仅需 CPU） |
| `test_pipeline_compile.py` | 真实算子（Superfiltering）compile 路径集成测试（需 GPU） |
| `test_real_operators.py` | 真实算子 benchmark，支持 argparse 传参，外层使用 `FileStorage` |

## 快速开始

```bash
cd /path/to/DataFlow
export HF_ENDPOINT=https://hf-mirror.com

# Dummy 算子测试（CPU 即可，约 30s）
python test/rayorch/test_accelerated_op.py

# Superfiltering benchmark — 4096 行，2/4/8 卡并行
python test/rayorch/test_real_operators.py --op superfiltering --rows 4096 --replicas 2 4 8

# Deita Quality — 256 行，2/4 卡并行
python test/rayorch/test_real_operators.py --op deita --rows 256 --replicas 2 4

# 跑所有算子
python test/rayorch/test_real_operators.py --op all --rows 1024

# 自定义 GPU 分配（如每个 replica 0.5 卡）
python test/rayorch/test_real_operators.py --op superfiltering --rows 2048 --replicas 4 8 16 --gpus-per-replica 0.5

# 结果保存到指定路径
python test/rayorch/test_real_operators.py --op superfiltering --rows 4096 --save-json my_results.json
```

## CLI 参数

```
--op {superfiltering,deita,all}   算子选择 (default: superfiltering)
--rows N                          Alpaca 数据行数 (default: 4096)
--replicas R [R ...]              并行副本数列表 (default: 2 4 8)
--gpus-per-replica G              每个副本的 GPU 数 (default: 1.0)
--save-json PATH                  JSON 结果保存路径 (default: bench_results.json)
```

串行 baseline（1 GPU）始终自动包含，无需手动指定。

## 输出说明

每个算子输出：
1. **Serial** — 单 GPU 串行时间（baseline）
2. **Parallel xN** — N 副本并行时间（分 cold start 和 warm compute）
3. **Correctness** — 并行结果与串行逐行对比（rtol=1e-3）
4. **Speedup** — warm compute 相对串行的加速倍数

结果同时保存为 JSON (`bench_results.json`)，方便后续分析。

## Benchmark 结果

> 以下结果在 8× NVIDIA A800-SXM4-80GB 上测得。

### SuperfilteringSampleEvaluator (gpt2, 124 M)

数据集：`tatsu-lab/alpaca`，4096 行，外层 storage = `FileStorage`

| 配置 | Warm 耗时 | 加速比 | Cold Start | 正确性 |
|------|-----------|--------|------------|--------|
| Serial (1 GPU) | 66.41s | 1.0x | — | baseline |
| 2 GPUs | 36.48s | 1.8x | 12.1s | ✓ match |
| 4 GPUs | 18.45s | 3.6x | 13.0s | ✓ match |
| 8 GPUs | 10.80s | 6.1x | 18.4s | ✓ match |

Superfiltering 基于 GPT-2 (124M)，单行推理极快（~60 it/s），Ray 调度/序列化开销
占比相对较高。更重的模型（如 Deita 7B）加速比会更接近线性。

### DeitaQualitySampleEvaluator (Llama-based, 7 B)

_待补充：需要先缓存 `hkust-nlp/deita-quality-scorer` 模型（~14 GB）_

## 设计说明

- **Cold start**：首次 `run()` 调用创建 Ray Actor 并加载模型（约 12-20s）。后续调用复用 warm actor。
- **数据量与加速比**：数据量越大，Ray 通信开销占比越低，加速比越接近线性。推荐 ≥1024 行做 benchmark。
- **Storage 分层**：外层 pipeline 使用 `FileStorage`（标准 DataFlow）。`InMemoryStorage` 仅在 Ray actor 内部的 `_OpRunner` 中使用——对调用方完全透明。
- **仅支持行独立算子**：需要跨行全局状态的算子（如基于完整相似度矩阵的语义去重）**不应**使用此 wrapper。
